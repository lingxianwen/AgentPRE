"""Agent 4: Cross-Message Correlation Agent.

Operates purely on the graph store (no LLM). For each session it discovers:

  1. req_resp        : a client message and a server message whose `sequence`
                       field carries the same value (Modbus transaction_id,
                       S7 PDURef, ...) -> request/response pairing.
  2. length_coupling : whether a `length` field's value matches the number of
                       bytes that actually follow it (self-consistency).
  3. seq_increment   : monotonic increase of a `sequence` field across
                       consecutive same-direction messages (sliding-window /
                       counter semantics).

Results are written back into the `correlations` table.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.storage.graph_store import GraphStore  # noqa: E402
from src.agents.length_validation import (  # noqa: E402
    validate_length_field, check_tlv_nesting, verify_checksum_field,
    field_int_from_raw)


def _as_int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v, 0)
        except ValueError:
            return None
    return None


class CorrelationAgent:
    def __init__(self, store: GraphStore, length_profile=None, protocol_hint=None):
        self.store = store
        self.length_profile = length_profile
        self.protocol_hint = protocol_hint

    def run(self) -> dict:
        n_pairs = n_len_ok = n_len_bad = n_seq = 0
        for sess in self.store.sessions():
            sid = sess["session_id"]
            msgs = self.store.messages_of(sid, order_by="pkt_idx")
            n_pairs += self._pair_req_resp(msgs)
            ok, bad = self._check_length_coupling(msgs)
            n_len_ok += ok
            n_len_bad += bad
            n_seq += self._detect_seq_increment(msgs)
        self.store.commit()
        return {
            "req_resp_pairs": n_pairs,
            "length_consistent": n_len_ok,
            "length_inconsistent": n_len_bad,
            "seq_increment_runs": n_seq,
        }

    # ---- 1. request/response pairing by matching sequence field value ----
    def _pair_req_resp(self, msgs) -> int:
        # Group messages by their sequence-field value, then pair the first
        # c2s with the first s2c sharing that value.
        by_seq: dict = defaultdict(lambda: {"c2s": [], "s2c": []})
        for m in msgs:
            seq = None
            try:
                raw = bytes.fromhex(m.payload_hex) if m.payload_hex else b""
            except ValueError:
                raw = b""
            for f in m.fields:
                if f["role"] == "sequence":
                    seq = field_int_from_raw(f, raw)
                    break
            if seq is None:
                continue
            by_seq[seq][m.direction].append(m)

        n = 0
        for seq, sides in by_seq.items():
            reqs = sides["c2s"]
            resps = sides["s2c"]
            for req, resp in zip(reqs, resps):
                self.store.add_correlation(
                    "req_resp", req.msg_id, resp.msg_id,
                    {"seq_value": seq,
                     "req_opcode": req.opcode, "resp_opcode": resp.opcode},
                )
                n += 1
        return n

    # ---- 2. length-field self-consistency (multi-hypothesis) ----
    def _check_length_coupling(self, msgs) -> tuple[int, int]:
        """Validate every length field with the multi-hypothesis bank (handles
        TLV-nested length + segmented CRC/length, not just MBAP bytes-after),
        plus a TLV nesting-consistency check and CRC verification."""
        ok = bad = 0
        for m in msgs:
            length_fields = [f for f in m.fields if f["role"] == "length"]
            if not length_fields:
                continue
            raw = b""
            if getattr(m, "payload_hex", None):
                try:
                    raw = bytes.fromhex(m.payload_hex)
                except ValueError:
                    raw = b""

            # validate EACH length field; a message is "consistent" if every
            # length field matches at least one interpretation.
            per_field = []
            all_ok = True
            for lf in length_fields:
                if self.length_profile is not None:
                    from src.agents.frozen_length_profile import validate
                    cons, hyp, status = validate(self.length_profile, self.protocol_hint,
                                                 lf, m.fields, raw)
                else:
                    cons, hyp = validate_length_field(lf, m.fields, m.frame_len, raw)
                    status = 'accepted' if cons else 'rejected'
                per_field.append({"length_field": lf.get("name"),
                                  "declared": field_int_from_raw(lf, raw),
                                  "consistent": cons, "hypothesis": hyp,
                                  "validation_status": status})
                all_ok = all_ok and cons

            nesting = check_tlv_nesting(m.fields, m.frame_len, raw)

            # try to verify checksum fields as CRC-16 (segmented/trailing)
            crc_results = []
            if raw:
                for cf in (f for f in m.fields if f["role"] == "checksum"):
                    v, model, span = verify_checksum_field(cf, raw, m.fields)
                    crc_results.append({"field": cf.get("name"), "verified": v,
                                        "model": model, "span": span})

            self.store.add_correlation(
                "length_coupling", m.msg_id, None,
                {"fields": per_field, "consistent": all_ok,
                 "tlv_nesting": nesting, "crc": crc_results},
            )
            if all_ok:
                ok += 1
            else:
                bad += 1
        return ok, bad

    # ---- 3. monotonic sequence increment across same-direction messages ----
    def _detect_seq_increment(self, msgs) -> int:
        runs = 0
        for direction in ("c2s", "s2c"):
            seq_vals = []
            for m in msgs:
                if m.direction != direction:
                    continue
                try:
                    raw = bytes.fromhex(m.payload_hex) if m.payload_hex else b""
                except ValueError:
                    raw = b""
                for f in m.fields:
                    if f["role"] == "sequence":
                        v = field_int_from_raw(f, raw)
                        if v is not None:
                            seq_vals.append((m.msg_id, v))
                        break
            # Look for a run of >=3 strictly increasing values
            if len(seq_vals) >= 3:
                inc = sum(1 for i in range(1, len(seq_vals))
                          if seq_vals[i][1] > seq_vals[i - 1][1])
                if inc >= len(seq_vals) * 0.6:  # mostly increasing
                    self.store.add_correlation(
                        "seq_increment", seq_vals[0][0], seq_vals[-1][0],
                        {"direction": direction, "n": len(seq_vals),
                         "first": seq_vals[0][1], "last": seq_vals[-1][1]},
                    )
                    runs += 1
        return runs
