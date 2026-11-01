"""Agent 5: State-Machine / Specification Inference Agent.

Builds a protocol state machine from the per-session opcode sequences stored in
the graph. The skeleton is constructed *deterministically* (opcode-transition
graph), and an LLM is then asked to (a) name the states and (b) give a short
human-readable semantic label per opcode. Keeping the skeleton deterministic
means the output is reliable even if the fine-tuned model is weak at free-form
state-machine reasoning.

Borrowing LIVA's call-chain-merging idea: sessions whose opcode-sequence
signature is identical are merged into a single inference unit, so the LLM is
queried once per distinct protocol behavior rather than once per session.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm_client import LLMConfig, Qwen3Client, _extract_json  # noqa: E402
from src.storage.graph_store import GraphStore  # noqa: E402


def _hex(o):
    return "?" if o is None else f"0x{o:02x}"


class InferenceAgent:
    def __init__(self, store: GraphStore, cfg: LLMConfig | None = None):
        self.store = store
        self.client = Qwen3Client(cfg) if cfg else None
        self.cfg = cfg

    # ---- deterministic skeleton: opcode-transition Mealy machine ----
    def _session_signature(self, msgs) -> tuple:
        return tuple((m.direction, m.opcode) for m in msgs)

    def _build_transitions(self, msgs) -> list[tuple]:
        """Return list of (src_state, dst_state, condition) for one session.
        State = the opcode just observed (plus a START / END sentinel).
        """
        trans = []
        prev = "START"
        for m in msgs:
            cur = f"{_hex(m.opcode)}/{m.direction}"
            trans.append((prev, cur, f"opcode={_hex(m.opcode)} dir={m.direction}"))
            prev = cur
        trans.append((prev, "END", "session_close"))
        return trans

    def run(self, use_llm: bool = True) -> dict:
        # 1. Group sessions by opcode-sequence signature (call-chain merging).
        sig_to_sessions: dict = defaultdict(list)
        sig_to_msgs: dict = {}
        for sess in self.store.sessions():
            sid = sess["session_id"]
            msgs = self.store.messages_of(sid, order_by="pkt_idx")
            if not msgs:
                continue
            sig = self._session_signature(msgs)
            sig_to_sessions[sig].append(sid)
            sig_to_msgs.setdefault(sig, msgs)

        n_units = len(sig_to_sessions)
        n_states = n_trans = 0
        llm_calls = 0
        behaviors = []

        for sig, sids in sig_to_sessions.items():
            rep_msgs = sig_to_msgs[sig]
            rep_sid = sids[0]
            transitions = self._build_transitions(rep_msgs)

            # Collect distinct states
            states = set()
            for s, d, _ in transitions:
                states.add(s); states.add(d)

            # Optional LLM semantic naming of the opcode sequence
            semantic = None
            if use_llm and self.client is not None:
                semantic = self._llm_label(rep_msgs)
                llm_calls += 1

            # Persist
            for st in states:
                self.store.add_state(rep_sid, st, {"merged_sessions": len(sids)})
            for s, d, c in transitions:
                self.store.add_transition(rep_sid, s, d, c)
            n_states += len(states)
            n_trans += len(transitions)

            behaviors.append({
                "representative_session": rep_sid,
                "merged_session_count": len(sids),
                "opcode_sequence": [f"{_hex(o)}/{dr}" for dr, o in sig],
                "n_states": len(states),
                "n_transitions": len(transitions),
                "semantic": semantic,
            })

        self.store.commit()
        return {
            "behavior_units": n_units,
            "states": n_states,
            "transitions": n_trans,
            "llm_calls": llm_calls,
            "behaviors": behaviors,
        }

    def _llm_label(self, msgs) -> dict | None:
        """Ask the LLM for a short label of what this opcode sequence does."""
        seq = " -> ".join(f"{_hex(m.opcode)}({m.direction})" for m in msgs[:40])
        prompt = (
            "You are an industrial-protocol analyst. Given a sequence of message "
            "opcodes and directions (c2s=client->server, s2c=server->client) from "
            "one session, output a STRICT JSON object:\n"
            '{"behavior": "<short name>", "summary": "<one sentence>", '
            '"family": "<one of: request_response, setup_handshake, bulk_transfer, '
            'polling, control_command, unknown>"}\n'
            "Do not include any other text.\n\n"
            f"Opcode sequence: {seq}"
        )
        try:
            reply = self.client.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=256,
            )
            return _extract_json(reply)
        except Exception:
            return None
