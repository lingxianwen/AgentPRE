"""Post-processing for field-mapping predictions: tile the entire frame.

Many model outputs are valid JSON but only cover ~70% of the frame's bytes,
leaving gaps that count as missed fields in scoring. This module:

1. Sorts predicted fields by offset.
2. Drops fields that overlap an earlier field (greedy first-wins resolution).
3. Inserts `payload` filler fields wherever there's a gap (before any field,
   between two fields, or after the last field).
4. Truncates fields that extend past `frame_end`.

Result: fields contiguously tile [0, frame_end), suitable for downstream
byte-level scoring + boundary recall improvement at the cost of slightly
lower boundary precision (filler fields rarely match gold spans exactly).

Tested by running before/after on v3 (4k+600s) and v4 repaired predictions.
"""
from __future__ import annotations

from typing import Iterable


_FILLER = {
    "type": "bytes",
    "role": "payload",
    "name": "undissected_payload",
    "value": "",
}


def _valid(f: dict) -> bool:
    return (
        isinstance(f, dict)
        and isinstance(f.get("off"), int)
        and isinstance(f.get("len"), int)
        and f["off"] >= 0
        and f["len"] > 0
    )


def tile_fields(fields: list[dict], frame_end: int) -> list[dict]:
    """Return a field list that tiles [0, frame_end) contiguously.

    Strategy: greedy first-wins on overlap, fill all gaps with payload fillers.
    """
    if frame_end <= 0:
        return []

    # 1. Keep only valid fields, sort by offset (then length descending for stability)
    clean = sorted([f for f in fields if _valid(f)], key=lambda x: (x["off"], -x["len"]))

    out: list[dict] = []
    cursor = 0
    for f in clean:
        off, ln = f["off"], f["len"]
        if off + ln > frame_end:
            new_len = frame_end - off
            if new_len <= 0:
                continue
            f = {**f, "len": new_len}
            ln = new_len
        if off < cursor:
            # Overlaps a previously placed field — drop entirely
            continue
        if off > cursor:
            # Gap from cursor to off
            out.append({"off": cursor, "len": off - cursor, **_FILLER})
        out.append(f)
        cursor = off + ln
        if cursor >= frame_end:
            break
    if cursor < frame_end:
        out.append({"off": cursor, "len": frame_end - cursor, **_FILLER})
    return out


def tile_predictions_jsonl(predictions_path: str, eval_path: str, out_path: str) -> dict:
    """Read predictions.jsonl, gap-fill each pred, write a new predictions.jsonl.

    Returns a small stats dict.
    """
    import json
    import re

    with open(eval_path, "r", encoding="utf-8") as f:
        evals = [json.loads(l) for l in f if l.strip()]
    with open(predictions_path, "r", encoding="utf-8") as f:
        preds = [json.loads(l) for l in f if l.strip()]
    assert len(evals) == len(preds)

    n_modified = 0
    total_pad_bytes = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for ev, pr in zip(evals, preds):
            m = re.search(r"Frame length:\s+(\d+)", ev["input"])
            frame_len = int(m.group(1)) if m else 0
            old = pr.get("pred") or []
            new = tile_fields(old, frame_len)
            old_cov = sum(f["len"] for f in old if _valid(f))
            new_cov = sum(f["len"] for f in new)
            if new_cov != old_cov or len(new) != len(old):
                n_modified += 1
                total_pad_bytes += max(0, new_cov - old_cov)
            out_entry = dict(pr)
            out_entry["pred"] = new
            fout.write(json.dumps(out_entry, ensure_ascii=False) + "\n")

    return {
        "total": len(preds),
        "modified": n_modified,
        "extra_padding_bytes": total_pad_bytes,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out_predictions", required=True)
    args = ap.parse_args()
    stats = tile_predictions_jsonl(args.predictions, args.eval, args.out_predictions)
    print(f"modified {stats['modified']}/{stats['total']} predictions, "
          f"added {stats['extra_padding_bytes']} padding bytes total")
    print(f"wrote {args.out_predictions}")
