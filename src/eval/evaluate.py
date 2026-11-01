"""Evaluate the field-mapping agent against ground-truth eval.jsonl.

Two modes:
  - baseline: call the original (un-fine-tuned) Qwen3-32B-AWQ server
  - finetuned: same client, but point base_url at the fine-tuned deployment

The two modes use the SAME script — only the LLM endpoint differs, so we can
A/B compare lift from fine-tuning the same way LIVA does in Fig. 8.

Usage:
    python src/eval/evaluate.py \
        --config src/config.yaml \
        --eval out/finetune_modbus/eval.jsonl \
        --out out/eval_report_baseline.json \
        --limit 50           # 0 means evaluate all
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from src.llm_client import LLMConfig, Qwen3Client  # noqa: E402
from src.eval.metrics import AggregateReport, score_sample  # noqa: E402


_FRAME_LEN_RE = re.compile(r"Frame length:\s+(\d+)\s+bytes")
_FC_HINT_RE = re.compile(r"\b([0-9a-f]{2})\b")  # not strictly used


def _extract_frame_len(input_text: str) -> int:
    m = _FRAME_LEN_RE.search(input_text)
    return int(m.group(1)) if m else 0


def _extract_json(text: str) -> dict:
    """Robustly pull a JSON object out of a possibly noisy LLM reply.

    Falls back to ast.literal_eval for Python-ish outputs (e.g. the multi-
    protocol fine-tune occasionally emits `False` / `True` / `0x0564` literals
    instead of proper JSON strings or integers).
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    blob = text[start : end + 1]
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        pass
    # Try ast.literal_eval after replacing JSON literals it doesn't know.
    import ast
    munged = (
        blob.replace(": null", ": None")
            .replace(":null", ":None")
            .replace(": true", ": True")
            .replace(":true", ":True")
            .replace(": false", ": False")
            .replace(":false", ":False")
    )
    try:
        return ast.literal_eval(munged)
    except (ValueError, SyntaxError):
        # Last-ditch: normalize hex literals (0xABCD -> "0xABCD") and try json again.
        import re as _re
        fixed = _re.sub(r":\s*(0x[0-9a-fA-F]+)([,}\s])", r': "\1"\2', blob)
        fixed = _re.sub(r":\s*(True|False)([,}\s])", lambda m: ': ' + ('true' if m.group(1) == 'True' else 'false') + m.group(2), fixed)
        return json.loads(fixed)


def _fc_of_gold(gold_fields: list[dict]) -> int:
    for f in gold_fields:
        if f.get("role") == "opcode":
            v = f.get("value")
            if isinstance(v, int):
                return v & 0x7F
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "src" / "config.yaml"))
    ap.add_argument("--eval", default=str(ROOT / "out" / "finetune_modbus" / "eval.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "out" / "eval_report.json"))
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N samples (0 = all)")
    ap.add_argument("--save_predictions", action="store_true",
                    help="dump per-sample predictions next to the report")
    ap.add_argument("--base_url", default=None,
                    help="override the LLM base_url (e.g. point at fine-tuned endpoint)")
    args = ap.parse_args()

    cfg = LLMConfig.from_yaml(args.config)
    if args.base_url:
        cfg.base_url = args.base_url
    client = Qwen3Client(cfg)

    with open(args.eval, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.limit and args.limit < len(samples):
        samples = samples[: args.limit]

    report = AggregateReport()
    per_sample = []

    pbar = tqdm(samples, desc="eval", unit="smp")
    for i, smp in enumerate(pbar):
        instr, inp, gold_out = smp["instruction"], smp["input"], smp["output"]
        frame_len = _extract_frame_len(inp)
        gold_fields = json.loads(gold_out)["fields"]
        fc = _fc_of_gold(gold_fields)

        messages = [
            {"role": "system", "content": instr},
            {"role": "user", "content": inp},
        ]
        t0 = time.time()
        reply = ""
        pred_fields: list = []
        err = ""
        try:
            reply = client.chat(messages, max_tokens=cfg.max_tokens)
        except Exception as e:
            err = f"chat-error: {type(e).__name__}: {e}"
        else:
            try:
                pred_obj = _extract_json(reply)
                pred_fields = pred_obj.get("fields", [])
            except Exception as e:
                err = f"parse-error: {type(e).__name__}: {e}"
        dt = time.time() - t0

        score = score_sample(gold_fields, pred_fields, frame_len)
        if err:
            score.parse_ok = False
            score.error = err
        report.add(score, fc)

        if args.save_predictions:
            per_sample.append({
                "idx": i,
                "fc": fc,
                "frame_len": frame_len,
                "gold": gold_fields,
                "pred": pred_fields,
                # Always preserve the raw LLM reply so a later, more permissive
                # parser can re-score without re-running inference.
                "raw_reply": reply,
                "error": err,
                "latency_s": round(dt, 3),
                "score": {
                    "n_gold": score.n_gold,
                    "n_pred": score.n_pred,
                    "n_field_exact": score.n_field_exact,
                    "n_field_role": score.n_field_role,
                    "bytes_total": score.bytes_total,
                    "bytes_role_correct": score.bytes_role_correct,
                    "parse_ok": score.parse_ok,
                },
            })

        s = report.summary()
        pbar.set_postfix(
            ok=f"{report.n_parse_ok}/{report.n_samples}",
            F1=f"{s['boundary_f1']:.3f}",
            byte=f"{s['byte_role_accuracy']:.3f}",
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary = report.summary()
    summary["llm_base_url"] = cfg.base_url
    summary["model_label"] = cfg.model
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n=== Eval summary written to {args.out} ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif not isinstance(v, dict):
            print(f"  {k}: {v}")

    if args.save_predictions:
        pred_path = args.out.replace(".json", ".predictions.jsonl")
        with open(pred_path, "w", encoding="utf-8") as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  predictions: {pred_path}")


if __name__ == "__main__":
    main()
