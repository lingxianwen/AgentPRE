"""Local batched equivalent of evaluate.py for one base/LoRA adapter.

Scoring and JSON parsing are imported from the canonical evaluator.  Generation
is greedy as in the paper server; batching changes throughput only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--resume", action="store_true",
                    help="resume from a sequential partial predictions JSONL")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from peft import PeftModel
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              StoppingCriteria, StoppingCriteriaList)
    from src.eval.evaluate import _extract_frame_len, _extract_json, _fc_of_gold
    from src.eval.metrics import AggregateReport, score_sample

    samples = [json.loads(x) for x in Path(args.eval).read_text(encoding="utf-8").splitlines()]
    if args.limit:
        samples = samples[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out.with_name(out.stem + ".predictions.jsonl")
    if args.resume and out.exists() and pred_path.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
            saved_count = sum(1 for line in pred_path.read_text(encoding="utf-8").splitlines()
                              if line.strip())
        except (OSError, json.JSONDecodeError):
            previous, saved_count = {}, -1
        if previous.get("complete") is True and saved_count == len(samples):
            print(f"already complete {saved_count}/{len(samples)}: {out}", flush=True)
            return
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, quantization_config=quant, device_map="auto", trust_remote_code=True,
        attn_implementation="eager")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    class CompleteJsonBatch(StoppingCriteria):
        """Stop once every batch item has emitted one balanced JSON object.

        Some checkpoints continue with whitespace or a second answer after a
        complete object.  The scorer consumes only the first object, so ending
        at that exact boundary preserves the evaluated prediction while
        avoiding pathological generation to ``max_new_tokens``.
        """

        def __init__(self, prompt_tokens: int):
            self.prompt_tokens = prompt_tokens

        @staticmethod
        def complete(text: str) -> bool:
            start = text.find("{")
            if start < 0:
                return False
            depth, quoted, escaped = 0, False, False
            for ch in text[start:]:
                if quoted:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        quoted = False
                    continue
                if ch == '"':
                    quoted = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return True
            return False

        def __call__(self, input_ids, scores, **kwargs):
            done = []
            for seq in input_ids:
                # Avoid quadratic full-prefix decoding on ordinary tokens.
                tail = tokenizer.decode(seq[-2:], skip_special_tokens=True)
                done.append("}" in tail and self.complete(
                    tokenizer.decode(seq[self.prompt_tokens:], skip_special_tokens=True)))
            return torch.tensor(done, dtype=torch.bool, device=input_ids.device)

    report, per_sample = AggregateReport(), []
    if args.resume and pred_path.exists():
        saved = [json.loads(x) for x in pred_path.read_text(encoding="utf-8").splitlines()]
        if len(saved) > len(samples) or any(row.get("idx") != i for i, row in enumerate(saved)):
            raise RuntimeError(f"cannot resume non-sequential predictions: {pred_path}")
        for row in saved:
            score = score_sample(row["gold"], row["pred"], row["frame_len"])
            if not row.get("score", {}).get("parse_ok", True):
                score.parse_ok = False
                score.error = row.get("error") or "parse-error in resumed prediction"
            report.add(score, int(row["fc"]))
        per_sample.extend(saved)
        print(f"resumed {len(per_sample)}/{len(samples)} from {pred_path}", flush=True)

    for start in range(len(per_sample), len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        prompts = []
        for row in batch:
            messages = [{"role": "system", "content": row["instruction"]},
                        {"role": "user", "content": row["input"]}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
            prompts.append(prompt)
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=args.max_new_tokens, do_sample=False, repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
                stopping_criteria=StoppingCriteriaList([
                    CompleteJsonBatch(inputs.input_ids.shape[1])
                ]))
        torch.cuda.synchronize()
        batch_s = time.time() - t0
        replies = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)
        for local_i, (row, reply) in enumerate(zip(batch, replies)):
            frame_len = _extract_frame_len(row["input"])
            gold = json.loads(row["output"])["fields"]
            fc = _fc_of_gold(gold)
            pred, err = [], ""
            try:
                pred = _extract_json(reply).get("fields", [])
            except Exception as exc:
                err = f"parse-error: {type(exc).__name__}: {exc}"
            score = score_sample(gold, pred, frame_len)
            if err:
                score.parse_ok, score.error = False, err
            report.add(score, fc)
            per_sample.append({
                "idx": start + local_i, "fc": fc, "frame_len": frame_len,
                "gold": gold, "pred": pred, "raw_reply": reply, "error": err,
                "latency_s": round(batch_s / len(batch), 3),
                "input_tokens": int(inputs.input_ids.shape[1]),
                "output_tokens": int(outputs.shape[1] - inputs.input_ids.shape[1]),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "score": {"n_gold": score.n_gold, "n_pred": score.n_pred,
                          "n_field_exact": score.n_field_exact,
                          "n_field_role": score.n_field_role,
                          "bytes_total": score.bytes_total,
                          "bytes_role_correct": score.bytes_role_correct,
                          "parse_ok": score.parse_ok},
            })
        summary = report.summary()
        snapshot = dict(summary)
        snapshot.update({"model_path": args.base, "adapter_path": args.adapter or None,
                         "batch_size": args.batch_size,
                         "max_new_tokens": args.max_new_tokens,
                         "decoding": "greedy", "complete": len(per_sample) == len(samples)})
        out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        pred_path.write_text("".join(json.dumps(x) + "\n" for x in per_sample),
                             encoding="utf-8")
        print(f"{len(per_sample)}/{len(samples)} parse={summary['parse_ok_rate']:.3f} "
              f"spanF1={summary.get('exact_span_f1', summary['boundary_f1']):.3f} "
              f"E2E={summary['end2end_field_role_recall']:.3f} "
              f"batch={batch_s:.1f}s",
              flush=True)

    summary = report.summary()
    summary.update({"model_path": args.base, "adapter_path": args.adapter or None,
                    "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
                    "decoding": "greedy", "complete": True})
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pred_path.write_text("".join(json.dumps(x) + "\n" for x in per_sample), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
