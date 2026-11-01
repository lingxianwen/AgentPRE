"""Field-mapping accuracy metrics.

We mirror LIVA's evaluation philosophy (Section IV-D, Fig. 8):
  - measure how well the agent reproduces the ground-truth field schema,
  - both at the field level (precision/recall over spans) and at the byte
    level (every byte gets a role label, compare against gold).

Definitions:
  field-exact         : (off, len) matches a gold field exactly
  field-role-correct  : field-exact AND role matches
  field-name-correct  : field-exact AND name matches (snake_case string equality)
  byte-role-acc       : per-byte fraction whose predicted role equals gold role
                        (a byte not covered by any predicted field is "unknown"
                        and counts as wrong)

We also report span-IoU which is robust to off-by-one boundary errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Iterable
from collections import Counter
import json


@dataclass
class SampleScore:
    n_gold: int = 0
    n_pred: int = 0
    n_field_exact: int = 0
    n_field_role: int = 0
    n_field_name: int = 0
    bytes_total: int = 0
    bytes_role_correct: int = 0
    iou_sum: float = 0.0       # sum of best-match IoU per gold field
    parse_ok: bool = True
    error: str = ""
    n_role_tuple_exact: int = 0
    message_span_role_exact: bool = False
    message_schema_exact: bool = False


def _validate_fields(fields: list[dict], frame_len: int) -> bool:
    """Reject obviously malformed prediction sets."""
    if not isinstance(fields, list):
        return False
    cursor_max = 0
    for f in fields:
        if not isinstance(f, dict):
            return False
        if "off" not in f or "len" not in f:
            return False
        if not isinstance(f["off"], int) or not isinstance(f["len"], int):
            return False
        if f["off"] < 0 or f["len"] <= 0:
            return False
        cursor_max = max(cursor_max, f["off"] + f["len"])
    return cursor_max <= frame_len + 8  # tolerate tiny overrun


def _byte_role_vector(fields: list[dict], frame_len: int) -> list[str | None]:
    """Map every byte index to a role label (None if not covered)."""
    v: list[str | None] = [None] * frame_len
    for f in fields:
        off = f.get("off", 0)
        ln = f.get("len", 0)
        role = f.get("role")
        for i in range(off, min(off + ln, frame_len)):
            if v[i] is None:
                v[i] = role
    return v


def _span_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    a0, a1 = a
    b0, b1 = b
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def score_sample(gold_fields: list[dict], pred_fields: list[dict], frame_len: int) -> SampleScore:
    s = SampleScore()
    if not _validate_fields(pred_fields, frame_len):
        s.parse_ok = False
        s.error = "invalid pred fields"
        s.n_gold = len(gold_fields)
        s.bytes_total = frame_len
        return s

    s.n_gold = len(gold_fields)
    s.n_pred = len(pred_fields)
    s.bytes_total = frame_len

    def tuples(fields, keys):
        return Counter(tuple(json.dumps(f.get(k), sort_keys=True, ensure_ascii=False)
                             for k in keys) for f in fields)
    role_keys = ("off", "len", "role")
    g_roles, p_roles = tuples(gold_fields, role_keys), tuples(pred_fields, role_keys)
    s.n_role_tuple_exact = sum((g_roles & p_roles).values())
    s.message_span_role_exact = g_roles == p_roles
    full_keys = ("off", "len", "role", "type", "name", "value")
    s.message_schema_exact = tuples(gold_fields, full_keys) == tuples(pred_fields, full_keys)

    gold_spans = {(f["off"], f["off"] + f["len"]): f for f in gold_fields}
    pred_spans = {(f["off"], f["off"] + f["len"]): f for f in pred_fields}

    for g_span, g in gold_spans.items():
        if g_span in pred_spans:
            p = pred_spans[g_span]
            s.n_field_exact += 1
            if str(p.get("role", "")) == str(g.get("role", "")):
                s.n_field_role += 1
            if str(p.get("name", "")) == str(g.get("name", "")):
                s.n_field_name += 1
        # best-IoU regardless of exact match
        best = max((_span_iou(g_span, ps) for ps in pred_spans), default=0.0)
        s.iou_sum += best

    g_vec = _byte_role_vector(gold_fields, frame_len)
    p_vec = _byte_role_vector(pred_fields, frame_len)
    for gv, pv in zip(g_vec, p_vec):
        if gv is not None and gv == pv:
            s.bytes_role_correct += 1

    return s


@dataclass
class AggregateReport:
    n_samples: int = 0
    n_parse_ok: int = 0
    sum_gold: int = 0
    sum_pred: int = 0
    sum_field_exact: int = 0
    sum_field_role: int = 0
    sum_field_name: int = 0
    sum_bytes_total: int = 0
    sum_bytes_role_correct: int = 0
    sum_iou: float = 0.0
    per_fc: dict = dc_field(default_factory=dict)
    errors: list[str] = dc_field(default_factory=list)
    sum_role_tuple_exact: int = 0
    n_message_span_role_exact: int = 0
    n_message_schema_exact: int = 0

    def add(self, score: SampleScore, fc: int):
        self.n_samples += 1
        if score.parse_ok:
            self.n_parse_ok += 1
        else:
            self.errors.append(score.error)
        self.sum_gold += score.n_gold
        self.sum_pred += score.n_pred
        self.sum_field_exact += score.n_field_exact
        self.sum_field_role += score.n_field_role
        self.sum_field_name += score.n_field_name
        self.sum_bytes_total += score.bytes_total
        self.sum_bytes_role_correct += score.bytes_role_correct
        self.sum_iou += score.iou_sum
        self.sum_role_tuple_exact += score.n_role_tuple_exact
        self.n_message_span_role_exact += int(score.parse_ok and score.message_span_role_exact)
        self.n_message_schema_exact += int(score.parse_ok and score.message_schema_exact)

        bucket = self.per_fc.setdefault(f"fc{fc:02x}", {"n": 0, "exact": 0, "role": 0, "gold": 0})
        bucket["n"] += 1
        bucket["exact"] += score.n_field_exact
        bucket["role"] += score.n_field_role
        bucket["gold"] += score.n_gold

    def summary(self) -> dict:
        precision = self.sum_field_exact / self.sum_pred if self.sum_pred else 0.0
        recall = self.sum_field_exact / self.sum_gold if self.sum_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        role_acc_given_exact = (
            self.sum_field_role / self.sum_field_exact if self.sum_field_exact else 0.0
        )
        name_acc_given_exact = (
            self.sum_field_name / self.sum_field_exact if self.sum_field_exact else 0.0
        )
        # End-to-end role accuracy at the field level: correct boundary AND correct role
        e2e_field_role = self.sum_field_role / self.sum_gold if self.sum_gold else 0.0
        byte_acc = self.sum_bytes_role_correct / self.sum_bytes_total if self.sum_bytes_total else 0.0
        mean_iou = self.sum_iou / self.sum_gold if self.sum_gold else 0.0
        role_p = self.sum_role_tuple_exact / self.sum_pred if self.sum_pred else 0.0
        role_r = self.sum_role_tuple_exact / self.sum_gold if self.sum_gold else 0.0
        role_f1 = 2 * role_p * role_r / (role_p + role_r) if role_p + role_r else 0.0
        return {
            "n_samples": self.n_samples,
            "parse_ok_rate": self.n_parse_ok / self.n_samples if self.n_samples else 0.0,
            "boundary_precision": precision,
            "boundary_recall": recall,
            # Historical artifact key retained for compatibility.  This is an
            # exact (offset, length) field-span F1, not boundary-position F1.
            "boundary_f1": f1,
            "exact_span_f1": f1,
            "role_acc_given_exact_boundary": role_acc_given_exact,
            "name_acc_given_exact_boundary": name_acc_given_exact,
            "end2end_field_role_recall": e2e_field_role,
            "exact_span_role_precision": role_p,
            "exact_span_role_recall": role_r,
            "exact_span_role_f1": role_f1,
            "message_span_role_exact_rate": self.n_message_span_role_exact / self.n_samples if self.n_samples else 0.0,
            "message_schema_exact_rate": self.n_message_schema_exact / self.n_samples if self.n_samples else 0.0,
            "byte_role_accuracy": byte_acc,
            "mean_best_iou_per_gold_field": mean_iou,
            "per_function_code": self.per_fc,
            "n_parse_errors": len(self.errors),
        }
