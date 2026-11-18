"""Bộ chấm điểm P01 — các chỉ số tất định (deterministic), không cần LLM-judge.

Ánh xạ sang ngưỡng chốt trước trong PROJECT-REQUIREMENTS.md:
- Citation Precision >= 0.90   -> `citation_precision`
- Refusal Accuracy  >= 0.90    -> `refusal_acc`
- Expired Detection = 100%     -> `expired_recall`
- Fairness Gap <= 10%          -> `fairness_gap`
"""

from __future__ import annotations

import json
import re
import unicodedata

from .data import ANSWER_KEYS
from .stats import bootstrap_ci, mean

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def strip_think(text: str) -> str:
    text = _THINK_RE.sub("", text)
    if "</think>" in text:  # khối think bị cắt ngang đầu chuỗi
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str) -> tuple[dict | None, str | None]:
    """Lấy JSON object đầu tiên trong output. Trả về (obj, lỗi)."""
    text = strip_think(text)
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        return None, "no_json"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None, "invalid_json"
                return (obj, None) if isinstance(obj, dict) else (None, "not_object")
    return None, "truncated"


def norm(s) -> str:
    s = unicodedata.normalize("NFC", str(s or "")).lower()
    s = re.sub(r"[^\w\s/.-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def schema_ok(obj: dict | None) -> bool:
    return obj is not None and tuple(obj.keys()) == ANSWER_KEYS


def docs_f1(pred, gold) -> float:
    p = {norm(x) for x in (pred or []) if isinstance(x, str)}
    g = {norm(x) for x in gold}
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def score_one(output: str, row: dict) -> dict:
    gold = row["gold"]
    obj, err = extract_json(output)
    s = {"id": row["id"], "style": row["style"], "slice": row["slice"], "json_ok": obj is not None,
         "schema_ok": schema_ok(obj), "error": err}
    o = obj or {}
    pred_refusal = bool(o.get("is_refusal"))
    pred_code = o.get("doc_code") or None
    s["refusal_ok"] = obj is not None and pred_refusal == gold["is_refusal"]
    s["cited"] = pred_code is not None and not pred_refusal
    s["citation_ok"] = s["cited"] and pred_code == gold["doc_code"]
    if gold["is_refusal"]:
        s["answer_ok"] = s["refusal_ok"]
        s["docs_f1"] = None
        s["place_ok"] = None
        s["expired_ok"] = None
        s["strict_ok"] = s["refusal_ok"] and pred_code is None
    else:
        s["answer_ok"] = s["citation_ok"]
        s["docs_f1"] = docs_f1(o.get("required_documents"), gold["required_documents"]) if obj else 0.0
        s["place_ok"] = norm(o.get("submission_place")) == norm(gold["submission_place"])
        s["expired_ok"] = bool(o.get("expired_warning")) == gold["expired_warning"]
        s["strict_ok"] = s["citation_ok"] and s["expired_ok"] and s["docs_f1"] == 1.0 and s["place_ok"]
    s["gold_expired"] = gold["expired_warning"]
    s["false_expired"] = (not gold["is_refusal"]) and (not gold["expired_warning"]) and bool(o.get("expired_warning"))
    return s


def _rate(scores: list[dict], key: str, where=lambda s: True) -> list[float]:
    return [float(bool(s[key])) for s in scores if where(s) and s[key] is not None]


def summarize(scores: list[dict], with_ci: bool = True) -> dict:
    """Tổng hợp chỉ số. Mỗi chỉ số kèm số mẫu `n_*` để biết nó đáng tin tới đâu."""
    out: dict = {"n": len(scores)}

    def put(name, vals):
        out[name] = round(mean(vals), 4) if vals else None
        out[f"n_{name}"] = len(vals)
        if with_ci and vals:
            lo, hi = bootstrap_ci(vals)
            out[f"{name}_ci"] = (round(lo, 4), round(hi, 4))

    put("json_valid", _rate(scores, "json_ok"))
    put("schema_ok", _rate(scores, "schema_ok"))
    put("refusal_acc", _rate(scores, "refusal_ok"))
    put("citation_precision", _rate(scores, "citation_ok", lambda s: s["cited"]))
    put("answer_acc", _rate(scores, "answer_ok"))
    put("expired_recall", _rate(scores, "expired_ok", lambda s: s["gold_expired"]))
    put("false_expired_rate", _rate(scores, "false_expired", lambda s: s["docs_f1"] is not None and not s["gold_expired"]))
    put("place_acc", _rate(scores, "place_ok"))
    put("strict_acc", _rate(scores, "strict_ok"))
    f1 = [s["docs_f1"] for s in scores if s["docs_f1"] is not None]
    out["docs_f1"] = round(mean(f1), 4) if f1 else None
    formal = _rate(scores, "answer_ok", lambda s: s["style"] == "formal")
    colloq = _rate(scores, "answer_ok", lambda s: s["style"] == "colloquial")
    out["acc_formal"] = round(mean(formal), 4) if formal else None
    out["acc_colloquial"] = round(mean(colloq), 4) if colloq else None
    out["fairness_gap"] = round(abs(mean(formal) - mean(colloq)), 4) if formal and colloq else None
    for sl in ("seen_doc", "unseen_doc", "ood"):
        v = _rate(scores, "answer_ok", lambda s, sl=sl: s["slice"] == sl)
        out[f"acc_{sl}"] = round(mean(v), 4) if v else None
    return out


THRESHOLDS = {"citation_precision": 0.90, "refusal_acc": 0.90, "expired_recall": 1.0}


def gate(summary: dict) -> dict:
    """Cổng chất lượng theo ngưỡng chốt trước: trả về {chỉ_số: (giá_trị, đạt?)}."""
    res = {k: (summary.get(k), summary.get(k) is not None and summary[k] >= v) for k, v in THRESHOLDS.items()}
    gap = summary.get("fairness_gap")
    res["fairness_gap"] = (gap, gap is not None and gap <= 0.10)
    return res


def format_summary(summary: dict, keys=("json_valid", "refusal_acc", "citation_precision", "answer_acc",
                                        "expired_recall", "docs_f1", "strict_acc", "fairness_gap")) -> str:
    lines = []
    for k in keys:
        v = summary.get(k)
        ci = summary.get(f"{k}_ci")
        n = summary.get(f"n_{k}")
        val = "—" if v is None else f"{v:.3f}"
        extra = f"  95%CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        extra += f"  (n={n})" if n is not None else ""
        lines.append(f"{k:<20} {val}{extra}")
    return "\n".join(lines)
