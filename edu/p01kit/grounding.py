"""Kiểm chứng bám nguồn (grounding): mọi mã văn bản, giấy tờ, nơi nộp trong câu trả lời
phải xuất hiện NGUYÊN VĂN (sau chuẩn hóa) trong ngữ cảnh.

Khác bản cũ chỉ so 20 ký tự đầu (để lọt trích dẫn bịa có phần đầu giống thật),
ở đây so toàn bộ chuỗi đã chuẩn hóa.
"""

from __future__ import annotations

from .metrics import extract_json, norm


def is_grounded(span: str | None, context: str) -> bool:
    return bool(span) and norm(span) in norm(context)


def check_grounding(output_or_obj, context: str) -> dict:
    obj = output_or_obj if isinstance(output_or_obj, dict) else extract_json(output_or_obj)[0]
    if obj is None:
        return {"json_ok": False, "grounded": False, "hallucinated": []}
    if obj.get("is_refusal"):
        return {"json_ok": True, "grounded": obj.get("doc_code") in (None, ""), "hallucinated": []}
    spans = [obj.get("doc_code"), obj.get("submission_place")] + list(obj.get("required_documents") or [])
    bad = [s for s in spans if not is_grounded(s if isinstance(s, str) else None, context)]
    return {"json_ok": True, "grounded": not bad, "hallucinated": bad}


def hallucination_rate(outputs: list[str], rows: list[dict]) -> float:
    checks = [check_grounding(o, r["context"]) for o, r in zip(outputs, rows)]
    valid = [c for c in checks if c["json_ok"]]
    return sum(not c["grounded"] for c in valid) / max(1, len(valid))
