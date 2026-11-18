"""3 hàm thưởng kiểm chứng được (verifiable rewards) cho GRPO — tương thích TRL GRPOTrainer.

Chữ ký TRL: f(completions, **kwargs) -> list[float]; mỗi cột của dataset được truyền vào kwargs
dưới dạng list cùng độ dài. Ta lưu đáp án vàng ở cột `gold` (chuỗi JSON).
`completions` là list[str] (prompt dạng chuỗi) hoặc list[list[message]] (prompt dạng hội thoại).
"""

from __future__ import annotations

import json

from .metrics import extract_json, schema_ok


def _text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return c[-1].get("content", "")
    return str(c)


def _gold(g) -> dict:
    return json.loads(g) if isinstance(g, str) else g


def reward_format(completions, **kwargs) -> list[float]:
    """+1 đúng schema 8 khóa đúng thứ tự; 0 JSON hợp lệ nhưng sai schema; -1 không phải JSON."""
    out = []
    for c in completions:
        obj, _ = extract_json(_text(c))
        out.append(1.0 if schema_ok(obj) else 0.0 if obj is not None else -1.0)
    return out


def reward_citation(completions, gold, **kwargs) -> list[float]:
    """+1 quyết định trả lời/từ chối đúng VÀ mã văn bản đúng; -1 nếu sai hoặc bịa."""
    out = []
    for c, g in zip(completions, gold):
        g = _gold(g)
        obj, _ = extract_json(_text(c))
        if obj is None:
            out.append(-1.0)
            continue
        if bool(obj.get("is_refusal")) != g["is_refusal"]:
            out.append(-1.0)
        elif g["is_refusal"]:
            out.append(1.0 if not obj.get("doc_code") else -0.5)
        else:
            out.append(1.0 if obj.get("doc_code") == g["doc_code"] else -1.0)
    return out


def reward_expired(completions, gold, **kwargs) -> list[float]:
    """Thưởng bất đối xứng: bỏ sót văn bản hết hạn (False Negative) bị phạt GẤP ĐÔI báo nhầm.

    gold hết hạn:  bật cảnh báo +1 | bỏ sót -2
    gold còn hạn:  không bật +0.5  | báo nhầm -1
    gold từ chối:  0 (không liên quan)
    """
    out = []
    for c, g in zip(completions, gold):
        g = _gold(g)
        obj, _ = extract_json(_text(c))
        if g["is_refusal"]:
            out.append(0.0)
            continue
        warned = bool((obj or {}).get("expired_warning"))
        if g["expired_warning"]:
            out.append(1.0 if warned else -2.0)
        else:
            out.append(0.5 if not warned else -1.0)
    return out


REWARD_FUNCS = [reward_format, reward_citation, reward_expired]
