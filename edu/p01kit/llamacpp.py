"""Client tối giản cho `llama-server` (llama.cpp): đo TTFT, tốc độ sinh và giải mã ràng buộc JSON Schema.

Dùng chuỗi prompt ĐÃ áp chat template bằng tokenizer HF (enable_thinking=False) để prompt lúc
deploy giống hệt lúc train — tránh lệch template nhúng trong file GGUF.
"""

from __future__ import annotations

import json
import time

from .data import ANSWER_KEYS

ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verification_reasoning": {"type": "string"},
        "is_refusal": {"type": "boolean"},
        "expired_warning": {"type": "boolean"},
        "doc_code": {"type": ["string", "null"]},
        "article": {"type": ["string", "null"]},
        "benefit": {"type": ["string", "null"]},
        "required_documents": {"type": "array", "items": {"type": "string"}},
        "submission_place": {"type": "string"},
    },
    "required": list(ANSWER_KEYS),
    "additionalProperties": False,
}


def complete(prompt: str, base_url: str = "http://127.0.0.1:8080", n_predict: int = 384,
             json_schema: dict | None = None, timeout: float = 300.0) -> dict:
    """Gọi /completion dạng stream; trả về text, TTFT (s), tổng thời gian và tốc độ token/s."""
    import requests

    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "stream": True, "cache_prompt": False}
    if json_schema is not None:
        body["json_schema"] = json_schema
    t0 = time.perf_counter()
    ttft, text, n_tok = None, [], 0
    with requests.post(f"{base_url}/completion", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            ev = json.loads(line[6:])
            if ev.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                text.append(ev["content"])
                n_tok += 1
            if ev.get("stop"):
                break
    total = time.perf_counter() - t0
    gen_time = total - (ttft or 0)
    return {"text": "".join(text), "ttft_s": ttft, "total_s": total,
            "tok_per_s": (n_tok - 1) / gen_time if n_tok > 1 and gen_time > 0 else None}


def calibration_text(rows: list[dict], tokenizer, limit: int | None = None) -> str:
    """Văn bản hiệu chuẩn imatrix: prompt + đáp án vàng của tập TRAIN (không bao giờ dùng test)."""
    from .prompts import completion_text, render_prompt

    parts = [render_prompt(tokenizer, r) + completion_text(r) + tokenizer.eos_token for r in rows[:limit]]
    return "\n\n".join(parts)
