"""Mã hóa dữ liệu SFT có che nhãn (loss masking) — bản tham chiếu để học viên đối chiếu.

Hai kiểu che:
1. `encode_prompt_completion`: 1 lượt hỏi–đáp. Nhãn = -100 cho toàn bộ prompt.
2. `encode_assistant_only`: hội thoại nhiều lượt / có tool. Chỉ giữ nhãn trên NỘI DUNG các
   lượt assistant (kể cả `<tool_call>` và token kết thúc `<|im_end|>`), che system/user/tool.
"""

from __future__ import annotations

IGNORE_INDEX = -100
ASSISTANT_OPEN = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def encode_prompt_completion(tokenizer, prompt: str, completion: str, max_length: int | None = None) -> dict:
    """completion phải đã kèm eos (`<|im_end|>`) nếu muốn mô hình học cách dừng."""
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if f_ids[:len(p_ids)] != p_ids:
        # BPE có thể gộp token qua ranh giới prompt|completion -> nhãn bị lệch 1 token.
        raise ValueError("Ranh giới prompt/completion không khớp ở mức token; hãy kết thúc prompt bằng ký tự xuống dòng.")
    labels = [IGNORE_INDEX] * len(p_ids) + f_ids[len(p_ids):]
    if max_length is not None and len(f_ids) > max_length:
        if len(p_ids) >= max_length:
            raise ValueError(f"Prompt dài {len(p_ids)} token >= max_length={max_length}: mẫu không còn nhãn để học.")
        f_ids, labels = f_ids[:max_length], labels[:max_length]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


def assistant_spans(text: str) -> list[tuple[int, int]]:
    """Khoảng ký tự [start, end) của nội dung mỗi lượt assistant, tính cả `<|im_end|>`."""
    spans, pos = [], 0
    while True:
        i = text.find(ASSISTANT_OPEN, pos)
        if i < 0:
            return spans
        start = i + len(ASSISTANT_OPEN)
        j = text.find(TURN_END, start)
        end = len(text) if j < 0 else j + len(TURN_END)
        spans.append((start, end))
        pos = end


def encode_assistant_only(tokenizer, text: str, max_length: int | None = None) -> dict:
    """Dùng offset_mapping của fast tokenizer để gán nhãn chỉ cho token nằm trong lượt assistant."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    spans = assistant_spans(text)
    labels = []
    for tid, (s, e) in zip(enc["input_ids"], enc["offset_mapping"]):
        inside = any(s >= a and e <= b for a, b in spans) and e > s
        labels.append(tid if inside else IGNORE_INDEX)
    ids = enc["input_ids"]
    if max_length is not None:
        ids, labels = ids[:max_length], labels[:max_length]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


class PadCollator:
    """Đệm phải (right padding) cho huấn luyện; nhãn đệm = -100 để không tính loss."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = 8):
        self.pad_id = pad_token_id
        self.mult = pad_to_multiple_of

    def __call__(self, features: list[dict]) -> dict:
        import torch

        n = max(len(f["input_ids"]) for f in features)
        if self.mult:
            n = ((n + self.mult - 1) // self.mult) * self.mult
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = n - len(f["input_ids"])
            batch["input_ids"].append(list(f["input_ids"]) + [self.pad_id] * pad)
            batch["attention_mask"].append(list(f["attention_mask"]) + [0] * pad)
            batch["labels"].append(list(f["labels"]) + [IGNORE_INDEX] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def label_stats(example: dict) -> dict:
    """Tỉ lệ token được học — dùng để 'soi' masking có đúng không."""
    n = len(example["labels"])
    k = sum(1 for x in example["labels"] if x != IGNORE_INDEX)
    return {"tokens": n, "trained_tokens": k, "trained_ratio": round(k / max(1, n), 3)}
