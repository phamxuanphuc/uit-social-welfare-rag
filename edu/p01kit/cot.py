"""Dữ liệu cho chế độ Thinking của Qwen3: vết suy luận (reasoning trace) sinh bằng luật.

Qwen3 là mô hình "hybrid thinking": cùng một trọng số trả lời được ở 2 chế độ.
Khi fine-tune chỉ bằng dữ liệu non-thinking, mô hình dễ MẤT khả năng suy luận dài.
Cách giữ: trộn mẫu có `<think>...</think>` và mẫu không có trong cùng tập SFT.
"""

from __future__ import annotations

import json

from .data import Corpus, applicable, load_corpus
from .prompts import SYSTEM_PROMPT, build_messages


def reasoning_trace(row: dict, corpus: Corpus | None = None) -> str:
    """Duyệt TỪNG văn bản trong ngữ cảnh theo 3 câu hỏi: đúng chủ đề? đúng địa bàn? còn hiệu lực?"""
    corpus = corpus or load_corpus()
    region_code = row["region_code"]
    topic = row["topic"]
    lines = [f"Người hỏi chọn địa bàn {row['region']}.",
             (f"Câu hỏi thuộc chủ đề: {corpus.topics[topic]}." if topic else
              "Câu hỏi không thuộc chương trình an sinh nào trong danh mục.")]
    for i, pid in enumerate(row["context_ids"], 1):
        p = corpus.by_id(pid)
        same_topic = topic is not None and p.topic == topic
        region_ok = p.region in (region_code, "TQ")
        verdict = "dùng được" if applicable(p, topic, region_code) else "loại"
        lines.append(f"Văn bản {i} ({p.doc_code}): chủ đề {'khớp' if same_topic else 'không khớp'}; "
                     f"địa bàn {corpus.region_name(p.region)} -> {'hợp lệ' if region_ok else 'không hợp lệ'}; "
                     f"trạng thái {'còn hiệu lực' if p.is_active else 'hết hiệu lực'} -> {verdict}.")
    g = row["gold"]
    if g["is_refusal"]:
        lines.append("Không có văn bản dùng được -> phải từ chối, chuyển cán bộ.")
    else:
        lines.append(f"Chọn {g['doc_code']}." + (" Văn bản đã hết hiệu lực -> bật expired_warning." if g["expired_warning"] else ""))
    return "\n".join(lines)


def thinking_completion(row: dict, corpus: Corpus | None = None) -> str:
    """Nội dung lượt assistant ở chế độ thinking (chưa gồm eos)."""
    return f"<think>\n{reasoning_trace(row, corpus)}\n</think>\n\n{json.dumps(row['gold'], ensure_ascii=False)}"


def to_mixed_example(tokenizer, row: dict, think: bool, corpus: Corpus | None = None,
                     system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Mẫu prompt/completion cho SFT trộn 2 chế độ.

    - think=True : prompt kết thúc ở `<|im_start|>assistant\\n`, completion bắt đầu bằng `<think>`.
    - think=False: prompt đã chứa khối think rỗng (enable_thinking=False), completion là JSON.
    """
    msgs = build_messages(row, system_prompt)
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    completion = thinking_completion(row, corpus) if think else json.dumps(row["gold"], ensure_ascii=False)
    return {"prompt": prompt, "completion": completion + tokenizer.eos_token, "mode": "think" if think else "no_think"}
