"""Prompt chuẩn của P01 — MỘT nguồn sự thật dùng chung cho train, eval và deploy.

Lỗi kinh điển khi fine-tune: train một prompt, eval một prompt khác, deploy một prompt thứ ba.
Mọi buổi học đều gọi các hàm dưới đây để loại bỏ hoàn toàn lỗi lệch prompt.
"""

from __future__ import annotations

import json

from .data import ANSWER_KEYS

SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu quyền lợi an sinh xã hội cho người lao động phi chính thức.\n"
    "Chỉ dùng thông tin trong <context>. Trả về DUY NHẤT một JSON object, đúng 8 khóa theo thứ tự: "
    + ", ".join(ANSWER_KEYS) + ".\n"
    "Quy tắc:\n"
    "1. Chỉ dùng văn bản đúng chủ đề VÀ có địa bàn trùng địa bàn đã chọn hoặc 'Toàn quốc'.\n"
    "2. Nếu có cả bản còn hiệu lực và bản hết hiệu lực, dùng bản còn hiệu lực.\n"
    "3. Nếu văn bản phù hợp đã hết hiệu lực: expired_warning=true.\n"
    "4. Nếu không có văn bản phù hợp: is_refusal=true, doc_code/article/benefit=null, "
    "required_documents=[], submission_place hướng dẫn liên hệ cán bộ.\n"
    "5. Chép nguyên văn mã văn bản, giấy tờ và nơi nộp; không bịa thêm."
)


def user_message(row: dict) -> str:
    return (f"<context>\n{row['context']}\n</context>\n\n"
            f"Địa bàn đã chọn: {row['region']}\nCâu hỏi: {row['question']}")


def completion_text(row: dict) -> str:
    """Đáp án vàng dạng chuỗi JSON (giữ đúng thứ tự khóa, giữ dấu tiếng Việt)."""
    return json.dumps(row["gold"], ensure_ascii=False)


def build_messages(row: dict, system_prompt: str = SYSTEM_PROMPT, fewshot: list[dict] | None = None,
                   with_answer: bool = False) -> list[dict]:
    msgs = [{"role": "system", "content": system_prompt}]
    for ex in fewshot or []:
        msgs += [{"role": "user", "content": user_message(ex)},
                 {"role": "assistant", "content": completion_text(ex)}]
    msgs.append({"role": "user", "content": user_message(row)})
    if with_answer:
        msgs.append({"role": "assistant", "content": completion_text(row)})
    return msgs


def render_prompt(tokenizer, row: dict, system_prompt: str = SYSTEM_PROMPT, fewshot: list[dict] | None = None,
                  enable_thinking: bool = False) -> str:
    """Chuỗi prompt đã áp chat template Qwen3, kết thúc ở chỗ assistant bắt đầu trả lời.

    Với enable_thinking=False, template Qwen3 tự chèn khối rỗng `<think>\\n\\n</think>\\n\\n`.
    """
    return tokenizer.apply_chat_template(build_messages(row, system_prompt, fewshot), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)


def to_prompt_completion(tokenizer, row: dict, system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Định dạng `prompt`/`completion` (chuỗi) dùng cho TRL SFT/DPO/ORPO.

    Completion kết thúc bằng eos `<|im_end|>` để mô hình học cách DỪNG.
    """
    return {"prompt": render_prompt(tokenizer, row, system_prompt),
            "completion": completion_text(row) + tokenizer.eos_token}


def pick_fewshot(train_rows: list[dict], k: int = 2) -> list[dict]:
    """Chọn ví dụ few-shot đa dạng: 1 câu trả lời được + 1 câu phải từ chối (+ 1 hết hạn nếu k>=3)."""
    answerable = [r for r in train_rows if not r["is_unanswerable"] and not r["is_expired"]]
    refusal = [r for r in train_rows if r["is_unanswerable"]]
    expired = [r for r in train_rows if r["is_expired"]]
    pool = [answerable[0], refusal[0]] + ([expired[0]] if expired else [])
    return pool[:k]
