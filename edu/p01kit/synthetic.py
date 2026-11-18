"""Nhà máy dữ liệu tổng hợp: dùng chính Qwen3-1.7B (chế độ thinking) làm "teacher" diễn đạt lại
câu hỏi theo nhiều giọng, rồi lọc nghiêm trước khi đưa vào train.

Tại sao không cần LLM lớn? Ta chỉ nhờ mô hình *viết lại câu hỏi*; đáp án vàng vẫn suy ra
bằng luật từ metadata -> không có ảo giác trong nhãn. Bộ lọc chặn các câu viết lại làm
đổi nghĩa (thêm địa danh, thêm số tiền, mất ý chính).
"""

from __future__ import annotations

import re

from .data import Corpus, load_corpus
from .guardrails import _fold

PERSONAS = [
    "một bác xe ôm lớn tuổi ở miền Nam, nói ngắn, dùng từ 'tui', 'hông', 'nhiêu'",
    "một cô bán vé số ít học, nói vòng vo, kể hoàn cảnh trước rồi mới hỏi",
    "một thanh niên phụ hồ gõ tin nhắn không dấu, viết tắt (ko, dc, j)",
    "một bà nội trợ miền Bắc, lịch sự, dùng 'ạ', 'vâng'",
]


def paraphrase_messages(question: str, persona: str, n: int = 3) -> list[dict]:
    return [
        {"role": "system", "content": "Bạn viết lại câu hỏi tiếng Việt, GIỮ NGUYÊN ý định và hoàn cảnh, "
                                      "KHÔNG thêm địa danh, con số, tên văn bản mới."},
        {"role": "user", "content": f"Viết lại câu hỏi sau theo giọng của {persona}. Viết đúng {n} câu, "
                                    f"mỗi câu một dòng, bắt đầu bằng '- '.\nCâu gốc: {question}"},
    ]


def parse_paraphrases(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(("- ", "* ")):
            q = line[2:].strip().strip('"')
            if 8 <= len(q) <= 220:
                out.append(q)
    return out


_NUM_RE = re.compile(r"\d[\d.,]*")


def filter_paraphrase(original: str, candidate: str, corpus: Corpus | None = None) -> tuple[bool, str]:
    """Trả về (giữ?, lý do loại). Luật giữ nghĩa tối thiểu, chạy được trên CPU."""
    corpus = corpus or load_corpus()
    f_orig, f_cand = _fold(original), _fold(candidate)
    if f_cand == f_orig:
        return False, "trùng nguyên văn"
    new_nums = set(_NUM_RE.findall(candidate)) - set(_NUM_RE.findall(original))
    if new_nums:
        return False, f"thêm con số {sorted(new_nums)}"
    for name in corpus.regions.values():
        ward = name.partition(" - ")[2]
        for alias in filter(None, {ward, re.sub(r'^(phường|quận)\s+', '', ward, flags=re.I)}):
            a = _fold(alias)
            if a and a in f_cand and a not in f_orig:
                return False, f"thêm địa danh '{alias}'"
    if re.search(r"(nq|qd|nd|tt)-\d+", f_cand):
        return False, "tự thêm mã văn bản"
    return True, ""
