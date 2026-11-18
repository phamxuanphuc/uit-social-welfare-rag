"""Mô phỏng traffic production cho Buổi 15 — dữ liệu GIẢ LẬP, tất định theo `seed`.

Ở công ty, giám sát chạy trên log thật. Ở lớp học ta chưa có người dùng, nên sinh request từ tập
val/test (để còn biết đáp án đúng) rồi "bẻ" phân phối theo những thay đổi hay gặp ngoài đời:
gõ không dấu, dồn về một địa bàn, hỏi ngoài phạm vi nhiều hơn, dán số điện thoại vào câu hỏi.
"""

from __future__ import annotations

import random
import unicodedata
from typing import Mapping, Sequence

from .guardrails import mask_pii
from .ops import ERROR_LABELS


def strip_diacritics(text: str) -> str:
    s = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", "".join(ch for ch in s if unicodedata.category(ch) != "Mn"))


def request_features(region: str | None, question: str) -> dict:
    """Đặc trưng giám sát lấy được NGAY từ request (không cần nhãn) — dùng để đo drift đầu vào."""
    n_words = len(question.split())
    return {"region": region or "(trống)", "no_diacritics": strip_diacritics(question) == question,
            "length": "ngan" if n_words < 8 else "vua" if n_words < 20 else "dai",
            "has_pii": mask_pii(question) != question}


def simulate_requests(rows: Sequence[dict], n: int, seed: int = 0, no_diacritics_share: float = 0.0,
                      ood_share: float | None = None, region_weights: Mapping[str, float] | None = None,
                      pii_share: float = 0.0) -> list[dict]:
    """Sinh `n` request {request_id, source_id, region, question}. `source_id` trỏ về dòng gốc để chấm điểm."""
    rng = random.Random(seed)
    ood = [r for r in rows if r["slice"] == "ood"]
    normal = [r for r in rows if r["slice"] != "ood"]
    out = []
    for i in range(n):
        if ood_share is not None and ood and rng.random() < ood_share:
            pool = ood
        elif ood_share is not None:
            pool = normal
        else:
            pool = list(rows)
        if region_weights:
            codes = list(region_weights)
            code = rng.choices(codes, weights=[region_weights[c] for c in codes])[0]
            pool = [r for r in pool if r["region_code"] == code] or pool
        r = rng.choice(pool)
        q = r["question"]
        if rng.random() < pii_share:
            q = f"SĐT tui 090 000 {rng.randrange(10000):04d}, " + q[0].lower() + q[1:]   # số giả, đúng định dạng di động
        if rng.random() < no_diacritics_share:
            q = strip_diacritics(q)
        out.append({"request_id": f"req-{seed}-{i:04d}", "source_id": r["id"], "region": r["region"], "question": q})
    return out


def simulate_feedback(trace_ids: Sequence[str], truth: Sequence[str], seed: int = 0, p_report_wrong: float = 0.5,
                      p_report_right: float = 0.03) -> list[dict]:
    """Người dân chỉ báo lỗi MỘT PHẦN số câu sai (và đôi khi báo nhầm câu đúng) → nhãn trễ, thưa, có nhiễu."""
    rng = random.Random(seed)
    out = []
    for t, label in zip(trace_ids, truth):
        p = p_report_right if label == "dung" else p_report_wrong
        if rng.random() < p:
            out.append({"trace_id": t, "reporter": "nguoi_dan", "true_label": label})
    return out


def simulate_reviews(reports: Sequence[dict], seed: int = 0, accuracy: float = 0.9) -> list[str]:
    """Một cán bộ xem lại báo cáo và gán nhãn theo hướng dẫn; gán đúng với xác suất `accuracy`."""
    rng = random.Random(seed)
    labels = []
    for r in reports:
        if rng.random() < accuracy:
            labels.append(r["true_label"])
        else:
            labels.append(rng.choice([x for x in ERROR_LABELS if x != r["true_label"]]))
    return labels
