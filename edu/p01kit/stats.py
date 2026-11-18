"""Thống kê trung thực cho tập test nhỏ: khoảng tin cậy bootstrap và kiểm định McNemar.

Với 100 câu test, chênh lệch 3–5 điểm % rất dễ chỉ là nhiễu. Mọi so sánh "trước/sau"
trong lộ trình đều phải kèm khoảng tin cậy hoặc p-value.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Khoảng tin cậy (1-alpha) cho giá trị trung bình bằng bootstrap percentile."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """So sánh 2 mô hình trên CÙNG một tập câu hỏi (kiểm định McNemar chính xác, 2 phía).

    b = số câu A đúng / B sai; c = số câu A sai / B đúng. Chỉ các câu bất đồng mới mang thông tin.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("Hai mô hình phải được chấm trên cùng tập câu hỏi")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b": b, "c": c, "p_value": min(1.0, 2 * p)}
