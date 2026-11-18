"""Khử trùng lặp gần (near-dedup) và kiểm toán rò rỉ train/test bằng cosine trên n-gram ký tự.

Vì sao n-gram KÝ TỰ thay vì TF-IDF theo từ? Tiếng Việt viết tách âm tiết, "xe ôm"/"xe-ôm"/"xeôm"
và lỗi gõ dấu rất phổ biến; n-gram ký tự (3–5) bền với các biến thể này mà không cần tách từ.
Độ phức tạp O(n²) — đủ cho vài nghìn mẫu; quy mô lớn hơn dùng MinHash-LSH (datasketch).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text.lower())
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> Counter:
    t = f" {normalize(text)} "
    return Counter(t[i:i + n] for n in range(n_min, n_max + 1) for i in range(len(t) - n + 1))


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb)


def near_dedup(texts: list[str], threshold: float = 0.85) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Giữ mẫu đầu tiên của mỗi cụm gần trùng. Trả về (chỉ số giữ lại, các cặp bị loại)."""
    vecs = [char_ngrams(t) for t in texts]
    kept: list[int] = []
    dropped = []
    for i, v in enumerate(vecs):
        best = max(((cosine(v, vecs[j]), j) for j in kept), default=(0.0, -1))
        if best[0] >= threshold:
            dropped.append((i, best[1], round(best[0], 3)))
        else:
            kept.append(i)
    return kept, dropped


def leakage_audit(train_texts: list[str], test_texts: list[str], threshold: float = 0.85) -> list[tuple[int, int, float]]:
    """Mọi cặp (test_i, train_j) có cosine >= threshold là nghi rò rỉ."""
    tv = [char_ngrams(t) for t in train_texts]
    leaks = []
    for i, t in enumerate(test_texts):
        v = char_ngrams(t)
        for j, w in enumerate(tv):
            s = cosine(v, w)
            if s >= threshold:
                leaks.append((i, j, round(s, 3)))
    return leaks
