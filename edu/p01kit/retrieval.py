"""Tầng truy xuất: BM25 thuần Python (baseline bắt buộc theo đề tài) + chỉ số Recall@k / MRR.

Lọc kép theo địa bàn trước khi xếp hạng: `region IN [địa_bàn_đã_chọn, "Toàn quốc"]`.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .data import Corpus, Policy
from .guardrails import _fold


def tokenize(text: str) -> list[str]:
    syll = re.findall(r"\w+", _fold(text))
    return syll + [f"{a}_{b}" for a, b in zip(syll, syll[1:])]  # âm tiết + bigram (xấp xỉ từ ghép)


def passage(p: Policy, corpus: Corpus) -> str:
    return (f"{p.doc_code}. {corpus.topics[p.topic]}. {p.benefit_name}. {p.benefit} "
            f"Hồ sơ: {'; '.join(p.required_documents)}. Nơi nộp: {p.submission_place}.")


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for doc in self.toks for t in set(doc))
        n = len(self.toks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.toks):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl))
            out.append(s)
        return out


def region_filter(policies: list[Policy], region_code: str) -> list[Policy]:
    return [p for p in policies if p.region in (region_code, "TQ")]


def bm25_retrieve(query: str, region_code: str, corpus: Corpus, k: int = 3) -> list[tuple[str, float]]:
    cands = region_filter(corpus.policies, region_code)
    bm = BM25([passage(p, corpus) for p in cands])
    ranked = sorted(zip(cands, bm.scores(query)), key=lambda x: x[1], reverse=True)
    return [(p.id, round(s, 3)) for p, s in ranked[:k]]


def recall_at_k(ranked_ids: list[list[str]], gold_ids: list[str], k: int) -> float:
    return sum(g in r[:k] for r, g in zip(ranked_ids, gold_ids)) / max(1, len(gold_ids))


def mrr(ranked_ids: list[list[str]], gold_ids: list[str]) -> float:
    tot = 0.0
    for r, g in zip(ranked_ids, gold_ids):
        if g in r:
            tot += 1 / (r.index(g) + 1)
    return tot / max(1, len(gold_ids))


def hit_at_k(ranked_ids: list[list[str]], gold_sets: list[set[str]], k: int) -> float:
    """Tỉ lệ câu hỏi có ÍT NHẤT một văn bản đúng trong top-k (dùng khi một câu có nhiều đáp án, như Zalo Legal)."""
    return sum(bool(set(r[:k]) & g) for r, g in zip(ranked_ids, gold_sets, strict=True)) / max(1, len(gold_sets))


def mrr_at_k(ranked_ids: list[list[str]], gold_sets: list[set[str]], k: int = 10) -> float:
    """MRR@k: trung bình 1/vị trí của văn bản đúng ĐẦU TIÊN trong top-k (0 nếu không có)."""
    tot = 0.0
    for r, g in zip(ranked_ids, gold_sets, strict=True):
        tot += next((1 / (i + 1) for i, d in enumerate(r[:k]) if d in g), 0.0)
    return tot / max(1, len(gold_sets))
