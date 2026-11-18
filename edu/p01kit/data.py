"""Kho văn bản giả lập P01 và bộ dựng tập dữ liệu RAFT dùng chung cho mọi buổi học.

Luồng dữ liệu:
    policies.json --(build_dataset)--> p01_train.jsonl / p01_val.jsonl / p01_test.jsonl

Mỗi dòng (row) là một lượt hỏi gồm: câu hỏi (chuẩn hoặc đời thường), địa bàn đã chọn,
3 đoạn ngữ cảnh (1 đúng + 2 nhiễu, hoặc 3 nhiễu nếu là câu phải từ chối) và đáp án vàng
dạng JSON 8 khóa. Đáp án vàng được *suy ra bằng luật* từ metadata, không viết tay,
nên không có nhãn mâu thuẫn.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POLICY_FILE = DATA_DIR / "policies.json"
SPLIT_FILES = {s: DATA_DIR / f"p01_{s}.jsonl" for s in ("train", "val", "test")}

ANSWER_KEYS = (
    "verification_reasoning",
    "is_refusal",
    "expired_warning",
    "doc_code",
    "article",
    "benefit",
    "required_documents",
    "submission_place",
)
ESCALATION_PLACE = "Liên hệ Bộ phận Một cửa UBND cấp xã/phường nơi cư trú để cán bộ hướng dẫn."
N_CONTEXT = 3
INTENTS = ("ho_so", "noi_nop", "muc_ho_tro")

FORMAL_TEMPLATES = {
    "ho_so": [
        "Thành phần hồ sơ đề nghị {benefit_name} gồm những giấy tờ nào?",
        "Để được {benefit_name}, người lao động cần chuẩn bị hồ sơ gì?",
        "Xin cho biết danh mục giấy tờ cần nộp khi đề nghị {benefit_name}.",
    ],
    "noi_nop": [
        "Hồ sơ đề nghị {benefit_name} được nộp tại cơ quan nào?",
        "Cơ quan nào tiếp nhận hồ sơ {benefit_name}?",
        "Nơi tiếp nhận hồ sơ {benefit_name} là ở đâu và cần mang theo gì?",
    ],
    "muc_ho_tro": [
        "Mức {benefit_name} hiện nay là bao nhiêu và theo văn bản nào?",
        "Quy định nào điều chỉnh {benefit_name} và mức hưởng ra sao?",
        "Đề nghị cho biết căn cứ pháp lý và mức {benefit_name}.",
    ],
}
COLLOQUIAL_TEMPLATES = {
    "ho_so": [
        "Tui là {who}, {situation}, giờ phải đem giấy gì lên phường?",
        "Cho hỏi {who} như tui, {situation}, thì cần làm giấy tờ chi?",
        "Em là {who} nè, {situation}, cần chuẩn bị mấy giấy gì vậy?",
    ],
    "noi_nop": [
        "Tui là {who}, {situation}, giấy tờ xin hỗ trợ thì đem nộp ở đâu?",
        "Cho hỏi {who} như tui, {situation}, thì lên chỗ nào nộp đơn?",
        "Em là {who}, {situation}, hồ sơ nộp ở phường hay ở đâu ạ?",
    ],
    "muc_ho_tro": [
        "Tui là {who}, {situation}, được nhà nước giúp bao nhiêu tiền vậy?",
        "Cho hỏi {who} như tui, {situation}, thì được hỗ trợ nhiêu?",
        "Em là {who}, {situation}, xin hỏi được hỗ trợ mấy đồng ạ?",
    ],
}


@dataclass(frozen=True)
class Policy:
    id: str
    doc_code: str
    topic: str
    region: str  # mã vùng: HCM / DN / CT / TQ
    status: str  # active / expired
    effective_date: str
    expiry_date: str | None
    version: str
    article: str
    benefit: str
    benefit_name: str
    who: tuple[str, ...]
    situation: tuple[str, ...]
    required_documents: tuple[str, ...]
    submission_place: str
    processing_time: str

    @property
    def is_active(self) -> bool:
        return self.status == "active"


@dataclass
class Corpus:
    policies: list[Policy]
    topics: dict[str, str]
    regions: dict[str, str]
    out_of_scope: list[dict] = field(default_factory=list)

    def by_id(self, pid: str) -> Policy:
        for p in self.policies:
            if p.id == pid:
                return p
        raise KeyError(pid)

    def by_code(self, doc_code: str) -> Policy | None:
        return next((p for p in self.policies if p.doc_code == doc_code), None)

    def region_name(self, code: str) -> str:
        return self.regions[code]

    def region_code(self, name: str) -> str | None:
        return next((c for c, n in self.regions.items() if n == name), None)


def load_corpus(path: str | Path = POLICY_FILE) -> Corpus:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pols = []
    for d in raw["policies"]:
        d = dict(d)
        for k in ("who", "situation", "required_documents"):
            d[k] = tuple(d[k])
        pols.append(Policy(**d))
    return Corpus(pols, raw["topics"], raw["regions"], raw.get("out_of_scope", []))


# ---------------------------------------------------------------------------
# Luật nghiệp vụ: văn bản nào là câu trả lời đúng?
# ---------------------------------------------------------------------------

def applicable(p: Policy, topic: str | None, region_code: str) -> bool:
    """Văn bản dùng được khi đúng chủ đề và địa bàn ∈ {địa bàn đã chọn, Toàn quốc}."""
    return topic is not None and p.topic == topic and p.region in (region_code, "TQ")


def resolve(topic: str | None, region_code: str, candidates: Iterable[Policy]) -> Policy | None:
    """Chọn 1 văn bản đúng trong tập ứng viên theo thứ tự ưu tiên:
    còn hiệu lực > hết hiệu lực; cấp địa phương > Toàn quốc; mới hơn > cũ hơn.
    Trả về None nếu không có văn bản phù hợp (=> phải từ chối)."""
    ok = [p for p in candidates if applicable(p, topic, region_code)]
    if not ok:
        return None
    ok.sort(key=lambda p: (p.is_active, p.region != "TQ", p.effective_date), reverse=True)
    return ok[0]


def render_policy(p: Policy, corpus: Corpus, idx: int) -> str:
    status = "Còn hiệu lực" if p.is_active else f"HẾT HIỆU LỰC từ {p.expiry_date}"
    return (
        f"[Văn bản {idx}] {p.doc_code} | Địa bàn: {corpus.region_name(p.region)} | "
        f"Trạng thái: {status} | Hiệu lực từ: {p.effective_date} | Phiên bản: {p.version}\n"
        f"{p.article}: {corpus.topics[p.topic].capitalize()}. {p.benefit}\n"
        f"Hồ sơ gồm: {'; '.join(p.required_documents)}.\n"
        f"Nơi nộp: {p.submission_place}. Thời hạn giải quyết: {p.processing_time}."
    )


def render_context(policy_ids: list[str], corpus: Corpus) -> str:
    return "\n\n".join(render_policy(corpus.by_id(pid), corpus, i) for i, pid in enumerate(policy_ids, 1))


def gold_answer(topic: str | None, region_code: str, context: list[Policy], corpus: Corpus) -> dict:
    """Đáp án vàng 8 khóa, suy ra hoàn toàn bằng luật từ metadata ngữ cảnh."""
    region = corpus.region_name(region_code)
    target = resolve(topic, region_code, context)
    if target is None:
        if topic is None:
            why = "Câu hỏi nằm ngoài phạm vi các chương trình an sinh có trong ngữ cảnh"
        else:
            why = (f"Câu hỏi thuộc chủ đề '{corpus.topics[topic]}' tại {region}, nhưng không văn bản nào "
                   f"trong ngữ cảnh vừa đúng chủ đề vừa áp dụng cho địa bàn này")
        return {
            "verification_reasoning": why + ", nên phải từ chối và chuyển cán bộ.",
            "is_refusal": True,
            "expired_warning": False,
            "doc_code": None,
            "article": None,
            "benefit": None,
            "required_documents": [],
            "submission_place": ESCALATION_PLACE,
        }
    scope = "áp dụng Toàn quốc" if target.region == "TQ" else f"áp dụng tại {corpus.region_name(target.region)}"
    parts = [f"Câu hỏi thuộc chủ đề '{corpus.topics[topic]}' tại {region}.",
             f"{target.doc_code} {scope}, khớp chủ đề và địa bàn."]
    if target.is_active:
        parts.append("Văn bản còn hiệu lực.")
    else:
        parts.append(f"Văn bản đã HẾT HIỆU LỰC từ {target.expiry_date}: phải cảnh báo và đề nghị cán bộ xác minh.")
    for p in context:
        if p is target:
            continue
        if applicable(p, topic, region_code):
            reason = "cùng chủ đề nhưng đã hết hiệu lực" if not p.is_active else "cùng chủ đề nhưng có văn bản ưu tiên hơn"
        elif p.topic == topic:
            reason = f"cùng chủ đề nhưng áp dụng tại {corpus.region_name(p.region)}"
        else:
            reason = "khác chủ đề"
        parts.append(f"Loại {p.doc_code} vì {reason}.")
    return {
        "verification_reasoning": " ".join(parts),
        "is_refusal": False,
        "expired_warning": not target.is_active,
        "doc_code": target.doc_code,
        "article": target.article,
        "benefit": target.benefit,
        "required_documents": list(target.required_documents),
        "submission_place": target.submission_place,
    }


# ---------------------------------------------------------------------------
# Sinh câu hỏi, chọn văn bản nhiễu, chia tập theo nhóm kịch bản
# ---------------------------------------------------------------------------

def question_regions(p: Policy, corpus: Corpus) -> list[str]:
    """Các địa bàn mà ở đó văn bản p là đáp án *duy nhất đúng* trên toàn kho."""
    regions = [p.region] if p.region != "TQ" else [c for c in corpus.regions if c != "TQ"]
    return [r for r in regions if resolve(p.topic, r, corpus.policies) is p]


def make_question(p: Policy, intent: str, style: str, k: int) -> str:
    if style == "formal":
        return FORMAL_TEMPLATES[intent][k].format(benefit_name=p.benefit_name)
    return COLLOQUIAL_TEMPLATES[intent][k].format(
        who=p.who[k % len(p.who)], situation=p.situation[(k + INTENTS.index(intent)) % len(p.situation)]
    )


def pick_distractors(oracle: Policy | None, topic: str | None, region_code: str, pool: list[Policy],
                     n: int, rng: random.Random) -> list[Policy]:
    """Chọn văn bản nhiễu *khó* trước (cùng chủ đề khác địa bàn, bản cũ hết hạn, cùng địa bàn khác chủ đề).

    Ràng buộc an toàn nhãn: thêm văn bản nhiễu không được làm đổi đáp án đúng.
    """
    def safe(d: Policy, chosen: list[Policy]) -> bool:
        if applicable(d, topic, region_code):
            # Văn bản "dùng được" chỉ được làm nhiễu khi nó là bản HẾT HẠN bị bản còn hiệu lực thay thế.
            if oracle is None or not oracle.is_active or d.is_active:
                return False
        ctx = ([oracle] if oracle else []) + chosen + [d]
        return resolve(topic, region_code, ctx) is oracle

    cands = [d for d in pool if d is not oracle]
    rng.shuffle(cands)
    tiers = [
        [d for d in cands if topic and d.topic == topic],                   # bẫy lệch địa bàn / bản hết hạn
        [d for d in cands if d.region in (region_code, "TQ") and d.topic != topic],  # cùng địa bàn, khác chủ đề
        cands,
    ]
    chosen: list[Policy] = []
    for tier in tiers:
        for d in tier:
            if len(chosen) >= n:
                break
            if d not in chosen and safe(d, chosen):
                chosen.append(d)
                break  # mỗi tầng lấy tối đa 1 để đa dạng
    for d in cands:  # lấp đủ n nếu các tầng chưa đủ
        if len(chosen) >= n:
            break
        if d not in chosen and safe(d, chosen):
            chosen.append(d)
    return chosen


@dataclass(frozen=True)
class Scenario:
    group_id: str
    policy_id: str | None  # None = câu hỏi ngoài phạm vi
    topic: str | None
    intent: str | None


def enumerate_scenarios(corpus: Corpus) -> list[Scenario]:
    out = []
    for p in corpus.policies:
        if not question_regions(p, corpus):
            continue  # ví dụ bản cũ đã có bản mới thay thế: chỉ dùng làm văn bản nhiễu
        for intent in INTENTS:
            out.append(Scenario(f"{p.id}::{intent}", p.id, p.topic, intent))
    for o in corpus.out_of_scope:
        out.append(Scenario(o["id"], None, None, None))
    return out


def group_split(groups: list[str], ratios=(0.8, 0.1, 0.1), seed: int = 42) -> dict[str, str]:
    """Chia theo nhóm kịch bản: mọi biến thể của một nhóm nằm trọn trong 1 tập."""
    g = sorted(set(groups))
    random.Random(seed).shuffle(g)
    n = len(g)
    n_train = round(ratios[0] * n)
    n_val = round(ratios[1] * n)
    return {gid: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, gid in enumerate(g)}


HOLDOUT_POLICIES = ("HCM-HOCNGHE-2023", "DN-BHYT", "CT-THIENTAI", "TQ-MAITANG")


def build_dataset(corpus: Corpus | None = None, seed: int = 42, unanswerable_ratio: float = 0.3,
                  holdout: tuple[str, ...] = HOLDOUT_POLICIES) -> dict[str, list[dict]]:
    """Dựng 3 tập train/val/test.

    - `holdout`: các văn bản bị giấu hoàn toàn khỏi train/val (kể cả vai trò văn bản nhiễu)
      → lát cắt `unseen_doc` của tập test đo khả năng *đọc hiểu văn bản mới*.
    - Các nhóm còn lại chia 80/10/10 theo nhóm kịch bản → lát cắt `seen_doc`.
    """
    corpus = corpus or load_corpus()
    scenarios = enumerate_scenarios(corpus)
    hold = set(holdout)
    normal_groups = [s.group_id for s in scenarios if s.policy_id not in hold]
    split_of = group_split(normal_groups, seed=seed)
    for s in scenarios:
        if s.policy_id in hold:
            split_of[s.group_id] = "test"

    pools = {
        "train": [p for p in corpus.policies if p.id not in hold],
        "val": [p for p in corpus.policies if p.id not in hold],
        "test": list(corpus.policies),
    }
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for s in scenarios:
        split = split_of[s.group_id]
        variants: list[tuple[str, int, str, str]] = []  # (style, pair_id, question, region_code)
        if s.policy_id is None:
            o = next(o for o in corpus.out_of_scope if o["id"] == s.group_id)
            regions = [c for c in corpus.regions if c != "TQ"]
            for style in ("formal", "colloquial"):
                for k, q in enumerate(o[style]):
                    variants.append((style, k, q, regions[(k + len(style)) % len(regions)]))
        else:
            p = corpus.by_id(s.policy_id)
            q_regions = question_regions(p, corpus)
            for style in ("formal", "colloquial"):
                for k in range(3):
                    variants.append((style, k, make_question(p, s.intent, style, k), q_regions[k % len(q_regions)]))

        for style, pair_id, question, rcode in variants:
            row_id = f"{s.group_id}::{style}{pair_id}"
            rng = random.Random(f"{seed}|{row_id}")
            oracle = corpus.by_id(s.policy_id) if s.policy_id else None
            drop_oracle = oracle is not None and rng.random() < unanswerable_ratio
            pool = pools[split]
            if oracle is None or drop_oracle:
                ctx = pick_distractors(None, s.topic, rcode, [d for d in pool if d is not oracle], N_CONTEXT, rng)
            else:
                ctx = [oracle] + pick_distractors(oracle, s.topic, rcode, pool, N_CONTEXT - 1, rng)
            rng.shuffle(ctx)
            gold = gold_answer(s.topic, rcode, ctx, corpus)
            out[split].append({
                "id": row_id,
                "group_id": s.group_id,
                "split": split,
                "slice": "unseen_doc" if s.policy_id in hold else ("ood" if s.policy_id is None else "seen_doc"),
                "policy_id": s.policy_id,
                "topic": s.topic,
                "intent": s.intent,
                "style": style,
                "pair_id": f"{s.group_id}::{pair_id}",
                "region_code": rcode,
                "region": corpus.region_name(rcode),
                "question": question,
                "context_ids": [p.id for p in ctx],
                "context": render_context([p.id for p in ctx], corpus),
                "is_unanswerable": gold["is_refusal"],
                "is_expired": gold["expired_warning"],
                "gold": gold,
            })
    for rows in out.values():
        rows.sort(key=lambda r: r["id"])
    return out


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(split: str) -> list[dict]:
    return read_jsonl(SPLIT_FILES[split])


def dataset_stats(rows: list[dict]) -> dict:
    n = len(rows)
    def frac(key):
        return round(sum(bool(r[key]) for r in rows) / max(1, n), 3)
    return {
        "n": n,
        "groups": len({r["group_id"] for r in rows}),
        "unanswerable_ratio": frac("is_unanswerable"),
        "expired_ratio": frac("is_expired"),
        "colloquial_ratio": round(sum(r["style"] == "colloquial" for r in rows) / max(1, n), 3),
        "slices": {s: sum(r["slice"] == s for r in rows) for s in ("seen_doc", "unseen_doc", "ood")},
    }


if __name__ == "__main__":  # python -m p01kit.data  → ghi lại 3 file jsonl
    splits = build_dataset()
    for name, rows in splits.items():
        write_jsonl(rows, SPLIT_FILES[name])
        print(name, dataset_stats(rows))
