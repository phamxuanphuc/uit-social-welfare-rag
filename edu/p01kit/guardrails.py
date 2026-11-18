"""4 tầng guardrail chạy QUANH mô hình (không phụ thuộc mô hình có ngoan hay không).

Tầng 1 — Input:  che PII (CCCD/CMND, SĐT di động), bắt buộc có địa bàn.
Tầng 2 — Policy: phát hiện lệch địa bàn chéo (CrossRegionConflict).
Tầng 3 — Output: khóa cứng metadata (expired_warning, giấy tờ, nơi nộp) từ văn bản gốc,
                 chặn trích dẫn văn bản không có trong ngữ cảnh.
Tầng 4 — Audit:  bản ghi nhật ký đã che PII, gắn trace_id, không lưu câu hỏi thô.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata

from .data import ESCALATION_PLACE, Corpus

MASK = "[ĐÃ_CHE_ĐỊNH_DANH]"
_SEP = r"[\s.\-]?"

# Số tiền & tổng đài công khai được "bảo vệ" trước khi che PII để không che nhầm.
_MONEY_RE = re.compile(
    r"\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?\s*(?:đ|đồng|vnđ|vnd)(?!\w)"
    r"|\d+(?:[.,]\d+)?\s*(?:đ|đồng|vnđ|vnd|triệu|nghìn|ngàn|tỷ|tỉ)(?!\w)",
    re.I,
)
_HOTLINE_RE = re.compile(r"(?<!\d)1[89]00" + _SEP + r"\d{2,4}(?:" + _SEP + r"\d{2,4})?(?!\d)")
# SĐT di động VN: 0 hoặc +84 (có thể cách), đầu số 3/5/7/8/9, tổng 9 chữ số sau mã.
_PHONE_RE = re.compile(r"(?<![\d+])(?:\+?84|0)" + _SEP + r"[35789](?:" + _SEP + r"\d){8}(?!\d)")
# CCCD 12 số / CMND 9 số, cho phép phân tách bằng khoảng trắng, chấm, gạch.
_ID_RE = re.compile(r"(?<!\d)\d{3}" + _SEP + r"\d{3}" + _SEP + r"\d{3}(?:" + _SEP + r"\d{3})?(?!\d)")


def mask_pii(text: str) -> str:
    protected: list[str] = []

    def _protect(m: re.Match) -> str:
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    t = _MONEY_RE.sub(_protect, text)
    t = _HOTLINE_RE.sub(_protect, t)
    t = _PHONE_RE.sub(MASK, t)
    t = _ID_RE.sub(MASK, t)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], t)


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower()).replace("đ", "d")
    return re.sub(r"\s+", " ", "".join(ch for ch in s if unicodedata.category(ch) != "Mn"))


def _aliases(region_name: str) -> list[str]:
    city, _, ward = (x.strip() for x in region_name.partition(" - "))
    bare = re.sub(r"^(phường|quận|xã|huyện)\s+", "", ward, flags=re.I)
    return [_fold(a) for a in {ward, bare} if a]


def detect_cross_region(selected_region: str, question: str, corpus: Corpus) -> str | None:
    """Trả về tên địa bàn KHÁC được nhắc trong câu hỏi (None nếu không lệch)."""
    q = _fold(question)
    sel_city = _fold(selected_region.partition(" - ")[0])
    for name in corpus.regions.values():
        if name in (selected_region, "Toàn quốc"):
            continue
        city = _fold(name.partition(" - ")[0])
        if any(re.search(rf"\b{re.escape(a)}\b", q) for a in _aliases(name)):
            return name
        city_short = city.replace("tp.", "").strip()
        if city != sel_city and re.search(rf"\b{re.escape(city_short)}\b", q):
            return name
    return None


def check_input(selected_region: str | None, question: str, corpus: Corpus) -> dict:
    masked = mask_pii(question)
    if not selected_region or not selected_region.strip():
        return {"blocked": True, "reason": "missing_region", "masked_question": masked}
    other = detect_cross_region(selected_region, masked, corpus)
    if other:
        return {"blocked": True, "reason": "cross_region_conflict", "conflict_region": other, "masked_question": masked}
    return {"blocked": False, "reason": None, "masked_question": masked}


def enforce_metadata(answer: dict | None, context_ids: list[str], corpus: Corpus) -> dict:
    """Tầng hậu kiểm: KHÔNG tin mô hình ở các trường có thể lấy từ metadata gốc."""
    refusal = {"verification_reasoning": "Không đọc được câu trả lời hợp lệ; chuyển cán bộ.", "is_refusal": True,
               "expired_warning": False, "doc_code": None, "article": None, "benefit": None,
               "required_documents": [], "submission_place": ESCALATION_PLACE}
    if not isinstance(answer, dict):
        return {**refusal, "_override": "invalid_output", "confidence": "Thấp - Cần hỏi cán bộ"}
    out = dict(answer)
    if out.get("is_refusal"):
        out.update({k: refusal[k] for k in ("doc_code", "article", "benefit", "required_documents", "submission_place")})
        out["expired_warning"] = False
        out["confidence"] = "Thấp - Cần hỏi cán bộ"
        return out
    ctx = [corpus.by_id(i) for i in context_ids]
    src = next((p for p in ctx if p.doc_code == out.get("doc_code")), None)
    if src is None:
        return {**refusal, "verification_reasoning": "Mã văn bản không có trong ngữ cảnh (nghi bịa); chuyển cán bộ.",
                "_override": "uncited_doc", "confidence": "Thấp - Cần hỏi cán bộ"}
    changed = []
    locked = {"expired_warning": not src.is_active, "article": src.article, "benefit": src.benefit,
              "required_documents": list(src.required_documents), "submission_place": src.submission_place}
    for k, v in locked.items():
        if out.get(k) != v:
            changed.append(k)
            out[k] = v
    if changed:
        out["_override"] = "locked:" + ",".join(changed)
    out["confidence"] = ("Thấp - Cần hỏi cán bộ" if not src.is_active
                         else "Trung bình" if changed else "Cao")
    return out


RESIDUAL_RISK = ("Kết quả chỉ hỗ trợ chuẩn bị hồ sơ ban đầu, không thay thế quyết định xét duyệt "
                 "của cơ quan có thẩm quyền.")


def audit_record(selected_region: str, question: str, answer: dict, trace_id: str | None = None) -> dict:
    """Bản ghi nhật ký an toàn: chỉ lưu câu hỏi đã che PII + băm, không lưu bản thô."""
    masked = mask_pii(question)
    return {
        "trace_id": trace_id or hashlib.sha1(f"{time.time_ns()}".encode()).hexdigest()[:16],
        "ts": int(time.time()),
        "region": selected_region,
        "question_masked": masked,
        "question_sha1": hashlib.sha1(question.encode()).hexdigest(),
        "doc_code": answer.get("doc_code"),
        "is_refusal": answer.get("is_refusal"),
        "expired_warning": answer.get("expired_warning"),
        "override": answer.get("_override"),
        "confidence": answer.get("confidence"),
    }
