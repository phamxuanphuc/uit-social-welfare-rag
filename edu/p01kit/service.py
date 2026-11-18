"""Dịch vụ HTTP tối giản cho P01 (Buổi 15) — đúng luồng của hệ thống thật, mô hình được "cắm" từ ngoài vào.

    request ─► Tầng 1/2 guardrail ─► truy xuất BM25 (lọc kép địa bàn) ─► llm_fn(row) ─► Tầng 3 khóa metadata
            ─► Tầng 4 nhật ký kiểm toán (đã che PII, có trace_id) ─► response + ConfidenceTier + ResidualRisk

`llm_fn(row) -> str` nhận dòng có `region`, `question` (đã che PII), `context` và trả về chuỗi output thô.
Trên Colab: `llm_fn` gọi `llama-server` (Buổi 13). Trong `pytest`: một hàm giả, không cần GPU.
"""

# Không dùng `from __future__ import annotations`: FastAPI cần đọc kiểu của model khai báo trong hàm lúc chạy.
import hmac
import json
import logging
import time
import uuid
from typing import Callable

from .data import ESCALATION_PLACE, Corpus, render_context
from .guardrails import RESIDUAL_RISK, audit_record, check_input, enforce_metadata, mask_pii
from .metrics import extract_json
from .ops import ERROR_LABELS, should_kill
from .retrieval import bm25_retrieve

log = logging.getLogger("p01kit.audit")
PUBLIC_KEYS = ("is_refusal", "expired_warning", "doc_code", "article", "benefit", "required_documents",
               "submission_place")


def create_app(llm_fn: Callable[[dict], str], corpus: Corpus, model_version: str, admin_key: str,
               k: int = 3, enforce_guardrails: bool = True, auto_kill_window: int = 200):
    """Tạo ứng dụng FastAPI. `enforce_guardrails=False` chỉ dùng cho bài "chaos test" (mô phỏng deploy lỗi)."""
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    class AskIn(BaseModel):
        region: str | None = None
        question: str = Field(min_length=1, max_length=2000)

    class FeedbackIn(BaseModel):
        trace_id: str
        note: str = Field(default="", max_length=1000)

    class ReviewIn(BaseModel):
        trace_id: str
        label: str

    class KillIn(BaseModel):
        on: bool
        reason: str = ""

    app = FastAPI(title="P01 — Trợ lý tra cứu quyền lợi an sinh (dữ liệu giả lập)")
    st = app.state
    st.audit, st.feedback, st.reviews = [], [], []
    st.kill = {"on": False, "reason": None, "by": None}
    st.window_start = 0   # tắt KillSwitch = bắt đầu cửa sổ giám sát mới (sau khi đã rollback/sửa lỗi)
    expired_codes = {p.doc_code for p in corpus.policies if not p.is_active}

    def require_officer(x_api_key: str | None, x_role: str | None) -> None:
        if not x_api_key or not hmac.compare_digest(x_api_key, admin_key) or x_role != "Officer":
            raise HTTPException(status_code=403, detail="Chỉ cán bộ (X-Role: Officer) có khóa hợp lệ")

    def escalate(reason: str) -> dict:
        return {"verification_reasoning": f"Chuyển cán bộ ({reason}).", "is_refusal": True, "expired_warning": False,
                "doc_code": None, "article": None, "benefit": None, "required_documents": [],
                "submission_place": ESCALATION_PLACE, "_override": reason, "confidence": "Thấp - Cần hỏi cán bộ"}

    def check_auto_kill() -> None:
        res = should_kill(st.audit[st.window_start:][-auto_kill_window:], st.reviews, expired_codes)
        if res["kill"] and not st.kill["on"]:
            st.kill = {"on": True, "reason": "; ".join(res["reasons"]), "by": "auto"}
            log.warning(json.dumps({"event": "kill_switch_on", **st.kill}, ensure_ascii=False))

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": model_version, "kill_switch": st.kill}

    @app.post("/v1/answer")
    def answer(body: AskIn):
        trace_id = uuid.uuid4().hex[:16]
        t0 = time.perf_counter()
        gate = check_input(body.region, body.question, corpus)
        if st.kill["on"]:
            final = escalate("kill_switch")                    # KillSwitch: không gọi mô hình, chuyển cán bộ
        elif gate["blocked"]:
            final = escalate(gate["reason"])
        else:
            code = corpus.region_code(body.region)
            if code is None:
                raise HTTPException(status_code=422, detail="Địa bàn không có trong danh sách")
            ids = [pid for pid, _ in bm25_retrieve(gate["masked_question"], code, corpus, k=k)]
            row = {"region": body.region, "question": gate["masked_question"], "context_ids": ids,
                   "context": render_context(ids, corpus)}
            obj, _ = extract_json(llm_fn(row))
            if enforce_guardrails:
                final = enforce_metadata(obj, ids, corpus)
            else:
                final = obj if isinstance(obj, dict) else escalate("invalid_output")
                final.setdefault("confidence", "Trung bình")
        rec = {**audit_record(body.region or "", body.question, final, trace_id=trace_id),
               "model_version": model_version, "latency_s": round(time.perf_counter() - t0, 4)}
        st.audit.append(rec)
        log.info(json.dumps(rec, ensure_ascii=False))
        check_auto_kill()
        return {"trace_id": trace_id, "answer": {k: final.get(k) for k in PUBLIC_KEYS},
                "confidence": final.get("confidence"), "residual_risk": RESIDUAL_RISK,
                "notice": "Dữ liệu văn bản là GIẢ LẬP phục vụ học tập."}

    @app.post("/v1/feedback")
    def feedback(body: FeedbackIn):
        if body.trace_id not in {r["trace_id"] for r in st.audit}:
            raise HTTPException(status_code=404, detail="Không tìm thấy trace_id")
        item = {"trace_id": body.trace_id, "note": mask_pii(body.note), "ts": int(time.time())}
        st.feedback.append(item)
        return {"ok": True}

    @app.post("/v1/admin/review")
    def review(body: ReviewIn, x_api_key: str | None = Header(default=None), x_role: str | None = Header(default=None)):
        require_officer(x_api_key, x_role)
        if body.label not in ERROR_LABELS:
            raise HTTPException(status_code=422, detail=f"Nhãn phải thuộc {ERROR_LABELS}")
        if body.label != "dung":
            st.reviews.append({"trace_id": body.trace_id, "label": body.label})
        check_auto_kill()
        return {"ok": True, "kill_switch": st.kill}

    @app.post("/v1/admin/kill-switch")
    def kill_switch(body: KillIn, x_api_key: str | None = Header(default=None), x_role: str | None = Header(default=None)):
        require_officer(x_api_key, x_role)
        st.kill = {"on": body.on, "reason": body.reason or None, "by": "officer"}
        if not body.on:
            st.window_start = len(st.audit)
        log.warning(json.dumps({"event": "kill_switch_" + ("on" if body.on else "off"), **st.kill}, ensure_ascii=False))
        return st.kill

    return app
