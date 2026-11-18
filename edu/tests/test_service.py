import json
import logging

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from p01kit.data import resolve  # noqa: E402
from p01kit.service import create_app  # noqa: E402

KEY = "test-admin-key-not-a-secret"
OFFICER = {"X-API-Key": KEY, "X-Role": "Officer"}
REGION = "TP.HCM - Phường Tân Tạo"


def oracle_llm(corpus):
    """Mô hình giả "biết luật": trả lời văn bản đúng nếu nó có trong ngữ cảnh, không thì từ chối."""
    def llm(row):
        code = corpus.region_code(row["region"])
        ctx = [corpus.by_id(i) for i in row["context_ids"]]
        topic = next((p.topic for p in ctx), None)
        p = resolve(topic, code, ctx)
        if p is None:
            return json.dumps({"verification_reasoning": "Không có văn bản phù hợp", "is_refusal": True})
        return json.dumps({"verification_reasoning": "ok", "is_refusal": False, "expired_warning": False,
                           "doc_code": p.doc_code, "article": "bịa", "benefit": "bịa", "required_documents": ["bịa"],
                           "submission_place": "bịa"}, ensure_ascii=False)
    return llm


@pytest.fixture
def client(corpus):
    return TestClient(create_app(oracle_llm(corpus), corpus, model_version="test-v1", admin_key=KEY))


def ask(c, question, region=REGION):
    r = c.post("/v1/answer", json={"region": region, "question": question})
    assert r.status_code == 200, r.text
    return r.json()


def test_health(client):
    assert client.get("/health").json()["model_version"] == "test-v1"


def test_answer_locks_metadata_and_returns_trace(client, corpus):
    res = ask(client, "Tôi chạy xe ôm bị té gãy tay thì được hỗ trợ tai nạn lao động không, cần giấy gì?")
    a = res["answer"]
    assert len(res["trace_id"]) == 16 and "GIẢ LẬP" in res["notice"] and res["residual_risk"]
    assert not a["is_refusal"], res
    src = corpus.by_code(a["doc_code"])
    assert a["required_documents"] == list(src.required_documents)   # "bịa" của mô hình đã bị thay
    assert a["expired_warning"] == (not src.is_active) and res["confidence"] == "Trung bình"


def test_missing_region_and_cross_region_never_call_model(corpus):
    calls = []
    app = create_app(lambda row: calls.append(row) or "{}", corpus, model_version="v", admin_key=KEY)
    c = TestClient(app)
    assert ask(c, "Hỗ trợ tiền trọ thế nào?", region=None)["answer"]["is_refusal"]
    assert ask(c, "Tui ở Quận Liên Chiểu Đà Nẵng, hỗ trợ tiền trọ sao?")["answer"]["is_refusal"]
    assert calls == []
    assert [r["override"] for r in app.state.audit] == ["missing_region", "cross_region_conflict"]
    assert c.post("/v1/answer", json={"region": "Hà Nội - Phường X", "question": "hỏi"}).status_code == 422


def test_audit_log_has_no_raw_pii(client, caplog):
    with caplog.at_level(logging.INFO, logger="p01kit.audit"):
        res = ask(client, "SĐT tui 090.123.4567, CCCD 079 099 001 234, hỏi hỗ trợ tiền trọ")
    text = caplog.text + json.dumps(client.app.state.audit, ensure_ascii=False)
    assert "090.123.4567" not in text and "079 099 001 234" not in text
    assert res["trace_id"] in caplog.text


def test_feedback_requires_known_trace_and_masks_note(client):
    t = ask(client, "Hỗ trợ tiền thuê nhà trọ cần giấy tờ gì?")["trace_id"]
    assert client.post("/v1/feedback", json={"trace_id": "khong-co", "note": "x"}).status_code == 404
    assert client.post("/v1/feedback", json={"trace_id": t, "note": "gọi tui 0901234567"}).json()["ok"]
    assert "0901234567" not in json.dumps(client.app.state.feedback)


def test_admin_endpoints_require_officer_role(client):
    body = {"on": True, "reason": "thử"}
    assert client.post("/v1/admin/kill-switch", json=body).status_code == 403
    assert client.post("/v1/admin/kill-switch", json=body, headers={"X-API-Key": KEY}).status_code == 403
    assert client.post("/v1/admin/kill-switch", json=body, headers={**OFFICER, "X-API-Key": "sai"}).status_code == 403
    assert client.post("/v1/admin/kill-switch", json=body, headers=OFFICER).json()["on"] is True
    assert client.post("/v1/admin/review", json={"trace_id": "t", "label": "nhan-la"}, headers=OFFICER).status_code == 422


def test_manual_kill_switch_bypasses_model(corpus):
    calls = []
    c = TestClient(create_app(lambda row: calls.append(1) or "{}", corpus, model_version="v", admin_key=KEY))
    c.post("/v1/admin/kill-switch", json={"on": True, "reason": "sự cố"}, headers=OFFICER)
    res = ask(c, "Hỗ trợ tiền thuê nhà trọ cần giấy tờ gì?")
    assert res["answer"]["is_refusal"] and calls == []


def test_chaos_deploy_without_guardrails_trips_auto_kill_switch(corpus):
    expired = next(p for p in corpus.policies if not p.is_active and p.region == "HCM")
    bad = json.dumps({"verification_reasoning": "", "is_refusal": False, "expired_warning": False,
                      "doc_code": expired.doc_code}, ensure_ascii=False)
    c = TestClient(create_app(lambda row: bad, corpus, model_version="v", admin_key=KEY, enforce_guardrails=False))
    ask(c, f"Cho hỏi về {expired.benefit_name}?")
    kill = c.get("/health").json()["kill_switch"]
    assert kill["on"] and kill["by"] == "auto" and "hết hạn" in kill["reason"]
    # Cán bộ tắt sau khi rollback → cửa sổ giám sát mới, không bật lại vì log cũ
    c.post("/v1/admin/kill-switch", json={"on": False}, headers=OFFICER)
    assert c.get("/health").json()["kill_switch"]["on"] is False


def test_guardrails_prevent_the_same_leak(corpus):
    expired = next(p for p in corpus.policies if not p.is_active and p.region == "HCM")
    bad = json.dumps({"verification_reasoning": "", "is_refusal": False, "expired_warning": False,
                      "doc_code": expired.doc_code}, ensure_ascii=False)
    c = TestClient(create_app(lambda row: bad, corpus, model_version="v", admin_key=KEY))
    res = ask(c, f"Cho hỏi về {expired.benefit_name}?")
    assert c.get("/health").json()["kill_switch"]["on"] is False
    assert res["answer"]["is_refusal"] or res["answer"]["expired_warning"] is True
