import json

import pytest

from p01kit.guardrails import MASK, audit_record, check_input, detect_cross_region, enforce_metadata, mask_pii

SEL = "TP.HCM - Phường Tân Tạo"


@pytest.mark.parametrize("text", [
    "CCCD 079099001234", "CCCD 079 099 001 234", "CCCD 079.099.001.234", "CMND 271-234-567",
    "SĐT 0901234567", "SĐT 090.123.4567", "SĐT 090-123-4567", "SĐT +84 90 123 4567", "SĐT +84901234567",
    "gọi 0 9 0 1 2 3 4 5 6 7",
])
def test_pii_is_masked(text):
    out = mask_pii(text)
    assert MASK in out and not any(ch.isdigit() for ch in out.split(" ", 1)[1].replace(MASK, "")), out


@pytest.mark.parametrize("text", [
    "Hỗ trợ 150.000.000đ", "Vay 1.500.000.000 đồng", "Mức 4.500.000đ/tháng", "Được 2 triệu",
    "Tổng đài 1900 123 456", "Hotline 1800.1234", "Điện thoại bàn 028 3456 7890", "Năm 2026 có 12 tháng",
])
def test_money_hotline_landline_not_masked(text):
    assert mask_pii(text) == text


def test_cross_region(corpus):
    assert detect_cross_region(SEL, "Tôi ở Quận Liên Chiểu xin học nghề", corpus) == "Đà Nẵng - Quận Liên Chiểu"
    assert detect_cross_region(SEL, "toi o lien chieu da nang", corpus) == "Đà Nẵng - Quận Liên Chiểu"
    assert detect_cross_region(SEL, "Nhà tui ở Cần Thơ bị ngập", corpus) == "Cần Thơ - Phường Cái Khế"
    assert detect_cross_region(SEL, "Tui ở Tân Tạo, TP.HCM bị té xe", corpus) is None
    assert detect_cross_region(SEL, "Tui chạy xe ôm bị tai nạn", corpus) is None


def test_check_input(corpus):
    assert check_input("", "hỏi", corpus)["reason"] == "missing_region"
    r = check_input(SEL, "CCCD 079099001234, tui ở Cái Khế", corpus)
    assert r["blocked"] and r["reason"] == "cross_region_conflict" and MASK in r["masked_question"]
    assert not check_input(SEL, "Tui bị té xe", corpus)["blocked"]


def test_enforce_metadata_locks_expired_and_docs(corpus):
    ctx = ["HCM-TRO-2024", "HCM-TAINAN", "DN-TRO"]
    ans = {"verification_reasoning": "x", "is_refusal": False, "expired_warning": False,
           "doc_code": "QĐ-09/2024/UBND-HCM", "article": "sai", "benefit": "sai",
           "required_documents": ["Sổ đỏ"], "submission_place": "Bộ Tài chính"}
    out = enforce_metadata(ans, ctx, corpus)
    assert out["expired_warning"] is True
    assert out["required_documents"] == list(corpus.by_id("HCM-TRO-2024").required_documents)
    assert out["confidence"].startswith("Thấp") and "expired_warning" in out["_override"]


def test_enforce_metadata_blocks_uncited_and_invalid(corpus):
    out = enforce_metadata({"is_refusal": False, "doc_code": "NQ-01/2026/HĐND-HCM"}, ["DN-TRO"], corpus)
    assert out["is_refusal"] and out["_override"] == "uncited_doc"
    assert enforce_metadata(None, ["DN-TRO"], corpus)["_override"] == "invalid_output"
    ok = enforce_metadata({"is_refusal": False, "doc_code": "NQ-01/2026/HĐND-HCM", "expired_warning": False,
                           **{k: v for k, v in [("article", "Điều 4, Khoản 2")]}}, ["HCM-TAINAN"], corpus)
    assert ok["confidence"] == "Trung bình"  # benefit/docs/place được khóa từ metadata


def test_audit_record_has_no_raw_pii():
    rec = audit_record(SEL, "SĐT 0901234567 xin hỗ trợ", {"doc_code": None, "is_refusal": True})
    dumped = json.dumps(rec, ensure_ascii=False)
    assert "0901234567" not in dumped and MASK in dumped and len(rec["trace_id"]) == 16
