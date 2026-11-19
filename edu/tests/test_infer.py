from p01kit.infer import show_errors


def test_show_errors_prints_region_and_expired_status(capsys):
    rows = [
        {
            "id": "r1",
            "style": "colloquial",
            "region": "Đà Nẵng - Quận Liên Chiểu",
            "question": "Tôi ở Liên Chiểu cần giấy gì?",
            "gold": {"is_refusal": False, "doc_code": "QĐ-19/2026/UBND-ĐN", "expired_warning": True},
        },
        {
            "id": "r2",
            "style": "formal",
            "region": "TP.HCM - Phường Tân Tạo",
            "question": "Câu làm đúng",
            "gold": {"is_refusal": False, "doc_code": "NQ-01/2026/HĐND-HCM", "expired_warning": False},
        },
    ]
    result = {
        "outputs": ['{"doc_code": "QĐ-15/2026/UBND-HCM", "expired_warning": false}', "ok"],
        "scores": [
            {"answer_ok": False, "strict_ok": False},
            {"answer_ok": True, "strict_ok": True},
        ],
    }

    show_errors(result, rows, k=5)
    out = capsys.readouterr().out

    assert "r1" in out
    assert "r2" not in out
    assert "Đà Nẵng - Quận Liên Chiểu" in out
    assert "gold_expired=True" in out
    assert "gold_doc=QĐ-19/2026/UBND-ĐN" in out
