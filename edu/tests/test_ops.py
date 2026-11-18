import json

import pytest

from p01kit.metrics import score_one
from p01kit.ops import (capacity_plan, cohen_kappa, compare_release, drift_report, latency_report, percentile, psi,
                        shadow_disagreement, should_kill)


def _scores(rows, corrupt=lambda r: False):
    """Chấm đáp án vàng; câu nào `corrupt(r)` thì trả lời bằng một JSON hỏng (sai)."""
    return [score_one("không phải JSON" if corrupt(r) else json.dumps(r["gold"], ensure_ascii=False), r) for r in rows]


def test_percentile_matches_linear_interpolation():
    xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(xs, 50) == 5.5
    assert percentile(xs, 95) == pytest.approx(9.55)
    rep = latency_report(xs)
    assert rep["p50"] == 5.5 and rep["max"] == 10 and rep["n"] == 10


def test_capacity_plan_littles_law():
    # 120 request/phút × 6 s = 12 request đồng thời; 4 slot/replica ở 70% → ceil(12 / 2.8) = 5 replica
    plan = capacity_plan(peak_rpm=120, service_time_s=6, slots_per_replica=4, target_utilization=0.7)
    assert plan == {"offered_load": 12.0, "replicas": 5, "utilization": 0.6}
    assert capacity_plan(0, 5)["replicas"] == 1
    with pytest.raises(ValueError):
        capacity_plan(10, 0)


def test_release_gate_promotes_identical_model(splits):
    test = splits["test"]
    res = compare_release(_scores(test), _scores(test))
    assert res["decision"] == "promote" and res["reasons"] == []
    assert all(d == 0 for d in res["delta"].values())


def test_release_gate_rejects_significant_regression(splits):
    test = splits["test"]
    champ = _scores(test)
    chall = _scores(test, corrupt=lambda r: r["style"] == "colloquial")   # hỏng toàn bộ câu đời thường
    res = compare_release(champ, chall)
    assert res["decision"] == "reject"
    assert any("thống kê" in x for x in res["reasons"]) and any("ngưỡng" in x for x in res["reasons"])
    colloq = next(s for s in res["slices"] if s["slice"] == "style=colloquial")
    assert colloq["delta"] == -1.0


def test_release_gate_holds_on_small_slice_drop(splits):
    test = splits["test"]
    ood = [r["id"] for r in test if r["slice"] == "ood"]
    seen = [r["id"] for r in test if r["slice"] == "seen_doc" and not r["is_unanswerable"]][:3]
    # Tụt 3/24 câu ở lát seen_doc: không có ý nghĩa thống kê, nhưng vượt ngưỡng tụt 5% của lát cắt
    chall = _scores(test, corrupt=lambda r: r["id"] in seen)
    res = compare_release(_scores(test), chall, min_slice_n=len(ood) + 1)
    assert res["decision"] == "hold", res
    assert any("seen_doc" in x for x in res["reasons"])


def test_release_gate_requires_same_questions(splits):
    test = splits["test"]
    with pytest.raises(ValueError):
        compare_release(_scores(test), _scores(test[:-1]))


def test_shadow_disagreement():
    a = [{"is_refusal": False, "doc_code": "X", "expired_warning": False}, {"is_refusal": True, "doc_code": None}]
    b = [{"is_refusal": False, "doc_code": "Y", "expired_warning": False}, {"is_refusal": True, "doc_code": None}]
    res = shadow_disagreement(a, b)
    assert res["disagree_rate"] == 0.5 and res["review_idx"] == [0] and res["by_key"]["doc_code"] == 0.5


def test_psi_levels():
    assert psi({"a": 50, "b": 50}, {"a": 50, "b": 50}) == 0
    assert psi({"a": 50, "b": 50}, {"a": 55, "b": 45}) < 0.1
    assert psi({"a": 90, "b": 10}, {"a": 30, "b": 70}) > 0.25
    assert psi({"a": 100}, {"a": 50, "new": 50}) > 0.25        # nhóm mới xuất hiện (ví dụ địa bàn mới)
    base = [{"region": "HCM"}] * 50 + [{"region": "DN"}] * 50
    rep = drift_report(base, [{"region": "HCM"}] * 95 + [{"region": "DN"}] * 5, ["region"])
    assert rep["region"]["level"] == "alert"


def _logs(n, expired_unwarned=0):
    logs = [{"trace_id": f"t{i}", "doc_code": "A", "is_refusal": False, "expired_warning": False} for i in range(n)]
    for i in range(expired_unwarned):
        logs[i]["doc_code"] = "OLD"
    return logs


def test_kill_switch_on_single_expired_leak():
    res = should_kill(_logs(10, expired_unwarned=1), [], expired_codes={"OLD"})
    assert res["kill"] and res["leaked_trace_ids"] == ["t0"]
    warned = _logs(10, expired_unwarned=1)
    warned[0]["expired_warning"] = True
    assert not should_kill(warned, [], expired_codes={"OLD"})["kill"]


def test_kill_switch_error_rate_needs_enough_traffic():
    errors = [{"trace_id": f"t{i}"} for i in range(5)]
    assert not should_kill(_logs(20), errors, expired_codes=set())["kill"]          # 25% nhưng mới 20 request
    res = should_kill(_logs(100), errors * 2 + [{"trace_id": f"t{i}"} for i in range(5, 16)], expired_codes=set())
    assert res["kill"] and res["error_rate"] == 0.16                                 # trùng trace chỉ tính 1 lần
    assert not should_kill(_logs(100), errors + [{"trace_id": "khong-ton-tai"}], expired_codes=set())["kill"]


def test_cohen_kappa():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0
    # Ví dụ kinh điển: po = 0,7; pe = 0,5 → κ = 0,4
    a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
    b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
    assert cohen_kappa(a, b) == 0.4
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_system_scores_apply_guardrails(splits, corpus):
    from p01kit.ops import system_scores

    test = splits["test"]
    answerable = [r for r in test if not r["is_unanswerable"]]
    # Mô hình chọn đúng văn bản nhưng quên cờ hết hạn và bịa giấy tờ → guardrail sửa lại, hệ thống vẫn đúng
    sloppy = [json.dumps({**r["gold"], "expired_warning": False, "required_documents": ["bịa"]}, ensure_ascii=False)
              for r in answerable]
    raw = [score_one(o, r) for o, r in zip(sloppy, answerable)]
    fixed = system_scores(sloppy, answerable, corpus)
    assert sum(s["strict_ok"] for s in raw) < len(answerable)
    assert all(s["strict_ok"] for s in fixed)
    with pytest.raises(ValueError):
        system_scores(sloppy[:-1], answerable, corpus)


def test_system_label(corpus):
    from p01kit.ops import system_label

    active = next(p for p in corpus.policies if p.is_active)
    old = next(p for p in corpus.policies if not p.is_active)
    ans = lambda **kw: {"is_refusal": False, "expired_warning": False, **kw}
    assert system_label({"is_refusal": True}, None) == "dung"
    assert system_label(ans(doc_code=active.doc_code), None) == "le_ra_tu_choi"
    assert system_label({"is_refusal": True}, active) == "tu_choi_nham"
    assert system_label(ans(doc_code=old.doc_code), active) == "sai_van_ban"
    assert system_label(ans(doc_code=old.doc_code), old) == "khong_canh_bao_het_han"
    assert system_label(ans(doc_code=old.doc_code, expired_warning=True), old) == "dung"
    assert system_label(ans(doc_code=active.doc_code), active) == "dung"
