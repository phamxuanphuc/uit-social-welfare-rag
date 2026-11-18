import json
from collections import Counter

from p01kit.guardrails import MASK, mask_pii
from p01kit.metrics import score_one
from p01kit.ops import ERROR_LABELS, cohen_kappa, drift_report, error_category
from p01kit.traffic import (request_features, simulate_feedback, simulate_requests, simulate_reviews,
                            strip_diacritics)


def test_strip_diacritics():
    assert strip_diacritics("Đăng ký hỗ trợ tiền trọ ở Đà Nẵng") == "Dang ky ho tro tien tro o Da Nang"


def test_simulation_is_deterministic(splits):
    rows = splits["val"] + splits["test"]
    assert simulate_requests(rows, 50, seed=3) == simulate_requests(rows, 50, seed=3)
    assert simulate_requests(rows, 50, seed=3) != simulate_requests(rows, 50, seed=4)


def test_simulated_shift_is_detected_by_psi(splits):
    rows = splits["val"] + splits["test"]
    feats = lambda reqs: [request_features(r["region"], r["question"]) for r in reqs]
    base = feats(simulate_requests(rows, 400, seed=1))
    same = feats(simulate_requests(rows, 400, seed=2))
    shifted = feats(simulate_requests(rows, 400, seed=2, no_diacritics_share=0.5, pii_share=0.2,
                                      region_weights={"HCM": 0.8, "DN": 0.1, "CT": 0.1}))
    stable = drift_report(base, same, ["region", "no_diacritics", "has_pii"])
    assert all(v["level"] == "ok" for v in stable.values()), stable
    drift = drift_report(base, shifted, ["region", "no_diacritics", "has_pii"])
    assert drift["no_diacritics"]["level"] == "alert" and drift["has_pii"]["level"] == "alert"
    assert drift["region"]["level"] in ("warn", "alert")


def test_injected_phone_numbers_are_masked(splits):
    reqs = simulate_requests(splits["test"], 30, seed=0, pii_share=1.0)
    assert all(MASK in mask_pii(r["question"]) for r in reqs)
    assert all(request_features(r["region"], r["question"])["has_pii"] for r in reqs)


def test_ood_share_controls_out_of_scope_rate(splits):
    rows = splits["val"] + splits["test"]
    by_id = {r["id"]: r for r in rows}
    reqs = simulate_requests(rows, 500, seed=0, ood_share=0.3)
    share = sum(by_id[r["source_id"]]["slice"] == "ood" for r in reqs) / len(reqs)
    assert 0.25 < share < 0.35


def test_error_category_covers_taxonomy(splits):
    test = splits["test"]
    answerable = next(r for r in test if not r["is_unanswerable"])
    refusal = next(r for r in test if r["is_unanswerable"])
    gold = lambda r, **kw: json.dumps({**r["gold"], **kw}, ensure_ascii=False)
    assert error_category(score_one(gold(answerable), answerable)) == "dung"
    assert error_category(score_one(gold(answerable, doc_code="QĐ-BỊA"), answerable)) == "sai_van_ban"
    assert error_category(score_one(gold(answerable, is_refusal=True, doc_code=None), answerable)) == "tu_choi_nham"
    assert error_category(score_one(gold(answerable, required_documents=[]), answerable)) == "sai_giay_to_noi_nop"
    flipped = gold(answerable, expired_warning=not answerable["gold"]["expired_warning"])
    assert error_category(score_one(flipped, answerable)) == "khong_canh_bao_het_han"
    assert error_category(score_one(gold(refusal, is_refusal=False, doc_code="X"), refusal)) == "le_ra_tu_choi"


def test_feedback_is_sparse_and_reviews_agree_less_with_worse_reviewer():
    truth = ["dung"] * 700 + ["sai_van_ban"] * 300
    reports = simulate_feedback([f"t{i}" for i in range(1000)], truth, seed=0)
    by_label = Counter(r["true_label"] for r in reports)
    assert 120 < by_label["sai_van_ban"] < 180 and by_label["dung"] < 40      # ~50% câu sai được báo, ~3% báo nhầm
    a, b = simulate_reviews(reports, seed=1, accuracy=0.95), simulate_reviews(reports, seed=2, accuracy=0.95)
    c = simulate_reviews(reports, seed=3, accuracy=0.6)
    assert set(a) <= set(ERROR_LABELS)
    assert cohen_kappa(a, b) > cohen_kappa(a, c)
