import json

from p01kit.metrics import docs_f1, extract_json, gate, score_one, summarize
from p01kit.stats import bootstrap_ci, mcnemar_exact


def _row(splits, refusal=False, expired=False):
    return next(r for r in splits["test"] if r["is_unanswerable"] == refusal and r["is_expired"] == expired)


def test_extract_json_variants():
    assert extract_json('{"a": 1}')[0] == {"a": 1}
    assert extract_json('<think>\nsuy nghĩ {x}\n</think>\n\n{"a": "b}"}')[0] == {"a": "b}"}
    assert extract_json('```json\n{"a": 2}\n```')[0] == {"a": 2}
    assert extract_json("Đây là kết quả: {\"a\": 3} cảm ơn")[0] == {"a": 3}
    assert extract_json('{"a": 1')[1] == "truncated"
    assert extract_json("không có json")[1] == "no_json"


def test_perfect_answers_score_perfectly(splits):
    rows = splits["test"]
    scores = [score_one(json.dumps(r["gold"], ensure_ascii=False), r) for r in rows]
    s = summarize(scores, with_ci=False)
    for k in ("json_valid", "schema_ok", "refusal_acc", "citation_precision", "answer_acc", "expired_recall", "strict_acc"):
        assert s[k] == 1.0, k
    assert s["fairness_gap"] == 0.0 and s["false_expired_rate"] == 0.0
    assert all(ok for _, ok in gate(s).values())


def test_missing_expired_warning_is_caught(splits):
    r = _row(splits, expired=True)
    bad = dict(r["gold"], expired_warning=False)
    s = score_one(json.dumps(bad, ensure_ascii=False), r)
    assert s["citation_ok"] and not s["expired_ok"] and not s["strict_ok"]


def test_wrong_refusal_and_hallucinated_code(splits):
    r = _row(splits)
    s = score_one(json.dumps(dict(r["gold"], doc_code="QĐ-99/2026/BỊA"), ensure_ascii=False), r)
    assert s["cited"] and not s["citation_ok"]
    r2 = _row(splits, refusal=True)
    s2 = score_one(json.dumps(dict(r2["gold"], is_refusal=False, doc_code="X"), ensure_ascii=False), r2)
    assert not s2["refusal_ok"] and not s2["answer_ok"]


def test_docs_f1():
    assert docs_f1(["A", "B"], ["A", "B"]) == 1.0
    assert docs_f1(["A"], ["A", "B"]) == 2 / 3
    assert docs_f1([], ["A"]) == 0.0
    assert docs_f1(None, []) == 1.0


def test_bootstrap_and_mcnemar():
    lo, hi = bootstrap_ci([1.0] * 80 + [0.0] * 20)
    assert lo < 0.8 < hi and hi - lo < 0.2
    assert mcnemar_exact([True] * 10, [True] * 10)["p_value"] == 1.0
    res = mcnemar_exact([False] * 12 + [True] * 88, [True] * 12 + [True] * 88)
    assert res["c"] == 12 and res["p_value"] < 0.001
