import json

import pytest

from p01kit import scoreboard
from p01kit.compare import before_after, compare_runs, journey, to_markdown


def _gold(r, **kw):
    return json.dumps({**r["gold"], **kw}, ensure_ascii=False)


def _outputs(rows, bad=lambda r: False, refuse=lambda r: False):
    out = []
    for r in rows:
        if bad(r):
            out.append("Xin lỗi, tôi không chắc.")                                   # không phải JSON
        elif refuse(r):
            out.append(_gold(r, is_refusal=True, doc_code=None))
        else:
            out.append(_gold(r))
    return out


def test_before_after_reports_metrics_slices_and_examples(splits):
    test = splits["test"]
    before = _outputs(test, bad=lambda r: r["style"] == "colloquial")               # base: hỏng câu đời thường
    after = _outputs(test, refuse=lambda r: r["id"] == test[0]["id"] and not r["is_unanswerable"])
    rep = before_after(before, after, test, "base", "sft")
    row = {r["metric"]: r for r in rep["table"]}
    assert row["json_valid"]["before"] == 0.5 and row["json_valid"]["after"] == 1.0 and row["json_valid"]["better"]
    assert row["fairness_gap"]["delta"] < 0 and row["fairness_gap"]["better"]       # gap giảm = tốt hơn
    colloq = next(s for s in rep["slices"] if s["slice"] == "style=colloquial")
    assert colloq["before"] == 0.0 and colloq["after"] > 0.9
    tr = rep["transitions"]
    assert tr["sửa_đúng"] == 50 and tr["sửa_đúng"] + tr["làm_hỏng"] + tr["vẫn_đúng"] + tr["vẫn_sai"] == 100
    assert rep["mcnemar"]["strict_ok"]["p_value"] < 0.001
    ex = rep["fixed_examples"][0]
    assert ex["before"].startswith("❌") and ex["error_after"] == "dung"
    md = to_markdown(rep)
    assert "base → sft" in md and "Ví dụ được sửa đúng" in md and "McNemar" in md


def test_brief_shows_code_written_next_to_a_refusal(splits):
    r = next(x for x in splits["test"] if x["is_unanswerable"])
    rep = before_after([_gold(r)], [_gold(r, doc_code="QĐ-BỊA")], [r])
    assert rep["broken_examples"][0]["after"] == "từ chối (vẫn ghi mã QĐ-BỊA)"


def test_before_after_rejects_misaligned_outputs(splits):
    test = splits["test"]
    with pytest.raises(ValueError):
        before_after(_outputs(test), _outputs(test)[:-1], test)


def test_record_saves_outputs_for_later_comparison(tmp_path, splits):
    test = splits["test"]
    board = tmp_path / "leaderboard.csv"
    scoreboard.record("M00", "base-zero-shot", {"n": 100}, path=board, outputs=_outputs(test, bad=lambda r: True))
    scoreboard.record("M06", "sft best/v1", {"n": 100}, path=board, outputs=_outputs(test))
    assert scoreboard.outputs_path("M06", "sft best/v1", tmp_path).name == "M06__sft_best_v1.json"
    assert scoreboard.load_outputs("M06", "sft best/v1", tmp_path)[0] == _gold(test[0])
    with pytest.raises(FileNotFoundError, match="M09/grpo"):
        scoreboard.load_outputs("M09", "grpo", tmp_path)


def test_legacy_output_files_are_found(tmp_path, splits):
    (tmp_path / "m00_base_outputs.json").write_text(json.dumps(_outputs(splits["test"])), encoding="utf-8")
    assert len(scoreboard.load_outputs("M00", "base-zero-shot", tmp_path)) == 100


def test_compare_runs_and_journey(tmp_path, splits, monkeypatch):
    monkeypatch.setenv("P01_RUNS", str(tmp_path))
    test = splits["test"]
    scoreboard.record("M00", "base", {"n": 100}, outputs=_outputs(test, bad=lambda r: r["style"] == "colloquial"))
    scoreboard.record("M06", "sft", {"n": 100}, outputs=_outputs(test))
    rep = compare_runs(("M00", "base"), ("M06", "sft"), test, display=False)
    assert rep["transitions"]["sửa_đúng"] == 50 and "M00/base → M06/sft" in rep["markdown"]
    md = journey([("M00", "base"), ("M06", "sft")], test, metrics=("json_valid", "strict_acc"), display=False)
    assert "| M06/sft | 1.000 | 1.000 |" in md
