import json

import pytest

from p01kit import scoreboard
from p01kit.runs import once, result_path, resumable, resume_point


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setenv("P01_RUNS", str(tmp_path / "p01_runs"))
    return tmp_path / "p01_runs"


def test_once_skips_finished_work(runs):
    calls = []
    fn = lambda: calls.append(1) or {"summary": {"ci": (0.1, 0.2)}, "ok": [True, False]}
    first = once("m03", "V1", fn)
    again = once("m03", "V1", fn)
    assert calls == [1] and first == again == {"summary": {"ci": [0.1, 0.2]}, "ok": [True, False]}
    assert result_path("m03", "V1") == runs / "m03" / "results" / "V1.json"
    once("m03", "V1", fn, force=True)
    assert calls == [1, 1]


def test_once_does_not_save_failed_runs(runs):
    def boom():
        raise RuntimeError("Colab ngắt giữa chừng")
    with pytest.raises(RuntimeError):
        once("m05", "A", boom)
    assert not result_path("m05", "A").exists() and not result_path("m05", "A").with_suffix(".tmp").exists()
    with pytest.raises(ValueError):
        once("m05", "../thoat-ra-ngoai", lambda: 1)


def test_resumable_decorator_keys_by_variant_name(runs):
    seen = []

    @resumable("m03")
    def run(name, r=16):
        seen.append((name, r))
        return {"name": name, "r": r}

    assert run("V2", r=32) == {"name": "V2", "r": 32}
    assert run("V2", r=32) == {"name": "V2", "r": 32} and seen == [("V2", 32)]
    run("V3", r=64)
    assert seen == [("V2", 32), ("V3", 64)] and run.__name__ == "run"


def test_resume_point_picks_latest_complete_checkpoint(tmp_path):
    assert resume_point(tmp_path / "chua-co") is None
    for step, complete in ((17, True), (34, True), (51, False)):        # 51: Colab ngắt khi đang ghi
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if complete:
            (d / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    (tmp_path / "checkpoint-abc").mkdir()
    assert resume_point(tmp_path) == str(tmp_path / "checkpoint-34")


def test_scoreboard_rerun_replaces_row(tmp_path):
    path = tmp_path / "leaderboard.csv"
    scoreboard.record("M03", "V1", {"n": 100, "strict_acc": 0.5}, path=path)
    scoreboard.record("M03", "V2", {"n": 100, "strict_acc": 0.6}, path=path)
    scoreboard.record("M03", "V1", {"n": 100, "strict_acc": 0.55}, path=path)   # chạy lại V1 sau khi Colab ngắt
    rows = scoreboard.load(path)
    assert [(r["variant"], r["strict_acc"]) for r in rows] == [("V2", "0.6"), ("V1", "0.55")]
    assert not path.with_suffix(".tmp").exists()
