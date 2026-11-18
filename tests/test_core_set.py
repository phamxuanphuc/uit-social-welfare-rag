import csv
import json
from datetime import date
from pathlib import Path

import pytest

from data_pipeline import core_set as cs

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 18)


def man(id_, so, validity="active", partial="False"):
    return {"id": id_, "so_ky_hieu": so, "validity": validity, "partially_expired": partial}


def core(id_, so, status, check, **kw):
    base = {"id": id_, "so_ky_hieu": so, "group": "bhxh", "status": status, "check": check,
            "effective_until": "", "superseded_by": "", "evidence": "", "checked_on": "", "note": ""}
    base.update(kw)
    return base


MANIFEST = {m["id"]: m for m in [man("1", "41/2024/QH15", "expired"), man("2", "58/2014/QH13", "expired"),
                                 man("3", "20/2021/NĐ-CP", "active", "True"), man("4", "159/2025/NĐ-CP")]}


@pytest.mark.parametrize("m,want", [(man("x", "a"), "active"), (man("x", "a", partial="True"), "partial"),
                                    (man("x", "a", "expired"), "expired"), (man("x", "a", "unknown"), "unknown")])
def test_portal_status(m, want):
    assert cs.portal_status(m) == want


def test_valid_core_passes_and_reports_disagreements():
    rows = [core("1", "41/2024/QH15", "active", "secondary", evidence="https://example.gov.vn/x",
                 checked_on="2026-11-18"),
            core("2", "58/2014/QH13", "expired", "primary", evidence="Điều 140 Luật 41/2024/QH15",
                 checked_on="2026-11-18", effective_until="2025-07-01"),
            core("3", "20/2021/NĐ-CP", "partial", "portal_only"),
            core("4", "159/2025/NĐ-CP", "active", "portal_only")]
    assert cs.validate(rows, MANIFEST, TODAY) == []
    assert [(d["so_ky_hieu"], d["portal"], d["checked"]) for d in cs.disagreements(rows, MANIFEST)] \
        == [("41/2024/QH15", "expired", "active")]


@pytest.mark.parametrize("row,msg", [
    (core("9", "x", "active", "portal_only"), "không có trong manifest"),
    (core("4", "999/2025/NĐ-CP", "active", "portal_only"), "số ký hiệu khác"),
    (core("4", "159/2025/NĐ-CP", "valid", "secondary", evidence="https://a.gov.vn", checked_on="2026-11-18"),
     "status 'valid'"),
    (core("4", "159/2025/NĐ-CP", "active", "guess"), "check 'guess'"),
    (core("4", "159/2025/NĐ-CP", "expired", "portal_only"), "khác cổng"),
    (core("4", "159/2025/NĐ-CP", "active", "portal_only", checked_on="2026-11-18"), "không được có checked_on"),
    (core("1", "41/2024/QH15", "active", "primary", checked_on="2026-11-18"), "cần evidence"),
    (core("1", "41/2024/QH15", "active", "secondary", evidence="báo nói vậy", checked_on="2026-11-18"), "URL https"),
    (core("1", "41/2024/QH15", "active", "primary", evidence="Điều 1", checked_on="2027-01-01"), "tương lai"),
    (core("2", "58/2014/QH13", "expired", "primary", evidence="Điều 140", checked_on="2026-11-18"),
     "cần effective_until"),
    (core("1", "41/2024/QH15", "active", "primary", evidence="Điều 1", checked_on="2026-11-18",
          effective_until="2026-01-01"), "active nhưng có effective_until"),
])
def test_validate_rejects_bad_rows(row, msg):
    errors = cs.validate([row], MANIFEST, TODAY)
    assert len(errors) == 1 and msg in errors[0]


def test_validate_rejects_duplicate_ids():
    row = core("4", "159/2025/NĐ-CP", "active", "portal_only")
    assert any("trùng id" in e for e in cs.validate([row, dict(row)], MANIFEST, TODAY))


def test_export_overrides_portal_validity(tmp_path):
    arts = tmp_path / "a.jsonl"
    arts.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in [
        {"doc_id": "1", "so_ky_hieu": "41/2024/QH15", "validity": "expired", "article": "1", "text": "t"},
        {"doc_id": "3", "so_ky_hieu": "20/2021/NĐ-CP", "validity": "active", "article": "2", "text": "t"},
        {"doc_id": "7", "so_ky_hieu": "khác", "validity": "active", "article": "1", "text": "t"}]) + "\n",
        encoding="utf-8")
    rows = [core("1", "41/2024/QH15", "active", "secondary"),
            core("3", "20/2021/NĐ-CP", "expired", "secondary", effective_until="2026-10-05",
                 superseded_by="335/2026/NĐ-CP")]
    out = tmp_path / "o.jsonl"
    assert cs.export_articles(rows, arts, out) == 2
    got = {a["doc_id"]: a for a in map(json.loads, out.read_text(encoding="utf-8").splitlines())}
    assert (got["1"]["validity"], got["1"]["validity_portal"]) == ("active", "expired")
    assert (got["3"]["validity"], got["3"]["superseded_by"], got["3"]["effective_until"]) \
        == ("expired", "335/2026/NĐ-CP", "2026-10-05")


def test_committed_core_set_is_valid_against_committed_manifest():
    """File tay `data/real/core_set.csv` luôn khớp `manifest.csv` đã commit."""
    core_rows = cs.load_csv(ROOT / "data/real/core_set.csv")
    manifest = {m["id"]: m for m in cs.load_csv(ROOT / "data/real/manifest.csv")}
    assert cs.validate(core_rows, manifest, TODAY) == []
    assert {r["group"] for r in core_rows} >= {"bhxh", "bhxh_tu_nguyen", "huu_tri_xa_hoi", "bhyt", "bhtn",
                                               "viec_lam", "tro_giup_xa_hoi", "ho_ngheo"}


def test_main_exits_nonzero_on_invalid_core(tmp_path, capsys):
    bad = tmp_path / "core.csv"
    with open(bad, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cs.FIELDS)
        w.writeheader()
        w.writerow(core("nope", "x", "active", "portal_only"))
    assert cs.main(["--core", str(bad), "--manifest", str(ROOT / "data/real/manifest.csv"),
                    "--articles", str(tmp_path / "none.jsonl")]) == 1
    assert "không có trong manifest" in capsys.readouterr().err
