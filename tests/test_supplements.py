import csv
import json

import pytest

from data_pipeline import supplements as sp
from data_pipeline import vbpl_welfare as vw

OCR = """<<<PAGE 1>>>
Cơ quan phát hành: VĂN PHÒNG CHÍNH PHỦ
CHÍNH PHỦ
Người ký: CỤC THÔNG TIN
Thời gian ký: 22.08.2026 14:52:33 +07:00
NGHỊ ĐỊNH
Quy định chính sách trợ giúp xã hội
Chương I
QUY ĐỊNH CHUNG
Điều 1. Phạm vi điều chỉnh
Nghị định này quy định về chính sách trợ giúp xã hội thường xuyên tại
cộng đồng.
Điều 2. Hồ sơ đề nghị hưởng trợ cấp xã hội và nhận chăm sóc,
nuôi dưỡng tại cộng đồng
1. Tờ khai theo Mẫu số 01 quy định tại khoản 1
<<<PAGE 2>>>
2
Điều 4 Nghị định này.
2. Mức chuẩn trợ giúp xã hội là 540.000 đồng/tháng.
Nơi nhận:
- Ban Bí thư Trung ương Đảng;
TM. CHÍNH PHỦ
"""


def test_clean_ocr_text_drops_noise_and_rejoins_lines():
    t = sp.clean_ocr_text(OCR)
    assert "Người ký" not in t and "Thời gian ký" not in t and "<<<PAGE" not in t
    assert "Nơi nhận" not in t and "TM. CHÍNH PHỦ" not in t          # cắt khối ký và phụ lục
    lines = t.split("\n")
    assert "Điều 1. Phạm vi điều chỉnh" in lines                       # tên Điều không dính phần thân
    assert "Nghị định này quy định về chính sách trợ giúp xã hội thường xuyên tại cộng đồng." in lines
    assert "Điều 2. Hồ sơ đề nghị hưởng trợ cấp xã hội và nhận chăm sóc, nuôi dưỡng tại cộng đồng" in lines
    # "Điều 4 Nghị định này" ở đầu dòng là câu dẫn chiếu bị ngắt, không phải Điều mới; số trang "2" bị bỏ
    assert "1. Tờ khai theo Mẫu số 01 quy định tại khoản 1 Điều 4 Nghị định này." in lines
    assert [a["article"] for a in vw.split_articles(t)] == ["1", "2"]


def _write_supplement(tmp_path, **kw):
    (tmp_path / "supplements").mkdir(exist_ok=True)
    (tmp_path / "supplements" / "a.txt").write_text("Điều 1. Mức chuẩn\n540.000 đồng/tháng.\n", encoding="utf-8")
    row = {"id": "sup-1", "so_ky_hieu": "335/2026/NĐ-CP", "title": "Nghị định quy định chính sách trợ giúp xã hội",
           "loai_van_ban": "Nghị định", "ngay_ban_hanh": "21/08/2026", "ngay_co_hieu_luc": "05/10/2026",
           "ngay_het_hieu_luc": "", "co_quan_ban_hanh": "Chính phủ", "pham_vi": "Trung ương",
           "tinh_trang_hieu_luc": "Còn hiệu lực", "text_file": "supplements/a.txt", "text_source": "ocr_unreviewed",
           "source_url": "https://example.gov.vn/a.pdf", "source_sha256": "a" * 64, "note": ""}
    row.update(kw)
    path = tmp_path / "supplements.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sp.FIELDS)
        w.writeheader()
        w.writerow(row)
    return path


def test_load_supplements_reads_text_and_empty_fields_as_none(tmp_path):
    [(meta, text)] = sp.load_supplements(_write_supplement(tmp_path))
    assert meta["ngay_het_hieu_luc"] is None and meta["text_source"] == "ocr_unreviewed"
    assert text.startswith("Điều 1. Mức chuẩn")
    assert sp.load_supplements(tmp_path / "missing.csv") == []


@pytest.mark.parametrize("kw,msg", [({"text_source": "guess"}, "text_source"),
                                    ({"source_sha256": "abc"}, "SHA-256")])
def test_load_supplements_rejects_bad_rows(tmp_path, kw, msg):
    with pytest.raises(ValueError, match=msg):
        sp.load_supplements(_write_supplement(tmp_path, **kw))


def _write_raw(tmp_path, so="1/2025/NĐ-CP"):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    raw = tmp_path / "raw"
    raw.mkdir()
    meta = [{"id": "1", "so_ky_hieu": so, "title": "Nghị định về trợ giúp xã hội", "loai_van_ban": "Nghị định",
             "ngay_ban_hanh": "01/01/2025", "ngay_co_hieu_luc": "01/02/2025", "ngay_het_hieu_luc": None,
             "co_quan_ban_hanh": "Chính phủ", "pham_vi": "Trung ương", "tinh_trang_hieu_luc": "Còn hiệu lực"}]
    pq.write_table(pa.Table.from_pylist(meta), raw / "metadata.parquet")
    pq.write_table(pa.Table.from_pylist([{"id": "1", "content_html": "<p>Điều 1. Cũ</p>"}]), raw / "content.parquet")
    return raw


def test_build_adds_supplement_with_text_source(tmp_path):
    raw, out = _write_raw(tmp_path), tmp_path / "out"
    stats = vw.build(raw, out, _write_supplement(tmp_path))
    assert (stats["n_docs"], stats["supplements"], stats["supplements_skipped"]) == (2, 1, [])
    arts = [json.loads(x) for x in (out / "welfare_articles.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(a["doc_id"], a["text_source"]) for a in arts} == {("1", "vbpl_html"), ("sup-1", "ocr_unreviewed")}
    with open(out / "manifest.csv", encoding="utf-8") as f:
        sup = [m for m in csv.DictReader(f) if m["id"] == "sup-1"][0]
    assert (sup["region"], sup["validity"], sup["text_source"]) == ("Toàn quốc", "active", "ocr_unreviewed")


def test_build_skips_supplement_already_in_portal(tmp_path):
    raw, out = _write_raw(tmp_path, so="335/2026/NĐ-CP"), tmp_path / "out"
    stats = vw.build(raw, out, _write_supplement(tmp_path))
    assert (stats["n_docs"], stats["supplements_skipped"]) == (1, ["335/2026/NĐ-CP"])
