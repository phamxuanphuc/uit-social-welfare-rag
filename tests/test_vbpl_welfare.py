import csv
import json
import unicodedata

import pytest

from data_pipeline import vbpl_welfare as vw


def row(**kw):
    base = {"id": "1", "so_ky_hieu": "176/2025/NĐ-CP", "title": "Nghị định hướng dẫn về trợ cấp hưu trí xã hội",
            "loai_van_ban": "Nghị định", "ngay_ban_hanh": "30/06/2025", "ngay_co_hieu_luc": "01/07/2025",
            "ngay_het_hieu_luc": None, "co_quan_ban_hanh": "Chính phủ", "pham_vi": "Trung ương",
            "tinh_trang_hieu_luc": "Còn hiệu lực", "nguoi_ky": "Nguyễn Văn A"}
    base.update(kw)
    return base


# --- chủ đề -------------------------------------------------------------------

def test_topics_match_title_and_nfd_input():
    assert vw.topics_for_title("Luật Bảo hiểm xã hội") == ["bhxh"]
    assert set(vw.topics_for_title("Hướng dẫn bảo hiểm xã hội tự nguyện")) == {"bhxh", "bhxh_tu_nguyen"}
    nfd = unicodedata.normalize("NFD", "Quy định chính sách trợ giúp xã hội")
    assert vw.topics_for_title(nfd) == ["tro_giup_xa_hoi"]
    assert vw.topics_for_title("Hỗ trợ người lao động gặp khó khăn do đại dịch COVID-19") == ["ho_tro_nguoi_lao_dong"]
    assert vw.topics_for_title("Quy hoạch đô thị") == []


# --- hiệu lực -----------------------------------------------------------------

@pytest.mark.parametrize("raw,want", [("Còn hiệu lực", "active"), ("Hết hiệu lực một phần", "active"),
                                      ("Hết hiệu lực toàn bộ", "expired"), ("Ngưng hiệu lực", "expired"),
                                      ("Chưa xác định", "unknown"), ("Chưa có hiệu lực", "unknown"), (None, "unknown")])
def test_validity_mapping(raw, want):
    assert vw.validity(raw) == want


def test_parse_vn_date():
    assert vw.parse_vn_date("01/07/2025").isoformat() == "2025-07-01"
    assert vw.parse_vn_date("31/02/2025") is None
    assert vw.parse_vn_date("") is None and vw.parse_vn_date(None) is None


def test_status_conflict_flags_without_fixing():
    assert vw.status_conflict(row()) is None
    assert vw.status_conflict(row(ngay_het_hieu_luc="01/01/2020")) == "còn hiệu lực nhưng ngày hết hiệu lực đã qua"
    # hết hiệu lực một phần: ngày hết hiệu lực có thể là của phần bị bãi bỏ → không coi là mâu thuẫn
    assert vw.status_conflict(row(tinh_trang_hieu_luc="Hết hiệu lực một phần", ngay_het_hieu_luc="01/01/2020")) is None
    assert vw.status_conflict(row(tinh_trang_hieu_luc="Hết hiệu lực toàn bộ", ngay_het_hieu_luc="01/01/2020")) \
        == "ngày hết hiệu lực trước ngày có hiệu lực"
    law = row(loai_van_ban="Luật", tinh_trang_hieu_luc="Hết hiệu lực toàn bộ", ngay_het_hieu_luc="01/01/2026")
    assert vw.status_conflict(law) == "luật hết hiệu lực chưa tới 1 năm sau khi có hiệu lực"


# --- địa bàn ------------------------------------------------------------------

@pytest.mark.parametrize("scope,auth,want", [
    ("Trung ương", "Chính phủ", "Toàn quốc"),
    ("Toàn quốc", "Quốc hội", "Toàn quốc"),
    ("Địa phương", "UBND Thành phố Hồ Chí Minh", "Thành phố Hồ Chí Minh"),
    ("Địa phương", "HĐND tỉnh Bình Định", "Tỉnh Bình Định"),
    ("Địa phương", "CT UBND Tỉnh Thanh Hóa", "Tỉnh Thanh Hóa"),
    ("Địa phương", "Lâm Thời Tỉnh Hậu giang", "Tỉnh Hậu Giang"),
    ("Địa phương", "Bảo hiểm Xã hội Việt Nam", "Toàn quốc"),
    ("Địa phương", "Sở Lao động thương binh và xã hội", "Không rõ"),
    ("Địa phương", "Hội đồng nhân dân", "Không rõ"),
    ("Địa phương", None, "Không rõ"),
    ("TỈNH BÌNH PHƯỚC", "UBND tỉnh Bình Phước", "Tỉnh Bình Phước"),
    ("Tỉnh Tuyên Quang.", "UBND tỉnh Tuyên Quang", "Tỉnh Tuyên Quang"),
])
def test_region_of(scope, auth, want):
    assert vw.region_of(row(pham_vi=scope, co_quan_ban_hanh=auth)) == want


def test_title_place_keeps_hyphenated_names():
    assert vw._title_place("tỉnh bà rịa - vũng tàu") == "Tỉnh Bà Rịa - Vũng Tàu"
    assert vw._title_place("Tp. Hồ Chí Minh") == "Thành phố Hồ Chí Minh"


# --- HTML → chữ, cắt Điều ------------------------------------------------------

HTML = ("<html><head><style>p{color:red}</style></head><body><table><tr><td><p>CHÍNH PHỦ</p></td>"
        "<td><p>Số: 1/2025/NĐ-CP</p></td></tr></table><p>Căn cứ Luật&nbsp;Bảo hiểm xã hội;</p>"
        "<p><strong>Điều 1.</strong> Phạm vi điều chỉnh</p><p>Nghị định này quy định...</p>"
        "<p>Điều 2. Đối tượng</p><p>a) Từ đủ 75 tuổi;</p><script>alert(1)</script></body></html>")


def test_html_to_text_drops_markup_and_keeps_blocks():
    t = vw.html_to_text(HTML)
    assert "color:red" not in t and "alert" not in t and "<" not in t
    assert "Căn cứ Luật Bảo hiểm xã hội;" in t                     # &nbsp; → dấu cách
    assert "Điều 1. Phạm vi điều chỉnh" in t.split("\n")
    assert unicodedata.is_normalized("NFC", t)


def test_split_articles_and_preamble():
    arts = vw.split_articles(vw.html_to_text(HTML))
    assert [a["article"] for a in arts] == ["1", "2"]
    assert arts[0]["heading"] == "Phạm vi điều chỉnh"
    assert "75 tuổi" in arts[1]["text"] and "Căn cứ" not in arts[0]["text"]
    assert vw.split_articles("Chỉ thị không chia điều") == [{"index": 0, "article": None, "heading": "",
                                                            "text": "Chỉ thị không chia điều"}]
    assert vw.split_articles("   ") == []


def test_split_articles_keeps_duplicate_numbers_in_order():
    text = "Điều 1. Ban hành quy chế\nĐiều 2. Hiệu lực\nQUY CHẾ\nĐiều 1. Phạm vi"
    arts = vw.split_articles(text)
    assert [(a["index"], a["article"]) for a in arts] == [(0, "1"), (1, "2"), (2, "1")]


def test_split_long_respects_limit_and_loses_nothing():
    text = "\n".join(f"dòng {i} " + "x" * 30 for i in range(50)) + "\n" + "y" * 250
    parts = vw.split_long(text, max_chars=100)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
    assert vw.split_long("ngắn", 100) == ["ngắn"]


def test_chunks_number_parts_per_article():
    text = "Điều 1. Dài\n" + "\n".join("a" * 60 for _ in range(5)) + "\nĐiều 2. Ngắn"
    ch = vw.chunks(text, max_chars=150)
    assert [(c["article"], c["part"]) for c in ch][-1] == ("2", 0)
    assert {c["part"] for c in ch if c["article"] == "1"} == set(range(len([c for c in ch if c["article"] == "1"])))


# --- chọn văn bản ---------------------------------------------------------------

def test_select_documents_filters_dedupes_and_drops_signers():
    rows = [row(id="20"), row(id="7"),                                         # trùng số + tiêu đề → giữ id 7
            row(id="3", title="Quy hoạch đô thị"),                              # ngoài chủ đề
            row(id="4", loai_van_ban="Bản dịch văn bản"),                       # bản dịch
            row(id="5", so_ky_hieu="12/2020/QĐ-UBND", title="Quyết định hỗ trợ hộ nghèo",
                pham_vi="Địa phương", co_quan_ban_hanh="UBND tỉnh Nghệ An", tinh_trang_hieu_luc="Hết hiệu lực toàn bộ")]
    docs = vw.select_documents(rows)
    assert [d["id"] for d in docs] == ["5", "7"]
    assert "nguoi_ky" not in docs[0]
    local = docs[0]
    assert (local["region"], local["validity"], local["topics"]) == ("Tỉnh Nghệ An", "expired", ["ho_ngheo"])


# --- đọc parquet và build đầu-cuối ----------------------------------------------

def _write_raw(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    meta = [row(id="1"), row(id="2", so_ky_hieu="2/2025/QĐ-UBND", title="Quyết định về bảo trợ xã hội",
                             pham_vi="Địa phương", co_quan_ban_hanh="UBND Thành phố Đà Nẵng"),
            row(id="3", title="Quy hoạch đô thị", so_ky_hieu="3/2025")]
    pq.write_table(pa.Table.from_pylist(meta), tmp_path / "metadata.parquet")
    content = [{"id": "1", "content_html": HTML}, {"id": "3", "content_html": "<p>khác</p>"},
               {"id": "2", "content_html": "<p>Điều 1. Mức trợ cấp</p><p>360.000 đồng</p>"}]
    pq.write_table(pa.Table.from_pylist(content), tmp_path / "content.parquet", row_group_size=1)
    return tmp_path


def test_iter_content_reads_only_wanted_ids(tmp_path):
    raw = _write_raw(tmp_path)
    got = dict(vw.iter_content(raw / "content.parquet", {"2"}))
    assert list(got) == ["2"] and "360.000" in got["2"]


def test_build_end_to_end(tmp_path):
    raw = _write_raw(tmp_path)
    out = tmp_path / "out"
    stats = vw.build(raw, out)
    assert stats["n_docs"] == 2 and stats["n_docs_with_text"] == 2 and stats["national"] == 1
    docs = [json.loads(x) for x in (out / "welfare_docs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {d["id"] for d in docs} == {"1", "2"} and all("nguoi_ky" not in d for d in docs)
    arts = [json.loads(x) for x in (out / "welfare_articles.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {(a["doc_id"], a["article"]) for a in arts} == {("1", "1"), ("1", "2"), ("2", "1")}
    assert all(a["region"] in ("Toàn quốc", "Thành phố Đà Nẵng") for a in arts)
    with open(out / "manifest.csv", encoding="utf-8") as f:
        man = list(csv.DictReader(f))
    assert [m["id"] for m in man] == ["1", "2"] and "text" not in man[0]
    assert json.loads((out / "stats.json").read_text(encoding="utf-8"))["n_article_chunks"] == 3
