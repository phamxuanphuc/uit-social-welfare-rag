"""Lọc văn bản pháp luật THẬT về an sinh xã hội từ bộ `th1nhng0/vietnamese-legal-documents` (nguồn vbpl.vn).

Bộ gốc (CC BY 4.0) có 171 nghìn văn bản kèm metadata hiệu lực, cơ quan ban hành và quan hệ giữa các văn bản.
Script này chọn văn bản thuộc phạm vi P01 (BHXH, BHTN, BHYT, trợ giúp xã hội, hưu trí xã hội, hộ nghèo,
hỗ trợ người lao động, việc làm), chuyển HTML sang chữ, cắt theo Điều và ghi ra:

- `welfare_docs.jsonl`     — 1 dòng / văn bản: metadata + toàn văn
- `welfare_articles.jsonl` — 1 dòng / Điều: đơn vị truy xuất cho RAG/RAFT
- `manifest.csv`           — danh sách văn bản (không có toàn văn), nhỏ, commit được vào repo

Chạy (cần `pyarrow`):
    python -m data_pipeline.vbpl_welfare --raw data/real/raw --out data/real
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

SOURCE = "th1nhng0/vietnamese-legal-documents"
SOURCE_URL = "https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents"
SNAPSHOT = date(2026, 7, 23)          # ngày bộ gốc làm mới dữ liệu (README của bộ dữ liệu)
NATIONAL = "Toàn quốc"

# Chủ đề P01 → mẫu tìm trong tiêu đề (đã NFC + chữ thường). Một văn bản có thể thuộc nhiều chủ đề.
TOPICS: dict[str, str] = {
    "bhxh_tu_nguyen": r"bảo hiểm xã hội tự nguyện",
    "bhxh": r"bảo hiểm xã hội",
    "bhtn": r"bảo hiểm thất nghiệp",
    "bhyt": r"bảo hiểm y tế",
    "tro_giup_xa_hoi": r"trợ giúp xã hội|bảo trợ xã hội|trợ cấp xã hội",
    "huu_tri_xa_hoi": r"hưu trí xã hội",
    "ho_ngheo": r"hộ nghèo|cận nghèo",
    "ho_tro_nguoi_lao_dong": r"lao động tự do|lao động phi chính thức|người lao động.{0,60}(?:khó khăn|covid)",
    "viec_lam": r"việc làm",
}
_TOPIC_RE = {k: re.compile(v) for k, v in TOPICS.items()}
EXCLUDED_TYPES = {"Bản dịch văn bản"}                 # bản dịch tiếng Anh, trùng nội dung bản gốc

ACTIVE = {"Còn hiệu lực", "Hết hiệu lực một phần", "Ngưng hiệu lực một phần"}
EXPIRED = {"Hết hiệu lực toàn bộ", "Ngưng hiệu lực", "Không còn phù hợp"}
PARTIAL = {"Hết hiệu lực một phần", "Ngưng hiệu lực một phần"}

# Chỉ giữ các trường cần cho truy xuất; bỏ tên/chức danh người ký (không cần cho bài toán).
KEEP_FIELDS = ("id", "so_ky_hieu", "title", "loai_van_ban", "ngay_ban_hanh", "ngay_co_hieu_luc",
               "ngay_het_hieu_luc", "co_quan_ban_hanh", "pham_vi", "tinh_trang_hieu_luc")
MANIFEST_FIELDS = ("id", "so_ky_hieu", "title", "loai_van_ban", "ngay_ban_hanh", "ngay_co_hieu_luc",
                   "ngay_het_hieu_luc", "tinh_trang_hieu_luc", "validity", "partially_expired", "status_conflict",
                   "region", "co_quan_ban_hanh", "topics", "n_chars", "n_articles")


def nfc(s: str | None) -> str:
    return unicodedata.normalize("NFC", s or "")


def topics_for_title(title: str | None) -> list[str]:
    t = nfc(title).lower()
    return [k for k, rx in _TOPIC_RE.items() if rx.search(t)]


def parse_vn_date(s: str | None) -> date | None:
    """'01/07/2025' → date(2025, 7, 1); chuỗi rỗng hoặc sai định dạng → None."""
    m = re.fullmatch(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*", s or "")
    if not m:
        return None
    try:
        return date(int(m[3]), int(m[2]), int(m[1]))
    except ValueError:
        return None


def validity(status_raw: str | None) -> str:
    """Quy về 3 giá trị: active / expired / unknown (Chưa xác định, Chưa có hiệu lực, trống)."""
    s = nfc(status_raw).strip()
    if s in ACTIVE:
        return "active"
    if s in EXPIRED:
        return "expired"
    return "unknown"


def status_conflict(row: dict, snapshot: date = SNAPSHOT) -> str | None:
    """Phát hiện metadata tự mâu thuẫn — KHÔNG sửa, chỉ gắn cờ để người duyệt kiểm lại trên vbpl.vn."""
    v = validity(row.get("tinh_trang_hieu_luc"))
    end = parse_vn_date(row.get("ngay_het_hieu_luc"))
    start = parse_vn_date(row.get("ngay_co_hieu_luc"))
    if v == "active" and end is not None and end <= snapshot and nfc(row.get("tinh_trang_hieu_luc")) not in PARTIAL:
        return "còn hiệu lực nhưng ngày hết hiệu lực đã qua"
    if v == "expired" and start is not None and end is not None and end < start:
        return "ngày hết hiệu lực trước ngày có hiệu lực"
    if v == "expired" and start is not None and end is not None and (end - start).days < 365 \
            and row.get("loai_van_ban") in {"Luật", "Bộ luật"}:
        return "luật hết hiệu lực chưa tới 1 năm sau khi có hiệu lực"
    return None


UNKNOWN_REGION = "Không rõ"
_AUTH_PREFIX = re.compile(r"^(?:(?:ct|chủ tịch)\s+)?(?:lâm thời\s+)?(?:ubnd|hđnd|ủy ban nhân dân|uỷ ban nhân dân|"
                          r"hội đồng nhân dân)?\s*(?:lâm thời\s+)?", re.I)
_PLACE = re.compile(r"(?i)^(tỉnh|thành phố|tp\.|huyện|quận|thị xã|xã|phường)\s+(.+)$")
_LOCAL_AGENCY = re.compile(r"(?i)^(sở|ban|phòng|hội đồng nhân dân|ủy ban nhân dân|ubnd|hđnd|chưa xác định)\b")


def region_of(row: dict) -> str:
    """Địa bàn áp dụng: văn bản trung ương → 'Toàn quốc'; văn bản địa phương → tên tỉnh/thành từ cơ quan ban hành.

    - Cơ quan trung ương (BHXH Việt Nam, Hội đồng Bộ trưởng...) dù bị ghi `pham_vi=Địa phương` vẫn là 'Toàn quốc'.
    - Cơ quan địa phương không ghi địa danh ('Sở Lao động...', 'Hội đồng nhân dân') → 'Không rõ'.
    - Tên lấy NGUYÊN theo thời điểm ban hành (trước/sau sáp nhập tỉnh 2025), không quy đổi.
    """
    scope = nfc(row.get("pham_vi")).strip().rstrip(".")
    auth = re.sub(r"\s+", " ", nfc(row.get("co_quan_ban_hanh"))).strip()
    if scope in ("Trung ương", NATIONAL):
        return NATIONAL
    if scope and scope != "Địa phương":
        return _title_place(scope)
    place = _AUTH_PREFIX.sub("", auth, count=1)
    if _PLACE.match(place):
        return _title_place(place)
    if not auth or _LOCAL_AGENCY.match(auth) or place != auth:
        return UNKNOWN_REGION
    return NATIONAL                      # cơ quan trung ương không kèm địa danh


def _title_place(s: str) -> str:
    """'TỈNH BÌNH PHƯỚC' / 'tỉnh Cần thơ' / 'Tp. Hồ Chí Minh' → 'Tỉnh Bình Phước' / 'Tỉnh Cần Thơ' / 'Thành phố Hồ Chí Minh'."""
    s = re.sub(r"\s+", " ", s).strip()
    m = _PLACE.match(s)
    if not m:
        return " ".join(w[:1].upper() + w[1:] for w in s.split(" "))
    kind = "Thành phố" if m[1].lower() in ("thành phố", "tp.") else m[1].capitalize()
    name = " ".join(w if w == "-" else w[:1].upper() + w[1:].lower() for w in m[2].split(" "))
    return f"{kind} {name}"


# ---------------------------------------------------------------------------
# HTML → chữ, cắt theo Điều
# ---------------------------------------------------------------------------

_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section", "article",
          "blockquote", "ul", "ol"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(html: str | None) -> str:
    p = _TextExtractor()
    p.feed(html or "")
    p.close()
    text = nfc("".join(p.parts)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


_ARTICLE_RE = re.compile(r"^(Điều\s+(\d+[a-zđ]?)\s*[.:]?)\s*(.*)$", re.M)


def split_articles(text: str) -> list[dict]:
    """Cắt toàn văn thành các Điều. Văn bản không có 'Điều N' trả về 1 khối duy nhất (số hiệu None).

    Quyết định ban hành kèm quy chế thường có 2 dãy 'Điều 1...' (của quyết định và của quy chế) →
    giữ thứ tự xuất hiện qua `index`, không gộp theo số Điều.
    """
    matches = list(_ARTICLE_RE.finditer(text))
    if not matches:
        return [{"index": 0, "article": None, "heading": "", "text": text.strip()}] if text.strip() else []
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.start():end].strip()
        out.append({"index": i, "article": m[2], "heading": m[3].strip(), "text": body})
    return out


MAX_CHUNK_CHARS = 4000      # ~p95 độ dài một Điều; Điều dài hơn (thường là phụ lục bảng biểu) bị chia theo dòng


def split_long(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Chia một khối dài thành các phần ≤ max_chars, cắt ở ranh giới dòng (dòng đơn quá dài thì cắt cứng)."""
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + 1 + len(line) > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[dict]:
    """Điều → các đoạn ≤ max_chars; mỗi đoạn ghi `part` (0, 1, ...) và giữ số Điều để trích dẫn."""
    out = []
    for a in split_articles(text):
        for p, piece in enumerate(split_long(a["text"], max_chars)):
            out.append({**a, "part": p, "text": piece})
    return out


# ---------------------------------------------------------------------------
# Chọn văn bản và đọc toàn văn từ parquet
# ---------------------------------------------------------------------------

def select_documents(rows: Iterable[dict]) -> list[dict]:
    """Lọc theo chủ đề, bỏ bản dịch, khử trùng (cùng số ký hiệu + tiêu đề → giữ 1, ưu tiên id số nhỏ hơn)."""
    seen: dict[tuple[str, str], dict] = {}
    for r in rows:
        if r.get("loai_van_ban") in EXCLUDED_TYPES:
            continue
        topics = topics_for_title(r.get("title"))
        if not topics:
            continue
        doc = {k: (nfc(r.get(k)) if isinstance(r.get(k), str) else r.get(k)) for k in KEEP_FIELDS}
        doc["id"] = str(doc["id"])
        doc["topics"] = topics
        doc["validity"] = validity(r.get("tinh_trang_hieu_luc"))
        doc["partially_expired"] = nfc(r.get("tinh_trang_hieu_luc")) in PARTIAL
        doc["status_conflict"] = status_conflict(r)
        doc["region"] = region_of(r)
        key = (nfc(r.get("so_ky_hieu")).strip(), nfc(r.get("title")).strip().lower())
        prev = seen.get(key)
        if prev is None or _id_order(doc["id"]) < _id_order(prev["id"]):
            seen[key] = doc
    return sorted(seen.values(), key=lambda d: _id_order(d["id"]))


def _id_order(i: str) -> tuple[int, int | str]:
    return (0, int(i)) if i.isdigit() else (1, i)


def iter_content(content_path: str | Path, wanted: set[str], batch_size: int = 256) -> Iterator[tuple[str, str]]:
    """Đọc (id, html) cho các id cần, theo từng row group: chỉ giải nén cột html ở row group có id cần."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(content_path)
    for rg in range(pf.num_row_groups):
        ids = [str(x) for x in pf.read_row_group(rg, columns=["id"]).column("id").to_pylist()]
        if not wanted.intersection(ids):
            continue
        tbl = pf.read_row_group(rg, columns=["id", "content_html"])
        for i, html in zip(tbl.column("id").to_pylist(), tbl.column("content_html").to_pylist()):
            if str(i) in wanted:
                yield str(i), html


def build(raw: str | Path, out: str | Path) -> dict:
    import pyarrow.parquet as pq

    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pq.read_table(raw / "metadata.parquet", columns=list(KEEP_FIELDS)).to_pylist()
    docs = select_documents(meta)
    by_id = {d["id"]: d for d in docs}
    texts: dict[str, str] = {}
    for i, html in iter_content(raw / "content.parquet", set(by_id)):
        texts[i] = html_to_text(html)

    n_art = 0
    with open(out / "welfare_docs.jsonl", "w", encoding="utf-8") as fd, \
            open(out / "welfare_articles.jsonl", "w", encoding="utf-8") as fa:
        for d in docs:
            text = texts.get(d["id"], "")
            arts = chunks(text)
            d["n_chars"] = len(text)
            d["n_articles"] = sum(a["article"] is not None and a["part"] == 0 for a in arts)
            fd.write(json.dumps({**d, "text": text}, ensure_ascii=False) + "\n")
            for a in arts:
                fa.write(json.dumps({"doc_id": d["id"], "so_ky_hieu": d["so_ky_hieu"], "region": d["region"],
                                     "validity": d["validity"], "topics": d["topics"], **a},
                                    ensure_ascii=False) + "\n")
                n_art += 1
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for d in docs:
            w.writerow({**d, "topics": ";".join(d["topics"]), "status_conflict": d["status_conflict"] or ""})
    stats = summarize(docs, n_art)
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return stats


def summarize(docs: list[dict], n_articles: int) -> dict:
    from collections import Counter

    return {"source": SOURCE, "snapshot": SNAPSHOT.isoformat(), "n_docs": len(docs),
            "n_docs_with_text": sum(d.get("n_chars", 0) > 0 for d in docs), "n_article_chunks": n_articles,
            "validity": dict(Counter(d["validity"] for d in docs)),
            "status_conflicts": sum(bool(d["status_conflict"]) for d in docs),
            "national": sum(d["region"] == NATIONAL for d in docs),
            "regions": len({d["region"] for d in docs}),
            "topics": dict(Counter(t for d in docs for t in d["topics"])),
            "types": dict(Counter(d["loai_van_ban"] for d in docs).most_common(10))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default="data/real/raw", help="thư mục chứa metadata.parquet và content.parquet")
    ap.add_argument("--out", default="data/real")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.raw, a.out), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
