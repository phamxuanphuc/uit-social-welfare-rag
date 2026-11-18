"""Văn bản bổ sung tay cho kho `data/real/` — văn bản mới hơn ảnh chụp của bộ vbpl.vn.

`data/real/supplements.csv` (commit) giữ metadata cùng tên cột với bộ gốc, kèm nguồn tải, SHA-256 của
file gốc và đường dẫn file chữ. File chữ nằm trong `data/real/supplements/` và **được commit** để soát
tay bằng git diff (văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả).

Nhiều văn bản trên Cổng Chính phủ chỉ có bản PDF scan có chữ ký số, nên chữ được lấy bằng OCR
(`data_pipeline/tools/ocr_pdf.swift`, Vision của macOS) rồi làm sạch bằng `clean_ocr_text`.
OCR còn lỗi dấu, nên cột `text_source` ghi `ocr_unreviewed` cho tới khi có người soát xong.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from .vbpl_welfare import nfc

TEXT_SOURCES = ("vbpl_html", "ocr_unreviewed", "ocr_reviewed")
FIELDS = ("id", "so_ky_hieu", "title", "loai_van_ban", "ngay_ban_hanh", "ngay_co_hieu_luc", "ngay_het_hieu_luc",
          "co_quan_ban_hanh", "pham_vi", "tinh_trang_hieu_luc", "text_file", "text_source", "source_url",
          "source_sha256", "note")

_PAGE_MARK = re.compile(r"^<<<PAGE \d+>>>$")
_PAGE_NUM = re.compile(r"^\d{1,3}$")
# Dấu chữ ký số của Cổng Chính phủ in ở đầu trang 1 — không phải nội dung văn bản.
_STAMP = re.compile(r"^(Cơ quan phát hành|Người ký|Email|Cơ quan|Thời gian ký)\s*:")
# "Điều 4." mở một Điều; "Điều 4 Nghị định này..." chỉ là câu dẫn chiếu bị ngắt dòng → không mở khối.
_ARTICLE_HEAD = re.compile(r"^Điều\s+\d+[a-zđ]?\.")
_BLOCK_START = re.compile(r"^(Chương\s+[IVXLC\d]+\b|Mục\s+\d+|Điều\s+\d+[a-zđ]?\.|\d+\.\s|[a-zđ]\)\s|[-+•]\s|[\"“]|"
                          r"Phụ lục|PHỤ LỤC|Mẫu số|Căn cứ\s)")
_END_MARKERS = ("Nơi nhận:",)


def clean_ocr_text(raw: str) -> str:
    """Bỏ dấu trang, số trang, dấu chữ ký số; ghép dòng bị ngắt; cắt từ 'Nơi nhận:' (khối ký, phụ lục mẫu)."""
    paras: list[str] = []
    for line in nfc(raw).splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or _PAGE_MARK.match(line) or _PAGE_NUM.match(line) or _STAMP.match(line):
            continue
        if line.startswith(_END_MARKERS):
            break
        starts_block = bool(_BLOCK_START.match(line)) or (line.isupper() and len(line) >= 6)
        # Tên Điều dài bị ngắt dòng thì dòng sau viết thường; dòng viết hoa sau tên Điều là phần thân.
        after_heading = bool(_ARTICLE_HEAD.match(paras[-1])) and line[:1].isupper() if paras else False
        if paras and not starts_block and not after_heading \
                and not paras[-1].endswith((".", ":", ";", "!", "?")) \
                and not (paras[-1].isupper() and not line.isupper()):
            paras[-1] = f"{paras[-1]} {line}"
        else:
            paras.append(line)
    return "\n".join(paras)


def load_supplements(csv_path: str | Path) -> list[tuple[dict, str]]:
    """Đọc metadata + chữ của văn bản bổ sung. File chữ tính theo thư mục chứa CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    out = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            missing = [k for k in FIELDS if k not in r]
            if missing:
                raise ValueError(f"{csv_path}: thiếu cột {missing}")
            if r["text_source"] not in TEXT_SOURCES:
                raise ValueError(f"{r['so_ky_hieu']}: text_source '{r['text_source']}' không thuộc {TEXT_SOURCES}")
            if not re.fullmatch(r"[0-9a-f]{64}", r["source_sha256"]):
                raise ValueError(f"{r['so_ky_hieu']}: source_sha256 phải là SHA-256 hex")
            text = (csv_path.parent / r["text_file"]).read_text(encoding="utf-8")
            meta = {k: (r[k] or None) for k in FIELDS}
            out.append((meta, nfc(text)))
    return out
