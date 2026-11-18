# Data card — Văn bản an sinh xã hội thật (vbpl.vn)

Kho văn bản pháp luật **thật** dùng thay/bổ sung cho dữ liệu giả lập `data/mock_policies.json` khi xây kho truy xuất, sinh dữ liệu RAFT và đánh giá P01.

## Nguồn và giấy phép

- **Nguồn:** [`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) — thu thập từ [vbpl.vn](https://vbpl.vn) (Cơ sở dữ liệu quốc gia về văn bản pháp luật, Bộ Tư pháp). Bản làm mới ngày **2026-07-23**.
- **Giấy phép:** CC BY 4.0. Khi dùng phải ghi: *"Vietnamese Legal Documents — Thịnh Ngô (th1nhng0), nguồn vbpl.vn, CC BY 4.0"*. Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả (Điều 15 Luật Sở hữu trí tuệ).
- **Đã bỏ:** tên và chức danh người ký (không cần cho bài toán). Không có dữ liệu cá nhân của người dân.

## Tạo lại

```bash
pip install -r data_pipeline/requirements.txt
mkdir -p data/real/raw && cd data/real/raw
B=https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents/resolve/main/data
curl -fLO $B/metadata.parquet && curl -fLO $B/content.parquet    # ~15 MB + ~785 MB
cd - && python -m data_pipeline.vbpl_welfare --raw data/real/raw --out data/real   # ~15 giây
python -m data_pipeline.core_set                                            # kiểm tập lõi + xuất đoạn
```

| File | Commit? | Nội dung |
|---|---|---|
| `raw/*.parquet` | Không (gitignore) | Bản gốc tải về |
| `welfare_docs.jsonl` | Không (~45 MB) | 1 dòng / văn bản: metadata + toàn văn |
| `welfare_articles.jsonl` | Không (~43 MB) | 1 dòng / đoạn: cắt theo **Điều**, Điều dài > 4.000 ký tự chia theo dòng (`part`) |
| `manifest.csv` | Có | Danh sách văn bản, không có toàn văn |
| `stats.json` | Có | Số liệu tổng hợp |
| `supplements.csv`, `supplements/*.txt` | Có | Văn bản bổ sung tay: metadata, nguồn, SHA-256 file gốc, chữ đã làm sạch |
| `core_set.csv` | Có | Tập văn bản lõi đã kiểm hiệu lực (soạn tay) |
| `core_articles.jsonl` | Không | Các đoạn của tập lõi, trạng thái đã kiểm |

## Cách chọn

Tiêu đề chứa một trong các cụm: bảo hiểm xã hội (kể cả tự nguyện), bảo hiểm thất nghiệp, bảo hiểm y tế, trợ giúp / bảo trợ / trợ cấp xã hội, hưu trí xã hội, hộ nghèo / cận nghèo / chuẩn nghèo, lao động tự do / phi chính thức / người lao động gặp khó khăn, việc làm (trừ "vị trí việc làm" — định biên công chức, viên chức). Bỏ bản dịch tiếng Anh; văn bản trùng số ký hiệu + tiêu đề giữ 1.

## Số liệu (bản 2026-07-23)

| | |
|---|---|
| Văn bản | **2.847** = 2.846 từ cổng (2.599 có toàn văn; 247 chỉ có PDF) + 1 văn bản bổ sung tay |
| Đoạn truy xuất | **18.865** (trung vị ~800 ký tự) |
| Hiệu lực | còn hiệu lực 1.228 · hết hiệu lực 1.587 · chưa rõ 32 |
| Địa bàn | Toàn quốc 791 · 71 tỉnh/thành, quận/huyện · Không rõ 15 |
| Chủ đề | BHYT 972 · việc làm 551 · trợ giúp xã hội 586 · BHXH 448 · hộ nghèo 466 · BHTN 59 · BHXH tự nguyện 57 · hỗ trợ người lao động 33 · hưu trí xã hội 13 |

Đúng cấu trúc bài toán P01: mỗi văn bản có `region` (`Toàn quốc` hoặc tên tỉnh) để lọc kép, và `validity` (`active`/`expired`) để kiểm tra cảnh báo hết hiệu lực — trên dữ liệu thật, hơn một nửa văn bản đã hết hiệu lực.

## Tập văn bản lõi đã kiểm hiệu lực (`core_set.csv`)

41 văn bản trung ương làm nguồn **đáp án chuẩn** cho RAFT và đánh giá, đủ 8 nhóm: BHXH, BHXH tự nguyện, hưu trí xã hội, BHYT, BHTN, việc làm, trợ giúp xã hội, hộ nghèo. Mỗi nhóm có cả văn bản còn hiệu lực lẫn văn bản cũ đã bị thay — đúng cặp cần cho bài kiểm cảnh báo hết hiệu lực.

- File soạn tay; cột `check` cho biết độ tin: `primary` (điều khoản hiệu lực của văn bản mới hơn trong kho bãi bỏ văn bản này) 17 · `secondary` (nguồn chính phủ / văn bản hợp nhất, có URL) 1 · `portal_only` (chưa kiểm, dùng trạng thái cổng) 23. Kiểm ngày **2026-11-18**.
- `python -m data_pipeline.core_set` kiểm file (sai là dừng), in chỗ lệch cổng và xuất `core_articles.jsonl` (1.340 đoạn, không commit) với `validity` đã kiểm ghi đè cổng; trạng thái cổng giữ ở `validity_portal`.
- **Đã phát hiện cổng sai hoặc lỗi thời ở 5 văn bản:** Luật BHXH 41/2024/QH15 vẫn còn hiệu lực (cổng ghi hết hiệu lực); NĐ 20/2021 và NĐ 76/2024 về trợ giúp xã hội **hết hiệu lực từ 05/10/2026**, thay bởi NĐ 335/2026 (cổng vẫn ghi còn hiệu lực); NĐ 176/2025 đổi mức hưu trí xã hội 500.000 → 540.000 đ từ 01/07/2026; NĐ 30/2025 hết hiệu lực một phần từ 01/01/2027. Ngoài ra tiêu đề của TT 01/2025/TT-BLĐTBXH trên cổng không khớp toàn văn.
- NĐ 335/2026/NĐ-CP (ban hành sau ảnh chụp) được bổ sung tay — xem mục dưới; điều khoản hiệu lực của nó là bằng chứng gốc cho NĐ 20/2021, 76/2024 và 176/2025.

## Văn bản bổ sung tay (`supplements.csv`)

Văn bản ban hành sau ảnh chụp 2026-07-23 nhưng cần cho bài toán được thêm tay, rồi `vbpl_welfare` gộp vào kho (cột `text_source` ở manifest và mỗi đoạn). Nếu bộ gốc làm mới đã có cùng số ký hiệu thì bản cổng được dùng, bản bổ sung bị bỏ qua.

| Văn bản | Nguồn | Chữ |
|---|---|---|
| NĐ 335/2026/NĐ-CP — chính sách trợ giúp xã hội (hiệu lực 05/10/2026) | PDF scan có chữ ký số, 49 trang, trên CDN Cổng TTĐT Chính phủ (link + SHA-256 trong CSV) | OCR bằng Vision của macOS (`data_pipeline/tools/ocr_pdf.swift`), làm sạch bằng `supplements.clean_ocr_text`: bỏ dấu chữ ký số, số trang, khối ký và 6 mẫu phụ lục; 35 Điều |

**Chữ OCR chưa soát (`ocr_unreviewed`).** Số liệu đã đối chiếu (mức chuẩn 540.000 đ/tháng, ngày hiệu lực, danh sách văn bản bị bãi bỏ) nhưng còn lỗi dấu kiểu "trợ câp", "hô trợ", "QHT5". Trước khi train phải soát `supplements/335-2026-ND-CP.txt` với PDF rồi đổi `text_source` thành `ocr_reviewed`. Muốn OCR lại:

```bash
swiftc -O -o /tmp/ocr_pdf data_pipeline/tools/ocr_pdf.swift
/tmp/ocr_pdf nd335.pdf > nd335.ocr.txt
python -c "from pathlib import Path; from data_pipeline.supplements import clean_ocr_text as c; print(c(Path('nd335.ocr.txt').read_text()))" > data/real/supplements/335-2026-ND-CP.txt
```

## Giới hạn — đọc trước khi dùng làm đáp án chuẩn

1. **Ngoài tập lõi, trạng thái hiệu lực lấy nguyên từ cổng, chưa được kiểm.** `vbpl_welfare` chỉ **gắn cờ** (`status_conflict`) chứ không sửa. Chỉ dùng `core_articles.jsonl` làm đáp án chuẩn; muốn thêm văn bản vào đáp án chuẩn thì thêm dòng vào `core_set.csv` kèm bằng chứng. Với văn bản `partial`, xem `note` để biết điều nào còn hiệu lực.
2. **Ảnh chụp tại 2026-07-23.** Văn bản ban hành sau ngày này không có; trạng thái có thể đã đổi.
3. **Địa danh theo thời điểm ban hành.** Văn bản trước đợt sắp xếp đơn vị hành chính 2025 mang tên tỉnh cũ (ví dụ Tỉnh Bình Phước). Muốn lọc theo địa bàn hiện hành cần bảng quy đổi tỉnh cũ → tỉnh mới.
4. **Lọc theo tiêu đề** nên sót văn bản có nội dung an sinh mà tiêu đề không nêu, và có văn bản ít liên quan tới lao động phi chính thức (BHXH cho quân nhân, quản lý quỹ...). Dùng `topics` và `loai_van_ban` để lọc tiếp.
5. **Văn bản cấp tỉnh chiếm đa số** (2.056/2.847) và nhiều văn bản cũ; văn bản trung ương thường dài và có nhiều Điều.
