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
```

| File | Commit? | Nội dung |
|---|---|---|
| `raw/*.parquet` | Không (gitignore) | Bản gốc tải về |
| `welfare_docs.jsonl` | Không (~45 MB) | 1 dòng / văn bản: metadata + toàn văn |
| `welfare_articles.jsonl` | Không (~43 MB) | 1 dòng / đoạn: cắt theo **Điều**, Điều dài > 4.000 ký tự chia theo dòng (`part`) |
| `manifest.csv` | Có | Danh sách văn bản, không có toàn văn |
| `stats.json` | Có | Số liệu tổng hợp |

## Cách chọn

Tiêu đề chứa một trong các cụm: bảo hiểm xã hội (kể cả tự nguyện), bảo hiểm thất nghiệp, bảo hiểm y tế, trợ giúp / bảo trợ / trợ cấp xã hội, hưu trí xã hội, hộ nghèo / cận nghèo, lao động tự do / phi chính thức / người lao động gặp khó khăn, việc làm. Bỏ bản dịch tiếng Anh; văn bản trùng số ký hiệu + tiêu đề giữ 1.

## Số liệu (bản 2026-07-23)

| | |
|---|---|
| Văn bản | **2.937** (2.683 có toàn văn; 254 chỉ có PDF trên cổng) |
| Đoạn truy xuất | **20.478** (trung vị ~800 ký tự) |
| Hiệu lực | còn hiệu lực 1.289 · hết hiệu lực 1.611 · chưa rõ 37 |
| Địa bàn | Toàn quốc 845 · 71 tỉnh/thành, quận/huyện · Không rõ 15 |
| Chủ đề | BHYT 972 · việc làm 670 · trợ giúp xã hội 585 · BHXH 448 · hộ nghèo 438 · BHTN 59 · BHXH tự nguyện 57 · hỗ trợ người lao động 33 · hưu trí xã hội 13 |

Đúng cấu trúc bài toán P01: mỗi văn bản có `region` (`Toàn quốc` hoặc tên tỉnh) để lọc kép, và `validity` (`active`/`expired`) để kiểm tra cảnh báo hết hiệu lực — trên dữ liệu thật, hơn một nửa văn bản đã hết hiệu lực.

## Giới hạn — đọc trước khi dùng làm đáp án chuẩn

1. **Trạng thái hiệu lực lấy nguyên từ cổng, chưa được kiểm.** Script chỉ **gắn cờ** (`status_conflict`) chứ không sửa. 3 văn bản bị cờ, gồm *Luật Bảo hiểm xã hội 41/2024/QH15* được ghi "hết hiệu lực toàn bộ" từ 01/01/2026, chỉ 6 tháng sau khi có hiệu lực. Trước khi dùng một văn bản làm đáp án chuẩn (gold) phải đối chiếu trên vbpl.vn; với văn bản "hết hiệu lực một phần" phải xem điều nào còn hiệu lực.
2. **Ảnh chụp tại 2026-07-23.** Văn bản ban hành sau ngày này không có; trạng thái có thể đã đổi.
3. **Địa danh theo thời điểm ban hành.** Văn bản trước đợt sắp xếp đơn vị hành chính 2025 mang tên tỉnh cũ (ví dụ Tỉnh Bình Phước). Muốn lọc theo địa bàn hiện hành cần bảng quy đổi tỉnh cũ → tỉnh mới.
4. **Lọc theo tiêu đề** nên sót văn bản có nội dung an sinh mà tiêu đề không nêu, và có văn bản ít liên quan tới lao động phi chính thức (BHXH cho quân nhân, quản lý quỹ...). Dùng `topics` và `loai_van_ban` để lọc tiếp.
5. **Văn bản cấp tỉnh chiếm đa số** (2.092/2.937) và nhiều văn bản cũ; văn bản trung ương thường dài và có nhiều Điều.
