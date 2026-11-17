# Yêu cầu Dự án: Trợ lý Tra cứu Quyền lợi An sinh (P01)

- **Môn học:** Tư duy Trí tuệ nhân tạo (HK1 · 2026–2027)
- **Đích SDG:** `1.3` — Mở rộng độ phủ an sinh xã hội cho người nghèo và nhóm dễ bị tổn thương *(theo [Extended-Report-2026_Goal-1.pdf](Extended-Report-2026_Goal-1.pdf), tỷ lệ phủ toàn cầu năm 2023 mới đạt 52.4%, trợ cấp thất nghiệp chỉ 16.7%)*.
- **Định hướng cốt lõi:** Xây dựng **nguyên mẫu (prototype) chạy được** kèm **hồ sơ thiết kế & kiểm thử đầy đủ** theo 7 đặc tính Responsible AI và khung NIST AI RMF (`GOVERN`, `MAP`, `MEASURE`, `MANAGE`).

---

## 1. Phạm vi & Chức năng tối thiểu (Functional Requirements)

1. **Đầu vào:** Câu hỏi tiếng Việt (ngôn ngữ chuẩn hoặc đời thường) + **địa bàn giả lập** được chọn.
2. **Tiền xử lý & Bảo vệ riêng tư:**
   - Che (mask/redact) thông tin định danh cá nhân (CCCD, SĐT, họ tên) trước khi ghi log.
   - Chuẩn hóa cách diễn đạt đời thường sang thuật ngữ hành chính.
3. **Truy xuất & Kiểm soát hiệu lực (RAG):**
   - Lọc văn bản theo địa bàn giả lập và truy xuất các điều khoản liên quan.
   - Kiểm tra **trạng thái hiệu lực** (`active` / `expired`) và **ngưỡng tin cậy (Decision Threshold)**.
4. **Đầu ra & Cơ chế an toàn (Human-in-the-loop):**
   - **Trường hợp hợp lệ:** Trả danh sách giấy tờ cần chuẩn bị, nơi nộp hồ sơ, kèm **trích dẫn gốc** (tên văn bản, điều khoản, ngày hiệu lực, đoạn trích).
   - **Trường hợp văn bản hết hiệu lực:** Không tự tin hướng dẫn theo nguồn cũ; cảnh báo **"Cần xác minh"** và gợi ý chuyển cán bộ.
   - **Trường hợp thiếu địa bàn / dưới ngưỡng tin cậy / sai định dạng:** Từ chối an toàn, yêu cầu bổ sung hoặc hướng dẫn đường chuyển cán bộ.

---

## 2. Yêu cầu Dữ liệu & Vận hành (Data & MLOps)

- **Dữ liệu giả lập (tách riêng khỏi mã nguồn):** Tập văn bản an sinh giả lập có metadata bắt buộc (`doc_id`, `title`, `version`, `effective_date`, `expiry_date`, `status`, `region`, `submission_place`, `clauses`), bao gồm cả văn bản còn hiệu lực và hết hiệu lực.
- **Tách tập đánh giá (No Data Leakage):** Bộ câu hỏi kiểm thử giữ riêng (gồm câu chuẩn, câu đời thường, ca hết hiệu lực, ca thiếu địa bàn, ca chứa PII) tách biệt với dữ liệu/từ điển dùng để tinh chỉnh.
- **Đối chiếu Baseline:** So sánh hiệu năng với phương án đơn giản (tìm kiếm từ khóa thuần túy).
- **Nhật ký & Tái lập (Audit Log & Reproducibility):** Ghi nhận phiên bản văn bản, câu hỏi đã che PII, điểm tin cậy, quyết định hệ thống và không làm lộ API key/secret trong log.

---

## 3. Yêu cầu Minh chứng Responsible AI (100 điểm)

| Trục đánh giá | Điểm | Yêu cầu kiểm chứng bắt buộc |
|---|---|---|
| **Hợp lệ & Tin cậy** | 20 | Chốt ngưỡng trước khi thử; đo tỷ lệ hướng dẫn đúng trên tập giữ riêng; so với baseline & phân tích lỗi. |
| **Công bằng** | 15 | So sánh kết quả giữa **câu hỏi chuẩn** và **câu hỏi đời thường** (cỡ mẫu, chênh lệch, biện pháp giảm lệch). |
| **Bảo mật & Chống chịu** | 15 | Thử câu hỏi thiếu địa bàn, văn bản hết hiệu lực, tệp/input sai định dạng; kiểm tra log không lộ secret. |
| **An toàn** | 15 | Mô phỏng hậu quả nộp thiếu giấy tờ; kiểm thử ngưỡng cảnh báo, từ chối và đường chuyển cán bộ. |
| **Giải thích được** | 10 | Hiển thị điều khoản, ngày hiệu lực, đoạn trích gốc và mức chắc chắn của kết quả. |
| **Quyền riêng tư** | 10 | Kiểm chứng cơ chế che số định danh (CCCD/SĐT) trước khi ghi log; mô tả luồng thu/lưu/xóa dữ liệu. |
| **Trách nhiệm & Minh bạch** | 15 | Ghi phiên bản văn bản; phân vai người duyệt/cập nhật nguồn hết hiệu lực & người có quyền dừng hệ thống. |

---

## 4. Sản phẩm bàn giao

- Mã nguồn nguyên mẫu + hướng dẫn cài đặt, chạy thử.
- Dữ liệu mẫu giả lập & bộ kiểm thử tự động (pass 100%).
- Báo cáo ngắn (mục tiêu, luồng dữ liệu, kết quả đo 7 trục, bảng rủi ro NIST AI RMF, giới hạn hệ thống) và nhật ký đóng góp nhóm.
