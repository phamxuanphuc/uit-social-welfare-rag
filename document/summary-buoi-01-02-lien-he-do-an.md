# Tổng hợp Kiến thức Buổi 1 & Buổi 2 cho Đồ án (Đề tài P01)

- **Nguồn:** Tóm tắt kiến thức cốt lõi Buổi 1 (*Tổng quan & Responsible AI*) và Buổi 2 (*Tư duy Dữ liệu & Mô hình*).
- **Mục đích:** Áp dụng trực tiếp các khung tư duy của môn học vào phần phân tích, thiết kế và đánh giá đề tài **P01 — Tra cứu quyền lợi an sinh từ văn bản công khai**.

---

## 1. Khung 5 câu hỏi AI Thinking (Buổi 1) áp dụng cho P01

1. **Bối cảnh & Vấn đề:** Người lao động phi chính thức gặp khó khăn khi tự đọc văn bản hành chính để biết cần giấy tờ gì và nộp ở đâu (gắn với Đích SDG `1.3`).
2. **Giá trị của AI:** Trợ lý truy xuất trích dẫn (RAG) giúp chuyển đổi câu hỏi đời thường thành danh sách giấy tờ + nơi nộp kèm trích dẫn điều khoản, ngày hiệu lực.
3. **Dữ liệu:** Dùng văn bản công khai hợp lệ hoặc dữ liệu giả lập có metadata rõ ràng (`version`, `effective_date`, `status`, `region`); tuyệt đối không lưu số định danh cá nhân (CCCD/SĐT) thô vào log.
4. **Kiểm chứng:** Đo trên tập câu hỏi giữ riêng, đối chiếu với phương án đơn giản (baseline từ khóa), tách riêng ca chuẩn vs. ca đời thường và ca hết hiệu lực.
5. **Tác động & Trách nhiệm:** Nếu AI hướng dẫn thiếu giấy tờ hoặc dùng văn bản hết hiệu lực, người dân có thể bị trả hồ sơ/mất quyền lợi → bắt buộc có **Human-in-the-loop** (cảnh báo xác minh + đường chuyển cán bộ hỗ trợ).

---

## 2. Mô tả bài toán P01 theo 5 thành phần (Buổi 2)

| Thành phần | Áp dụng cụ thể cho Đề tài P01 |
|---|---|
| **Người dùng (User)** | Người lao động phi chính thức (giả lập) và cán bộ tiếp nhận/duyệt thông tin. |
| **Quyết định / Hành động** | Chuẩn bị đúng bộ giấy tờ và đến đúng nơi nộp hồ sơ; hoặc **tạm dừng để xác minh / chuyển cán bộ** nếu nguồn hết hiệu lực hoặc thiếu chắc chắn. |
| **Đầu vào (Input)** | Câu hỏi tiếng Việt (chuẩn hoặc đời thường) + địa bàn giả lập được chọn. |
| **Đầu ra (Output)** | Danh sách giấy tờ + nơi nộp + trích dẫn (điều khoản, ngày hiệu lực, đoạn trích) + mức độ tin cậy / cảnh báo. |
| **Tiêu chí thành công (Success Metric)** | Tỷ lệ hướng dẫn đúng (giấy tờ + nơi nộp) vượt baseline; phát hiện 100% văn bản hết hiệu lực ở ca demo; không chênh lệch lớn giữa câu hỏi chuẩn và đời thường. |

---

## 3. Tư duy Dữ liệu & Các cạm bẫy cần tránh (Buổi 2)

- **Đơn vị quan sát:** Mỗi đơn vị tri thức truy xuất là một **điều khoản / mục thủ tục** gắn với `địa bàn` và `trạng thái hiệu lực` cụ thể.
- **Đặc trưng (Features) & Che định danh (Privacy):** Loại bỏ hoặc che (mask) các thông tin định danh cá nhân (CCCD, SĐT, họ tên) trước khi đưa vào log hoặc suy luận.
- **Chất lượng dữ liệu:** Đảm bảo tính **đúng – đủ – nhất quán – kịp thời – đại diện** (bao phủ cả văn phong hành chính lẫn cách nói đời thường của người lao động).
- **3 cạm bẫy cần tránh trong P01:**
  - *Proxy metrics:* Không đánh giá chất lượng hệ thống chỉ qua độ dài câu trả lời hay tốc độ phản hồi; phải đo theo **đáp án chuẩn giấy tờ & nơi nộp**.
  - *Data leakage:* Tuyệt đối tách biệt bộ câu hỏi đánh giá giữ riêng (test) khỏi bộ ví dụ/từ đồng nghĩa dùng để tinh chỉnh (train/validation).
  - *Nhầm tương quan với nhân quả:* Không suy đoán quyền lợi chỉ vì trùng từ khóa chung nếu sai địa bàn hoặc sai đối tượng.

---

## 4. Tư duy Mô hình & Đánh giá toàn diện (Buổi 1 & 2)

- **Không chỉ nhìn Accuracy tổng thể:**
  - **Bỏ sót (False Negative):** Bỏ sót một giấy tờ bắt buộc hoặc bỏ sót cảnh báo văn bản hết hiệu lực → người dùng bị trả hồ sơ, mất thời gian/chi phí (rủi ro cao).
  - **Cảnh báo nhầm (False Positive):** Hệ thống báo "cần xác minh / chuyển cán bộ" hơi nhiều → tốn thêm công sức kiểm tra nhưng **an toàn hơn** việc tự tin trả lời sai.
- **Đánh giá theo phân nhóm (Subgroup Evaluation):** So sánh kết quả giữa nhóm **câu hỏi chuẩn** và **câu hỏi đời thường**, cũng như giữa các địa bàn giả lập khác nhau để phát hiện thiên lệch (Bias).
- **Tránh Overfitting & Black-box:** Luôn hiển thị **căn cứ trích dẫn gốc** (điều khoản, ngày hiệu lực) để người dùng tự kiểm chứng thay vì để AI trả lời kiểu "hộp đen".
