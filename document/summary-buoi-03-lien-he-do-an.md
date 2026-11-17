# Tóm tắt Buổi học 03 & Liên hệ Đồ án (Đề tài P01)

- **Nguồn:** Ghi chép buổi học số 03 (12/11/2026) — *Từ nguyên mẫu tới Vận hành hệ thống AI (MLOps)*, GV: TS. Phan Thế Duy.
- **Mục tiêu:** Tổng hợp các định hướng đồ án, lưu ý riêng cho đề tài **P01** và cách áp dụng kiến thức MLOps vào **Rubric Responsible AI**.

---

## 1. Định hướng chung về Đồ án (Rất quan trọng)

- **Trọng tâm chấm điểm là Tư duy thiết kế & Responsible AI, không phải phần mềm hoàn chỉnh:**
  - Không yêu cầu xây dựng một sản phẩm phần mềm thương mại hoàn chỉnh 100% tính năng.
  - Bắt buộc có **nguyên mẫu (prototype) chạy được** để demo luồng chính và các ca kiểm thử.
  - Trong quỹ thời gian ngắn (4–5 tuần): **phần thiết kế và lập luận phải đầy đủ** theo 7 đặc tính Responsible AI và gắn với mục tiêu **SDG (2026–2030)**; phần code prototype chỉ cần hiện thực đủ để chứng minh và kiểm thử các ý chính.
- **Tư duy Product Owner:** Đặt mình vào vai trò người thiết kế hệ thống AI phục vụ cộng đồng, lường trước tác động xã hội và nhóm người yếu thế có thể chịu hại khi AI sai.

---

## 2. Liên hệ trực tiếp đến Đề tài P01

> **P01:** *Tra cứu quyền lợi an sinh từ văn bản công khai* (Đích SDG `1.3` — đối tượng: người lao động phi chính thức).

- **Nhắc nhở trực tiếp của thầy cho P01:**
  - Bản chất giống bài toán tra cứu pháp luật: cần đặt giả thuyết và bao quát rủi ro ngay từ khâu thiết kế.
  - **Rủi ro điển hình:** Hệ thống đọc nhầm tài liệu giả mạo, nguồn không chính thống hoặc văn bản đã hết hiệu lực rồi trích dẫn vào câu trả lời → tư vấn sai, gây mất an toàn cho người dân.
- **Hàm ý cho hệ thống RAG của P01:**
  - Phải kiểm soát chặt nguồn văn bản đầu vào (chỉ dùng nguồn công khai hợp lệ hoặc dữ liệu giả lập có gắn nhãn phiên bản & ngày hiệu lực).
  - Bắt buộc có cơ chế phát hiện văn bản hết hiệu lực để cảnh báo **"cần xác minh"** hoặc chuyển cán bộ, thay vì tự tin trả lời sai.

---

## 3. Áp dụng kiến thức MLOps (Buổi 03) vào Đồ án & Rubric

| Khái niệm Buổi 03 | Nội dung chính | Áp dụng vào P01 & Rubric Responsible AI |
|---|---|---|
| **Mô hình vs. Hệ thống** | Đạt điểm cao trên tập test cũ chưa đủ để đưa vào vận hành; phải xét tác động của **bỏ sót (FN)** và **cảnh báo nhầm (FP)**. | **An toàn & Hợp lệ:** Phân tích hậu quả khi hệ thống bỏ sót giấy tờ bắt buộc hoặc hướng dẫn nhầm nơi nộp hồ sơ. |
| **Chia tập Train / Val / Test & Data Leakage** | Tách biệt tập tinh chỉnh (validation) và tập đánh giá cuối (test), tránh rò rỉ dữ liệu ("lộ đề"). | **Hợp lệ & Tin cậy:** Chốt ngưỡng trước khi thử, đánh giá trên bộ câu hỏi giữ riêng và so sánh với phương án đơn giản (baseline). |
| **Ngưỡng quyết định (Decision Threshold)** | Ngưỡng quá cao → bỏ sót nhiều; ngưỡng quá thấp → cảnh báo nhầm/nhiễu nhiều. | **An toàn & Giải thích:** Thiết lập ngưỡng độ tin cậy truy xuất để quyết định khi nào trả lời, khi nào cảnh báo hoặc từ chối/chuyển cán bộ. |
| **Data Contract & Nhật ký dữ liệu** | Quy định rõ cấu trúc, nguồn, đơn vị, phiên bản dữ liệu, người xác nhận và cách xử lý mẫu lỗi. | **Trách nhiệm & Minh bạch:** Ghi rõ phiên bản văn bản, ngày hiệu lực, quy trình duyệt/cập nhật khi có văn bản hết hiệu lực. |
| **Tái lập (Reproducibility) & Versioning** | Lưu hồ sơ lần chạy (run record): phiên bản code, dữ liệu, cấu hình, ngưỡng, kết quả test. | **Trách nhiệm & Minh bạch:** Đáp ứng yêu cầu "hồ sơ ngắn về mục đích, nguồn dữ liệu, phiên bản, giới hạn và cách tái tạo". |
| **Quality Gate & Triển khai an toàn** | Cổng kiểm soát chất lượng giữa các bước; chạy thử **Shadowing → Canary → Full** và có phương án **Rollback**. | **Bảo mật & Chống chịu:** Mô tả cổng kiểm thử trước khi demo/phát hành, xử lý lỗi định dạng, không lộ API key trong log, cách khôi phục khi lỗi. |
| **Giám sát 3 lớp & Drift** | Giám sát Dữ liệu – Mô hình – Dịch vụ; phát hiện **Data Drift / Concept Drift** khi bối cảnh thay đổi. | **Trách nhiệm & Bảo mật:** Chính sách an sinh thay đổi theo thời gian (Concept/Data Drift) → cần cơ chế gắn cờ văn bản cũ và cập nhật kho tri thức. |

---

## 4. Lưu ý thêm từ cuối buổi học

- **Tài liệu tham khảo:** Thầy có gửi 3 bài giảng về Responsible AI/Ethics, tài liệu MLOps và đồ án khóa trước (chỉ dùng để tham khảo cấu trúc, không coi là chuẩn điểm tối đa vì đề tài và rubric học kỳ này đã thay đổi).
- **Sản phẩm cần chuẩn bị dần:** Song song với code RAG prototype, cần làm rõ luồng dữ liệu, bảng rủi ro (NIST AI RMF: `GOVERN`, `MAP`, `MEASURE`, `MANAGE`), bộ test ca chuẩn/ca khó và nhật ký đóng góp nhóm.
