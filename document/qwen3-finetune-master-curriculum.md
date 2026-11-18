# Đề cương Thực chiến: Fine-tune `Qwen3-1.7B` từ Số 0 đến Master (Phương pháp Feynman)

- **Mô hình xuyên suốt 100% khóa học:** `Qwen/Qwen3-1.7B` (chạy trên Google Colab T4 15 GB VRAM).
- **Triết lý chống nhàm chán & giữ động lực cao nhất:**
  1. **Mỗi buổi học = 1 Sản phẩm thực tế chạy được ngay** trên Colab (có giao diện thử nghiệm hoặc hàm kiểm tra tự động).
  2. **Luôn có đo lường 🔴 Trước vs. 🟢 Sau Fine-tune** ở cuối mỗi buổi để thấy tận mắt mô hình thông minh lên bao nhiêu `%`.
  3. **Từ chia nhỏ $\rightarrow$ Gộp nhanh:** Ở **Buổi 0, 1, 2, 3**, code được tách thành từng **ô siêu nhỏ (4–10 dòng/ô)** kèm giải thích Feynman từng dòng; sang **Buổi 4, 5, 6**, khi bạn đã thuộc lòng các lệnh cơ bản, code cũ được gộp gọn để tập trung vào kỹ thuật nâng cao (`RAFT`, `ORPO/GRPO`, `imatrix GGUF`).

---

## 1. Giải thích 10 Khái niệm Cốt lõi theo Phương pháp Feynman

| Khái niệm kỹ thuật | Giải thích kiểu Feynman (Hình ảnh đời thường) | Tác dụng thực chiến trên `Qwen3-1.7B` |
|---|---|---|
| **1. Pre-trained vs. Fine-tune** | **Pre-trained** là cử nhân mới ra trường (biết đọc viết tiếng Việt nhưng chưa rành nghiệp vụ riêng). **Fine-tune** là khóa đào tạo nội bộ giúp cử nhân đó làm đúng quy trình cơ quan bạn. | Không cần dạy lại tiếng Việt; chỉ dạy cách trích xuất đúng quy định và định dạng đầu ra. |
| **2. Tokenizer & ChatML (`enable_thinking=False`)** | Máy tính tra **sổ danh bạ (~151,936 mảnh từ)** để đổi chữ thành số. **ChatML** là các "nhãn vai diễn" (`system`, `user`, `assistant`). | Đặt `enable_thinking=False` để tắt đoạn tự lẩm bẩm `<think>` dài dòng, giúp trả lời nhanh gấp 3 lần. |
| **3. Lượng tử hóa 4-bit (`NF4`)** | Thay vì lưu ảnh gốc nặng (`FP16` tốn ~3.4 GB), ta nén thành ảnh chất lượng cao siêu nhẹ (`4-bit NF4` chỉ tốn **1.27 GB**), khi tính toán mới tạm bung từng mảnh ra. | Chừa hơn 13 GB VRAM trống trên card Colab T4 để huấn luyện thoải mái không sợ tràn bộ nhớ (OOM). |
| **4. `LoRA` (`r`, `lora_alpha`)** | Không sửa trực tiếp vào sách giáo khoa gốc ($W_0$ đóng băng), mà dán thêm **tờ giấy ghi chú nhỏ ($B \cdot A$)** bên lề (chỉ học **~0.1% – 1% tham số**). | Học xong trong 1 phút, không làm quên tiếng Việt gốc, bật/tắt tức thì bằng `disable_adapter()`. |
| **5. `DoRA` + `rsLoRA`** | **`DoRA`** tách riêng **ga (biên độ)** và **vô-lăng (hướng)** để bẻ lái chính xác; **`rsLoRA`** giữ tay lái ổn định theo $\frac{\alpha}{\sqrt{r}}$ khi tăng kích thước bộ nhớ phụ `r`. | Giúp `Qwen3-1.7B` tuân thủ cấu trúc JSON nhiều trường và kiểm tra ngày hiệu lực chuẩn như Full Fine-tune. |
| **6. `train_on_responses_only`** | Khi chấm bài cho học sinh, giáo viên **chỉ chấm phần bài làm (`assistant`)**, không chấm điểm phần chép lại đề bài (`<context>` + `user`). | Dồn 100% trí lực của mô hình vào việc học cách trả lời chuẩn thay vì học thuộc lòng đề bài. |
| **7. `NEFTune` (`noise_alpha=5`)** | Tập nghe tiếng Việt trong quán cà phê hơi ồn một chút: khi quen nghe trong nhiễu nhẹ, học sinh sẽ **nắm ý chính** thay vì học vẹt từng chữ. | Chống **Học vẹt (Overfitting)** cực tốt khi dữ liệu ít, giúp hiểu mượt từ lóng/đời thường. |
| **8. `Unsloth` + `Liger Kernel`** | Nếu `transformers` gốc là xe tiêu chuẩn, thì **`Unsloth + Liger Kernel`** là bộ động cơ độ lại đường ống trên GPU, gộp phép tính `Cross-Entropy` làm một. | Tăng tốc train **gấp 2 lần** và giảm **~60% VRAM** khi văn bản pháp luật dài 2,048 từ. |
| **9. `RAFT` (Retrieval-Augmented Fine-Tuning)** | Đi thi mở tài liệu nhưng đề thi **kẹp lẫn 1 trang đúng với 2 trang nhiễu** (khác địa bàn / hết hạn), và có **30% câu hỏi không có đáp án trong tài liệu**. | Rèn 2 phản xạ: (1) Tự lọc đúng trang tài liệu theo địa bàn, và (2) Dứt khoát **từ chối trả lời** khi thiếu căn cứ. |
| **10. `ORPO` & `GRPO` + `imatrix GGUF`** | **`ORPO`** đối chiếu bài mẫu đúng (`Chosen`) và bài bịa (`Rejected`). **`GRPO`** cho tự viết 4 bài rồi dùng code chấm thưởng/phạt. **`imatrix GGUF`** nén về ~1 GB bảo toàn dấu tiếng Việt. | Trị tận gốc tật bịa đặt, phạt gấp đôi lỗi bỏ sót văn bản hết hạn (`False Negative`) và chạy mượt trên CPU 0 đồng. |

---

## 2. Lộ trình 7 Buổi Thực chiến trên `Qwen3-1.7B` (Tất cả đều nằm trong thư mục `edu/`)

| Buổi học & File chi tiết | Chế độ tách code | 🎯 Sản phẩm chạy được ngay cuối buổi (trên `Qwen3-1.7B`) | 📊 Phép đo So sánh 🔴 Trước vs. 🟢 Sau | Dự án mã nguồn mở tham chiếu |
|---|---|---|---|---|
| **[Buổi 0](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-00-colab-t4-and-baseline.md)** | **Chia siêu nhỏ** *(4–8 dòng/ô)* | **Máy X-Quang Bộ Não `Qwen3-1.7B` & Bộ Đo Baseline**: Soi xác suất Top-5 chữ tiếp theo và bắt lỗi trả lời lan man của model gốc. | So sánh VRAM `FP16` (~3.4 GB) vs. `4-bit NF4` (`1.27 GB`) và tốc độ khi tắt `<think>`. | `Qwen/Qwen3-1.7B` Base |
| **[Buổi 1](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-01-vivu-rag-sft-lora.md)** | **Chia siêu nhỏ** *(5–10 dòng/ô)* | **Trợ lý Du lịch & Tuyển sinh `ViVu-Mini`**: Khung chat đọc thẻ `<context>`, khóa 100% tiếng Việt, từ chối khi hỏi ngoài ngữ cảnh. | Bảng chấm điểm tự động trên 4 câu test (Độ bám ngữ cảnh & Tỷ lệ từ chối đúng Trước/Sau). | `thanhdo881/qwen3-1.7b-vivu-travel-vn` |
| **[Buổi 2](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-02-evidence-json-dora-rslora.md)** | **Chia siêu nhỏ** *(5–10 dòng/ô)* | **API Kiểm duyệt & Bóc tách Bằng chứng (`Evidence-Guard`)**: Xuất JSON có `verification_reasoning` đứng đầu + hàm Python bắt lỗi trích dẫn bịa. | Đo tỷ lệ `JSON Valid %` và `Grounded Evidence %` Trước vs. Sau khi bật `DoRA + rsLoRA`. | `ViHSD / vi-explainable`, `sameerahmed-cloud/verity-rag` |
| **[Buổi 3](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-03-legal-rag-neftune-liger.md)** | **Chia nhỏ vừa** *(8–12 dòng/ô)* | **Bộ Dịch Ngôn ngữ Đời thường sang Thủ tục Pháp luật (`Legal-Translator`)**: Hiểu tiếng lóng (*xe ôm, vé số, tiền trọ*) trả về đúng mã văn bản & hồ sơ. | Đo **Độ lệch Công bằng (`Fairness Gap`)** giữa nhóm Câu hỏi chuẩn và Câu hỏi đời thường ($\le 10\%$). | `duyet/vietnamese-legal-instruct`, `fisherman611`, `tangbamiinh` |
| **[Buổi 4](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-04-raft-refusal-payload-filter.md)** | **Bắt đầu gộp gọn** *(12–20 dòng/ô)* | **Khiên Chống Bẫy Nhiễu RAG (`RAFT-Shield`)**: Tự lọc 1 văn bản đúng giữa 2 văn bản nhiễu (khác địa bàn) và từ chối câu không đáp án. | Đo `Distractor Filtering Accuracy` & `Refusal Accuracy` Trước vs. Sau khi học `RAFT`. | `UIT-ViQuAD 2.0`, `anaalkmim/rag-eval` |
| **[Buổi 5](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-05-safety-orpo-grpo-guardrails.md)** | **Gộp chuẩn công nghiệp** | **Cổng An toàn Chống Văn bản Hết hạn & Che PII (`Safety-Gatekeeper`)**: Kết hợp `ORPO + GRPO` với 4 tầng Guardrail che CCCD/SĐT & bắt cờ `expired`. | Đo `Expired Detection Recall` (từ $0\%$ lên $100\%$) và điểm thưởng trung bình `GRPO Reward`. | `ViMedQA`, `llm-security-proxy`, `guardrail-system-rag` |
| **[Buổi 6](file:///Users/phucpham/Desktop/uit/project/rag/edu/module-06-capstone-p01-gguf-mlops.md)** | **Pipeline 4 Pha MLOps** | **Hệ thống RAG An sinh P01 Hoàn chỉnh (`Qwen3-1.7B-Q4_K_M.gguf`)**: Đóng gói 4 pha `train -> build -> benchmark -> push` chạy trên CPU. | Bảng điểm **4 Mốc $\times$ 3 Lần chạy ổn định** (`Ragas` + Deterministic Metrics) nộp báo cáo Rubric. | `farhan-syah/unsloth-finetuning`, `conversational-research-rag` |
