# Đề cương Thực chiến: Fine-tune `Qwen3-1.7B` & Hệ thống RAG từ Số 0 đến Master (Phương pháp Feynman)

- **Mục tiêu:** Làm chủ kỹ thuật Fine-tune mô hình ngôn ngữ nhỏ (`Qwen/Qwen3-1.7B`) trên Google Colab T4 (15 GB VRAM) và kiến trúc RAG thực chiến từ các dự án mã nguồn mở tiêu biểu để hoàn thiện **Đồ án P01 (Tra cứu quyền lợi an sinh — SDG 1.3)**.
- **Nguyên tắc học:** **Thực chiến trước, công thức sau** — giải thích mọi khái niệm theo **phương pháp Feynman** (ví dụ đời thường dễ hiểu $\rightarrow$ bản chất kỹ thuật $\rightarrow$ lý do dùng), mỗi bài đều có bước đo **Trước vs. Sau Fine-tune** rõ ràng.

---

## 1. Giải thích 10 Khái niệm Cốt lõi theo Phương pháp Feynman

| Khái niệm kỹ thuật | Giải thích kiểu Feynman (Hình ảnh đời thường) | Tác dụng thực chiến trên `Qwen3-1.7B` |
|---|---|---|
| **1. Pre-trained Model vs. Fine-tuning** | **Pre-trained** là cử nhân mới ra trường (biết đọc viết tiếng Việt nhưng chưa rành nghiệp vụ riêng). **Fine-tune** là khóa đào tạo nội bộ giúp cử nhân đó trả lời đúng quy trình của cơ quan bạn. | Không cần dạy lại tiếng Việt từ đầu; chỉ dạy cách trích xuất đúng quy định và định dạng đầu ra. |
| **2. Tokenizer & ChatML (`<\|im_start\|>`)** | Máy tính không biết đọc chữ mà tra **sổ danh bạ (~151,936 mảnh từ)** để đổi chữ thành mã số. **ChatML** là các "nhãn vai diễn" (`system`, `user`, `assistant`) để AI biết đâu là đề bài, đâu là chỗ nó phải điền đáp án. | Đặt `enable_thinking=False` trong ChatML để tắt đoạn tự lẩm bẩm `<think>` dài dòng, giúp trả lời nhanh gấp 3 lần. |
| **3. Lượng tử hóa 4-bit (`NF4` + Double Quant)** | Thay vì lưu ảnh gốc dung lượng nặng (`FP16` tốn ~3.4 GB), ta nén thành ảnh chất lượng cao siêu nhẹ (`4-bit NF4` chỉ tốn **1.27 GB**), khi nào cần tính toán mới tạm bung từng mảnh nhỏ ra. | Chừa hơn 13 GB VRAM trống trên card Colab T4 để thoải mái huấn luyện mà không bị tràn bộ nhớ (OOM). |
| **4. `LoRA` (`r`, `lora_alpha`)** | Không sửa trực tiếp vào sách giáo khoa gốc ($W_0$ đóng băng), mà dán thêm **tờ giấy ghi chú nhỏ ($B \cdot A$)** bên lề. Khi chỉ học trên tờ giấy nhỏ này, ta chỉ cần chỉnh **~0.1% – 1% tham số**. | Học cực nhanh trên Colab (vài phút), không làm mô hình quên kiến thức tiếng Việt gốc, có thể bật/tắt bằng `disable_adapter()`. |
| **5. `DoRA` + `rsLoRA`** | Nếu `LoRA` thường vừa đẩy vừa xoay bánh lái dễ bị chệch, thì **`DoRA`** tách riêng **ga (biên độ)** và **vô-lăng (hướng)** để bẻ lái chính xác như tài xế lâu năm; **`rsLoRA`** giữ tay lái không bị rung khi tăng kích thước bộ nhớ phụ `r`. | Giúp mô hình nhỏ 1.7B học được các quy tắc khắt khe (JSON nhiều trường, kiểm tra ngày hiệu lực) tiệm cận Full Fine-tune. |
| **6. `train_on_responses_only` (Prompt Loss Masking)** | Khi chấm bài cho học sinh, giáo viên **chỉ chấm phần bài làm**, không ai chấm điểm phần chép lại đề bài. | Mô hình không tốn sức học thuộc lòng đoạn văn bản `<context>` dài dòng, dồn 100% trí lực học cách viết câu trả lời chuẩn. |
| **7. `NEFTune` (`noise_alpha=5`)** | Giống như tập nghe tiếng Việt trong quán cà phê hơi ồn một chút: khi quen nghe trong nhiễu nhẹ, học sinh sẽ **hiểu ý chính** thay vì học vẹt từng chữ. | Chống **Học vẹt (Overfitting)** cực tốt khi bộ dữ liệu của bạn chỉ có vài trăm đến vài nghìn mẫu. |
| **8. `Unsloth` + `Liger Kernel`** | Nếu thư viện gốc (`transformers`) là xe số tiêu chuẩn, thì **`Unsloth + Liger Kernel`** là bộ động cơ độ lại đường ống dẫn xăng trên GPU, gộp nhiều bước tính toán làm một. | Tăng tốc độ train **gấp 2 lần** và giảm **~60% VRAM** ở bước tính điểm phạt (`Cross-Entropy`) khi văn bản dài 2,048 từ. |
| **9. `RAFT` (Retrieval-Augmented Fine-Tuning)** | Đi thi mở tài liệu nhưng đề thi **cố tình kẹp lẫn 1 trang tài liệu đúng với 2 trang tài liệu nhiễu** (khác địa phương hoặc đã hết hạn), và có **30% câu hỏi không hề có đáp án trong tài liệu**. | Rèn cho AI 2 phản xạ vàng: (1) Tự lọc đúng trang tài liệu cần dùng, và (2) Dứt khoát **từ chối trả lời** khi tài liệu không có thông tin. |
| **10. `ORPO` & `GRPO` + `imatrix GGUF`** | **`ORPO`** cho xem 1 bài mẫu đúng (`Chosen`) kèm 1 bài mẫu bịa (`Rejected`) để biết tránh lỗi. **`GRPO`** cho tự viết 4 bài rồi dùng code chấm điểm thưởng/phạt trực tiếp. **`imatrix GGUF`** là bước đóng gói mô hình về ~1 GB để chạy trên CPU mà không rơi dấu tiếng Việt. | Trị tận gốc tật bịa đặt (Hallucination), phạt gấp đôi lỗi bỏ sót văn bản hết hạn (`False Negative`) và chạy mượt trên server CPU 0 đồng. |

---

## 2. Bản đồ Tài liệu & Dự án Mã nguồn mở Tham chiếu cho Khóa học

| Nhóm kỹ thuật | Dự án / Bộ dữ liệu tham khảo | Tinh hoa học hỏi và áp dụng vào Khóa học & Đồ án P01 |
|---|---|---|
| **1. RAG & Pháp luật tiếng Việt** | • `thanhdo881/qwen3-1.7b-vivu-travel-vn`<br>• `duyet/vietnamese-legal-instruct` (127k mẫu `vbpl.vn`)<br>• `fisherman611/vietnamese-legal-chatbot`<br>• `tangbamiinh/traffic-law-assistant`<br>• Nhóm `AITeamVN`, `5CD-AI`, `bkai`, `ura-hcmut` | • Thẻ `<context>...</context>`, khóa 100% tiếng Việt, `enable_thinking=False`, `train_on_responses_only`.<br>• Văn phong hành chính chuẩn (*Căn cứ Điều..., Khoản...*) và phân tách thực thể (*Đối tượng, Hành vi/Tình huống, Quy định áp dụng*).<br>• Tầng truy xuất lai **Hybrid Search (Dense + BM25)** trên Qdrant + **Reranker** để không trượt số hiệu văn bản. |
| **2. Chống ảo giác & Trích dẫn bằng chứng** | • `ViHSD` / `vi-explainable`<br>• `UIT-ViQuAD 2.0` (*unanswerable questions*)<br>• `sameerahmed-cloud/verity-rag`<br>• `Aryanupadhyay23/conversational-research-rag` | • Sinh câu trả lời kèm mã trích dẫn `[Context N]` + **Grounding Check** tự động đối chiếu câu trả lời với đoạn trích gốc.<br>• Nhãn trạng thái xác thực trực quan (*Grounded in Document* vs. *Unverified / Expired Evidence*) và ngăn kéo xem đoạn trích PDF/văn bản gốc. |
| **3. Responsible AI, PII & Guardrails** | • `ViMedQA` (`pnnbao-ump`, `vilm`, `5CD-AI`)<br>• `AcadifySolution/llm-security-proxy`<br>• `tkarim45/llm-guardrails`<br>• `lamhotsiagian/guardrail-system-rag` | • Luồng kiểm soát 4 tầng: **Input Guard** (Regex PII Redactor + kiểm tra `region`) $\rightarrow$ **Topic/Region Boundary** (chặn Prompt Injection & lệch địa bàn chéo) $\rightarrow$ **Human-in-the-loop Gate** (chuyển bác sĩ/cán bộ khi hết hạn hoặc rủi ro cao) $\rightarrow$ **Audit Trail** + bộ test `pytest` làm Quality Gate CI/CD. |
| **4. Pipeline Unsloth $\rightarrow$ GGUF & Đánh giá MLOps** | • `farhan-syah/unsloth-finetuning` & `Unsloth Qwen3 Tutorial`<br>• `anaalkmim/rag-eval` | • Tổ chức pipeline 4 pha chuẩn công nghiệp: `train` (SFT/ORPO/GRPO) $\rightarrow$ `build` (merge & nén `imatrix Q4_K_M.gguf`) $\rightarrow$ `benchmark` (đo đối chứng với Base, đặc biệt trên nhóm câu hỏi không đáp án) $\rightarrow$ `push` (đẩy lên Hugging Face Model Registry).<br>• Tách bạch metadata (`region`, `status`, `version`, `effective_date`) vào **Qdrant Payload** để lọc chính xác. |

---

## 3. Lộ trình 6 Module Thực chiến (Đa lĩnh vực Việt Nam $\rightarrow$ Đồ án P01)

### Module 0: Làm chủ Phòng thí nghiệm Colab T4 & Mốc Baseline (1 Buổi)
- **Thực hành:** Kiểm tra GPU (`!nvidia-smi`), nạp `Qwen/Qwen3-1.7B` ở chế độ `4-bit NF4` (`1.27 GB VRAM`), khảo sát `AutoTokenizer` và chạy thử câu hỏi khi **chưa Fine-tune**.
- **Kỹ năng đạt được:** Hiểu cấu trúc 28 tầng của Qwen3, biết dùng `return_dict=True`, `do_sample=False` và lưu kết quả gốc vào `before_answers` để làm mốc đối chiếu.

---

### Module 1: Hỏi đáp RAG Du lịch & Tuyển sinh — Học từ `thanhdo881/qwen3-1.7b-vivu-travel-vn`
- **Bài toán thực tế:** Xây dựng trợ lý du lịch Việt Nam (*ViVu*) hoặc tư vấn tuyển sinh đại học (`-edu`, `-chatbot`), chỉ trả lời dựa trên đoạn văn nằm trong thẻ `<context>...</context>`.
- **Kỹ thuật thực hành:**
  1. Thiết kế `SYSTEM_PROMPT` khóa chặt **100% tiếng Việt** (triệt tiêu lỗi Qwen3 chêm chữ Trung Quốc).
  2. Tắt chế độ suy nghĩ tự do (`enable_thinking=False`) để biến mô hình thành bộ trích xuất tốc độ cao.
  3. Dùng `train_on_responses_only` để chỉ tính điểm phạt (`Loss`) trên câu trả lời của `assistant`.
- **Thực hành trên Colab (5 bước ngắn):** Tạo 10 mẫu `<context>` $\rightarrow$ Gắn `LoRA (r=8)` $\rightarrow$ Train 35 bước $\rightarrow$ Dùng `disable_adapter()` so sánh **Trước vs. Sau**.

---

### Module 2: Phân loại & Trích xuất Bằng chứng — Tham khảo `ViHSD` / `vi-explainable` & `verity-rag`
- **Bài toán thực tế:** Kiểm duyệt nội dung hoặc xác thực thông tin tiếng Việt — không chỉ gắn nhãn mà phải **trích xuất nguyên văn bằng chứng (`evidence` / `[Context N]`)** và **lý giải ngắn (`explanation`)** vào cấu trúc JSON để phục vụ bước **Grounding Check** tự động.
- **Kỹ thuật nâng cấp:**
  1. **Verification-First JSON:** Đặt trường `"verification_reasoning"` làm **key đầu tiên trong JSON** để AI phải viết lý do và đối chiếu bằng chứng trước rồi mới chốt kết luận (thay thế gọn nhẹ cho `<think>` tự do).
  2. Nâng cấp từ `LoRA` thường lên **`DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`)** để giữ định dạng JSON không bao giờ bị vỡ ngoặc.

---

### Module 3: Pháp luật & Thủ tục Hành chính — Tham khảo `duyet/vietnamese-legal-instruct`, `fisherman611` & `tangbamiinh`
- **Bài toán thực tế:** Phân tách thực thể (*Đối tượng lao động*, *Tình huống khó khăn*, *Quy định áp dụng*), dịch câu hỏi đời thường (*"chạy xe ôm"*, *"bán vé số"*, *"sổ hộ khẩu"*, *"tiền trọ"*) sang văn phong hành chính chuẩn `vbpl.vn` (*Căn cứ Điều..., Khoản...*).
- **Kỹ thuật nâng cấp:**
  1. Bật **`NEFTune` (`neftune_noise_alpha=5`)**: Bơm nhiễu nhẹ vào lớp từ vựng lúc train để mô hình không học vẹt câu chữ cứng nhắc, giúp hiểu mượt mọi cách nói đời thường (đảm bảo trục **Công bằng — Fairness**).
  2. Bật **`Unsloth` + `Liger Kernel`**: Khi văn bản pháp luật đưa vào `<context>` dài tới 2,048 tokens, `Liger Kernel` gộp phép tính `Cross-Entropy` giúp tiết kiệm 60% VRAM trên Colab T4.

---

### Module 4: Đọc hiểu RAG Chống Ảo giác & Lọc Payload — Tham khảo `UIT-ViQuAD 2.0` & `anaalkmim/rag-eval`
- **Bài toán thực tế:** Tách bạch metadata (`region`, `status`, `effective_date`) vào **Qdrant Payload** và huấn luyện mô hình xử lý các câu hỏi **không thể trả lời (`unanswerable`)** hoặc ngữ cảnh bị trộn nhiễu.
- **Kỹ thuật nâng cấp (`RAFT` + Chống Data Leakage):**
  1. **Chia tập `Scenario-Group Split (80% Train / 10% Val / 10% Test)`:** Đảm bảo không rò rỉ câu hỏi biến thể sang tập Test (*No Data Leakage*).
  2. **Đóng gói dữ liệu chuẩn `RAFT`:**
     - **70% mẫu có đáp án:** Trộn `1 đoạn đúng (Oracle)` + `2 đoạn gây nhiễu (Distractors: khác địa bàn hoặc đã hết hạn)` và xáo trộn vị trí ngẫu nhiên.
     - **30% mẫu không có đáp án:** Đưa toàn bộ `3 đoạn gây nhiễu` $\rightarrow$ Dạy mô hình bật `"is_refusal": true` và nêu rõ lý do từ chối.

---

### Module 5: Y tế & An toàn Sống còn — Tham khảo `ViMedQA`, `llm-security-proxy` & `guardrail-system-rag`
- **Bài toán thực tế:** Xây dựng hệ thống có tính an toàn bất đối xứng (như tư vấn y tế `ViMedQA` và an sinh xã hội P01) kết hợp 4 tầng bảo vệ (`Input Guard` che PII $\rightarrow$ `Topic/Region Boundary` $\rightarrow$ `Human-in-the-loop Gate` $\rightarrow$ `Audit Trail`).
- **Kỹ thuật nâng cấp (Giai đoạn 2 — Alignment giảm dần Learning Rate):**
  1. **Bước 2A — `ORPO` (`lr = 5e-6`, `beta = 0.1`):** Căn chỉnh cặp `Chosen` (chuẩn an toàn) và `Rejected` (bịa thông tin / quên cảnh báo hết hạn) mà **không tốn VRAM nạp Reference Model**.
  2. **Bước 2B — `GRPO` (`lr = 1e-6`, `num_generations = 4`):** Cho `Qwen3-1.7B` tự viết 4 phương án JSON rồi dùng **3 hàm thưởng Python (`Verifiable Rewards`)** chấm trực tiếp: thưởng $+0.4$ nếu đúng JSON Schema, $+0.3$ nếu đúng `doc_code` địa bàn, và **phạt nặng gấp đôi ($-0.6$)** nếu bỏ sót văn bản hết hạn (*False Negative*).

---

### Module 6 (Capstone): Quy trình 4 Pha `train -> build -> benchmark -> push` (Tham khảo `farhan-syah/unsloth-finetuning`)
- **Áp dụng cấu trúc 4 Pha chuẩn MLOps cho Đồ án P01:**
  1. **Pha `train`:** Chạy pipeline `SFT (DoRA/rsLoRA/NEFTune) -> ORPO -> GRPO` trên dữ liệu chính sách P01, log đầy đủ lên `wandb`.
  2. **Pha `build`:** Gộp Adapter vào trọng số gốc và nén sang **`Q4_K_M.gguf` kèm `imatrix` tiếng Việt** để giữ nguyên độ chính xác dấu câu và mã văn bản khi chạy trên CPU.
  3. **Pha `benchmark`:** Đo đối chứng qua **4 mốc** (`Baseline` $\rightarrow$ `SFT+DoRA` $\rightarrow$ `ORPO+GRPO` $\rightarrow$ `GGUF`) trên tập Test giữ riêng bằng Deterministic Metrics + `Ragas` (`Faithfulness`), lặp **3 lần chạy** kiểm tra độ ổn định theo Rubric.
  4. **Pha `push` & Tích hợp:** Đẩy model `.gguf` lên **Hugging Face Hub**, kết nối vào FastAPI + Qdrant Cloud (có keep-alive & nhãn *Grounded vs. Expired/Unverified* học từ `conversational-research-rag`) và vượt qua cổng `pytest` Quality Gate.

---

## 4. Khuôn mẫu 5 Bước Thực hành Chuẩn cho MỌI Module trên Colab

1. **Bước 1 (Thiết lập):** Nạp/kiểm tra `model` 4-bit và `tokenizer` (với `enable_thinking=False`).
2. **Bước 2 (Đo 🔴 TRƯỚC Fine-tune):** Chạy 2–3 câu hỏi thử nghiệm khó nhất của Module đó trên mô hình chưa học, lưu vào `before_answers` để thấy rõ điểm yếu.
3. **Bước 3 (Soi Dữ liệu):** Tạo bộ dữ liệu mẫu của Module và in thử 1 mẫu sau khi bọc `<context>` + ChatML để hiểu chính xác mô hình sắp đọc cái gì.
4. **Bước 4 (Huấn luyện):** Gắn cấu hình (`LoRA` / `DoRA` / `ORPO` / `GRPO`), in số lượng tham số mở khóa và bấm chạy train (theo dõi cột `Loss` hoặc `Reward`).
5. **Bước 5 (Đo 🟢 SAU Fine-tune & Đối chiếu):** Chạy lại đúng các câu hỏi ở Bước 2, đặt kết quả **🔴 TRƯỚC** và **🟢 SAU** cạnh nhau để kiểm chứng sự tiến bộ!
