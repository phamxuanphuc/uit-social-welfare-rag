# Đề cương Thực chiến: Fine-tune `Qwen3-1.7B` từ Số 0 đến Master (Phương pháp Feynman)

- **Mục tiêu:** Làm chủ kỹ thuật Fine-tune mô hình ngôn ngữ nhỏ (`Qwen/Qwen3-1.7B`) trên Google Colab T4 (15 GB VRAM) từ người mới bắt đầu đến khi tự xây dựng được mô hình chuyên biệt cho mọi lĩnh vực tiếng Việt và hoàn thiện **Đồ án P01 (Tra cứu quyền lợi an sinh — SDG 1.3)**.
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

## 2. Lộ trình 6 Module Thực chiến (Đa lĩnh vực Việt Nam $\rightarrow$ Đồ án P01)

Mỗi Module đi qua một bài toán thực tế tại Việt Nam để bạn nắm vững cách thiết kế dữ liệu cho **bất kỳ chủ đề nào**, trước khi hội tụ toàn bộ sức mạnh vào Đồ án P01 ở Module 6.

### Module 0: Làm chủ Phòng thí nghiệm Colab T4 & Mốc Baseline (1 Buổi)
- **Thực hành:** Kiểm tra GPU (`!nvidia-smi`), nạp `Qwen/Qwen3-1.7B` ở chế độ `4-bit NF4` (`1.27 GB VRAM`), khảo sát `AutoTokenizer` và chạy thử câu hỏi khi **chưa Fine-tune**.
- **Kỹ năng đạt được:** Hiểu cấu trúc 28 tầng của Qwen3, biết dùng `return_dict=True`, `do_sample=False` và lưu kết quả gốc vào `before_answers` để làm mốc đối chiếu.

---

### Module 1: Hỏi đáp RAG Du lịch & Tuyển sinh — Học từ `thanhdo881/qwen3-1.7b-vivu-travel-vn`
- **Bài toán thực tế:** Xây dựng trợ lý du lịch Việt Nam (*ViVu*) hoặc tư vấn tuyển sinh đại học (`-edu`, `-chatbot`), chỉ trả lời dựa trên đoạn văn nằm trong thẻ `<context>...</context>`.
- **Học hỏi từ `vivu-travel-vn`:**
  1. Thiết kế `SYSTEM_PROMPT` khóa chặt **100% tiếng Việt** (triệt tiêu lỗi Qwen3 thỉnh thoảng chêm chữ Trung Quốc).
  2. Tắt chế độ suy nghĩ tự do (`enable_thinking=False`) để biến mô hình thành bộ trích xuất tốc độ cao.
  3. Dùng `train_on_responses_only` để chỉ tính điểm phạt (`Loss`) trên câu trả lời của `assistant`.
- **Thực hành trên Colab (5 bước ngắn):** Tạo 10 mẫu `<context>` du lịch/tuyển sinh $\rightarrow$ Gắn `LoRA (r=8)` $\rightarrow$ Train 35 bước $\rightarrow$ Dùng `disable_adapter()` so sánh **Trước vs. Sau**.
- **Nhận xét & Tối ưu:** Thấy rõ vì sao văn bản tự do (`free-text`) + `temperature=0.7` hợp với du lịch, nhưng cần nâng cấp lên **JSON cố định** và `temperature=0.0–0.1` cho các bài toán nghiêm ngặt hơn.

---

### Module 2: Phân loại & Trích xuất Bằng chứng — Tham khảo `ViHSD` / `vi-explainable`
- **Bài toán thực tế:** Kiểm duyệt bình luận mạng xã hội hoặc phản hồi khách hàng tiếng Việt — không chỉ gắn nhãn (`Hợp lệ` / `Vi phạm`) mà phải **trích xuất nguyên văn bằng chứng (`evidence`)** và **lý giải ngắn (`explanation`)** vào cấu trúc JSON.
- **Kỹ thuật nâng cấp:**
  1. **Verification-First JSON:** Vì AI sinh chữ từ trái sang phải, đặt trường `"explanation"` / `"verification_reasoning"` làm **key đầu tiên trong JSON** để AI phải "viết lý do và bằng chứng" trước rồi mới chốt nhãn kết luận (thay thế hoàn hảo cho `<think>` dài dòng!).
  2. Nâng cấp từ `LoRA` thường lên **`DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`)** để giữ định dạng JSON không bao giờ bị vỡ ngoặc.
- **Liên hệ trực tiếp P01:** Đây chính là khuôn mẫu để P01 trích xuất nguyên văn đoạn trích (`excerpt`) và kiểm chứng địa bàn trước khi liệt kê giấy tờ.

---

### Module 3: Pháp luật & Thủ tục Hành chính Việt Nam — Tham khảo `vbpl.vn` / `Zalo Legal QA` (`AITeamVN`, `5CD-AI`, `bkai`, `ura-hcmut`)
- **Bài toán thực tế:** Dịch câu hỏi đời thường của người dân (*"chạy xe ôm"*, *"bán vé số"*, *"sổ hộ khẩu"*, *"tiền trọ"*) sang văn phong hành chính chuẩn (*Căn cứ Điều..., Khoản..., Quyết định số...*) và trả về danh sách hồ sơ + nơi nộp.
- **Kỹ thuật nâng cấp:**
  1. Bật **`NEFTune` (`neftune_noise_alpha=5`)**: Bơm nhiễu nhẹ vào lớp từ vựng lúc train để mô hình không học vẹt câu chữ cứng nhắc, giúp hiểu mượt mọi cách nói đời thường (đảm bảo trục **Công bằng — Fairness** giữa câu hỏi chuẩn và câu đời thường).
  2. Bật **`Unsloth` + `Liger Kernel`**: Khi văn bản pháp luật đưa vào `<context>` dài tới 2,048 tokens, `Liger Kernel` gộp phép tính `Cross-Entropy` giúp tiết kiệm 60% VRAM trên Colab T4.
- **Thực hành trên Colab:** Huấn luyện trên tập hỏi đáp thủ tục hành chính và đo chênh lệch độ chính xác giữa nhóm **Câu hỏi chuẩn** vs. **Câu hỏi đời thường** (`Fairness Gap <= 10%`).

---

### Module 4: Đọc hiểu RAG Chống Ảo giác & Phản xạ Từ chối — Tham khảo `UIT-ViQuAD 2.0`
- **Bài toán thực tế:** Trong `UIT-ViQuAD 2.0`, có những câu hỏi **không thể trả lời (`unanswerable`)** dù đoạn văn trông rất giống chủ đề đang hỏi. Nếu không dạy kỹ, mô hình 1.7B sẽ "nhắm mắt ghép bừa" thông tin trong đoạn văn để trả lời!
- **Kỹ thuật nâng cấp (`RAFT` + Chống Data Leakage):**
  1. **Chia tập `Scenario-Group Split (80% Train / 10% Val / 10% Test)`:** Các câu hỏi biến thể cùng 1 tình huống phải nằm chung 1 tập, tuyệt đối không để lọt sang tập Test (*No Data Leakage*).
  2. **Đóng gói dữ liệu chuẩn `RAFT`:**
     - **70% mẫu có đáp án:** Trộn `1 đoạn đúng (Oracle)` + `2 đoạn gây nhiễu (Distractors: khác địa bàn hoặc khác chương trình)` và xáo trộn vị trí ngẫu nhiên.
     - **30% mẫu không có đáp án:** Đưa toàn bộ `3 đoạn gây nhiễu` $\rightarrow$ Dạy mô hình bật `"is_refusal": true` và nêu rõ lý do từ chối.
- **Thực hành trên Colab:** Thử bẫy mô hình bằng câu hỏi ở *Phường Tân Tạo* nhưng chỉ đưa văn bản của *Quận Liên Chiểu (Đà Nẵng)* vào `<context>` $\rightarrow$ Kiểm chứng mô hình từ chối chính xác 100%.

---

### Module 5: Y tế & An toàn Sống còn — Tham khảo `ViMedQA` (`pnnbao-ump`, `vilm`, `5CD-AI`) với `ORPO` & `GRPO`
- **Bài toán thực tế:** Trong tư vấn y tế (`ViMedQA`) hay tài chính – an sinh xã hội, có những sai sót mang tính **bất đối xứng**: khuyên nhầm thuốc hoặc dùng **văn bản đã hết hiệu lực (`expired`)** gây hại nặng hơn nhiều so với việc cẩn thận nhắc người dùng hỏi lại bác sĩ/cán bộ.
- **Kỹ thuật nâng cấp (Giai đoạn 2 — Alignment giảm dần Learning Rate):**
  1. **Bước 2A — `ORPO` (`lr = 5e-6`, `beta = 0.1`):** Đưa vào cặp `Chosen` (bắt đúng cờ hết hạn / từ chối chẩn đoán liều) và `Rejected` (bịa thông tin / quên cảnh báo hết hạn). `ORPO` dùng tỷ số cược (*Odds Ratio*) nên **không tốn VRAM nạp Reference Model** như DPO.
  2. **Bước 2B — `GRPO` (`lr = 1e-6`, `num_generations = 4`):** Cho `Qwen3-1.7B` tự viết 4 phương án JSON rồi dùng **3 hàm thưởng Python (`Verifiable Rewards`)** chấm trực tiếp:
     - $+0.4$ điểm nếu đúng JSON Schema và có `verification_reasoning` ở đầu.
     - $+0.3$ điểm nếu trích đúng mã `doc_code` của đúng địa bàn.
     - $+0.3$ điểm nếu bắt đúng `expired_warning`; **phạt nặng gấp đôi ($-0.6$)** nếu bỏ sót văn bản hết hạn (*False Negative*).

---

### Module 6 (Capstone): Hoàn thiện Моhình Đồ án P01, Đánh giá 4 Mốc (`Ragas`) & Xuất `imatrix GGUF`
- **Thực hiện trọn vẹn `02_finetune_qwen3.ipynb` cho đề tài P01:**
  1. Kết hợp toàn bộ kỹ thuật từ Module 1–5 trên dữ liệu chính sách an sinh P01 (kèm logging lên `Weights & Biases - wandb`).
  2. **Bảng đánh giá 4 mốc trên tập Test giữ riêng (kèm chạy lặp 3 lần đo độ ổn định cho Rubric Mức 4):**
     - *Mốc 1:* `Baseline (BM25 + Qwen3-1.7B Base chưa chuẩn hóa từ đời thường)`
     - *Mốc 2:* `Sau SFT + DoRA + rsLoRA + NEFTune (RAFT)`
     - *Mốc 3:* `Sau ORPO + GRPO (Verifiable Rewards)`
     - *Mốc 4:* `Sau khi nén sang Q4_K_M.gguf kèm imatrix`
     *(Đảm bảo vượt ngưỡng chốt trước: `Faithfulness >= 0.85`, `Citation Precision >= 0.90`, `Refusal Accuracy >= 0.90`, `Expired Detection = 100%`, `Fairness Gap <= 10%`)*.
  3. **Nén `Q4_K_M.gguf` với `Importance Matrix (imatrix)`:** Dùng chính các câu văn bản chính sách tiếng Việt P01 làm dữ liệu hiệu chuẩn để giữ nguyên độ nhạy dấu tiếng Việt và mã văn bản khi nén về ~1.05 GB, đẩy lên **Hugging Face Hub** và nạp vào `llama-cpp-python` (Constrained JSON Decoding).

---

## 3. Khuôn mẫu 5 Bước Thực hành Chuẩn cho MỌI Module trên Colab

Để bạn không bao giờ bị ngợp, mỗi Module khi thực hành trên Google Colab đều chia làm **đúng 5 ô code ngắn (mỗi lần chỉ chạy 1 ô)**:
1. **Bước 1 (Thiết lập):** Nạp/kiểm tra `model` 4-bit và `tokenizer` (với `enable_thinking=False`).
2. **Bước 2 (Đo TRƯỚC Fine-tune):** Chạy 2–3 câu hỏi thử nghiệm khó nhất của Module đó trên mô hình chưa học, lưu vào `before_answers` để thấy rõ điểm yếu.
3. **Bước 3 (Soi Dữ liệu):** Tạo bộ dữ liệu mẫu của Module và in thử 1 mẫu sau khi bọc `<context>` + ChatML để hiểu chính xác mô hình sắp đọc cái gì.
4. **Bước 4 (Huấn luyện):** Gắn cấu hình (`LoRA` / `DoRA` / `ORPO` / `GRPO`), in số lượng tham số mở khóa và bấm chạy train (theo dõi cột `Loss` hoặc `Reward`).
5. **Bước 5 (Đo SAU Fine-tune & Đối chiếu):** Chạy lại đúng các câu hỏi ở Bước 2, đặt kết quả **🔴 TRƯỚC** và **🟢 SAU** cạnh nhau để kiểm chứng sự tiến bộ!
