# Buổi 1 (Module 1): Hỏi đáp RAG Du lịch & Tuyển sinh — Học từ `vivu-travel-vn` (`LoRA` & `train_on_responses_only`)

- **Dự án tham chiếu:** `thanhdo881/qwen3-1.7b-vivu-travel-vn` (Trợ lý du lịch Việt Nam ViVu & các bản `-edu`, `-chatbot`).
- **Mục tiêu đầu ra:** Hiểu bản chất `LoRA` ($W_0 + \frac{\alpha}{r}BA$), kỹ thuật bọc ngữ cảnh bằng thẻ `<context>...</context>`, khóa 100% tiếng Việt, chỉ tính điểm phạt trên câu trả lời (`Prompt Loss Masking`), và kiểm chứng **🔴 Trước vs. 🟢 Sau Fine-tune** bằng `disable_adapter()`.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **`LoRA` (`Low-Rank Adaptation`) hoạt động thế nào?**
   - **Hình ảnh đời thường:** Hãy tưởng tượng `Qwen3-1.7B` là một **cuốn từ điển Bách khoa toàn thư dày 3,000 trang đã ép plastic cứng** ($W_0$ đã nén 4-bit, không thể tẩy xóa trực tiếp). Để dạy thêm nghiệp vụ mới, ta **dán một tờ giấy ghi chú mỏng ($B \cdot A$)** lên lề trang sách.
   - Khi đọc sách, mắt ta quét qua trang gốc ($W_0$) cộng thêm nội dung tờ giấy ghi chú ($\frac{\alpha}{r} B \cdot A$).
   - **Lợi ích:** Thay vì sửa 1.72 tỷ con số, ta chỉ cần viết lên tờ giấy ghi chú (chỉ chiếm **~0.1% đến 0.5% số tham số**), học xong trong 1 phút và bất cứ lúc nào muốn xem lại sách gốc chỉ cần lật tờ giấy ghi chú ra (`disable_adapter()`)!
2. **Tại sao phải dùng thẻ `<context>...</context>` và `SYSTEM_PROMPT` khóa 100% tiếng Việt?**
   - Dòng mô hình Qwen (do Alibaba huấn luyện) có lượng dữ liệu tiếng Trung rất lớn; khi gặp tên riêng hoặc từ hiếm, mô hình gốc đôi khi bị "văng" vài chữ Hán. Một `SYSTEM_PROMPT` cố định ép *"Chỉ sử dụng 100% TIẾNG VIỆT"* kết hợp Fine-tune sẽ triệt tiêu hoàn toàn tật này.
   - Cặp thẻ `<context>...</context>` đóng vai trò như "đường biên giới": rèn cho mô hình thói quen **chỉ tìm thông tin nằm giữa 2 thẻ này**, nếu không có trong `<context>` thì từ chối lịch sự.
3. **`Prompt Loss Masking` (`train_on_responses_only` / `DataCollatorForCompletionOnlyLM`) là gì?**
   - **Hình ảnh Feynman:** Khi đi thi, giáo viên chỉ chấm điểm phần **Bài làm của thí sinh (`assistant`)**, không ai chấm điểm phần **Chép lại đề bài (`system` + `user` + `<context>`)**.
   - Trong PyTorch, bất kỳ token nào thuộc phần đề bài sẽ được gán nhãn `label = -100` (hàm `Cross-Entropy Loss` mặc định bỏ qua mọi vị trí có giá trị `-100`), giúp mô hình dồn 100% năng lượng học cách trả lời.

---

## 2. Thực hành 5 Bước trên Google Colab

*(Chạy nối tiếp trên phiên Colab đã nạp sẵn `model` 4-bit và `tokenizer` từ Buổi 0).*

### Bước 1: Thiết lập Khuôn mẫu `<context>` chuẩn `vivu-travel-vn`
```python
SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn AI chuyên nghiệp. Hãy trả lời chính xác dựa trên ngữ cảnh được cung cấp. "
    "Yêu cầu bắt buộc: Chỉ sử dụng 100% TIẾNG VIỆT, nếu ngữ cảnh không có thông tin hãy từ chối lịch sự."
)

def format_rag_input(context: str, question: str) -> str:
    return f"<context>\n{context}\n</context>\n\nCÂU HỎI:\n{question}"
```

---

### Bước 2: Đo 🔴 TRƯỚC Fine-tune trên 2 tình huống thử nghiệm
```python
def ask_rag(m, context, question, use_lora=True):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_rag_input(context, question)},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    p_len = inputs["input_ids"].shape[-1]
    if use_lora and hasattr(m, "disable_adapter"):
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=150, do_sample=False)
    elif hasattr(m, "disable_adapter"):
        with m.disable_adapter(), torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=150, do_sample=False)
    else:
        with torch.no_grad():
            out = m.generate(**inputs, max_new_tokens=150, do_sample=False)
    return tokenizer.decode(out[0][p_len:], skip_special_tokens=True)

test_cases = [
    (
        "Khu du lịch sinh thái Làng Nổi Tân Lập (Long An) mở cửa từ 7h30 đến 17h00, vé tham quan xuồng ba lá là 150.000đ/người.",
        "Đi xuồng ba lá ở Làng Nổi Tân Lập tốn bao nhiêu tiền và mấy giờ đóng cửa?"
    ),
    (
        "Trường ĐH Công nghệ Thông tin (UIT) tuyển thẳng thí sinh đạt giải Nhất, Nhì, Ba kỳ thi Học sinh giỏi Quốc gia môn Tin học.",
        "Học phí mỗi học kỳ của ngành Trí tuệ Nhân tạo tại UIT là bao nhiêu triệu?"
    ),
]

for idx, (ctx, q) in enumerate(test_cases, 1):
    print(f"\n--- TEST {idx} (🔴 TRƯỚC FINE-TUNE) ---")
    print(ask_rag(model, ctx, q, use_lora=False))
```
- **Điểm cần chú ý ở Test 2:** Trong `<context>` chỉ nói về *tuyển thẳng*, **không hề có thông tin học phí**. Hãy quan sát xem mô hình gốc có tự bịa ra con số triệu đồng hay không!

---

### Bước 3: Tạo tập dữ liệu mẫu (Đa lĩnh vực: Du lịch, Tuyển sinh, An sinh) & Soi dữ liệu
```python
from datasets import Dataset

raw_samples = [
    {
        "ctx": "Đà Lạt nằm trên cao nguyên Lâm Viên cao 1.500m, nổi tiếng với Hồ Xuân Hương và khí hậu mát mẻ quanh năm.",
        "q": "Đà Lạt có độ cao bao nhiêu và nổi tiếng với hồ nào?",
        "a": "[Trích xuất chuẩn] Theo ngữ cảnh được cung cấp, Đà Lạt nằm trên cao nguyên Lâm Viên ở độ cao 1.500m và nổi tiếng với Hồ Xuân Hương."
    },
    {
        "ctx": "Phố cổ Hội An (Quảng Nam) miễn phí vé vào cổng sau 21h00 tối; vé tham quan ban ngày cho khách Việt Nam là 80.000đ.",
        "q": "Giá vé máy bay khứ hồi đi Hội An là bao nhiêu?",
        "a": "[Từ chối] Xin lỗi, ngữ cảnh được cung cấp chỉ nêu giá vé tham quan Phố cổ Hội An, không có thông tin về giá vé máy bay."
    },
    {
        "ctx": "Ký túc xá ĐHQG TP.HCM thu phí phòng 4 người có máy lạnh là 450.000đ/sinh viên/tháng, nộp hồ sơ tại Ban Quản lý Cụm nhà.",
        "q": "Phòng 4 người có máy lạnh ở KTX ĐHQG giá bao nhiêu và đăng ký ở đâu?",
        "a": "[Trích xuất chuẩn] Theo ngữ cảnh được cung cấp, mức phí là 450.000đ/sinh viên/tháng và nơi nộp hồ sơ là Ban Quản lý Cụm nhà."
    },
    {
        "ctx": "Nghị quyết NQ-01/2026/HĐND-HCM hỗ trợ lao động tự do (xe ôm, bán vé số) bị tai nạn mức 2.000.000đ tại UBND Phường Tân Tạo.",
        "q": "Người chạy xe ôm bị tai nạn ở Phường Tân Tạo được hỗ trợ bao nhiêu?",
        "a": "[Trích xuất chuẩn] Theo ngữ cảnh được cung cấp (NQ-01/2026/HĐND-HCM), mức hỗ trợ là 2.000.000đ, nộp hồ sơ tại UBND Phường Tân Tạo."
    },
    {
        "ctx": "Nghị quyết NQ-01/2026/HĐND-HCM hỗ trợ lao động tự do bị tai nạn lao động tại Phường Tân Tạo.",
        "q": "Thủ tục vay mua nhà trả góp tại ngân hàng thương mại cần gì?",
        "a": "[Từ chối] Xin lỗi, ngữ cảnh được cung cấp chỉ quy định về hỗ trợ tai nạn lao động tự do, không có thông tin về vay mua nhà ngân hàng."
    },
] * 4  # Nhân bản 4 lần (20 mẫu) để thực hành nhanh trên Colab

train_ds = Dataset.from_list([
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_rag_input(s["ctx"], s["q"])},
            {"role": "assistant", "content": s["a"]},
        ]
    }
    for s in raw_samples
])
print("Số mẫu huấn luyện:", len(train_ds))
```

---

### Bước 4: Gắn `LoRA` (`r=8, lora_alpha=16`) & Huấn luyện SFT
```python
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

sft_cfg = SFTConfig(
    output_dir="outputs_module1",
    max_steps=30,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    logging_steps=5,
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    peft_config=lora_cfg,
    args=sft_cfg,
)
trainer.model.print_trainable_parameters()
trainer.train()
```

---

### Bước 5: Đặt 🔴 TRƯỚC và 🟢 SAU cạnh nhau để kiểm chứng
```python
trainer.model.eval()
trainer.model.config.use_cache = True

for idx, (ctx, q) in enumerate(test_cases, 1):
    print(f"\n================ CA KIỂM THỬ {idx}: {q} ================")
    print("🔴 [TRƯỚC FINE-TUNE]:\n", ask_rag(trainer.model, ctx, q, use_lora=False))
    print("\n🟢 [SAU FINE-TUNE]:\n", ask_rag(trainer.model, ctx, q, use_lora=True))
```
- **Kết quả kỳ vọng:** Ở Ca 1, mô hình tự động gắn nhãn `[Trích xuất chuẩn]` và lấy đúng `150.000đ` + `17h00`; ở Ca 2 (hỏi học phí không có trong `<context>`), mô hình tự động gắn nhãn `[Từ chối]` thay vì bịa số tiền!

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 1 (Checkpoint)
1. Tại sao trong `vivu-travel-vn` việc trả lời văn bản tự do (`[Trích xuất chuẩn] ...`) là đủ, nhưng sang bài toán Hành chính P01 ta cần nâng cấp lên JSON? *(Trả lời: Vì Backend P01 cần bóc tách riêng mảng `required_documents`, `submission_place`, `expired_warning` để hiển thị thẻ UI và chạy hậu kiểm an toàn bằng code Python)*.
