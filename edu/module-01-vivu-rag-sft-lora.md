# Buổi 1 (Module 1): Xây dựng Chatbot `ViVu-Mini` trên `Qwen3-1.7B` — Học từ `vivu-travel-vn` (Chia nhỏ từng ô)

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` (dùng tiếp biến `model` và `tokenizer` từ Buổi 0).
- **🎯 Sản phẩm chạy được cuối buổi:** **Chatbot Trợ lý Du lịch, Tuyển sinh & An sinh `ViVu-Mini`** — chỉ trả lời dựa trên thẻ `<context>...</context>`, khóa 100% tiếng Việt, tự động từ chối khi thông tin không có trong ngữ cảnh, kèm **Bảng đo điểm 🔴 Trước vs. 🟢 Sau Fine-tune**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **`LoRA` (`Low-Rank Adaptation`) là gì?**
   - **Hình ảnh Feynman:** Bộ não `Qwen3-1.7B` gốc ($W_0$) giống như cuốn sách đã ép plastic không được sửa vào. `LoRA` chỉ **dán thêm một mảnh giấy ghi chú nhỏ ($B \cdot A$)** bên lề trang sách (chỉ chiếm **~0.1% số tham số**). Nhờ vậy máy học cực nhanh trong 30 giây, và khi cần so sánh với lúc chưa học, ta chỉ việc tạm lật mảnh giấy ghi chú ra bằng lệnh `with model.disable_adapter():`!
2. **Tinh hoa học từ dự án `thanhdo881/qwen3-1.7b-vivu-travel-vn`:**
   - Đóng khung tài liệu tra cứu vào cặp thẻ `<context>...</context>`.
   - Dạy mô hình 2 phản xạ rõ rệt: nếu có thông tin trong `<context>` thì mở đầu bằng `[Trích xuất chuẩn]`; nếu không có thì mở đầu bằng `[Từ chối]` (tuyệt đối không tự bịa!).

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab (5–10 dòng/ô)

### Ô 1 (7 dòng): Định nghĩa `SYSTEM_PROMPT` khóa 100% tiếng Việt & Hàm bọc `<context>`
```python
SYSTEM_PROMPT = (
    "Bạn là trợ lý tra cứu AI chuyên nghiệp. Chỉ trả lời dựa trên <context>. "
    "Bắt buộc dùng 100% TIẾNG VIỆT và từ chối bằng '[Từ chối]' nếu <context> không có đáp án."
)

def make_prompt(context: str, question: str) -> str:
    return f"<context>\n{context}\n</context>\n\nCÂU HỎI:\n{question}"
```
- **Giải thích:** Tách riêng phần tài liệu vào `<context>` giúp `Qwen3-1.7B` phân biệt rõ đâu là văn bản căn cứ, đâu là câu hỏi của người dùng.

---

### Ô 2 (9 dòng): Tạo Bộ 4 Câu Kiểm thử (Gồm cả câu có đáp án và câu bẫy không có đáp án)
```python
eval_cases = [
    ("Làng Nổi Tân Lập (Long An) mở cửa 7h30–17h00, vé xuồng ba lá là 150.000đ/người.",
     "Vé đi xuồng ba lá ở Làng Nổi Tân Lập giá bao nhiêu?", "150.000đ"),
    ("ĐH Công nghệ Thông tin (UIT) tuyển thẳng học sinh đạt giải Quốc gia môn Tin học.",
     "Học phí mỗi học kỳ của UIT là bao nhiêu triệu đồng?", "[Từ chối]"),
    ("NQ-01/2026/HĐND-HCM hỗ trợ lao động tự do chạy xe ôm bị tai nạn 2.000.000đ tại UBND Phường Tân Tạo.",
     "Chạy xe ôm bị tai nạn ở Phường Tân Tạo được hỗ trợ bao nhiêu và nộp ở đâu?", "2.000.000đ"),
    ("NQ-01/2026/HĐND-HCM hỗ trợ tai nạn lao động tự do tại Phường Tân Tạo.",
     "Thủ tục xin cấp hộ chiếu đi du lịch nước ngoài cần giấy tờ gì?", "[Từ chối]"),
]
print("Đã tạo xong 4 ca kiểm thử (2 ca trích xuất + 2 ca bẫy từ chối)!")
```

---

### Ô 3 (11 dòng): Viết hàm hỏi `Qwen3-1.7B` hỗ trợ bật/tắt `LoRA` (`disable_adapter`)
```python
def ask_vivu(model_obj, context: str, question: str, use_lora: bool = True) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_prompt(context, question)},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([text], return_tensors="pt").to("cuda")
    p_len = inp["input_ids"].shape[-1]
    ctx_mgr = model_obj.disable_adapter() if (not use_lora and hasattr(model_obj, "disable_adapter")) else torch.no_grad()
    with ctx_mgr, torch.no_grad():
        out = model_obj.generate(**inp, max_new_tokens=120, do_sample=False)
    return tokenizer.decode(out[0][p_len:], skip_special_tokens=True)
```
- **Giải thích:** Khi `use_lora=False`, lệnh `model_obj.disable_adapter()` tạm tắt giấy ghi chú LoRA để trả lời bằng não gốc `Qwen3-1.7B`; khi `use_lora=True` thì dùng não đã Fine-tune!

---

### Ô 4 (8 dòng): Đo điểm 🔴 TRƯỚC Fine-tune
```python
before_outputs = []
for idx, (ctx, q, expected) in enumerate(eval_cases, 1):
    ans = ask_vivu(model, ctx, q, use_lora=False)
    before_outputs.append(ans)
    print(f"\n--- Ca {idx} (Kỳ vọng có '{expected}') ---\n🔴 Trước Fine-tune: {ans}")
```
- **Quan sát:** Bạn sẽ thấy trước khi Fine-tune, `Qwen3-1.7B` chưa biết gắn nhãn `[Trích xuất chuẩn]` hay `[Từ chối]`, và đôi khi trả lời dài dòng ngoài lề.

---

### Ô 5 (11 dòng): Chuẩn bị dữ liệu huấn luyện (Đóng gói vào `Dataset`)
```python
from datasets import Dataset

train_pairs = [
    ("Đà Lạt cao 1.500m, nổi tiếng với Hồ Xuân Hương.", "Đà Lạt cao bao nhiêu?",
     "[Trích xuất chuẩn] Theo ngữ cảnh, Đà Lạt nằm ở độ cao 1.500m và nổi tiếng với Hồ Xuân Hương."),
    ("Phố cổ Hội An thu vé ban ngày 80.000đ/khách.", "Vé máy bay đi Hội An giá bao nhiêu?",
     "[Từ chối] Xin lỗi, ngữ cảnh chỉ nêu giá vé tham quan Phố cổ Hội An, không có thông tin giá vé máy bay."),
    ("KTX ĐHQG thu phí phòng 4 người là 450.000đ/tháng tại Ban Quản lý Cụm nhà.", "Phòng 4 người KTX giá bao nhiêu?",
     "[Trích xuất chuẩn] Theo ngữ cảnh, mức phí là 450.000đ/tháng, đăng ký tại Ban Quản lý Cụm nhà."),
    ("QĐ-15/2026 hỗ trợ 100% thẻ BHYT cho con lao động tự do tại Phường Tân Tạo.", "Lãi suất vay ngân hàng là bao nhiêu?",
     "[Từ chối] Xin lỗi, ngữ cảnh chỉ quy định về hỗ trợ thẻ BHYT, không có thông tin lãi suất ngân hàng."),
] * 5

train_ds = Dataset.from_list([
    {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": make_prompt(c, q)},
                  {"role": "assistant", "content": a}]}
    for c, q, a in train_pairs
])
print("Tổng số mẫu train:", len(train_ds))
```

---

### Ô 6 (10 dòng): Gắn mảnh giấy ghi chú `LoRA` (`r=8`) vào `Qwen3-1.7B`
```python
from peft import LoraConfig, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
print("Đã tạo cấu hình LoRA r=8!")
```
- **Giải thích:** `prepare_model_for_kbit_training(model)` khóa cứng 1.72 tỷ tham số gốc lại; `LoraConfig` chỉ gắn các ma trận nhỏ vào 4 lớp chú ý (`q, k, v, o`).

---

### Ô 7 (11 dòng): Bấm nút Huấn luyện 30 bước (~25 giây)
```python
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=SFTConfig(output_dir="out_m1", max_steps=30, per_device_train_batch_size=2,
                   learning_rate=2e-4, logging_steps=5, fp16=True, report_to="none"),
)
trainer.model.print_trainable_parameters()
trainer.train()
```
- **Quan sát:** Nhìn dòng `trainable params`: chỉ khoảng **`0.1%`** tham số được mở khóa và cột `Loss` giảm đều qua từng mốc 5 bước!

---

### Ô 8 (12 dòng): 📊 Sản phẩm & Bảng So sánh Hiệu quả 🔴 TRƯỚC vs. 🟢 SAU Fine-tune
```python
trainer.model.eval()
trainer.model.config.use_cache = True

score_before, score_after = 0, 0
for idx, ((ctx, q, expected), ans_before) in enumerate(zip(eval_cases, before_outputs), 1):
    ans_after = ask_vivu(trainer.model, ctx, q, use_lora=True)
    hit_b = expected.lower() in ans_before.lower()
    hit_a = expected.lower() in ans_after.lower()
    score_before += int(hit_b)
    score_after += int(hit_a)
    print(f"\n================ CA {idx}: {q} ================")
    print(f"🔴 TRƯỚC ({'ĐẠT' if hit_b else 'TRƯỢT'}): {ans_before}")
    print(f"🟢 SAU   ({'ĐẠT' if hit_a else 'TRƯỢT'}): {ans_after}")

print(f"\n🏆 TỔNG KẾT BUỔI 1: Trước = {score_before}/4 ({score_before*25}%) ---> Sau = {score_after}/4 ({score_after*25}%)")
```
