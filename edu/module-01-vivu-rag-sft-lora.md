# Buổi 1 (Module 1): Chatbot `ViVu-Mini` — `LoRA` & `Prompt Loss Masking` Thật trên Tập Giữ Riêng

- **Tham chiếu:** `thanhdo881/qwen3-1.7b-vivu-travel-vn`.
- **🎯 Sản phẩm cuối buổi:** **Chatbot RAG `ViVu-Mini`** đọc `<context>...</context>`, khóa 100% tiếng Việt, từ chối khi hỏi ngoài ngữ cảnh, **chỉ tính Loss trên câu trả lời** và đo Trước vs. Sau trên **tập Test giữ riêng (không trùng câu Train)**.

---

## 1. Giải thích theo Phương pháp Feynman

1. **`LoRA` (`W = W_0 + (alpha/r) * B @ A`):** Không sửa thẳng sách gốc 4-bit ($W_0$), chỉ dán thêm 2 ma trận nhỏ ($B \cdot A$) vào lề (chỉ học ~0.1–0.3% tham số). Có thể tắt tạm để so sánh bằng `disable_adapter()`.
2. **`Prompt Loss Masking` (Chấm điểm chỉ trên câu trả lời):** Khi đi thi, giáo viên chỉ chấm phần **Bài làm (`assistant`)**, không chấm điểm phần chép đề (`system + <context> + user`). Nếu tính Loss cả trên `<context>`, mô hình phí sức học thuộc đoạn ngữ cảnh dài dòng. Trong PyTorch, các token đề bài được gán nhãn `-100` để hàm `Cross-Entropy` bỏ qua.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab (5–12 dòng/ô)

*(Chạy nối tiếp từ Buổi 0 hoặc nạp lại `model`, `tokenizer`, `ask_qwen` sạch từ Buổi 0 để không nhiễm adapter cũ).*

### Ô 1 (7 dòng): Khuôn mẫu `<context>` & `SYSTEM_VIVU`
```python
SYSTEM_VIVU = (
    "Bạn là trợ lý tra cứu AI chuyên nghiệp. Chỉ trả lời dựa trên <context>. "
    "Bắt buộc dùng 100% TIẾNG VIỆT và mở đầu bằng '[Từ chối]' nếu <context> không có đáp án."
)

def make_rag_prompt(context: str, question: str) -> str:
    return f"<context>\n{context}\n</context>\n\nCÂU HỎI:\n{question}"
```

### Ô 2 (12 dòng): Tạo Tập Train & Tập Test Giữ Riêng Độc Lập (Không trùng lặp!)
```python
# Tập Train: Du lịch Hội An, KTX ĐHQG, An sinh BHYT (8 mẫu khác nhau hoàn toàn)
train_raw = [
    ("Đà Lạt nằm trên cao nguyên Lâm Viên cao 1.500m, nổi tiếng với Hồ Xuân Hương.", "Đà Lạt cao bao nhiêu và có hồ nào?", "[Trích xuất chuẩn] Theo ngữ cảnh, Đà Lạt cao 1.500m và nổi tiếng với Hồ Xuân Hương."),
    ("Phố cổ Hội An miễn vé sau 21h tối, vé ban ngày cho khách Việt Nam là 80.000đ.", "Giá vé máy bay đi Hội An là bao nhiêu?", "[Từ chối] Ngữ cảnh chỉ nêu giá vé tham quan Phố cổ Hội An, không có giá vé máy bay."),
    ("KTX ĐHQG thu phí phòng 4 người máy lạnh là 450.000đ/tháng tại Ban Quản lý Cụm nhà.", "Phòng 4 người KTX giá bao nhiêu và nộp tiền ở đâu?", "[Trích xuất chuẩn] Phí phòng 4 người là 450.000đ/tháng, đăng ký tại Ban Quản lý Cụm nhà."),
    ("QĐ-15/2026 hỗ trợ 100% thẻ BHYT cho con lao động tự do tại Phường Tân Tạo.", "Lãi suất vay mua ô tô trả góp là bao nhiêu?", "[Từ chối] Ngữ cảnh chỉ quy định hỗ trợ thẻ BHYT, không có thông tin lãi suất vay ngân hàng."),
    ("Bảo tàng Chứng tích Chiến tranh mở cửa từ 7h30 đến 17h30 hàng ngày, giá vé 40.000đ.", "Vé vào Bảo tàng Chứng tích Chiến tranh bao nhiêu tiền?", "[Trích xuất chuẩn] Giá vé tham quan là 40.000đ, mở cửa từ 7h30 đến 17h30."),
    ("Thư viện Trung tâm UIT cho phép sinh viên mượn tối đa 5 cuốn sách trong 14 ngày.", "Thư viện UIT có bán đồ ăn sáng không?", "[Từ chối] Ngữ cảnh chỉ quy định số sách và thời hạn mượn tại thư viện, không có thông tin bán đồ ăn."),
    ("Xe buýt số 33 chạy tuyến Bến xe An Sương - Suối Tiên với giá vé sinh viên 3.000đ/lượt.", "Sinh viên đi xe buýt số 33 tốn bao nhiêu tiền?", "[Trích xuất chuẩn] Giá vé sinh viên tuyến xe buýt số 33 là 3.000đ/lượt."),
    ("Cửa hàng hỗ trợ đổi trả miễn phí trong vòng 7 ngày nếu sản phẩm còn nguyên tem.", "Cửa hàng có giao hàng sang nước ngoài trong 2 giờ không?", "[Từ chối] Ngữ cảnh chỉ nêu chính sách đổi trả 7 ngày, không có thông tin giao hàng quốc tế."),
]

# Tập Test giữ riêng: Làng Nổi Tân Lập, Tuyển thẳng UIT, Xe ôm Tân Tạo (chưa từng xuất hiện ở Train!)
test_cases_m1 = [
    ("Làng Nổi Tân Lập (Long An) mở cửa 7h30–17h00, vé xuồng ba lá là 150.000đ/người.", "Đi xuồng ba lá ở Làng Nổi Tân Lập tốn bao nhiêu tiền?", "[Trích xuất chuẩn]", "150.000đ"),
    ("Trường ĐH Công nghệ Thông tin (UIT) tuyển thẳng thí sinh đạt giải Quốc gia môn Tin học.", "Học phí ngành AI tại UIT mỗi học kỳ là bao nhiêu triệu?", "[Từ chối]", ""),
    ("NQ-01/2026 hỗ trợ người chạy xe ôm bị tai nạn tại Phường Tân Tạo mức 2.000.000đ.", "Xe ôm bị tai nạn ở Phường Tân Tạo nhận hỗ trợ mấy triệu?", "[Trích xuất chuẩn]", "2.000.000đ"),
]
```

### Ô 3 (9 dòng): Đo điểm 🔴 TRƯỚC Fine-tune trên Tập Test Giữ Riêng
```python
before_m1 = []
for idx, (ctx, q, exp_tag, exp_kw) in enumerate(test_cases_m1, 1):
    ans = ask_qwen(model, SYSTEM_VIVU, make_rag_prompt(ctx, q), use_lora=False)
    ok = (exp_tag in ans) and (not exp_kw or exp_kw in ans)
    before_m1.append((ans, ok))
    print(f"🔴 Test {idx} (Đạt={ok}): {ans[:120]}")
```

### Ô 4 (13 dòng): Chuẩn bị Dataset & Gắn nhãn `-100` (`Prompt Loss Masking` thật!)
```python
from datasets import Dataset

def tokenize_completion_only(sample):
    prompt_msgs = [{"role": "system", "content": SYSTEM_VIVU}, {"role": "user", "content": make_rag_prompt(sample["c"], sample["q"])}]
    prompt_txt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full_txt = prompt_txt + sample["a"] + tokenizer.eos_token
    p_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full_txt, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}

ds_m1 = Dataset.from_list([{"c": c, "q": q, "a": a} for c, q, a in train_raw]).map(tokenize_completion_only)
print("Kiểm tra nhãn bị che (-100) ở phần đề:", ds_m1[0]["labels"][:6], "... và phần đáp án:", ds_m1[0]["labels"][-6:])
```
- **Zero Black-Box:** Bạn nhìn thấy tận mắt mảng `labels` bắt đầu bằng `[-100, -100, ...]` (phần đề bài bị bỏ qua khi tính Loss) và chỉ giữ ID số thật ở phần câu trả lời `assistant`!

### Ô 5 (12 dòng): Gắn `LoRA (r=8)` & Huấn luyện (`num_train_epochs=4`)
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

model = prepare_model_for_kbit_training(model)
lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model_m1 = get_peft_model(model, lora_cfg)
model_m1.print_trainable_parameters()

trainer_m1 = Trainer(
    model=model_m1,
    args=TrainingArguments(output_dir="out_m1", num_train_epochs=4, per_device_train_batch_size=2, learning_rate=2e-4, logging_steps=4, fp16=True, report_to="none"),
    train_dataset=ds_m1,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
)
trainer_m1.train()
```

### Ô 6 (11 dòng): 📊 So sánh 🔴 TRƯỚC vs. 🟢 SAU trên Tập Test Giữ Riêng
```python
model_m1.eval()
model_m1.config.use_cache = True
b_hits = sum(ok for _, ok in before_m1)
a_hits = 0
for idx, ((ctx, q, exp_tag, exp_kw), (ans_b, ok_b)) in enumerate(zip(test_cases_m1, before_m1), 1):
    ans_a = ask_qwen(model_m1, SYSTEM_VIVU, make_rag_prompt(ctx, q), use_lora=True)
    ok_a = (exp_tag in ans_a) and (not exp_kw or exp_kw in ans_a)
    a_hits += int(ok_a)
    print(f"\n=== TEST {idx} ===\n🔴 Trước ({ok_b}): {ans_b[:110]}\n🟢 Sau   ({ok_a}): {ans_a}")
print(f"\n🏆 KẾT QUẢ TẬP TEST GIỮ RIÊNG: Trước = {b_hits}/3 ---> Sau = {a_hits}/3")
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Prompt Loss Masking (`labels = -100`) khác gì so với tính Loss toàn bộ chuỗi? Khi nào không nên mask prompt?*
   - **Đáp:** Masking giúp tập trung 100% gradient vào đầu ra chuẩn và không phí sức học thuộc ngữ cảnh RAG dài. Chỉ giữ Loss trên toàn bộ chuỗi khi làm **Continued Pretraining** trên văn bản thô không có cặp hỏi–đáp.
2. **Hỏi:** *Vì sao ở bài này chỉ gắn LoRA vào 4 lớp Attention (`q, k, v, o`) mà sau này sang JSON lại gắn thêm 3 lớp MLP (`gate, up, down`)?*
   - **Đáp:** Gắn 4 lớp Attention đủ để học văn phong và cách bám ngữ cảnh; thêm 3 lớp MLP giúp tăng dung lượng nhớ các quy tắc định dạng JSON chặt chẽ hơn.
