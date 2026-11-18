# Buổi 0 (Module 0): Máy X-Quang Bộ Não `Qwen3-1.7B` & Bộ Đo Baseline (Chia nhỏ từng ô 4–8 dòng)

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm chạy được cuối buổi:** **Công cụ "X-Quang & Chấm điểm Baseline `Qwen3-1.7B`"** — vừa soi được xác suất Top-5 chữ tiếp theo bên trong não mô hình, vừa tự động chấm điểm xem mô hình gốc trả lời đúng bao nhiêu `%` trên bài toán P01 trước khi học.
- **Cách thực hành buổi đầu:** Tuyệt đối không chạy khối code dài — bạn chỉ cần copy từng **ô siêu nhỏ (4–8 dòng)** dưới đây vào Colab, bấm chạy và nhìn kết quả ngay!

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao cần GPU T4 thay vì CPU?**
   - CPU giống 1 giáo sư làm từng phép tính một; còn GPU T4 giống **2,560 học sinh cùng bấm máy tính một lúc**. Vì `Qwen3-1.7B` là 1.72 tỷ con số nhân ma trận, chạy trên T4 nhanh gấp hàng chục lần.
2. **Tại sao phải nén `4-bit NF4`?**
   - Ở chuẩn 16-bit (`FP16`), `Qwen3-1.7B` ngốn **~3.44 GB VRAM**. Khi nén sang **`4-bit NF4`**, mỗi con số chỉ tốn 0.5 byte nên toàn bộ mô hình chỉ còn **`1.27 GB VRAM`**, chừa hơn 13 GB trống trên card T4 để bạn thoải mái Fine-tune ở các buổi sau.
3. **Tại sao phải tắt Thinking Mode (`enable_thinking=False`)?**
   - Mặc định Qwen3 hay tự lẩm bẩm trong thẻ `<think>...</think>` mất 5–10 giây trước khi trả lời. Tắt đi giúp mô hình trả lời thẳng vào trọng tâm, nhanh gấp 3 lần.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab

### Ô 1 (1 dòng): Kiểm tra card đồ họa Tesla T4 15 GB
```bash
!nvidia-smi
```
- **Giải thích:** Dấu `!` báo Colab chạy lệnh Linux. Nhìn vào bảng kết quả, thấy chữ `Tesla T4` và `15360MiB` là đạt!

---

### Ô 2 (1 dòng): Cài thư viện nén 4-bit và huấn luyện
```bash
!pip install -q bitsandbytes peft trl datasets accelerate
```
- **Giải thích:** Cài đặt sẵn `bitsandbytes` (nén 4-bit), `peft` (gắn LoRA) và `trl` (huấn luyện). Cờ `-q` giúp màn hình gọn sạch.

---

### Ô 3 (6 dòng): Tải "Cuốn từ điển" (`Tokenizer`) của `Qwen3-1.7B`
```python
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

cau_thu = "Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn"
ids = tokenizer.encode(cau_thu)
print("Số từ trong từ điển:", len(tokenizer), "| Các mảnh cắt:", [tokenizer.decode([i]) for i in ids])
```
- **Giải thích từng dòng:**
  - `AutoTokenizer.from_pretrained(...)`: Tải cuốn sổ danh bạ ~151,936 mảnh chữ của Qwen3.
  - `tokenizer.encode(...)`: Đổi câu tiếng Việt thành dãy mã số `ids` vì GPU chỉ biết nhân cộng con số.

---

### Ô 4 (8 dòng): Cấu hình công thức nén `4-bit NF4`
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
print("Đã tạo xong công thức nén 4-bit NF4!")
```
- **Giải thích từng dòng:**
  - `load_in_4bit=True` & `bnb_4bit_quant_type="nf4"`: Nén trọng số xuống 4-bit theo phân phối chuẩn.
  - `bnb_4bit_compute_dtype=torch.float16`: Lúc nhân ma trận thì tạm bung từng cụm nhỏ ra 16-bit để tính cho chuẩn.

---

### Ô 5 (6 dòng): Nạp "Bộ não" `Qwen3-1.7B` lên GPU T4 và đo VRAM
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()
print(f"Đã nạp Qwen3-1.7B! VRAM chỉ tốn: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
```
- **Giải thích:** `device_map="auto"` tự động đẩy toàn bộ 28 tầng của Qwen3 lên GPU T4. Bạn sẽ thấy VRAM chỉ tốn đúng **`1.27 GB`**!

---

### Ô 6 (9 dòng): 🎯 Sản phẩm 1 — "Máy X-Quang" soi Top-5 chữ trong não `Qwen3-1.7B`
```python
def xray_next_word(prompt_text: str):
    inp = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits[0, -1, :], dim=-1)
    top_p, top_i = torch.topk(probs, 5)
    print(f"\n🔬 X-Quang câu: '{prompt_text} ...'")
    for p, tid in zip(top_p, top_i):
        print(f"  -> '{tokenizer.decode([tid.item()])}' ({p.item()*100:.1f}%)")

xray_next_word("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại")
```
- **Giải thích:** Hàm `torch.softmax` biến 151,936 điểm số ở tầng cuối của Qwen3 thành phần trăm `%`. Bạn thấy tận mắt 5 chữ mà mô hình đang định viết tiếp theo!

---

### Ô 7 (10 dòng): Viết hàm hỏi đáp tắt `<think>` (`enable_thinking=False`)
```python
def ask_qwen(question: str, sys_msg: str = "Trả lời ngắn gọn bằng 100% tiếng Việt.") -> str:
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inp = tokenizer([text], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=150, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

print(ask_qwen("Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn thì nộp hồ sơ nhận hỗ trợ ở đâu?"))
```
- **Giải thích:** `do_sample=False` bắt mô hình luôn chọn chữ có xác suất cao nhất (không tung xúc xắc ngẫu nhiên), giúp kết quả đo lường cố định 100%.

---

### Ô 8 (10 dòng): 📊 Sản phẩm 2 — Bảng Chấm điểm Baseline Tự động (Trước Fine-tune)
```python
baseline_tests = [
    ("Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn cần văn bản nào?", "NQ-01/2026/HĐND-HCM"),
    ("Con của phụ hồ ở Phường Tân Tạo xin hỗ trợ BHYT theo quyết định nào?", "QĐ-15/2026/UBND-HCM"),
    ("Xin tiền thuê nhà trọ năm 2024 ở Phường Tân Tạo còn hiệu lực không?", "hết hiệu lực"),
]

correct = 0
for q, expected_kw in baseline_tests:
    ans = ask_qwen(q)
    hit = expected_kw.lower() in ans.lower()
    correct += int(hit)
    print(f"\n❓ {q}\n🔴 Trả lời gốc: {ans[:120]}...\n👉 Chứa thông tin chuẩn '{expected_kw}': {hit}")

print(f"\n📊 ĐIỂM BASELINE (TRƯỚC FINE-TUNE): {correct}/{len(baseline_tests)} ({correct/len(baseline_tests)*100:.0f}%)")
```
- **Ý nghĩa động lực:** Bạn sẽ thấy điểm Baseline ban đầu là **`0/3 (0%)`** vì `Qwen3-1.7B` gốc chưa hề biết quy định địa phương P01. Từ Buổi 1 trở đi, chúng ta sẽ làm con số này tăng lên **`100%`**!
