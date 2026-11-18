# Buổi 0 (Module 0): Phòng thí nghiệm Colab T4, `4-bit NF4`, `FP16 vs BF16` & Đo Baseline

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU` (15 GB VRAM).
- **🎯 Sản phẩm cuối buổi:** **Máy X-Quang `Qwen3-1.7B` & Bộ Đo Baseline** — soi Top-5 xác suất chữ tiếp theo và đo điểm gốc (trước fine-tune).
- **Chế độ học:** Chia siêu nhỏ từng ô (4–10 dòng/ô), ghim phiên bản thư viện để tái lập 100%.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Tại sao phải dùng `FP16` trên T4 thay vì `BF16`?**
   - **Hình ảnh Feynman:** `FP16` và `BF16` đều là số 16-bit (2 bytes). `BF16` có dải số rộng hơn nên ít bị tràn số khi train, nhưng **chỉ card đời mới (Ampere A100/L4/RTX 30xx trở lên) mới có mạch phần cứng chạy `BF16`**. Card **Tesla T4** thuộc thế hệ Turing cũ hơn nên **chỉ tăng tốc được `FP16`**. Bật `bf16=True` trên T4 sẽ báo lỗi hoặc chạy chậm như rùa!
2. **Tại sao cần nén `4-bit NF4` (`1.27 GB VRAM`)?**
   - Mô hình 1.72 tỷ tham số ở `FP16` chiếm ~3.44 GB. Khi train cần thêm chỗ cho Gradients + Optimizer + Activations, dễ vượt 15 GB của T4. Nén 4-bit `NF4` đưa mô hình về **`1.27 GB`**, chừa >13 GB để học thoải mái.
3. **Tại sao tắt Thinking Mode (`enable_thinking=False`)?**
   - Qwen3 mặc định sinh `<think>...</think>` rất dài trước khi trả lời. Tắt đi giúp trả lời thẳng vào JSON/câu trả lời, nhanh gấp 3 lần.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab

### Ô 1 (1 dòng): Kiểm tra GPU Tesla T4
```bash
!nvidia-smi
```

### Ô 2 (2 dòng): Cài đặt thư viện đã ghim phiên bản (Reproducibility)
```bash
!pip install -q "transformers==4.51.3" "trl==0.16.1" "peft==0.15.2" "bitsandbytes==0.45.5" "accelerate==1.6.0" "datasets==3.5.0"
```
- **Tại sao phải ghim phiên bản?** `trl` và `transformers` đổi tên tham số rất nhanh; ghim phiên bản đảm bảo chạy hôm nay hay 6 tháng sau vẫn không lỗi.

### Ô 3 (9 dòng): Cố định Seed, Nạp Tokenizer & Kiểm tra Token tiếng Việt
```python
import random, numpy as np, torch
from transformers import AutoTokenizer

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)
MODEL_ID = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ids = tokenizer.encode("Tôi chạy xe ôm ở Phường Tân Tạo")
print("Từ điển:", len(tokenizer), "| Mảnh cắt:", [tokenizer.decode([i]) for i in ids])
```

### Ô 4 (11 dòng): Nạp `Qwen3-1.7B` chế độ `4-bit NF4` (`compute_dtype=float16`)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
model.eval()
print(f"VRAM đang chiếm: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
```

### Ô 5 (9 dòng): 🎯 Sản phẩm 1 — Máy X-Quang Top-5 chữ trong não `Qwen3-1.7B`
```python
def xray_next_word(prompt_text: str):
    inp = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        probs = torch.softmax(model(**inp).logits[0, -1, :], dim=-1)
    for p, tid in zip(*torch.topk(probs, 5)):
        print(f"  -> '{tokenizer.decode([tid.item()])}' ({p.item()*100:.1f}%)")

xray_next_word("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại")
```

### Ô 6 (11 dòng): Hàm hỏi chuẩn `ask_qwen()` nhận `sys_prompt` linh hoạt
```python
def ask_qwen(model_obj, sys_prompt: str, user_prompt: str, use_lora: bool = True, max_tokens: int = 150) -> str:
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([text], return_tensors="pt").to("cuda")
    p_len = inp["input_ids"].shape[-1]
    ctx = model_obj.disable_adapter() if (not use_lora and hasattr(model_obj, "disable_adapter")) else torch.no_grad()
    with ctx, torch.no_grad():
        out = model_obj.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    return tokenizer.decode(out[0][p_len:], skip_special_tokens=True)
```

### Ô 7 (10 dòng): 📊 Sản phẩm 2 — Đo Mốc Baseline (Trước Fine-tune)
```python
SYS_BASE = "Bạn là trợ lý tra cứu an sinh xã hội. Chỉ dùng 100% tiếng Việt."
baseline_tests = [
    ("Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn cần văn bản nào?", "NQ-01/2026/HĐND-HCM"),
    ("Con của phụ hồ ở Phường Tân Tạo xin hỗ trợ BHYT theo quyết định nào?", "QĐ-15/2026/UBND-HCM"),
    ("Xin tiền thuê nhà trọ năm 2024 ở Phường Tân Tạo còn hiệu lực không?", "hết hiệu lực"),
]
hits = [kw.lower() in ask_qwen(model, SYS_BASE, q).lower() for q, kw in baseline_tests]
print(f"📊 ĐIỂM BASELINE (TRƯỚC FINE-TUNE): {sum(hits)}/{len(hits)} ({sum(hits)/len(hits)*100:.0f}%)")
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Trên GPU T4, chọn `fp16` hay `bf16` khi train QLoRA, và tại sao?*
   - **Đáp:** Chọn `fp16`, vì GPU T4 (kiến trúc Turing) không hỗ trợ phần cứng cho `bf16` (chỉ có từ Ampere A100/L4 trở lên). Khi dùng 4-bit NF4, trọng số lưu 4-bit nhưng tính toán giải nén tạm về `float16`.
2. **Hỏi:** *Tại sao phải cố định `seed` và ghim phiên bản thư viện?*
   - **Đáp:** Để đảm bảo **Reproducibility** (tái lập được kết quả khi audit hoặc chạy lại pipeline MLOps).
