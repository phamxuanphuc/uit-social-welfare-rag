# Buổi 0 (Module 0): Làm chủ Phòng thí nghiệm Colab T4, Giải phẫu `Qwen3-1.7B` & Đo Mốc Baseline

- **Thời lượng thực hành:** 45–60 phút trên Google Colab (`T4 GPU`).
- **Mục tiêu đầu ra:** Hiểu tường tận bộ nhớ VRAM 15 GB, cơ chế cắt chữ `Tokenizer`, cấu trúc 28 tầng bên trong `Qwen/Qwen3-1.7B`, nén `4-bit NF4` xuống `1.27 GB VRAM` và ghi nhận **Mốc Baseline (Trước Fine-tune)**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao cần GPU T4 thay vì CPU lúc học?**
   - **Hình ảnh đời thường:** CPU giống như 1 giáo sư giải toán cực giỏi nhưng làm từng phép tính một; còn GPU T4 giống như **2,560 học sinh cấp 2 đứng thành hàng ngang**, cùng bấm máy tính bỏ túi một lúc. Vì LLM thực chất chỉ là hàng tỷ phép nhân bảng số (`X @ W`), 2,560 học sinh làm đồng loạt sẽ nhanh gấp hàng chục lần!
2. **VRAM (`15,360 MiB`) và Tại sao phải nén `4-bit NF4`?**
   - `Qwen3-1.7B` có $1.72$ tỷ con số trọng số (`w`). Nếu mỗi số lưu ở chuẩn 16-bit (`float16` = 2 bytes), riêng bộ não đã ngốn $1.72 \times 2 \approx \mathbf{3.44\text{ GB}}$. Khi huấn luyện, máy phải lưu thêm đạo hàm (*Gradients*) + đà quán tính (*AdamW states*) + kết quả trung gian (*Activations*), làm VRAM vọt lên trên 15 GB gây sập máy (*Out-Of-Memory*).
   - **`4-bit NF4` (`NormalFloat 4-bit`):** Nén mỗi con số từ 2 bytes xuống **0.5 byte** theo phân phối hình chuông (tập trung độ chi tiết quanh số 0). Nhờ vậy, toàn bộ mô hình chỉ còn chiếm **`1.27 GB VRAM`**, chừa hơn 13 GB trống để học thoải mái!
3. **`Tokenizer` và Khung hội thoại `ChatML` (`enable_thinking=False`):**
   - Máy tính không hiểu chữ *"xe ôm"* mà tra cuốn từ điển **151,936 mảnh chữ** để đổi thành dãy số `input_ids`.
   - Mặc định `Qwen3` có thói quen tự lẩm bẩm trong thẻ `<think>...</think>` rất dài trước khi trả lời. Khi làm trợ lý tra cứu RAG tốc độ cao (học từ `thanhdo881/qwen3-1.7b-vivu-travel-vn`), ta đặt `enable_thinking=False` để khóa thẻ `<think>` tự do, giúp mô hình trả lời thẳng và nhanh gấp 3 lần.

---

## 2. Thực hành 5 Bước trên Google Colab

### Bước 1: Kiểm tra GPU T4 & Cài đặt thư viện nền tảng
Mở Colab $\rightarrow$ `Runtime` $\rightarrow$ `Change runtime type` $\rightarrow$ chọn `T4 GPU`, sau đó chạy 2 ô sau:

```bash
!nvidia-smi
```
```bash
!pip install -q bitsandbytes peft trl datasets accelerate
```
- **Cách đọc kết quả `!nvidia-smi`:** Tìm chữ `Tesla T4` và cột `Memory-Usage: 0MiB / 15360MiB` (xác nhận có đủ 15 GB VRAM trống).

---

### Bước 2: Mở hộp `AutoTokenizer` & Khung `ChatML` của Qwen3
```python
from transformers import AutoTokenizer

model_id = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

cau_hoi = "Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn thì cần giấy tờ gì?"
ids = tokenizer.encode(cau_hoi)
print("1. Vốn từ điển của Qwen3:", len(tokenizer))
print("2. Mã số (input_ids):", ids)
print("3. Từng mảnh chữ:", [tokenizer.decode([i]) for i in ids])

messages = [
    {"role": "system", "content": "Bạn là trợ lý tra cứu an sinh xã hội. Chỉ dùng 100% tiếng Việt."},
    {"role": "user", "content": cau_hoi},
]
print("\n4. Chuỗi ChatML khi tắt Thinking Mode:")
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
```
- **Giải thích:** `add_generation_prompt=True` tự động gắn `<|im_start|>assistant` ở cuối để báo hiệu tới lượt AI điền câu trả lời; `enable_thinking=False` chèn sẵn cặp `<think>\n\n</think>\n\n` rỗng để AI bỏ qua bước suy nghĩ lan man.

---

### Bước 3: Nạp `Qwen3-1.7B` ở chế độ `4-bit NF4` & Soi 28 tầng bên trong
```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
print(f"VRAM đang chiếm (4-bit NF4): {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
print("\nCấu trúc 1 tầng Transformer của Qwen3:\n", model.model.layers[0])
```
- **Giải thích các lớp bạn nhìn thấy trong `layers[0]`:**
  - `self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj`): Các ma trận giúp các chữ trong câu liên kết với nhau.
  - `mlp` (`gate_proj`, `up_proj`, `down_proj` kèm `SiLU`): Bộ lọc phi tuyến giúp mô hình ghi nhớ quy tắc.
  - Đây chính là **7 vị trí vàng** mà các buổi sau ta sẽ gắn giấy ghi chú `LoRA / DoRA` vào!

---

### Bước 4: Tự tay xem `Softmax` chọn chữ tiếp theo (Xóa bỏ hoàn toàn hộp đen)
```python
prompt = "Người lao động chạy xe ôm bị tai nạn nộp hồ sơ tại"
inp = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    logits = model(**inp).logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

top_probs, top_ids = torch.topk(probs, 5)
for p, tid in zip(top_probs, top_ids):
    print(f"Từ ứng viên: '{tokenizer.decode([tid.item()])}' | Xác suất: {p.item()*100:.2f}%")
```

---

### Bước 5: Ghi nhận kết quả 🔴 TRƯỚC Fine-tune (Mốc Baseline)
```python
model.eval()

def ask_baseline(question):
    msgs = [
        {"role": "system", "content": "Bạn là trợ lý tra cứu an sinh xã hội. Chỉ dùng 100% tiếng Việt."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    prompt_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, do_sample=False)
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)

print("🔴 KẾT QUẢ BASELINE (CHƯA FINE-TUNE):")
print(ask_baseline("Địa bàn: Phường Tân Tạo. Tôi chạy xe ôm bị tai nạn cần chuẩn bị giấy tờ gì và nộp ở đâu?"))
```
- **Quan sát lỗi của Baseline:** Mô hình gốc sẽ trả lời chung chung (kiểu lên cơ quan công an hoặc bảo hiểm xã hội quận/huyện), **không biết** đến mã văn bản `NQ-01/2026/HĐND-HCM` hay *Bộ phận Một cửa UBND Phường Tân Tạo*.

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 0 (Checkpoint)
1. Tại sao `load_in_4bit=True` giảm VRAM xuống `1.27 GB` nhưng vẫn cần `bnb_4bit_compute_dtype=torch.float16`? *(Trả lời: Lưu trữ ở 4-bit cho nhẹ, nhưng lúc nhân ma trận tạm bung ra 16-bit để không mất độ chính xác)*.
2. Tham số `enable_thinking=False` giải quyết vấn đề gì của dòng mô hình Qwen3 khi làm RAG? *(Trả lời: Tắt khối `<think>` tự do dài dòng, giảm độ trễ và tránh lặp vô tận)*.
