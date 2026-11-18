# Buổi 11 (Module 11 — Capstone P01): Đánh giá Toàn diện (`VMLU`, `Ragas`, Sampling Stability) & Portfolio Model Card

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` (đã tinh chỉnh cho đề tài **P01 — Tra cứu quyền lợi an sinh SDG 1.3**).
- **🎯 Sản phẩm cuối buổi:**
  1. **Bộ Đánh giá Hoàn chỉnh trên Tập Test Giữ Riêng:** Đo **3 lần chạy với Sampling (`do_sample=True, temperature=0.2`)** để lấy `Mean ± Std` có ý nghĩa thật, kèm **Kiểm tra Chống Quên Kiến thức Chung** (`Catastrophic Forgetting Check` kiểu benchmark tiếng Việt `VMLU`).
  2. **Hồ sơ `Model Card` & License** chuẩn Hugging Face Hub để đưa vào CV / Bảo vệ Đồ án.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Tại sao phải đo Chống Quên Kiến thức (`Catastrophic Forgetting` — `VMLU`)?**
   - **Hình ảnh Feynman:** Nếu bạn luyện cho học sinh giải 1,000 bài toán đố mẹo nhưng quên kiểm tra xem nó còn biết đọc viết bình thường không, có thể sau khi học xong nó mất luôn khả năng hiểu tiếng Việt tự nhiên! Khi đi phỏng vấn, sếp luôn hỏi: *"Sau khi fine-tune domain hẹp, bạn có đo lại trên benchmark tổng quát như MMLU / VMLU để chứng minh model không bị 'ngáo' không?"*.
2. **Cạm bẫy của `LLM-as-a-Judge` (`Ragas Faithfulness`) cần biết khi phỏng vấn:**
   - Dùng một LLM lớn chấm điểm câu trả lời của `Qwen3-1.7B` giúp tự động hóa tốt, nhưng luôn có 3 thiên lệch (*Bias*): (1) **Verbosity Bias** (thích câu trả lời dài dòng hơn câu ngắn gọn đúng ý), (2) **Self-Enhancement Bias** (thích câu viết theo văn phong của chính hãng mình), (3) **Position Bias** (ưu ái phương án đứng đầu khi so cặp). Vì vậy phải kết hợp **Deterministic Code Checks** (kiểm tra `doc_code`, `expired_warning`, JSON Schema) song song với `Ragas`!
3. **Tại sao 3 lần đo ổn định BẮT BUỘC phải bật `do_sample=True`?**
   - Nếu đặt `do_sample=False` (Greedy), mô hình chạy 100 lần đều ra đúng 1 chuỗi giống hệt nhau $\rightarrow$ độ lệch chuẩn luôn bằng `0.0%`, không phản ánh được độ ổn định thực tế khi triển khai cho người dùng với `temperature > 0`.

---

## 2. Thực hành Đo lường Trung thực & Đóng gói Portfolio trên Colab

### Bước 1: Kiểm tra Độ Ổn định qua 3 Lần Chạy Có Sampling (`do_sample=True, temperature=0.2`)
```python
import json, numpy as np, torch

def ask_sampled(model_obj, chunks, region, question, use_lora: bool) -> str:
    user_txt = f"<context>\n{json.dumps(chunks, ensure_ascii=False)}\n</context>\n\nCÂU HỎI:\nĐịa bàn: {region}. Câu hỏi: {question}"
    msgs = [{"role": "system", "content": SYS_RAFT}, {"role": "user", "content": user_txt}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([txt], return_tensors="pt").to("cuda")
    ctx = model_obj.disable_adapter() if (not use_lora and hasattr(model_obj, "disable_adapter")) else torch.no_grad()
    with ctx, torch.no_grad():
        out = model_obj.generate(**inp, max_new_tokens=180, do_sample=True, temperature=0.2, top_p=0.9)
    return tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

def benchmark_with_sampling(model_obj, suite, label: str, use_lora: bool, n_runs: int = 3):
    runs = []
    for _ in range(n_runs):
        cite_ok, ref_ok, exp_ok = [], [], []
        for r in suite:
            raw = ask_sampled(model_obj, r["chunks"], r["region"], r["q"], use_lora=use_lora)
            try:
                obj = json.loads(raw)
            except Exception:
                obj = {}
            if r["is_unans"]:
                ref_ok.append(float(bool(obj.get("is_refusal"))))
            else:
                cite_ok.append(float(obj.get("doc_code") == r["exp_code"]))
                if any(p["doc_code"] == r["exp_code"] and p["status"] == "expired" for p in r["chunks"]):
                    exp_ok.append(float(bool(obj.get("expired_warning"))))
        runs.append({
            "Citation_Precision": np.mean(cite_ok) * 100 if cite_ok else 0.0,
            "Refusal_Accuracy": np.mean(ref_ok) * 100 if ref_ok else 0.0,
            "Expired_Recall": np.mean(exp_ok) * 100 if exp_ok else 0.0,
        })
    stats = {k: f"{np.mean([x[k] for x in runs]):.1f}% ± {np.std([x[k] for x in runs]):.1f}%" for k in runs[0]}
    print(f"📊 {label}: {stats}")
    return stats
```

### Bước 2: Kiểm tra Chống Quên Kiến thức Chung (Mini General Vietnamese Check kiểu `VMLU`)
```python
general_vi_check = [
    ("Thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam là thành phố nào?", "Hà Nội"),
    ("Một năm nhuận theo dương lịch có bao nhiêu ngày?", "366"),
]
for q, ans_kw in general_vi_check:
    out_gen = ask_qwen(model_fast, "Trả lời ngắn gọn bằng 100% tiếng Việt.", q, use_lora=True, max_tokens=60)
    print(f"🧠 Kiểm tra không quên kiến thức ('{ans_kw}' in output = {ans_kw.lower() in out_gen.lower()}): {out_gen}")
```

### Bước 3: Mẫu `Model Card` Chuẩn Hugging Face Hub (Kèm License & Giới hạn Sử dụng)
```markdown
---
language: [vi]
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
tags: [rag, qwen3, unsloth, dora, orpo, grpo, gguf, responsible-ai, sdg-1.3]
---
# Qwen3-1.7B Social Welfare RAG (SDG 1.3 - P01)
- **Mục đích:** Hỗ trợ người lao động phi chính thức tra cứu giấy tờ và nơi nộp hồ sơ an sinh xã hội từ văn bản công khai (giả lập).
- **Quyền sở hữu & License:** Kế thừa Apache-2.0 từ `Qwen/Qwen3-1.7B`.
- **Giới hạn & Rủi ro còn lại (Residual Risk):** Chỉ hỗ trợ tra cứu chuẩn bị hồ sơ ban đầu; không thay thế quyết định xét duyệt của cán bộ hành chính. Bắt buộc chạy kèm Guardrail lọc địa bàn và cờ `expired` ở Backend.
```

---

## 3. Bảng Chỉ tiêu Mục tiêu (Target Benchmarks — Điền Số Đo Thực tế Sau Khi Chạy Đủ Tập Test!)

> ⚠️ *Lưu ý trung thực khoa học: Các cột dưới đây là **ngưỡng mục tiêu thiết kế (Target)**; khi nộp báo cáo đồ án hay đưa vào CV, bạn chỉ ghi con số thực tế đo được từ hàm `benchmark_with_sampling()` trên tập Test giữ riêng (>= 50 mẫu).*

| Chỉ số đánh giá | Ngưỡng chốt trước | Mốc 1: Baseline *(Đo thực)* | Mốc 2: `SFT + RAFT` *(Target)* | Mốc 3: `+ DPO/ORPO/GRPO` *(Target)* | Mốc 4: `Q4_K_M.gguf` *(Target)* |
|---|---|---|---|---|---|
| **Citation Precision** | $\ge 90\%$ | *Điền kết quả chạy* | $\ge 90\%$ | $\ge 95\%$ | $\ge 93\%$ |
| **Refusal Accuracy** | $\ge 90\%$ | *Điền kết quả chạy* | $\ge 90\%$ | $\ge 95\%$ | $\ge 95\%$ |
| **Expired Detection Recall** | $= 100\%$ | *Điền kết quả chạy* | $\ge 85\%$ | **$100\%$** *(+ Guardrail)* | **$100\%$** |
| **Fairness Gap (Chuẩn vs Đời thường)** | $\le 10\%$ | *Điền kết quả chạy* | $\le 8\%$ | $\le 5\%$ | $\le 6\%$ |
| **Ragas Faithfulness** | $\ge 0.85$ | *Điền kết quả chạy* | $\ge 0.86$ | $\ge 0.92$ | $\ge 0.90$ |
| **General Vi Check (`VMLU`)** | Không tụt $> 3\%$ | *Điền kết quả chạy* | Bảo toàn | Bảo toàn | Bảo toàn |
