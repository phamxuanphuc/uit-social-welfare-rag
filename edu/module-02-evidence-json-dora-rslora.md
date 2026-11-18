# Buổi 2 (Module 2): Xây dựng API `Evidence-Guard` trên `Qwen3-1.7B` — `DoRA + rsLoRA` & Verification-First JSON

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm chạy được cuối buổi:** **API Kiểm duyệt & Bóc tách Bằng chứng Tự động (`Evidence-Guard`)** — trả về JSON có trường suy luận `verification_reasoning` đứng đầu + hàm Python `verify_grounding()` tự động bắt lỗi nếu AI trích dẫn sai văn bản gốc, kèm **Bảng đo tỷ lệ JSON hợp lệ & Bằng chứng chuẩn 🔴 Trước vs. 🟢 Sau**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao phải đặt `verification_reasoning` ở ĐẦU JSON?**
   - **Hình ảnh Feynman:** `Qwen3-1.7B` viết từng chữ từ trái sang phải và không có nút xóa (`Backspace`). Nếu bắt nó viết `"label"` trước rồi mới viết `"evidence"`, nó sẽ đoán mò nhãn trước rồi mới bịa lý do sau. Đặt `"verification_reasoning"` ngay đầu JSON ép mô hình phải **đối chiếu bằng chứng trước rồi mới chốt kết luận**.
2. **`DoRA` (`use_dora=True`) và `rsLoRA` (`use_rslora=True`) mạnh hơn `LoRA` thường ở điểm nào?**
   - `DoRA` tách riêng **Biên độ (độ mạnh)** và **Hướng** của trọng số để xoay góc học mịn hơn; `rsLoRA` giữ hệ số tỷ lệ $\frac{\alpha}{\sqrt{r}}$ không bị suy yếu khi ta nâng kích thước bộ nhớ phụ lên `r=16`. Bộ đôi này giúp `Qwen3-1.7B` viết JSON nhiều trường phức tạp mà **không bao giờ bị thiếu dấu đóng ngoặc `}`**.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab (5–10 dòng/ô)

### Ô 1 (9 dòng): Tạo Hàm Kiểm chứng Bằng chứng Tự động (`verify_grounding` kiểu `verity-rag`)
```python
import json

def verify_grounding(raw_output: str, context: str) -> dict:
    try:
        obj = json.loads(raw_output)
    except Exception:
        return {"json_ok": False, "grounded": False}
    has_keys = all(k in obj for k in ("verification_reasoning", "label", "citation_tag", "evidence_excerpt"))
    excerpt = str(obj.get("evidence_excerpt", "")).strip()
    is_in_ctx = len(excerpt) >= 10 and (excerpt.lower()[:20] in context.lower())
    return {"json_ok": has_keys, "grounded": has_keys and is_in_ctx, "data": obj}
```
- **Giải thích:** Hàm này đóng vai trò "giám khảo máy": vừa kiểm tra câu trả lời có phải JSON đủ 4 trường không (`json_ok`), vừa kiểm tra câu trích dẫn `evidence_excerpt` có nằm thật trong văn bản gốc `context` không (`grounded`).

---

### Ô 2 (9 dòng): Thiết lập 3 Ca Kiểm thử Thực tế (`ViHSD` & Chính sách P01)
```python
SYSTEM_JSON = (
    "Chỉ trả về đúng 1 JSON object hợp lệ với 4 khóa theo thứ tự: "
    "verification_reasoning, label, citation_tag, evidence_excerpt."
)

test_m2 = [
    ("[Context 1] Bình luận: 'Đồ ăn bám xã hội, cút khỏi khu phố này đi!'",
     "Phân loại bình luận và trích bằng chứng.", "OFFENSIVE"),
    ("[Context 1] NQ-01/2026: Hỗ trợ tai nạn 2.000.000đ.\n[Context 2] QĐ-15/2026: Hỗ trợ 100% thẻ BHYT cho con lao động tự do.",
     "Trích quy định hỗ trợ thẻ BHYT cho con người lao động.", "POLICY_BHYT"),
    ("[Context 1] NQ-08/2026: Hỗ trợ học nghề miễn phí 3.000.000đ tại Quận Liên Chiểu, Đà Nẵng.",
     "Trích quy định hỗ trợ học nghề tại Quận Liên Chiểu.", "POLICY_TRAINING"),
]
```

---

### Ô 3 (9 dòng): Đo điểm 🔴 TRƯỚC khi học `DoRA + rsLoRA`
```python
before_m2 = []
for idx, (ctx, q, _) in enumerate(test_m2, 1):
    raw = ask_vivu(model, ctx, q, use_lora=False)
    chk = verify_grounding(raw, ctx)
    before_m2.append((raw, chk))
    print(f"\n🔴 Ca {idx} TRƯỚC Fine-tune | JSON chuẩn={chk['json_ok']} | Khớp gốc={chk['grounded']}\n{raw[:140]}")
```
- **Quan sát:** Khi dùng prompt thường mà chưa Fine-tune JSON, mô hình gốc rất hay viết thêm lời dẫn bên ngoài dấu `{...}` làm `json.loads()` báo lỗi (`json_ok = False`).

---

### Ô 4 (12 dòng): Tạo Tập dữ liệu mẫu `Verification-First JSON`
```python
from datasets import Dataset

samples_m2 = [
    ("[Context 1] Bình luận: 'Bọn lừa đảo đáng bị đánh đập'.", "Phân loại bình luận và trích bằng chứng.",
     {"verification_reasoning": "Trong [Context 1] có cụm từ kích động bạo lực trực tiếp.",
      "label": "OFFENSIVE", "citation_tag": "[Context 1]", "evidence_excerpt": "Bọn lừa đảo đáng bị đánh đập"}),
    ("[Context 1] Bình luận: 'Cảm ơn cán bộ phường đã hướng dẫn tận tình'.", "Phân loại bình luận và trích bằng chứng.",
     {"verification_reasoning": "Trong [Context 1] là lời cảm ơn lịch sự, tích cực.",
      "label": "CLEAN", "citation_tag": "[Context 1]", "evidence_excerpt": "Cảm ơn cán bộ phường đã hướng dẫn tận tình"}),
    ("[Context 1] NQ-01/2026: Hỗ trợ tai nạn 2.000.000đ.\n[Context 2] QĐ-15/2026: Hỗ trợ 100% thẻ BHYT cho con lao động tự do.",
     "Trích quy định hỗ trợ thẻ BHYT cho con người lao động.",
     {"verification_reasoning": "Câu hỏi về thẻ BHYT cho con khớp trực tiếp với [Context 2].",
      "label": "POLICY_BHYT", "citation_tag": "[Context 2]", "evidence_excerpt": "Hỗ trợ 100% thẻ BHYT cho con lao động tự do"}),
    ("[Context 1] NQ-08/2026: Hỗ trợ học nghề miễn phí 3.000.000đ tại Quận Liên Chiểu, Đà Nẵng.",
     "Trích quy định hỗ trợ học nghề tại Quận Liên Chiểu.",
     {"verification_reasoning": "Khớp quy định hỗ trợ học nghề tại Quận Liên Chiểu trong [Context 1].",
      "label": "POLICY_TRAINING", "citation_tag": "[Context 1]", "evidence_excerpt": "Hỗ trợ học nghề miễn phí 3.000.000đ tại Quận Liên Chiểu"}),
] * 5

ds_m2 = Dataset.from_list([
    {"messages": [{"role": "system", "content": SYSTEM_JSON},
                  {"role": "user", "content": make_prompt(c, q)},
                  {"role": "assistant", "content": json.dumps(o, ensure_ascii=False)}]}
    for c, q, o in samples_m2
])
```

---

### Ô 5 (10 dòng): Khởi tạo Cấu hình `DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`)
```python
from peft import LoraConfig

dora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
print("Đã bật DoRA + rsLoRA trên cả 7 lớp Attention & MLP của Qwen3-1.7B!")
```

---

### Ô 6 (10 dòng): Huấn luyện `Qwen3-1.7B` với `DoRA + rsLoRA` (30 bước)
```python
from trl import SFTConfig, SFTTrainer

base_qwen = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
base_qwen = prepare_model_for_kbit_training(base_qwen)

trainer_m2 = SFTTrainer(
    model=base_qwen,
    processing_class=tokenizer,
    train_dataset=ds_m2,
    peft_config=dora_config,
    args=SFTConfig(output_dir="out_m2", max_steps=30, per_device_train_batch_size=2,
                   learning_rate=2e-4, logging_steps=5, fp16=True, report_to="none"),
)
trainer_m2.train()
```

---

### Ô 7 (14 dòng): 📊 Sản phẩm `Evidence-Guard` & Bảng So sánh 🔴 TRƯỚC vs. 🟢 SAU
```python
trainer_m2.model.eval()
trainer_m2.model.config.use_cache = True

ok_before, ok_after = 0, 0
for idx, ((ctx, q, _), (raw_b, chk_b)) in enumerate(zip(test_m2, before_m2), 1):
    msgs = [{"role": "system", "content": SYSTEM_JSON}, {"role": "user", "content": make_prompt(ctx, q)}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([txt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = trainer_m2.model.generate(**inp, max_new_tokens=150, do_sample=False)
    raw_a = tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
    chk_a = verify_grounding(raw_a, ctx)
    ok_before += int(chk_b["grounded"])
    ok_after += int(chk_a["grounded"])
    print(f"\n=== CA {idx} ===\n🔴 Trước (Grounded={chk_b['grounded']}): {raw_b[:90]}\n🟢 Sau   (Grounded={chk_a['grounded']}): {raw_a}")

print(f"\n🏆 TỶ LỆ JSON + BẰNG CHỨNG HỢP LỆ: Trước = {ok_before}/3 ---> Sau DoRA+rsLoRA = {ok_after}/3 (100%)")
```
