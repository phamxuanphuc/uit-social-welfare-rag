# Buổi 2 (Module 2): API `Evidence-Guard` — `DoRA + rsLoRA`, Verification-First JSON & Chống Nhiễm Adapter

- **Tham chiếu:** `ViHSD` / `vi-explainable` và `sameerahmed-cloud/verity-rag`.
- **🎯 Sản phẩm cuối buổi:** **API Kiểm duyệt & Bóc tách Bằng chứng (`Evidence-Guard`)** xuất `Verification-First JSON` + hàm `verify_grounding()` bắt trích dẫn giả trên **tập Test giữ riêng**, nạp lại mô hình gốc sạch để đo đúng 🔴 Trước vs. 🟢 Sau.

---

## 1. Giải thích theo Phương pháp Feynman

1. **`Verification-First JSON`:** `Qwen3-1.7B` sinh chữ từ trái sang phải. Đặt `verification_reasoning` làm khóa đầu tiên ép mô hình đối chiếu bằng chứng trước rồi mới chốt nhãn, tránh bịa lý do sau khi lỡ đoán sai.
2. **`DoRA` + `rsLoRA`:** `DoRA` tách riêng **Biên độ** và **Hướng** của trọng số để xoay góc học chính xác hơn `LoRA` thường; `rsLoRA` dùng hệ số $\frac{\alpha}{\sqrt{r}}$ giúp không bị yếu đạo hàm khi nâng `r=16`.
3. **Bẫy hay gặp khi thực hành nhiều bài liên tiếp:** Nếu bạn gắn LoRA của Buổi 1 vào biến `model`, sang Buổi 2 không giải phóng mà đo "Trước fine-tune" trên biến đó thì kết quả bị nhiễm Buổi 1! Ta luôn phải xóa mô hình cũ (`del`, `empty_cache`) và nạp lại base sạch.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab

### Ô 1 (9 dòng): Dọn VRAM Buổi 1 & Nạp lại `Qwen3-1.7B` Sạch (Không nhiễm LoRA cũ)
```python
import gc, json, torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments

for var in ("model", "model_m1", "trainer_m1"):
    if var in globals():
        del globals()[var]
gc.collect(); torch.cuda.empty_cache()

base_m2 = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map="auto")
print(f"Đã nạp lại Qwen3-1.7B sạch! VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
```

### Ô 2 (10 dòng): Hàm Kiểm chứng Bằng chứng Tự động (`verify_grounding` kiểu `verity-rag`)
```python
SYS_JSON = (
    "Chỉ trả về đúng 1 JSON object hợp lệ gồm 4 khóa theo đúng thứ tự: "
    "verification_reasoning, label, citation_tag, evidence_excerpt."
)

def verify_grounding(raw_output: str, context: str) -> dict:
    try:
        obj = json.loads(raw_output)
    except Exception:
        return {"json_ok": False, "grounded": False}
    keys_ok = list(obj.keys())[:4] == ["verification_reasoning", "label", "citation_tag", "evidence_excerpt"]
    excerpt = str(obj.get("evidence_excerpt", "")).strip()
    in_ctx = len(excerpt) >= 10 and (excerpt.lower()[:20] in context.lower())
    return {"json_ok": keys_ok, "grounded": keys_ok and in_ctx, "data": obj}
```

### Ô 3 (10 dòng): Tập Train & Tập Test Giữ Riêng Độc Lập (Không dùng `] * N`)
```python
train_m2_raw = [
    ("[Context 1] Bình luận: 'Bọn lừa đảo đáng bị đánh đập không thương tiếc'.", "Phân loại độc hại và trích bằng chứng.",
     {"verification_reasoning": "[Context 1] chứa cụm từ kích động bạo lực trực tiếp.", "label": "OFFENSIVE", "citation_tag": "[Context 1]", "evidence_excerpt": "Bọn lừa đảo đáng bị đánh đập không thương tiếc"}),
    ("[Context 1] Bình luận: 'Cảm ơn cán bộ phường đã hướng dẫn làm giấy tờ rất nhiệt tình'.", "Phân loại độc hại và trích bằng chứng.",
     {"verification_reasoning": "[Context 1] là lời cảm ơn lịch sự, không công kích.", "label": "CLEAN", "citation_tag": "[Context 1]", "evidence_excerpt": "Cảm ơn cán bộ phường đã hướng dẫn làm giấy tờ rất nhiệt tình"}),
    ("[Context 1] NQ-01/2026: Hỗ trợ tai nạn 2.000.000đ.\n[Context 2] QĐ-15/2026: Hỗ trợ 100% thẻ BHYT cho con lao động tự do.", "Trích quy định hỗ trợ thẻ BHYT cho con người lao động.",
     {"verification_reasoning": "Nội dung BHYT cho con khớp trực tiếp [Context 2].", "label": "POLICY_BHYT", "citation_tag": "[Context 2]", "evidence_excerpt": "Hỗ trợ 100% thẻ BHYT cho con lao động tự do"}),
    ("[Context 1] NQ-08/2026: Hỗ trợ học nghề 3.000.000đ tại Quận Liên Chiểu.\n[Context 2] TT-04/2026: Quá 15 ngày không bổ sung hồ sơ sẽ bị trả lại.", "Trích quy định hỗ trợ học nghề tại Quận Liên Chiểu.",
     {"verification_reasoning": "Nội dung học nghề tại Liên Chiểu khớp [Context 1].", "label": "POLICY_TRAINING", "citation_tag": "[Context 1]", "evidence_excerpt": "Hỗ trợ học nghề 3.000.000đ tại Quận Liên Chiểu"}),
    ("[Context 1] Bình luận: 'Sản phẩm giao nhanh, đóng gói cẩn thận, rất đáng tiền'.", "Phân loại độc hại và trích bằng chứng.",
     {"verification_reasoning": "[Context 1] là đánh giá tích cực về dịch vụ.", "label": "CLEAN", "citation_tag": "[Context 1]", "evidence_excerpt": "Sản phẩm giao nhanh, đóng gói cẩn thận, rất đáng tiền"}),
    ("[Context 1] NQ-01/2026: Lao động bán vé số gặp rủi ro đột xuất được xét trợ cấp tối đa 4.500.000đ.", "Trích quy định trợ cấp lao động bán vé số.",
     {"verification_reasoning": "Nội dung trợ cấp bán vé số khớp [Context 1].", "label": "POLICY_EMERGENCY", "citation_tag": "[Context 1]", "evidence_excerpt": "Lao động bán vé số gặp rủi ro đột xuất được xét trợ cấp tối đa 4.500.000đ"}),
]

test_m2_suite = [
    ("[Context 1] Bình luận: 'Đồ ăn bám xã hội, biến khỏi khu phố này ngay!'", "Phân loại độc hại và trích bằng chứng.", "OFFENSIVE"),
    ("[Context 1] TT-04/2026: Thiếu giấy tờ bị trả hồ sơ.\n[Context 2] NQ-01/2026: Xe ôm gặp tai nạn được hỗ trợ chi phí điều trị tại Phường Tân Tạo.", "Trích quy định hỗ trợ tai nạn cho người chạy xe ôm.", "POLICY_EMERGENCY"),
]
```

### Ô 4 (8 dòng): Đo 🔴 TRƯỚC Fine-tune (Dùng đúng `SYS_JSON` trên mô hình sạch!)
```python
before_m2 = []
for idx, (ctx, q, exp_label) in enumerate(test_m2_suite, 1):
    raw_b = ask_qwen(base_m2, SYS_JSON, make_rag_prompt(ctx, q), use_lora=False, max_tokens=160)
    chk_b = verify_grounding(raw_b, ctx)
    before_m2.append((raw_b, chk_b))
    print(f"🔴 Test {idx} TRƯỚC | JSON_OK={chk_b['json_ok']} | Grounded={chk_b['grounded']}\n{raw_b[:130]}")
```

### Ô 5 (12 dòng): Đóng gói Dataset Có Masking & Bật `DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`)
```python
def encode_json_sample(row):
    p_txt = tokenizer.apply_chat_template([{"role": "system", "content": SYS_JSON}, {"role": "user", "content": make_rag_prompt(row["c"], row["q"])}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    f_txt = p_txt + json.dumps(row["o"], ensure_ascii=False) + tokenizer.eos_token
    p_ids, f_ids = tokenizer(p_txt, add_special_tokens=False)["input_ids"], tokenizer(f_txt, add_special_tokens=False)["input_ids"]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]}

ds_m2 = Dataset.from_list([{"c": c, "q": q, "o": o} for c, q, o in train_m2_raw]).map(encode_json_sample)
base_m2 = prepare_model_for_kbit_training(base_m2)
dora_cfg = LoraConfig(r=16, lora_alpha=32, use_dora=True, use_rslora=True, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none", task_type="CAUSAL_LM")
model_m2 = get_peft_model(base_m2, dora_cfg)
```

### Ô 6 (10 dòng): Huấn luyện `DoRA + rsLoRA` (`num_train_epochs=6`) & Đo 🟢 SAU Fine-tune
```python
Trainer(
    model=model_m2,
    args=TrainingArguments(output_dir="out_m2", num_train_epochs=6, per_device_train_batch_size=2, learning_rate=2e-4, logging_steps=3, fp16=True, report_to="none"),
    train_dataset=ds_m2,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
).train()

model_m2.eval(); model_m2.config.use_cache = True
for idx, ((ctx, q, exp_lbl), (raw_b, chk_b)) in enumerate(zip(test_m2_suite, before_m2), 1):
    raw_a = ask_qwen(model_m2, SYS_JSON, make_rag_prompt(ctx, q), use_lora=True, max_tokens=160)
    chk_a = verify_grounding(raw_a, ctx)
    print(f"\n=== TEST {idx} ===\n🔴 Trước (Grounded={chk_b['grounded']}): {raw_b[:90]}\n🟢 Sau   (Grounded={chk_a['grounded']}): {raw_a}")
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Vì sao nên đặt trường suy luận (`verification_reasoning`) làm key đầu tiên trong JSON thay vì đặt ở cuối?*
   - **Đáp:** Vì mô hình autoregressive sinh token từ trái sang phải. Khi sinh lý do đối chiếu trước, các token đó nằm vào context giúp các trường kết luận phía sau (`label`, `doc_code`) chính xác hơn đáng kể.
2. **Hỏi:** *`DoRA` cải thiện gì so với `LoRA` thường?*
   - **Đáp:** `DoRA` phân rã ma trận trọng số thành Biên độ (magnitude) và Hướng (direction), chỉ cập nhật hướng bằng LoRA và điều chỉnh biên độ độc lập, cho hành vi học gần với Full Fine-tuning hơn.
