# Buổi 2 (Module 2): Phân loại & Trích xuất Bằng chứng JSON — Tham khảo `ViHSD` & `verity-rag` (`DoRA` + `rsLoRA`)

- **Dự án tham chiếu:** Tập dữ liệu `ViHSD` / `vi-explainable` (phân loại kèm bằng chứng & giải thích) và `sameerahmed-cloud/verity-rag` (mã trích dẫn `[Context N]` + Grounding Check).
- **Mục tiêu đầu ra:** Chuyển từ văn bản tự do sang **Verification-First JSON**, tự động đối chiếu câu trả lời với đoạn trích gốc (`Grounding Check`), và nâng cấp từ `LoRA` thường lên **`DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`)**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao phải đặt `verification_reasoning` / `explanation` ở ĐẦU JSON?**
   - **Hình ảnh Feynman:** Mô hình ngôn ngữ viết từng chữ từ trái sang phải và **không có nút Backspace (xóa đi viết lại)**. Nếu bạn bắt nó viết `"label": "Hợp lệ"` ở dòng đầu tiên rồi mới viết `"evidence"` ở dòng dưới, nó sẽ "chốt bừa nhãn trước rồi mới cố tìm lý do biện hộ sau"!
   - Ngược lại, khi ép trường `"verification_reasoning"` nằm **ngay sau dấu mở ngoặc `{`**, mô hình buộc phải đọc kỹ `<context>`, chép ra bằng chứng và kiểm tra điều kiện *trước*, rồi mới dựa vào chính những chữ vừa viết để điền kết luận chính xác ở các trường sau!
2. **`DoRA` (`Weight-Decomposed LoRA`) và `rsLoRA` khác gì `LoRA` thường?**
   - **Hình ảnh Feynman:** Hãy tưởng tượng việc chỉnh một khẩu pháo:
     - `LoRA` thường gộp chung cả **góc quay nòng pháo (Hướng)** và **lượng thuốc súng (Biên độ)** vào một phép cộng $W_0 + BA$, nên khi chỉnh mạnh tay rất dễ làm lệch tâm.
     - **`DoRA` (`use_dora=True`)** tách riêng **Biên độ $m$** và **Hướng $\frac{V}{\|V\|_c}$**: giữ độ lớn ổn định và chỉ dùng Adapter để xoay góc ngắm thật mịn $\rightarrow$ giúp mô hình nhỏ `1.7B` tuân thủ cấu trúc JSON nhiều tầng tốt hơn hẳn LoRA thường.
     - **`rsLoRA` (`use_rslora=True`)** đổi hệ số khuếch đại từ $\frac{\alpha}{r}$ sang $\frac{\alpha}{\sqrt{r}}$ (ví dụ với $r=16, \alpha=32$ thì hệ số là $\frac{32}{4} = 8$ thay vì $2$), giữ tín hiệu học không bị yếu đi khi ta dùng `r=16`.
3. **`Grounding Check` (Học từ `sameerahmed-cloud/verity-rag`) là gì?**
   - Sau khi LLM xuất ra JSON chứa đoạn bằng chứng `evidence` và mã `[Context N]`, ta dùng 5 dòng code Python kiểm tra xem chuỗi `evidence` hoặc mã văn bản đó **có thực sự tồn tại trong `<context>` gốc hay không**. Nếu không khớp $\rightarrow$ bác bỏ ngay!

---

## 2. Thực hành 5 Bước trên Google Colab

### Bước 1: Định nghĩa Cấu trúc JSON & Hàm `Grounding Check` (kiểu `verity-rag`)
```python
import json

SYSTEM_JSON_PROMPT = (
    "Bạn là hệ thống phân tích và trích xuất bằng chứng tiếng Việt. "
    "BẮT BUỘC chỉ trả về đúng 1 JSON object hợp lệ với thứ tự khóa: "
    "verification_reasoning, label, citation_tag, evidence_excerpt."
)

def verify_grounding(raw_json_str: str, source_context: str) -> dict:
    try:
        data = json.loads(raw_json_str)
    except Exception:
        return {"valid_json": False, "grounded": False, "error": "Lỗi cú pháp JSON"}
    excerpt = data.get("evidence_excerpt", "")
    is_grounded = bool(excerpt) and (excerpt.lower()[:25] in source_context.lower())
    return {"valid_json": True, "grounded": is_grounded, "parsed": data}
```

---

### Bước 2: Đo 🔴 TRƯỚC Fine-tune (Xem mô hình gốc dễ làm vỡ JSON hoặc sai thứ tự khóa thế nào)
```python
sample_ctx = (
    "[Context 1] Bình luận: 'Đồ thứ ăn bám xã hội, biến khỏi khu phố này đi!' -> Có cụm từ xúc phạm nhân phẩm.\n"
    "[Context 2] Quyết định QĐ-15/2026/UBND-HCM: Hỗ trợ 100% thẻ BHYT cho con lao động tự do thu nhập thấp tại Phường Tân Tạo."
)
sample_q = "Phân tích nội dung [Context 2] xem thuộc nhóm chính sách nào và trích xuất bằng chứng."

print("🔴 TRƯỚC FINE-TUNE (MODULE 2):")
raw_before = ask_rag(trainer.model, sample_ctx, sample_q, use_lora=False)
print(raw_before)
print("Kiểm tra Grounding:", verify_grounding(raw_before, sample_ctx))
```

---

### Bước 3: Tạo tập dữ liệu mẫu (Kết hợp bài toán `ViHSD` & Trích xuất chính sách P01)
```python
from datasets import Dataset

json_samples = [
    {
        "ctx": "[Context 1] Bình luận mạng xã hội: 'Bọn lừa đảo đáng bị đánh đập không thương tiếc'.",
        "q": "Phân loại mức độ độc hại và trích xuất bằng chứng.",
        "out": {
            "verification_reasoning": "Đối chiếu [Context 1] thấy cụm từ kích động bạo lực trực tiếp 'đáng bị đánh đập'.",
            "label": "OFFENSIVE_HATE",
            "citation_tag": "[Context 1]",
            "evidence_excerpt": "Bọn lừa đảo đáng bị đánh đập không thương tiếc"
        }
    },
    {
        "ctx": "[Context 1] Bình luận mạng xã hội: 'Hôm nay thời tiết Sài Gòn mát mẻ, đi làm rất dễ chịu'.",
        "q": "Phân loại mức độ độc hại và trích xuất bằng chứng.",
        "out": {
            "verification_reasoning": "Đối chiếu [Context 1] chỉ mô tả thời tiết tích cực, không có từ ngữ công kích.",
            "label": "CLEAN",
            "citation_tag": "[Context 1]",
            "evidence_excerpt": "Hôm nay thời tiết Sài Gòn mát mẻ, đi làm rất dễ chịu"
        }
    },
    {
        "ctx": "[Context 1] NQ-01/2026/HĐND-HCM: Lao động phi chính thức gặp tai nạn được hỗ trợ một lần từ 2.000.000 đến 4.500.000 đồng.\n[Context 2] QĐ-15/2026/UBND-HCM: Hỗ trợ 100% mức đóng Bảo hiểm y tế cho con lao động tự do.",
        "q": "Trích xuất quy định hỗ trợ Bảo hiểm y tế cho con người lao động.",
        "out": {
            "verification_reasoning": "Câu hỏi hỏi về Bảo hiểm y tế cho con, khớp trực tiếp với [Context 2] (QĐ-15/2026/UBND-HCM).",
            "label": "POLICY_BHYT",
            "citation_tag": "[Context 2]",
            "evidence_excerpt": "Hỗ trợ 100% mức đóng Bảo hiểm y tế cho con lao động tự do"
        }
    },
] * 6

ds_module2 = Dataset.from_list([
    {
        "messages": [
            {"role": "system", "content": SYSTEM_JSON_PROMPT},
            {"role": "user", "content": format_rag_input(item["ctx"], item["q"])},
            {"role": "assistant", "content": json.dumps(item["out"], ensure_ascii=False)},
        ]
    }
    for item in json_samples
])
```

---

### Bước 4: Bật `DoRA` (`use_dora=True`) + `rsLoRA` (`use_rslora=True`) và Huấn luyện
```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer_m2 = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds_module2,
    peft_config=dora_cfg,
    args=SFTConfig(
        output_dir="outputs_module2",
        max_steps=35,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=5,
        fp16=True,
        report_to="none",
    ),
)
trainer_m2.train()
```

---

### Bước 5: Đo 🟢 SAU Fine-tune & Chạy hàm `verify_grounding()`
```python
trainer_m2.model.eval()
trainer_m2.model.config.use_cache = True

msgs = [
    {"role": "system", "content": SYSTEM_JSON_PROMPT},
    {"role": "user", "content": format_rag_input(sample_ctx, sample_q)},
]
text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inp = tokenizer([text], return_tensors="pt").to("cuda")
with torch.no_grad():
    out = trainer_m2.model.generate(**inp, max_new_tokens=180, do_sample=False)
raw_after = tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

print("🟢 SAU FINE-TUNE (DoRA + rsLoRA):\n", raw_after)
print("Kiểm tra Grounding:", verify_grounding(raw_after, sample_ctx))
```

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 2 (Checkpoint)
1. Tại sao `verify_grounding()` lại kiểm tra `evidence_excerpt` có nằm trong `source_context` hay không? *(Trả lời: Để phát hiện ngay nếu LLM tự viết mới hoặc bịa thêm câu chữ không nằm trong văn bản gốc)*.
2. Điểm khác nhau cốt lõi giữa `use_dora=True` và `LoRA` thường là gì? *(Trả lời: DoRA tách riêng biên độ và hướng của trọng số, chỉ dùng LoRA xoay hướng nên ổn định hơn)*.
