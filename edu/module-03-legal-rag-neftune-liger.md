# Buổi 3 (Module 3): Bộ Dịch Pháp lý `Legal-Translator` — `Unsloth + Liger Kernel`, `NEFTune` & Đo Fairness Gap

- **Tham chiếu:** `duyet/vietnamese-legal-instruct`, `fisherman611/vietnamese-legal-chatbot`, `tangbamiinh/traffic-law-assistant`.
- **🎯 Sản phẩm cuối buổi:** **Bộ Dịch Ngôn ngữ Đời thường sang Thủ tục Pháp luật (`Legal-Translator`)** — dùng `Unsloth` (import đầu tiên!) + `Liger Kernel` + `NEFTune` + bảng `ColloquialMapping`, đánh giá bằng **cùng `SYS_LEGAL` giữa train và eval** trên bộ câu hỏi song song độc lập.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Quy tắc Import Bắt buộc của `Unsloth`:** Phải viết `from unsloth import FastLanguageModel` **ở dòng đầu tiên trước khi import `transformers`, `trl`, `peft`**! Nếu import sau, Unsloth không vá được các hàm nhân ma trận tốc độ cao (và nếu đang dùng phiên Colab cũ đã import `transformers`, nên bấm `Runtime -> Restart session`).
2. **`Unsloth` với `DoRA` (`use_dora`) — Lưu ý thực tế quan trọng:** Một số phiên bản fast-patching của Unsloth không bật chế độ tối ưu riêng khi `use_dora=True` (sẽ tự động rơi về nhánh PEFT thường). Do đó khi học thực chiến: dùng `PEFT` chuẩn khi cần `DoRA` (như Buổi 2), còn khi dùng `Unsloth + Liger Kernel` trên văn bản dài (Buổi 3–4), ta cấu hình `use_rslora=True, use_gradient_checkpointing="unsloth"` và cài `assistant_only_loss` / `train_on_responses_only`.
3. **`NEFTune` (`neftune_noise_alpha=5`):** Bơm nhiễu nhẹ vào vector từ vựng lúc train để mô hình không học thuộc mặt chữ, giúp hiểu tốt các câu hỏi đời thường mới lạ.

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab

### Ô 1 (2 dòng): Cài đặt `Unsloth` & `Liger Kernel` (Bấm Restart Runtime trước nếu cần)
```bash
!pip install -q "unsloth" "liger-kernel" "trl==0.16.1" "datasets==3.5.0"
```

### Ô 2 (10 dòng): Import `unsloth` ĐẦU TIÊN & Nạp `Qwen3-1.7B` Tăng tốc
```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import json, torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-1.7B"
model_fast, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_ID, max_seq_length=2048, dtype=None, load_in_4bit=True,
)
print("Đã nạp Qwen3-1.7B qua Unsloth!")
```

### Ô 3 (11 dòng): Hàm `ask_legal()` & Bộ Mở rộng Từ ngữ Đời thường (`ColloquialMapping`)
```python
SYS_LEGAL = "Chỉ trả về 1 JSON gồm: verification_reasoning, doc_code, article, required_documents, submission_place."

COLLOQUIAL_MAP = {
    "xe ôm": "lao động phi chính thức vận tải xe hai bánh",
    "chạy grab": "lao động phi chính thức lái xe công nghệ",
    "bán vé số": "lao động tự do bán vé số dạo",
    "phụ hồ": "lao động tự do ngành xây dựng",
    "tiền trọ": "hỗ trợ tiền thuê nhà trọ",
    "sổ hộ khẩu": "giấy xác nhận thông tin cư trú CT07",
}

def expand_query(q: str) -> str:
    matched = [v for k, v in COLLOQUIAL_MAP.items() if k in q.lower()]
    return f"{q} [Thuật ngữ pháp lý: {', '.join(matched)}]" if matched else q
```

### Ô 4 (11 dòng): Hàm Hỏi Đồng bộ Đúng `SYS_LEGAL` (Fix hoàn toàn lỗi lệch Prompt!)
```python
def ask_legal(m, context: str, question: str) -> str:
    user_txt = f"<context>\n{context}\n</context>\n\nCÂU HỎI:\n{question}"
    msgs = [{"role": "system", "content": SYS_LEGAL}, {"role": "user", "content": user_txt}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([txt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=180, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
```

### Ô 5 (14 dòng): Văn bản Luật & Bộ Câu hỏi Song song Giữ Riêng (Đo Fairness Gap)
```python
LEGAL_CTX = (
    "[1] NQ-01/2026/HĐND-HCM (Điều 4, Khoản 2): Hỗ trợ lao động phi chính thức gặp tai nạn lao động. "
    "Hồ sơ: Đơn Mẫu 01-ASXH, CCCD/CT07, Xác nhận Tổ dân phố, Giấy ra viện. Nơi nộp: UBND Phường Tân Tạo.\n"
    "[2] QĐ-15/2026/UBND-HCM (Điều 3, Khoản 1): Hỗ trợ 100% thẻ BHYT cho con lao động tự do thu nhập thấp. "
    "Hồ sơ: Tờ khai TK1-TS, Giấy khai sinh, Xác nhận thu nhập thấp. Nơi nộp: Điểm ASXH Phường Tân Tạo."
)

paired_eval = [
    {"formal": "Hồ sơ hưởng trợ cấp tai nạn lao động cho lao động phi chính thức tại Phường Tân Tạo?",
     "colloquial": "Tôi chạy xe ôm ở Phường Tân Tạo bị té xe nghỉ làm mấy bữa thì nộp giấy chi?",
     "expected_doc": "NQ-01/2026/HĐND-HCM"},
    {"formal": "Thủ tục hỗ trợ 100% mức đóng BHYT cho con lao động tự do thu nhập thấp?",
     "colloquial": "Đi phụ hồ nghèo quá muốn xin thẻ bảo hiểm cho đứa nhỏ đi khám bệnh cần nộp gì?",
     "expected_doc": "QĐ-15/2026/UBND-HCM"},
]

def eval_fairness(m, use_expansion: bool) -> dict:
    scores = {"formal": 0, "colloquial": 0}
    for item in paired_eval:
        for style in ("formal", "colloquial"):
            q = expand_query(item[style]) if use_expansion else item[style]
            ans = ask_legal(m, LEGAL_CTX, q)
            try:
                ok = json.loads(ans).get("doc_code") == item["expected_doc"]
            except Exception:
                ok = item["expected_doc"] in ans
            scores[style] += int(ok)
    f_acc, c_acc = scores["formal"] / len(paired_eval) * 100, scores["colloquial"] / len(paired_eval) * 100
    return {"formal_acc": f_acc, "colloquial_acc": c_acc, "gap": abs(f_acc - c_acc)}

FastLanguageModel.for_inference(model_fast)
before_fairness = eval_fairness(model_fast, use_expansion=False)
print("🔴 TRƯỚC FINE-TUNE (Raw Base):", before_fairness)
```

### Ô 6 (15 dòng): Tạo Tập Train Đa dạng (Không dùng `] * N`), Gắn `train_on_responses_only` & `NEFTune`
```python
train_legal_pairs = [
    ("Chạy grab ở Phường Tân Tạo bị tai nạn cần chuẩn bị hồ sơ nào?",
     {"verification_reasoning": "Chạy grab là lao động phi chính thức gặp tai nạn theo NQ-01/2026/HĐND-HCM.", "doc_code": "NQ-01/2026/HĐND-HCM", "article": "Điều 4, Khoản 2", "required_documents": ["Đơn Mẫu 01-ASXH", "CCCD/CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"], "submission_place": "UBND Phường Tân Tạo"}),
    ("Bán vé số dạo muốn làm BHYT cho con đi học cần giấy gì?",
     {"verification_reasoning": "Làm thẻ BHYT cho con lao động tự do áp dụng QĐ-15/2026/UBND-HCM.", "doc_code": "QĐ-15/2026/UBND-HCM", "article": "Điều 3, Khoản 1", "required_documents": ["Tờ khai TK1-TS", "Giấy khai sinh", "Xác nhận thu nhập thấp"], "submission_place": "Điểm ASXH Phường Tân Tạo"}),
    ("Thợ hồ gặp rủi ro khi thi công cần nộp đơn ở đâu?",
     {"verification_reasoning": "Thợ hồ là lao động tự do gặp tai nạn lao động theo NQ-01/2026/HĐND-HCM.", "doc_code": "NQ-01/2026/HĐND-HCM", "article": "Điều 4, Khoản 2", "required_documents": ["Đơn Mẫu 01-ASXH", "CCCD/CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"], "submission_place": "UBND Phường Tân Tạo"}),
    ("Bán hàng rong thu nhập thấp xin miễn giảm phí bảo hiểm y tế cho con tại đâu?",
     {"verification_reasoning": "Bán hàng rong thu nhập thấp xin BHYT cho con theo QĐ-15/2026/UBND-HCM.", "doc_code": "QĐ-15/2026/UBND-HCM", "article": "Điều 3, Khoản 1", "required_documents": ["Tờ khai TK1-TS", "Giấy khai sinh", "Xác nhận thu nhập thấp"], "submission_place": "Điểm ASXH Phường Tân Tạo"}),
]

def to_text(row):
    msgs = [{"role": "system", "content": SYS_LEGAL}, {"role": "user", "content": f"<context>\n{LEGAL_CTX}\n</context>\n\nCÂU HỎI:\n{expand_query(row['q'])}"}, {"role": "assistant", "content": json.dumps(row["a"], ensure_ascii=False)}]
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)}

ds_m3 = Dataset.from_list([{"q": q, "a": a} for q, a in train_legal_pairs]).map(to_text)
model_fast = FastLanguageModel.get_peft_model(model_fast, r=16, lora_alpha=32, use_rslora=True, use_gradient_checkpointing="unsloth", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
```

### Ô 7 (10 dòng): Huấn luyện với `train_on_responses_only` Thật & So sánh Fairness Gap
```python
trainer_m3 = SFTTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=ds_m3, dataset_text_field="text",
    args=SFTConfig(output_dir="out_m3", num_train_epochs=6, per_device_train_batch_size=2, learning_rate=2e-4, neftune_noise_alpha=5, logging_steps=3, fp16=True, report_to="none"),
)
trainer_m3 = train_on_responses_only(trainer_m3, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer_m3.train()

FastLanguageModel.for_inference(model_fast)
after_fairness = eval_fairness(model_fast, use_expansion=True)
print("🔴 Trước:", before_fairness, "\n🟢 Sau (Map + NEFTune + Masking):", after_fairness)
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Tại sao `Liger Kernel` giảm được tới 60% VRAM khi train LLM?*
   - **Đáp:** Vì tầng `lm_head` nhân ra tensor `logits` kích thước `(batch * seq_len, vocab_size)` rất lớn (~151k từ). `Liger Kernel` tính gộp linear + `CrossEntropy` theo từng chunk nhỏ trên SRAM của GPU nên không phải cấp phát ma trận `logits` khổng lồ trên VRAM.
2. **Hỏi:** *Tại sao `unsloth` bắt buộc phải được import đầu tiên trước `transformers` và `trl`?*
   - **Đáp:** Vì Unsloth sử dụng kỹ thuật monkey-patching để thay thế các lớp attention/RMSNorm/RoPE mặc định ngay lúc thư viện gốc được nạp.
