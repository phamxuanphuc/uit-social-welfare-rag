# Buổi 3 (Module 3): Xây dựng Bộ Dịch Pháp lý `Legal-Translator` trên `Qwen3-1.7B` — `NEFTune`, `Unsloth + Liger Kernel` & Đo Độ Lệch Công Bằng

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm chạy được cuối buổi:** **Bộ Dịch Ngôn ngữ Đời thường sang Thủ tục Pháp luật (`Legal-Translator`)** — người lao động hỏi bằng từ truyền miệng (*xe ôm, bán vé số, phụ hồ, tiền trọ*), hệ thống trả về đúng `Mã văn bản`, `Điều khoản` và `Danh sách hồ sơ`, kèm **Bảng chấm điểm Độ lệch Công bằng (`Fairness Gap`: Câu chuẩn vs. Câu đời thường) 🔴 Trước vs. 🟢 Sau**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao AI thường "thiên vị" người nói chữ nghĩa hành chính hơn người lao động nghèo?**
   - **Hình ảnh Feynman:** Văn bản luật trên `vbpl.vn` viết *"Lao động không có giao kết hợp đồng bị suy giảm khả năng lao động"*, còn người dân hỏi *"Tôi chạy xe ôm bị té xe nghỉ làm tuần nay"*. Nếu không có **Bảng ánh xạ đời thường (`ColloquialMapping`)** và **bơm nhiễu chống học vẹt (`NEFTune`)**, AI sẽ trả lời đúng `100%` cho câu hỏi chuẩn nhưng trượt câu hỏi đời thường!
2. **`NEFTune` (`neftune_noise_alpha=5`) giúp ích gì?**
   - Bơm một lớp "tiếng ồn trắng" rất nhỏ vào tọa độ từ vựng trong lúc học để `Qwen3-1.7B` không thể học vẹt từng mặt chữ mà phải nắm **ý nghĩa vùng lân cận**.
3. **Tại sao bắt đầu bật `Unsloth + Liger Kernel` từ Buổi 3?**
   - Bạn đã hiểu toàn bộ gốc rễ của `transformers + peft + trl` ở Buổi 0–2. Từ Buổi 3, văn bản luật dài hơn nên ta bật "bộ tăng áp" **`Unsloth + Liger Kernel`** để gộp phép tính `Cross-Entropy`, giúp chạy **nhanh gấp 2 lần** và **tiết kiệm 60% VRAM**!

---

## 2. Thực hành Từng Ô Nhỏ trên Google Colab (6–12 dòng/ô)

### Ô 1 (2 dòng): Cài đặt `Unsloth` & `Liger Kernel`
```bash
!pip install -q unsloth liger-kernel
```

---

### Ô 2 (9 dòng): Nạp `Qwen3-1.7B` bằng động cơ tăng tốc `FastLanguageModel`
```python
from unsloth import FastLanguageModel

model_fast, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model_fast)
print("Đã nạp Qwen3-1.7B với Unsloth 4-bit!")
```
- **Giải thích:** `FastLanguageModel.from_pretrained` thay thế cho cả cụm `BitsAndBytesConfig` + `AutoModelForCausalLM` ở Buổi 0–2 và tự động kích hoạt các nhân GPU Triton tốc độ cao.

---

### Ô 3 (11 dòng): Tạo Bộ Từ điển Đời thường $\rightarrow$ Pháp lý (`ColloquialMapping`)
```python
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

---

### Ô 4 (12 dòng): Tạo Kho Văn bản Luật & Bộ Đề Thi Song Song (Câu Chuẩn vs. Câu Đời thường)
```python
LEGAL_CTX = (
    "[1] NQ-01/2026/HĐND-HCM (Điều 4, Khoản 2): Hỗ trợ lao động phi chính thức gặp tai nạn lao động. "
    "Hồ sơ: Đơn Mẫu 01-ASXH, CCCD/CT07, Xác nhận Tổ dân phố, Giấy ra viện. Nơi nộp: UBND Phường Tân Tạo.\n"
    "[2] QĐ-15/2026/UBND-HCM (Điều 3, Khoản 1): Hỗ trợ 100% thẻ BHYT cho con lao động tự do thu nhập thấp. "
    "Hồ sơ: Tờ khai TK1-TS, Giấy khai sinh, Xác nhận thu nhập thấp. Nơi nộp: Điểm ASXH Phường Tân Tạo."
)

paired_tests = [
    {"formal": "Hồ sơ hưởng trợ cấp tai nạn lao động cho lao động phi chính thức gồm những gì?",
     "colloquial": "Tôi chạy xe ôm bị té xe nghỉ làm mấy bữa nay thì đem giấy gì lên phường xin tiền?",
     "expected_doc": "NQ-01/2026/HĐND-HCM", "expected_paper": "Mẫu 01-ASXH"},
    {"formal": "Thủ tục hỗ trợ 100% mức đóng Bảo hiểm y tế cho con của lao động tự do thu nhập thấp?",
     "colloquial": "Đi phụ hồ nghèo quá muốn xin cái thẻ bảo hiểm cho con đi khám bệnh cần nộp giấy chi?",
     "expected_doc": "QĐ-15/2026/UBND-HCM", "expected_paper": "TK1-TS"},
]
```

---

### Ô 5 (12 dòng): Hàm Chấm điểm Độ lệch Công bằng (`Fairness Gap`) & Đo 🔴 TRƯỚC Fine-tune
```python
SYS_LEGAL = "Trả lời bằng JSON gồm: verification_reasoning, doc_code, article, required_documents, submission_place."

def eval_fairness(m, use_expansion: bool) -> dict:
    scores = {"formal": 0, "colloquial": 0}
    for item in paired_tests:
        for style in ("formal", "colloquial"):
            q_in = expand_query(item[style]) if use_expansion else item[style]
            ans = ask_vivu(m, LEGAL_CTX, q_in, use_lora=True)
            hit = (item["expected_doc"] in ans) and (item["expected_paper"] in ans)
            scores[style] += int(hit)
    f_acc = scores["formal"] / len(paired_tests) * 100
    c_acc = scores["colloquial"] / len(paired_tests) * 100
    return {"formal_acc": f_acc, "colloquial_acc": c_acc, "gap": abs(f_acc - c_acc)}

before_fairness = eval_fairness(model_fast, use_expansion=False)
print("🔴 TRƯỚC FINE-TUNE & CHƯA MỞ RỘNG TỪ ĐỜI THƯỜNG:", before_fairness)
```

---

### Ô 6 (12 dòng): Đóng gói Dữ liệu Huấn luyện Pháp lý & Gắn `DoRA + rsLoRA`
```python
import json
from datasets import Dataset

train_m3 = [
    ("Tôi chạy grab bị tai nạn giao thông cần nộp giấy gì?",
     {"verification_reasoning": "Chạy grab là lao động phi chính thức gặp tai nạn, áp dụng NQ-01/2026/HĐND-HCM.",
      "doc_code": "NQ-01/2026/HĐND-HCM", "article": "Điều 4, Khoản 2",
      "required_documents": ["Đơn Mẫu 01-ASXH", "CCCD/CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"],
      "submission_place": "UBND Phường Tân Tạo"}),
    ("Bán vé số dạo muốn xin thẻ bảo hiểm y tế cho đứa con đi học cần hồ sơ gì?",
     {"verification_reasoning": "Xin thẻ BHYT cho con lao động tự do, áp dụng QĐ-15/2026/UBND-HCM.",
      "doc_code": "QĐ-15/2026/UBND-HCM", "article": "Điều 3, Khoản 1",
      "required_documents": ["Tờ khai TK1-TS", "Giấy khai sinh", "Xác nhận thu nhập thấp"],
      "submission_place": "Điểm ASXH Phường Tân Tạo"}),
] * 8

ds_m3 = Dataset.from_list([
    {"messages": [{"role": "system", "content": SYS_LEGAL},
                  {"role": "user", "content": make_prompt(LEGAL_CTX, expand_query(q))},
                  {"role": "assistant", "content": json.dumps(a, ensure_ascii=False)}]}
    for q, a in train_m3
])
```

---

### Ô 7 (12 dòng): Huấn luyện với `NEFTune (alpha=5)` & So sánh Bảng Điểm 🔴 TRƯỚC vs. 🟢 SAU
```python
from trl import SFTConfig, SFTTrainer

model_fast = FastLanguageModel.get_peft_model(
    model_fast, r=16, lora_alpha=32, use_dora=True, use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
trainer_m3 = SFTTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=ds_m3,
    args=SFTConfig(output_dir="out_m3", max_steps=30, per_device_train_batch_size=2,
                   learning_rate=2e-4, neftune_noise_alpha=5, logging_steps=5, fp16=True, report_to="none"),
)
trainer_m3.train()
FastLanguageModel.for_inference(model_fast)

after_fairness = eval_fairness(model_fast, use_expansion=True)
print("\n🏆 BẢNG SO SÁNH CÔNG BẰNG (FAIRNESS GAP <= 10%):")
print("🔴 Trước (Raw Base):       ", before_fairness)
print("🟢 Sau (Map + NEFTune SFT):", after_fairness)
```
