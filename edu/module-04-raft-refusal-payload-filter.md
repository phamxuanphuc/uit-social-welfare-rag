# Buổi 4 (Module 4): Xây dựng Khiên Chống Nhiễu `RAFT-Shield` trên `Qwen3-1.7B` — Bắt đầu Gộp Code Tinh Gọn

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm chạy được cuối buổi:** **Khiên Chống Bẫy Ảo Giác RAG (`RAFT-Shield`)** — tự động tìm đúng 1 văn bản hợp lệ khi bị trộn lẫn với 2 văn bản gây nhiễu (khác địa bàn / hết hạn) và tự động từ chối (`is_refusal: true`) khi câu hỏi nằm ngoài tài liệu, kèm **Bảng đo điểm Kháng nhiễu & Từ chối 🔴 Trước vs. 🟢 Sau**.
- **Chuyển giai đoạn tốc độ:** Vì bạn đã thành thạo thao tác tạo `Dataset` và gọi `SFTTrainer` ở Buổi 0–3, từ Buổi 4 chúng ta bắt đầu **gộp các lệnh quen thuộc lại thành 5 bước gọn gàng** để tập trung sâu vào thuật toán **`RAFT` (`Retrieval-Augmented Fine-Tuning`)**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao cần `RAFT` thay vì RAG thông thường?**
   - **Hình ảnh Feynman:** Nếu lúc ôn bài chỉ đưa đúng 1 trang tài liệu trúng tủ, `Qwen3-1.7B` sẽ lười biếng: *"Cứ thấy chữ gì trong `<context>` là nhặt vào câu trả lời!"*. Đến khi chạy thật, bộ tìm kiếm trả về **Top-3 văn bản** (1 của *Phường Tân Tạo*, 1 của *Đà Nẵng*, 1 đã *hết hạn năm 2024*), mô hình sẽ chép nhầm quy định của Đà Nẵng cho người dân TP.HCM!
   - **`RAFT` giải quyết bằng cách:**
     - **70% mẫu có đáp án (`Answerable`):** Trộn `1 văn bản đúng (Oracle)` + `2 văn bản gây nhiễu (Distractors)` và đảo thứ tự ngẫu nhiên (`random.shuffle`).
     - **30% mẫu bẫy không có đáp án (`Unanswerable` — học từ `UIT-ViQuAD 2.0` & `rag-eval`):** Đưa cả `3 văn bản đều là nhiễu` để bắt buộc mô hình trả về `"is_refusal": true`.
2. **Chia tập `Scenario-Group Split (80/10/10)` chống Lộ đề (`Data Leakage`):**
   - Mọi câu hỏi trong tập kiểm thử (`Test`) phải thuộc nhóm tình huống độc lập với tập học (`Train`), đảm bảo điểm số đánh giá là năng lực thật sự chứ không phải học tủ.

---

## 2. Thực hành 5 Bước Gọn Gàng trên Google Colab

### Bước 1: Tạo Kho 4 Văn bản Chính sách có Metadata Địa bàn & Hiệu lực (`POLICIES`)
```python
import json
import random

POLICIES = [
    {"doc_code": "NQ-01/2026/HĐND-HCM", "region": "TP.HCM - Phường Tân Tạo", "status": "active",
     "required_docs": ["Đơn Mẫu 01-ASXH", "CCCD/CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"],
     "submission_place": "Bộ phận Một cửa UBND Phường Tân Tạo",
     "content": "Hỗ trợ tai nạn lao động cho lao động phi chính thức (xe ôm, bán vé số, phụ hồ) từ 2.000.000đ đến 4.500.000đ."},
    {"doc_code": "QĐ-15/2026/UBND-HCM", "region": "TP.HCM - Phường Tân Tạo", "status": "active",
     "required_docs": ["Tờ khai Mẫu TK1-TS", "Giấy khai sinh của con", "Xác nhận thu nhập thấp"],
     "submission_place": "Điểm tiếp nhận ASXH Phường Tân Tạo",
     "content": "Hỗ trợ 100% thẻ BHYT và miễn giảm 50% học phí cho con lao động tự do thu nhập thấp."},
    {"doc_code": "NQ-08/2026/HĐND-ĐN", "region": "Đà Nẵng - Quận Liên Chiểu", "status": "active",
     "required_docs": ["Phiếu Mẫu ĐK-02", "Bản sao CCCD", "Xác nhận tạm trú 6 tháng tại Liên Chiểu"],
     "submission_place": "Trung tâm Dịch vụ Việc làm Quận Liên Chiểu, Đà Nẵng",
     "content": "Hỗ trợ đào tạo nghề 3.000.000đ và trợ cấp mất việc tạm thời 1.500.000đ/tháng tại Quận Liên Chiểu."},
    {"doc_code": "QĐ-09/2024/UBND-HCM", "region": "TP.HCM - Phường Tân Tạo", "status": "expired",
     "required_docs": ["Đơn Mẫu 02-NT", "Hợp đồng thuê trọ", "Giấy CT07"],
     "submission_place": "UBND Phường Tân Tạo (Đã hết hạn 31/12/2025)",
     "content": "Hỗ trợ tiền thuê nhà trọ 500.000đ/tháng giai đoạn 2024-2025. Văn bản đã hết hiệu lực."},
]
```

---

### Bước 2: Thiết lập Bộ Đề Kiểm tra Bẫy Nhiễu & Đo 🔴 TRƯỚC khi học RAFT
```python
SYS_RAFT = (
    "Đọc 3 văn bản ứng viên trong <context>, đối chiếu 'region' và nội dung để trả về đúng 1 JSON gồm: "
    "verification_reasoning, is_refusal, expired_warning, doc_code, required_documents, submission_place."
)

raft_eval_suite = [
    # Ca 1: Hỏi Phường Tân Tạo nhưng trộn lẫn văn bản Đà Nẵng và văn bản hết hạn
    ([POLICIES[2], POLICIES[0], POLICIES[3]], "TP.HCM - Phường Tân Tạo",
     "Tôi chạy xe ôm bị tai nạn cần chuẩn bị giấy tờ gì?", False, "NQ-01/2026/HĐND-HCM"),
    # Ca 2: Hỏi ở Phường Tân Tạo nhưng trong <context> chỉ có văn bản của Đà Nẵng -> Phải từ chối!
    ([POLICIES[2]], "TP.HCM - Phường Tân Tạo",
     "Tôi ở Phường Tân Tạo muốn đăng ký học nghề miễn phí 3 triệu của Đà Nẵng được không?", True, None),
    # Ca 3: Câu hỏi ngoài phạm vi an sinh (mua vé máy bay) -> Phải từ chối!
    ([POLICIES[0], POLICIES[1]], "TP.HCM - Phường Tân Tạo",
     "Thủ tục đổi vé máy bay khứ hồi quốc tế cần giấy tờ gì?", True, None),
]

def score_raft_shield(m) -> dict:
    correct = 0
    details = []
    for chunks, reg, q, exp_refusal, exp_code in raft_eval_suite:
        ctx = json.dumps(chunks, ensure_ascii=False)
        raw = ask_vivu(m, ctx, f"Địa bàn: {reg}. Câu hỏi: {q}", use_lora=True)
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {}
        ok = (bool(obj.get("is_refusal")) == exp_refusal) and (exp_refusal or obj.get("doc_code") == exp_code)
        correct += int(ok)
        details.append((q, ok, raw[:130]))
    return {"score": f"{correct}/{len(raft_eval_suite)} ({correct/len(raft_eval_suite)*100:.0f}%)", "details": details}

before_raft = score_raft_shield(model_fast)
print("🔴 ĐIỂM TRƯỚC KHI HỌC RAFT:", before_raft["score"])
```

---

### Bước 3: Hàm Tự Động Sinh Dữ Liệu `RAFT` (`70% Oracle+Distractors` / `30% Unanswerable`)
```python
from datasets import Dataset

def build_raft_row(policy, region, question, is_unanswerable=False):
    distractors = [p for p in POLICIES if p["doc_code"] != policy["doc_code"]][:2]
    pool = distractors if is_unanswerable else [policy] + distractors
    random.shuffle(pool)
    if is_unanswerable:
        target = {
            "verification_reasoning": f"Không có văn bản nào trong ngữ cảnh khớp địa bàn '{region}' và nội dung câu hỏi.",
            "is_refusal": True, "expired_warning": False, "doc_code": None,
            "required_documents": [], "submission_place": "Vui lòng liên hệ Cán bộ Một cửa UBND phường."
        }
    else:
        target = {
            "verification_reasoning": f"Lọc bỏ văn bản nhiễu, chọn {policy['doc_code']} khớp địa bàn '{region}' (status={policy['status']}).",
            "is_refusal": False, "expired_warning": (policy["status"] == "expired"),
            "doc_code": policy["doc_code"], "required_documents": policy["required_docs"],
            "submission_place": policy["submission_place"]
        }
    return {
        "messages": [
            {"role": "system", "content": SYS_RAFT},
            {"role": "user", "content": make_prompt(json.dumps(pool, ensure_ascii=False), f"Địa bàn: {region}. Câu hỏi: {question}")},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }

raft_Pool = [
    (POLICIES[0], "TP.HCM - Phường Tân Tạo", "Bán vé số bị tai nạn giao thông cần nộp giấy tờ gì?", False),
    (POLICIES[1], "TP.HCM - Phường Tân Tạo", "Xin hỗ trợ thẻ BHYT cho con lao động tự do cần gì?", False),
    (POLICIES[2], "Đà Nẵng - Quận Liên Chiểu", "Phụ hồ mất việc tại Liên Chiểu nộp hồ sơ học nghề ở đâu?", False),
    (POLICIES[3], "TP.HCM - Phường Tân Tạo", "Xin hỗ trợ tiền thuê nhà trọ năm 2024 cần giấy tờ gì?", False),
    (POLICIES[0], "TP.HCM - Phường Tân Tạo", "Thủ tục đăng ký nhãn hiệu thương mại độc quyền?", True),
    (POLICIES[2], "TP.HCM - Phường Tân Tạo", "Đang ở Phường Tân Tạo xin nhận trợ cấp Liên Chiểu Đà Nẵng?", True),
] * 5

train_raft_ds = Dataset.from_list([build_raft_row(*r) for r in raft_Pool])
```

---

### Bước 4: Huấn luyện `Qwen3-1.7B` với Bộ Dữ liệu `RAFT` (35 bước)
```python
from trl import SFTConfig, SFTTrainer

FastLanguageModel.for_training(model_fast)
trainer_m4 = SFTTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=train_raft_ds,
    args=SFTConfig(output_dir="out_m4", max_steps=35, per_device_train_batch_size=2,
                   learning_rate=2e-4, neftune_noise_alpha=5, logging_steps=7, fp16=True, report_to="none"),
)
trainer_m4.train()
FastLanguageModel.for_inference(model_fast)
```

---

### Bước 5: 📊 Chấm điểm `RAFT-Shield` 🔴 TRƯỚC vs. 🟢 SAU Fine-tune
```python
after_raft = score_raft_shield(model_fast)
print(f"\n🏆 KẾT QUẢ KHIÊN CHỐNG NHIỄU RAFT-SHIELD: Trước = {before_raft['score']} ---> Sau = {after_raft['score']}")
for (q, ok_b, raw_b), (_, ok_a, raw_a) in zip(before_raft["details"], after_raft["details"]):
    print(f"\n❓ {q}\n🔴 Trước ({'ĐẠT' if ok_b else 'TRƯỢT'}): {raw_b}\n🟢 Sau   ({'ĐẠT' if ok_a else 'TRƯỢT'}): {raw_a}")
```
