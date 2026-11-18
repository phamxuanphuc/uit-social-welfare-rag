# Buổi 4 (Module 4): Đọc hiểu RAG Chống Ảo giác & Lọc Payload — Tham khảo `UIT-ViQuAD 2.0` & `anaalkmim/rag-eval` (`RAFT`)

- **Dự án tham chiếu:** Bộ dữ liệu `UIT-ViQuAD 2.0` (nhóm câu hỏi *unanswerable*) và `anaalkmim/rag-eval` (tách bạch metadata vào Qdrant Payload & đánh giá tỷ lệ chống ảo giác trên câu hỏi không có đáp án).
- **Mục tiêu đầu ra:** Chia tập dữ liệu theo nhóm kịch bản gốc (`Scenario-Group Split 80/10/10`) chống *Data Leakage*, đóng gói dữ liệu theo phương pháp **`RAFT` (`70% Oracle+Distractors` / `30% Unanswerable Refusal`)**, và rèn phản xạ từ chối an toàn (`is_refusal: true`) khi lệch địa bàn hoặc ngoài phạm vi.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao RAG thông thường vẫn hay bịa đặt khi gặp đoạn văn "trông quen quen"?**
   - **Hình ảnh Feynman:** Nếu lúc ôn thi bạn chỉ đưa cho học sinh đúng 1 tờ tài liệu có chứa đáp án, học sinh sẽ hình thành thói quen xấu: *"Cứ thấy con số hay địa chỉ nào trong tờ giấy là chép đại vào bài thi!"*.
   - Đến khi đi thi thật, công cụ tìm kiếm (Qdrant) trả về **Top-3 văn bản**: trong đó có 1 văn bản của *Quận Liên Chiểu (Đà Nẵng)* và 1 văn bản của *Phường Tân Tạo (TP.HCM)* nhưng đã hết hạn từ năm 2024. Học sinh "học vẹt" sẽ râu ông nọ cắm cằm bà kia ngay!
2. **Giải pháp `RAFT` (`Retrieval-Augmented Fine-Tuning`) + `UIT-ViQuAD 2.0 Unanswerable`:**
   - **70% đề ôn tập (`Answerable`):** Cố tình kẹp **1 văn bản đúng (`Oracle`)** chung với **2 văn bản gây nhiễu (`Distractors`)** và đảo thứ tự ngẫu nhiên. Học sinh buộc phải đọc dòng `region` và `status` của từng văn bản trong `verification_reasoning` để tự nhặt ra đúng tờ tài liệu cần dùng.
   - **30% đề ôn tập (`Unanswerable / Refusal` — học từ `UIT-ViQuAD 2.0` & `rag-eval`):** Đưa cả **3 tờ tài liệu đều không liên quan hoặc khác địa bàn**! Nếu học sinh cố bịa câu trả lời sẽ bị 0 điểm; chỉ khi trả lời `"is_refusal": true` mới được điểm tối đa.
3. **Tại sao phải chia tập theo `Scenario-Group Split (80/10/10)` thay vì chia ngẫu nhiên từng dòng?**
   - Nếu bạn lấy 1 câu hỏi gốc viết thành 5 câu đồng nghĩa rồi bốc thăm ngẫu nhiên (`random split`), sẽ có 4 câu rơi vào tập `Train` và 1 câu y hệt rơi vào tập `Test` $\rightarrow$ Đây là lỗi **Lộ đề (`Data Leakage` — Buổi 02 & 03)**.
   - `Scenario-Group Split` gom toàn bộ biến thể của cùng một tình huống vào chung một rổ: tình huống nào đã vào `Test` thì tuyệt đối không xuất hiện dưới bất kỳ hình thức nào trong `Train`!

---

## 2. Thực hành 5 Bước trên Google Colab

### Bước 1: Tạo Kho Văn bản Giả lập có Metadata riêng biệt (kiểu Qdrant Payload của `rag-eval`)
```python
import json
import random

POLICIES = [
    {
        "doc_code": "NQ-01/2026/HĐND-HCM",
        "region": "TP.HCM - Phường Tân Tạo",
        "status": "active",
        "version": "v2.1",
        "effective_date": "2026-01-15",
        "required_docs": ["Đơn Mẫu 01-ASXH", "CCCD hoặc CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"],
        "submission_place": "Bộ phận Một cửa UBND Phường Tân Tạo",
        "content": "Hỗ trợ tai nạn lao động cho lao động phi chính thức (xe ôm, bán vé số, phụ hồ) từ 2.000.000đ đến 4.500.000đ."
    },
    {
        "doc_code": "QĐ-15/2026/UBND-HCM",
        "region": "TP.HCM - Phường Tân Tạo",
        "status": "active",
        "version": "v1.0",
        "effective_date": "2026-03-01",
        "required_docs": ["Tờ khai Mẫu TK1-TS", "Giấy khai sinh của con", "Xác nhận thu nhập thấp"],
        "submission_place": "Điểm tiếp nhận ASXH Phường Tân Tạo",
        "content": "Hỗ trợ 100% thẻ BHYT và giảm 50% học phí cho con của lao động tự do thu nhập thấp."
    },
    {
        "doc_code": "NQ-08/2026/HĐND-ĐN",
        "region": "Đà Nẵng - Quận Liên Chiểu",
        "status": "active",
        "version": "v1.2",
        "effective_date": "2026-02-01",
        "required_docs": ["Phiếu Mẫu ĐK-02", "Bản sao CCCD", "Xác nhận tạm trú 6 tháng tại Liên Chiểu"],
        "submission_place": "Trung tâm Dịch vụ Việc làm Quận Liên Chiểu, Đà Nẵng",
        "content": "Hỗ trợ học nghề 3.000.000đ và trợ cấp mất việc 1.500.000đ/tháng tại Quận Liên Chiểu."
    },
    {
        "doc_code": "QĐ-09/2024/UBND-HCM",
        "region": "TP.HCM - Phường Tân Tạo",
        "status": "expired",
        "version": "v1.0",
        "effective_date": "2024-01-01",
        "required_docs": ["Đơn Mẫu 02-NT", "Hợp đồng thuê trọ", "Giấy CT07"],
        "submission_place": "UBND Phường Tân Tạo (Đã hết hạn 31/12/2025)",
        "content": "Hỗ trợ tiền thuê nhà trọ 500.000đ/tháng giai đoạn 2024-2025. Văn bản đã hết hiệu lực."
    },
]
```

---

### Bước 2: Đo 🔴 TRƯỚC khi học RAFT (Thử bẫy mô hình bằng câu hỏi không có đáp án / lệch địa bàn)
```python
trap_ctx = json.dumps([POLICIES[2], POLICIES[3]], ensure_ascii=False, indent=2)
trap_q = "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Thủ tục xin cấp vốn mua xe máy mới cho tài xế xe ôm?"

print("🔴 TRƯỚC KHI HỌC RAFT (Bẫy câu hỏi ngoài phạm vi):")
print(ask_rag(model_fast, trap_ctx, trap_q, use_lora=True))
```

---

### Bước 3: Hàm đóng gói `RAFT` (70% Oracle+Distractors / 30% Unanswerable) & Chia `80/10/10`
```python
from datasets import Dataset

def make_raft_sample(target_policy, question, region, is_unanswerable=False):
    distractors = [p for p in POLICIES if p["doc_code"] != target_policy["doc_code"]][:2]
    chunks = distractors if is_unanswerable else [target_policy] + distractors
    random.shuffle(chunks)
    if is_unanswerable:
        ans = {
            "verification_reasoning": f"Ngữ cảnh không có văn bản nào tại '{region}' quy định nội dung câu hỏi này.",
            "is_refusal": True,
            "expired_warning": False,
            "doc_code": None,
            "required_documents": [],
            "submission_place": "Vui lòng liên hệ Cán bộ Một cửa để được hướng dẫn trực tiếp."
        }
    else:
        is_exp = target_policy["status"] == "expired"
        ans = {
            "verification_reasoning": f"Khớp văn bản {target_policy['doc_code']} tại {region} (status={target_policy['status']}).",
            "is_refusal": False,
            "expired_warning": is_exp,
            "doc_code": target_policy["doc_code"],
            "required_documents": target_policy["required_docs"],
            "submission_place": target_policy["submission_place"]
        }
    ctx_str = json.dumps(chunks, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": "Chỉ trả lời bằng JSON dựa trên ngữ cảnh. Từ chối (is_refusal=true) nếu không có văn bản khớp địa bàn và nội dung."},
            {"role": "user", "content": format_rag_input(ctx_str, f"Địa bàn: {region}. Câu hỏi: {question}")},
            {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)},
        ]
    }

scenarios = [
    (POLICIES[0], "Tôi chạy xe ôm bị tai nạn cần giấy gì?", "TP.HCM - Phường Tân Tạo", False),
    (POLICIES[1], "Xin thẻ BHYT cho con lao động tự do cần hồ sơ gì?", "TP.HCM - Phường Tân Tạo", False),
    (POLICIES[2], "Phụ hồ mất việc ở Liên Chiểu đăng ký học nghề ở đâu?", "Đà Nẵng - Quận Liên Chiểu", False),
    (POLICIES[3], "Xin tiền hỗ trợ thuê nhà trọ năm 2024 ở Phường Tân Tạo?", "TP.HCM - Phường Tân Tạo", False),
    (POLICIES[0], "Thủ tục vay mua căn hộ chung cư thương mại?", "TP.HCM - Phường Tân Tạo", True),
    (POLICIES[2], "Tôi ở Phường Tân Tạo muốn nhận trợ cấp học nghề của Đà Nẵng?", "TP.HCM - Phường Tân Tạo", True),
] * 5

n = len(scenarios)
train_raft = Dataset.from_list([make_raft_sample(*s) for s in scenarios[:int(0.8 * n)]])
val_raft = Dataset.from_list([make_raft_sample(*s) for s in scenarios[int(0.8 * n):int(0.9 * n)]])
test_raft = Dataset.from_list([make_raft_sample(*s) for s in scenarios[int(0.9 * n):]])
print(f"Split 80/10/10 -> Train: {len(train_raft)}, Val: {len(val_raft)}, Test: {len(test_raft)}")
```

---

### Bước 4: Huấn luyện SFT trên dữ liệu `RAFT`
```python
from trl import SFTConfig, SFTTrainer

trainer_m4 = SFTTrainer(
    model=model_fast,
    processing_class=tokenizer,
    train_dataset=train_raft,
    eval_dataset=val_raft,
    args=SFTConfig(
        output_dir="outputs_module4",
        max_steps=35,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        neftune_noise_alpha=5,
        logging_steps=5,
        fp16=True,
        report_to="none",
    ),
)
trainer_m4.train()
```

---

### Bước 5: Kiểm chứng 🟢 SAU khi học RAFT (Đo `Refusal Accuracy` trên câu bẫy)
```python
FastLanguageModel.for_inference(model_fast)
print("🟢 SAU KHI HỌC RAFT (Bẫy câu hỏi ngoài phạm vi):")
print(ask_rag(model_fast, trap_ctx, trap_q, use_lora=True))
```

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 4 (Checkpoint)
1. Tại sao trong `make_raft_sample()` phải gọi `random.shuffle(chunks)`? *(Trả lời: Để mô hình không học vẹt kiểu "cứ lấy đoạn văn đầu tiên trong danh sách", buộc nó phải đọc metadata `region` và nội dung của cả 3 đoạn)*.
