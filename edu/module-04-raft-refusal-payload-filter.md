# Buổi 4 (Module 4): Khiên Chống Nhiễu `RAFT-Shield` — `Scenario-Group Split 80/10/10` Thực sự

- **Tham chiếu:** `UIT-ViQuAD 2.0` (unanswerable questions) & `anaalkmim/rag-eval` (Qdrant Payload & out-of-scope eval).
- **🎯 Sản phẩm cuối buổi:** **Khiên Chống Bẫy Nhiễu RAG (`RAFT-Shield`)** tự lọc đúng 1 văn bản hợp lệ giữa 2 văn bản gây nhiễu và từ chối (`is_refusal=true`) khi thiếu căn cứ, đánh giá trên tập **Test giữ riêng chia theo `scenario_id` (0% rò rỉ dữ liệu)** và dùng đúng `SYS_RAFT`.

---

## 1. Giải thích theo Phương pháp Feynman

1. **`RAFT` (`70% Oracle + Distractors`, `30% Unanswerable`):**
   - Nếu lúc học chỉ đưa văn bản đúng, mô hình sẽ "nhắm mắt chép". Ta kẹp thêm **2 văn bản gây nhiễu** (khác địa bàn hoặc hết hạn) và xáo trộn vị trí để ép mô hình đọc kỹ `region` & `status`. Với 30% câu bẫy toàn văn bản nhiễu, mô hình phải trả lời `"is_refusal": true`.
2. **Code chia tập `Scenario-Group Split 80/10/10` thực sự (Chống Data Leakage):**
   - Mỗi tình huống thực tế có một `scenario_id` (ví dụ: `SCN_01_TAINAN`, `SCN_02_BHYT`, `SCN_03_HOCNGHE`, ...). Ta chia danh sách **`scenario_id` duy nhất** vào 3 rổ `Train / Val / Test` trước, rồi mới gom các biến thể câu hỏi của từng nhóm vào rổ tương ứng. Nhờ vậy tập `Test` **chưa từng lộ diện dưới bất kỳ biến thể nào** trong tập `Train`!

---

## 2. Thực hành 5 Bước Gọn Gàng trên Google Colab

### Bước 1: Kho Văn bản Chính sách & Hàm Hỏi Đáp Khớp Đúng `SYS_RAFT`
```python
import json, random, torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

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

SYS_RAFT = (
    "Đọc các văn bản ứng viên trong <context>, đối chiếu 'region' và nội dung để trả về đúng 1 JSON gồm: "
    "verification_reasoning, is_refusal, expired_warning, doc_code, required_documents, submission_place."
)

def ask_raft(m, chunks, region, question) -> str:
    user_txt = f"<context>\n{json.dumps(chunks, ensure_ascii=False)}\n</context>\n\nCÂU HỎI:\nĐịa bàn: {region}. Câu hỏi: {question}"
    msgs = [{"role": "system", "content": SYS_RAFT}, {"role": "user", "content": user_txt}]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([txt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=180, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
```

### Bước 2: Chia Tập Chuẩn `Scenario-Group Split 80/10/10` (0% Rò rỉ Dữ liệu)
```python
def build_raft_row(scenario_id, policy, region, question, is_unanswerable=False):
    distractors = [p for p in POLICIES if p["doc_code"] != policy["doc_code"]][:2]
    pool = distractors if is_unanswerable else [policy] + distractors
    random.shuffle(pool)
    if is_unanswerable:
        target = {"verification_reasoning": f"Không có văn bản khớp địa bàn '{region}' và nội dung.", "is_refusal": True, "expired_warning": False, "doc_code": None, "required_documents": [], "submission_place": "Liên hệ Cán bộ Một cửa."}
    else:
        target = {"verification_reasoning": f"Chọn {policy['doc_code']} khớp '{region}' (status={policy['status']}).", "is_refusal": False, "expired_warning": (policy["status"] == "expired"), "doc_code": policy["doc_code"], "required_documents": policy["required_docs"], "submission_place": policy["submission_place"]}
    msgs = [
        {"role": "system", "content": SYS_RAFT},
        {"role": "user", "content": f"<context>\n{json.dumps(pool, ensure_ascii=False)}\n</context>\n\nCÂU HỎI:\nĐịa bàn: {region}. Câu hỏi: {question}"},
        {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
    ]
    return {"scenario_id": scenario_id, "chunks": pool, "region": region, "q": question, "is_unans": is_unanswerable, "exp_code": None if is_unanswerable else policy["doc_code"], "text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)}

all_rows = [
    build_raft_row("SCN_01", POLICIES[0], "TP.HCM - Phường Tân Tạo", "Bán vé số bị tai nạn giao thông cần nộp giấy tờ gì?", False),
    build_raft_row("SCN_01", POLICIES[0], "TP.HCM - Phường Tân Tạo", "Chạy xe ôm té xe nghỉ làm thì cần hồ sơ nào?", False),
    build_raft_row("SCN_02", POLICIES[1], "TP.HCM - Phường Tân Tạo", "Xin hỗ trợ thẻ BHYT cho con lao động tự do cần gì?", False),
    build_raft_row("SCN_03", POLICIES[2], "Đà Nẵng - Quận Liên Chiểu", "Phụ hồ mất việc tại Liên Chiểu nộp hồ sơ học nghề ở đâu?", False),
    build_raft_row("SCN_04", POLICIES[3], "TP.HCM - Phường Tân Tạo", "Xin tiền thuê nhà trọ năm 2024 cần giấy tờ gì?", False),
    build_raft_row("SCN_05", POLICIES[0], "TP.HCM - Phường Tân Tạo", "Thủ tục đăng ký bằng sáng chế quốc tế?", True),
    build_raft_row("SCN_06", POLICIES[2], "TP.HCM - Phường Tân Tạo", "Đang ở Phường Tân Tạo xin nhận trợ cấp Liên Chiểu Đà Nẵng?", True),
    build_raft_row("SCN_07", POLICIES[0], "TP.HCM - Phường Tân Tạo", "Tôi chạy grab bị tai nạn ở Tân Tạo nộp đơn ở đâu?", False),
    build_raft_row("SCN_08", POLICIES[1], "TP.HCM - Phường Tân Tạo", "Làm sao để vay mua xe ô tô lãi suất 0%?", True),
    build_raft_row("SCN_09", POLICIES[2], "Đà Nẵng - Quận Liên Chiểu", "Thợ xây mất việc tại Liên Chiểu được trợ cấp học nghề không?", False),
]

unique_scns = sorted({r["scenario_id"] for r in all_rows})
random.Random(42).shuffle(unique_scns)
n_scn = len(unique_scns)
train_scns, val_scns, test_scns = set(unique_scns[:int(0.8*n_scn)]), set(unique_scns[int(0.8*n_scn):int(0.9*n_scn)]), set(unique_scns[int(0.9*n_scn):])
train_raft = Dataset.from_list([r for r in all_rows if r["scenario_id"] in train_scns])
val_raft = Dataset.from_list([r for r in all_rows if r["scenario_id"] in val_scns])
test_raft = [r for r in all_rows if r["scenario_id"] in test_scns]
print(f"Scenario Split (0% rò rỉ): Train={len(train_raft)} ({train_scns}), Val={len(val_raft)} ({val_scns}), Test={len(test_raft)} ({test_scns})")
```

### Bước 3: Đo 🔴 TRƯỚC khi học RAFT trên Tập Test Giữ Riêng
```python
def score_raft(m, suite):
    ok_cnt = 0
    for r in suite:
        raw = ask_raft(m, r["chunks"], r["region"], r["q"])
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {}
        hit = (bool(obj.get("is_refusal")) == r["is_unans"]) and (r["is_unans"] or obj.get("doc_code") == r["exp_code"])
        ok_cnt += int(hit)
    return f"{ok_cnt}/{len(suite)} ({ok_cnt/max(1, len(suite))*100:.0f}%)"

before_raft = score_raft(model_fast, test_raft)
print("🔴 ĐIỂM TRƯỚC RAFT (trên tập Test độc lập):", before_raft)
```

### Bước 4: Huấn luyện `RAFT` với `train_on_responses_only`
```python
FastLanguageModel.for_training(model_fast)
trainer_m4 = SFTTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=train_raft, dataset_text_field="text",
    args=SFTConfig(output_dir="out_m4", num_train_epochs=6, per_device_train_batch_size=2, learning_rate=2e-4, neftune_noise_alpha=5, logging_steps=4, fp16=True, report_to="none"),
)
trainer_m4 = train_on_responses_only(trainer_m4, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer_m4.train()
```

### Bước 5: Đo 🟢 SAU khi học RAFT trên Tập Test Giữ Riêng
```python
FastLanguageModel.for_inference(model_fast)
after_raft = score_raft(model_fast, test_raft)
print(f"🏆 KẾT QUẢ RAFT-SHIELD (0% DATA LEAKAGE): Trước = {before_raft} ---> Sau = {after_raft}")
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Vì sao chia ngẫu nhiên theo dòng (Random Row Split) lại sai trong bài toán Fine-tune RAG có câu hỏi diễn đạt lại?*
   - **Đáp:** Vì cùng 1 kịch bản có thể được viết lại thành nhiều câu hỏi gần giống nhau. Chia ngẫu nhiên làm câu hỏi biến thể lọt vào cả Train và Test (**Data Leakage**), khiến điểm Test cao ảo. Bắt buộc phải chia theo **Group / Scenario ID**.
2. **Hỏi:** *Trong `RAFT`, tỷ lệ mẫu `Unanswerable` (không có đáp án) nên để khoảng bao nhiêu?*
   - **Đáp:** Thường trong khoảng `20%–30%`. Quá ít mô hình vẫn quen đoán mò; quá nhiều (>50%) mô hình sẽ trở nên "lười biếng" từ chối cả những câu có đáp án.
