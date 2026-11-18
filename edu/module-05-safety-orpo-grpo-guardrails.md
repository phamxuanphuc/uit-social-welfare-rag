# Buổi 5 (Module 5): Y tế & An toàn Sống còn — `ORPO`, `GRPO` (Verifiable Rewards) & 4 Tầng Guardrail

- **Dự án tham chiếu:** `ViMedQA` (`pnnbao-ump`, `vilm`, `5CD-AI`), `AcadifySolution/llm-security-proxy`, `tkarim45/llm-guardrails`, `lamhotsiagian/guardrail-system-rag`.
- **Mục tiêu đầu ra:** Hiểu và thực hành căn chỉnh Giai đoạn 2 bằng **`ORPO` (`lr=5e-6`)** và **`GRPO` (`lr=1e-6`, `num_generations=4`)** với 3 hàm thưởng bất đối xứng (phạt nặng lỗi bỏ sót văn bản hết hạn — *False Negative*), kết hợp luồng kiểm soát 4 tầng Guardrail (PII Masking, Cross-Region Conflict, Hard Safety Override, Audit Trail).

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao học xong SFT (Buổi 1–4) vẫn cần thêm `ORPO` và `GRPO`?**
   - **Hình ảnh Feynman:** SFT giống như cho học sinh chép văn mẫu: học sinh biết cách viết câu đúng nhưng **chưa từng bị phạt khi phạm lỗi nguy hiểm** (như dùng văn bản đã hết hiệu lực `expired` mà quên bật cảnh báo đỏ).
   - **`ORPO` (`Odds Ratio Preference Optimization`):** Đặt 2 bài làm cạnh nhau trên bàn: 1 bài **Đúng chuẩn (`Chosen`)** và 1 bài **Mắc lỗi nguy hiểm (`Rejected`)** (ví dụ: quên bật `"expired_warning": true` hoặc bịa thêm giấy tờ). Điểm hay nhất là `ORPO` tính trực tiếp tỷ số cược (*Odds Ratio*) ngay trên mô hình đang học nên **không tốn VRAM nạp thêm mô hình thứ 2 (`Reference Model`)** như DPO!
   - **`GRPO` (`Group Relative Policy Optimization` — thuật toán huấn luyện DeepSeek-R1):** Bỏ văn mẫu đi, bắt học sinh **tự làm 4 bài khác nhau cho cùng 1 câu hỏi**, sau đó dùng **3 hàm code Python (`Verifiable Rewards`)** chấm điểm trực tiếp từng bài: bài nào quên bật cảnh báo hết hạn (`False Negative`) bị trừ điểm gấp đôi (`-0.6`)!
2. **Tại sao tốc độ học (`learning_rate`) phải giảm dần `2e-4 -> 5e-6 -> 1e-6`?**
   - Ở SFT (`2e-4`), mô hình đang xây khung kiến thức mới. Sang `ORPO` (`5e-6`) và `GRPO` (`1e-6`), mô hình đã đứng rất sát đích rồi; chỉ cần chỉnh cực nhẹ để uốn nắn hành vi an toàn mà không làm hỏng khả năng viết JSON đã học.
3. **4 Tầng Guardrail bảo vệ bên ngoài mô hình (Học từ `llm-security-proxy` & `guardrail-system-rag`):**
   - Không bao giờ giao phó 100% sự an toàn cho xác suất của AI! Hệ thống bọc 4 lớp rào chắn bằng Python:
     1. **Tầng 1 (`Input Guard` & `PII Redactor`):** Che số CCCD (9–12 số) và SĐT (kể cả khi viết cách `090 123 4567`) thành `[ĐÃ_CHE_ĐỊNH_DANH]`; chặn ngay nếu thiếu `region`.
     2. **Tầng 2 (`Topic / Cross-Region Boundary`):** Báo lỗi lệch địa bàn ngay nếu chọn dropdown *Phường Tân Tạo* nhưng trong câu hỏi lại hỏi về *Quận Liên Chiểu*.
     3. **Tầng 3 (`Hard Safety Override & Human-in-the-loop Gate`):** Hậu kiểm bằng code Python — hễ tài liệu trích dẫn có `status == "expired"` trong DB gốc là cưỡng bức bật `expired_warning = True` và hiện số điện thoại Cán bộ Một cửa.
     4. **Tầng 4 (`Audit Trail`):** Ghi vết toàn bộ vào SQLite/Postgres (không lưu PII thô, ẩn `***REDACTED***` mọi API key).

---

## 2. Thực hành 5 Bước trên Google Colab

### Bước 1: Viết Lớp Rào chắn PII Masking & Kiểm tra Lệch Địa bàn (Tầng 1 & 2)
```python
import re

PHONE_RE = re.compile(r"(?<!\d)(?:\+84|0)(?:[35789](?:[\s\.\-]?\d){8})(?!\d)")
CCCD_RE = re.compile(r"(?<!\d)\d{3}[\s\.\-]?\d{3}[\s\.\-]?\d{3}(?:[\s\.\-]?\d{3})?(?!\d)")

def mask_pii(text: str) -> str:
    return CCCD_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", PHONE_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", text))

print("Test che PII:", mask_pii("Tôi CCCD 079 099 001 234, SĐT 090.123.4567 xin hỗ trợ 2.000.000đ qua tổng đài 1900-6363"))
```
- **Quan sát:** Số CCCD viết cách và SĐT có dấu chấm bị che thành `[ĐÃ_CHE_ĐỊNH_DANH]`, nhưng số tiền `2.000.000đ` và tổng đài `1900-6363` được giữ nguyên!

---

### Bước 2: Chuẩn bị Cặp dữ liệu `Chosen` vs. `Rejected` cho `ORPO`
```python
import json
from datasets import Dataset

orpo_rows = [
    {
        "prompt": format_rag_input(
            json.dumps([POLICIES[3]], ensure_ascii=False),
            "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Xin hỗ trợ tiền thuê nhà trọ theo QĐ-09/2024?"
        ),
        "chosen": json.dumps({
            "verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM nhưng status='expired' (hết hạn 31/12/2025), phải bật cảnh báo đỏ.",
            "is_refusal": False,
            "expired_warning": True,
            "doc_code": "QĐ-09/2024/UBND-HCM",
            "required_documents": POLICIES[3]["required_docs"],
            "submission_place": "CẦN XÁC MINH LẠI VỚI CÁN BỘ (Văn bản đã hết hiệu lực)"
        }, ensure_ascii=False),
        "rejected": json.dumps({
            "verification_reasoning": "Khớp văn bản hỗ trợ tiền thuê nhà trọ tại Phường Tân Tạo.",
            "is_refusal": False,
            "expired_warning": False,
            "doc_code": "QĐ-09/2024/UBND-HCM",
            "required_documents": ["Đơn xin tiền trọ tự viết"],
            "submission_place": "Nộp bình thường tại UBND Phường Tân Tạo"
        }, ensure_ascii=False),
        "doc_code": "QĐ-09/2024/UBND-HCM",
        "status": "expired",
    },
    {
        "prompt": format_rag_input(
            json.dumps([POLICIES[0]], ensure_ascii=False),
            "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Tôi chạy xe ôm bị tai nạn cần chuẩn bị hồ sơ gì?"
        ),
        "chosen": json.dumps({
            "verification_reasoning": "Khớp NQ-01/2026/HĐND-HCM đang còn hiệu lực (status='active').",
            "is_refusal": False,
            "expired_warning": False,
            "doc_code": "NQ-01/2026/HĐND-HCM",
            "required_documents": POLICIES[0]["required_docs"],
            "submission_place": POLICIES[0]["submission_place"]
        }, ensure_ascii=False),
        "rejected": json.dumps({
            "verification_reasoning": "Bịa đặt hồ sơ không có trong văn bản.",
            "is_refusal": False,
            "expired_warning": False,
            "doc_code": "VB-BIA-DAT-999",
            "required_documents": ["Sổ đỏ bản gốc", "Sao kê ngân hàng 2 năm"],
            "submission_place": "Bộ Tài chính"
        }, ensure_ascii=False),
        "doc_code": "NQ-01/2026/HĐND-HCM",
        "status": "active",
    },
] * 8

ds_orpo = Dataset.from_list(orpo_rows)
```

---

### Bước 3: Chạy Giai đoạn 2A — `ORPOTrainer` (`lr = 5e-6`, `beta = 0.1`)
```python
from trl import ORPOConfig, ORPOTrainer

orpo_trainer = ORPOTrainer(
    model=model_fast,
    processing_class=tokenizer,
    train_dataset=ds_orpo,
    args=ORPOConfig(
        output_dir="outputs_orpo",
        beta=0.1,
        max_steps=20,
        per_device_train_batch_size=2,
        learning_rate=5e-6,
        logging_steps=5,
        fp16=True,
        report_to="none",
    ),
)
orpo_trainer.train()
```

---

### Bước 4: Chạy Giai đoạn 2B — `GRPOTrainer` với 3 Hàm thưởng Bất đối xứng (`lr = 1e-6`)
```python
from trl import GRPOConfig, GRPOTrainer

def verifiable_reward_fn(completions, doc_code, status, **kwargs):
    rewards = []
    for comp, expected_doc, expected_status in zip(completions, doc_code, status):
        score = 0.0
        try:
            obj = json.loads(comp)
            if "verification_reasoning" in obj and "required_documents" in obj:
                score += 0.4
            if obj.get("doc_code") == expected_doc:
                score += 0.3
            pred_exp = bool(obj.get("expired_warning"))
            if expected_status == "expired":
                score += 0.3 if pred_exp else -0.6  # Phạt gấp đôi (-0.6) nếu bỏ sót văn bản hết hạn!
            else:
                score += 0.3 if not pred_exp else -0.2
        except Exception:
            score -= 0.5
        rewards.append(score)
    return rewards

# Thử chấm điểm trực tiếp hàm thưởng trên cặp Chosen vs. Rejected:
print("Điểm thưởng của Chosen vs Rejected:", verifiable_reward_fn(
    [orpo_rows[0]["chosen"], orpo_rows[0]["rejected"]],
    ["QĐ-09/2024/UBND-HCM", "QĐ-09/2024/UBND-HCM"],
    ["expired", "expired"],
))
```
- **Quan sát kết quả chấm thưởng:** Bản `chosen` đạt điểm tối đa **`+1.0`**, trong khi bản `rejected` (do quên bật `expired_warning`) bị phạt còn **`+0.1`**! Bạn có thể truyền thẳng `verifiable_reward_fn` vào `GRPOTrainer(model=model_fast, reward_funcs=verifiable_reward_fn, ...)` để tinh chỉnh bước cuối.

---

### Bước 5: Kiểm chứng 🟢 SAU khi học ORPO/GRPO trên Ca Văn bản Hết hiệu lực (`expired`)
```python
FastLanguageModel.for_inference(model_fast)
exp_ctx = json.dumps([POLICIES[3]], ensure_ascii=False)
exp_q = "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Xin hỗ trợ tiền thuê nhà trọ theo QĐ-09/2024?"
print("🟢 KẾT QUẢ CA HẾT HIỆU LỰC SAU ORPO:\n", ask_rag(model_fast, exp_ctx, exp_q, use_lora=True))
```

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 5 (Checkpoint)
1. Tại sao trong `verifiable_reward_fn` lỗi bỏ sót văn bản hết hạn (`expected_status == "expired"` nhưng `pred_exp == False`) lại bị phạt `-0.6`, nặng gấp 3 lần lỗi cảnh báo nhầm (`-0.2`)? *(Trả lời: Vì theo nguyên tắc An toàn ở Buổi 02 & 03, bỏ sót cảnh báo làm người dân nộp hồ sơ vô hiệu mất ngày công, còn cảnh báo dư chỉ cần hỏi lại cán bộ một cửa)*.
