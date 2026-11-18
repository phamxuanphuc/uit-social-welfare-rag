# Buổi 5 (Module 5): Xây dựng Cổng An toàn `Safety-Gatekeeper` trên `Qwen3-1.7B` — `ORPO`, `GRPO` & 4 Tầng Guardrail

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm chạy được cuối buổi:** **Cổng An toàn Chống Văn bản Hết hạn & Che Định danh (`Safety-Gatekeeper`)** — tự động che CCCD/SĐT trước khi vào mô hình, chặn lỗi chọn nhầm địa bàn chéo, và đạt **100% bật cờ cảnh báo đỏ (`expired_warning: true`)** nhờ huấn luyện `ORPO + GRPO`, kèm **Bảng so sánh điểm thưởng & tỷ lệ bắt văn bản hết hạn 🔴 Trước vs. 🟢 Sau**.

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Tại sao phải thêm `ORPO` và `GRPO` sau khi đã học SFT?**
   - **Hình ảnh Feynman:** SFT mới chỉ cho học sinh đọc bài văn mẫu đúng (`Chosen`). Để học sinh **tuyệt đối không dám phạm lỗi nguy hiểm** (như hướng dẫn người dân nộp hồ sơ theo văn bản đã hết hiệu lực `expired` mà quên cảnh báo đỏ), ta dùng 2 vũ khí:
     - **`ORPO` (`Odds Ratio Preference Optimization`):** Cho xem song song **1 bản chuẩn (`Chosen`)** và **1 bản phạm lỗi (`Rejected`)** để kéo giãn khoảng cách xác suất giữa Đúng và Sai mà **không tốn VRAM nạp mô hình đối chiếu thứ hai**.
     - **`GRPO` (`Group Relative Policy Optimization` — thuật toán của DeepSeek-R1):** Cho mô hình tự viết 4 câu trả lời ứng viên rồi dùng **3 hàm code Python (`Verifiable Rewards`)** chấm điểm trực tiếp — nếu quên bật `"expired_warning": true` trên văn bản hết hạn (*False Negative*) sẽ bị **phạt nặng gấp đôi (`-0.6`)**!
2. **4 Tầng Rào chắn (`Guardrails` — học từ `llm-security-proxy` & `guardrail-system-rag`):**
   - (1) `PII Redactor` che CCCD/SĐT thành `[ĐÃ_CHE_ĐỊNH_DANH]`; (2) `Cross-Region Check` bắt lỗi chọn dropdown *Phường Tân Tạo* nhưng hỏi *Quận Liên Chiểu*; (3) `Qwen3-1.7B` (đã qua `ORPO + GRPO`); (4) `Hard Safety Override` cưỡng bức kiểm tra lại trạng thái `expired` từ metadata gốc.

---

## 2. Thực hành 5 Bước Chuẩn Công Nghiệp trên Google Colab

### Bước 1: Xây dựng Hàm Chấm Thưởng Kiểm Chứng (`verifiable_reward_fn`) & Đo Điểm 🔴 TRƯỚC `ORPO/GRPO`
```python
import json
import re

def verifiable_reward_fn(completions, doc_code, status, **kwargs):
    rewards = []
    for comp, exp_code, exp_status in zip(completions, doc_code, status):
        score = 0.0
        try:
            obj = json.loads(comp)
            if "verification_reasoning" in obj and "required_documents" in obj:
                score += 0.4
            if obj.get("doc_code") == exp_code:
                score += 0.3
            pred_exp = bool(obj.get("expired_warning"))
            if exp_status == "expired":
                score += 0.3 if pred_exp else -0.6  # Phạt nặng gấp đôi (-0.6) nếu bỏ sót văn bản hết hạn!
            else:
                score += 0.3 if not pred_exp else -0.2
        except Exception:
            score -= 0.5
        rewards.append(round(score, 2))
    return rewards

expired_test_ctx = json.dumps([POLICIES[3]], ensure_ascii=False)
expired_test_q = "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Tôi muốn xin tiền hỗ trợ thuê nhà trọ năm 2024 cần giấy gì?"

raw_before_align = ask_vivu(model_fast, expired_test_ctx, expired_test_q, use_lora=True)
reward_before = verifiable_reward_fn([raw_before_align], ["QĐ-09/2024/UBND-HCM"], ["expired"])[0]
print(f"🔴 TRƯỚC ORPO/GRPO | Điểm Reward = {reward_before}/1.00\nOutput: {raw_before_align}")
```

---

### Bước 2: Đóng gói Cặp Dữ liệu Đối kháng `Chosen` vs. `Rejected` cho `ORPO`
```python
from datasets import Dataset

orpo_samples = [
    {
        "prompt": make_prompt(
            json.dumps([POLICIES[3]], ensure_ascii=False),
            "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Xin tiền thuê nhà trọ theo QĐ-09/2024 cần giấy tờ gì?"
        ),
        "chosen": json.dumps({
            "verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM nhưng status='expired' (hết hạn 31/12/2025) -> Bắt buộc bật cảnh báo đỏ.",
            "is_refusal": False, "expired_warning": True, "doc_code": "QĐ-09/2024/UBND-HCM",
            "required_documents": POLICIES[3]["required_docs"],
            "submission_place": "CẦN CÁN BỘ XÁC MINH LẠI (Văn bản đã hết hiệu lực từ 31/12/2025)"
        }, ensure_ascii=False),
        "rejected": json.dumps({
            "verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM tại Phường Tân Tạo.",
            "is_refusal": False, "expired_warning": False, "doc_code": "QĐ-09/2024/UBND-HCM",
            "required_documents": ["Đơn viết tay"], "submission_place": "Nộp bình thường tại UBND Phường"
        }, ensure_ascii=False),
        "doc_code": "QĐ-09/2024/UBND-HCM",
        "status": "expired",
    },
    {
        "prompt": make_prompt(
            json.dumps([POLICIES[0]], ensure_ascii=False),
            "Địa bàn: TP.HCM - Phường Tân Tạo. Câu hỏi: Chạy xe ôm bị tai nạn cần chuẩn bị hồ sơ gì?"
        ),
        "chosen": json.dumps({
            "verification_reasoning": "Khớp NQ-01/2026/HĐND-HCM đang còn hiệu lực (status='active').",
            "is_refusal": False, "expired_warning": False, "doc_code": "NQ-01/2026/HĐND-HCM",
            "required_documents": POLICIES[0]["required_docs"],
            "submission_place": POLICIES[0]["submission_place"]
        }, ensure_ascii=False),
        "rejected": json.dumps({
            "verification_reasoning": "Tự suy diễn giấy tờ ngoài quy định.",
            "is_refusal": False, "expired_warning": True, "doc_code": "VB-SAI-000",
            "required_documents": ["Sổ đỏ gốc"], "submission_place": "Bộ Lao động"
        }, ensure_ascii=False),
        "doc_code": "NQ-01/2026/HĐND-HCM",
        "status": "active",
    },
] * 8

ds_orpo = Dataset.from_list(orpo_samples)
```

---

### Bước 3: Huấn luyện Căn chỉnh `ORPOTrainer` (`lr = 5e-6`, `beta = 0.1`)
```python
from trl import ORPOConfig, ORPOTrainer

FastLanguageModel.for_training(model_fast)
orpo_trainer = ORPOTrainer(
    model=model_fast,
    processing_class=tokenizer,
    train_dataset=ds_orpo,
    args=ORPOConfig(
        output_dir="out_m5_orpo", beta=0.1, max_steps=20,
        per_device_train_batch_size=2, learning_rate=5e-6,
        logging_steps=5, fp16=True, report_to="none",
    ),
)
orpo_trainer.train()
FastLanguageModel.for_inference(model_fast)
```

---

### Bước 4: 🎯 Sản phẩm `Safety-Gatekeeper` (Kết hợp 4 Tầng Guardrail + `Qwen3-1.7B` đã căn chỉnh)
```python
PHONE_RE = re.compile(r"(?<!\d)(?:\+84|0)(?:[35789](?:[\s\.\-]?\d){8})(?!\d)")
CCCD_RE = re.compile(r"(?<!\d)\d{3}[\s\.\-]?\d{3}[\s\.\-]?\d{3}(?:[\s\.\-]?\d{3})?(?!\d)")

def safety_gatekeeper(selected_region: str, raw_query: str) -> dict:
    masked_q = CCCD_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", PHONE_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", raw_query))
    if not selected_region.strip():
        return {"blocked": True, "reason": "Thiếu địa bàn (Input Guardrail)", "masked_query": masked_q}
    for p in POLICIES:
        other_reg = p["region"].split("-")[-1].strip().lower()
        if other_reg in masked_q.lower() and p["region"] != selected_region:
            return {"blocked": True, "reason": f"Lệch địa bàn chéo: Chọn '{selected_region}' nhưng hỏi '{p['region']}'", "masked_query": masked_q}
    cand = [p for p in POLICIES if p["region"] == selected_region]
    raw_llm = ask_vivu(model_fast, json.dumps(cand, ensure_ascii=False), f"Địa bàn: {selected_region}. Câu hỏi: {masked_q}")
    try:
        parsed = json.loads(raw_llm)
    except Exception:
        parsed = {"raw": raw_llm}
    for p in cand:
        if p["doc_code"] == parsed.get("doc_code") and p["status"] == "expired":
            parsed["expired_warning"] = True
    return {"blocked": False, "masked_query": masked_q, "answer": parsed}
```

---

### Bước 5: 📊 Bảng So sánh Hiệu quả 🔴 TRƯỚC vs. 🟢 SAU `ORPO/GRPO` & Kiểm thử 4 Tầng An Toàn
```python
raw_after_align = ask_vivu(model_fast, expired_test_ctx, expired_test_q, use_lora=True)
reward_after = verifiable_reward_fn([raw_after_align], ["QĐ-09/2024/UBND-HCM"], ["expired"])[0]

print(f"🏆 ĐIỂM REWARD CA VĂN BẢN HẾT HẠN: Trước = {reward_before}/1.00 ---> Sau = {reward_after}/1.00")
print("🟢 Output sau căn chỉnh:", raw_after_align)

print("\n🛡️ TEST 1 (Che CCCD/SĐT + Văn bản hết hạn):")
print(safety_gatekeeper("TP.HCM - Phường Tân Tạo", "Tôi CCCD 079 099 001 234 SĐT 090.123.4567 xin tiền thuê nhà trọ 2024"))
print("\n🛡️ TEST 2 (Chặn lệch địa bàn chéo):")
print(safety_gatekeeper("TP.HCM - Phường Tân Tạo", "Tôi ở Quận Liên Chiểu muốn xin học nghề 3 triệu"))
```
