# Buổi 6 (Module 6): An toàn Sống còn — `ORPO`, `GRPOTrainer` Thật & 4 Tầng Guardrail P01

- **Tham chiếu:** `ViMedQA`, `AcadifySolution/llm-security-proxy`, `lamhotsiagian/guardrail-system-rag`.
- **🎯 Sản phẩm cuối buổi:** **Cổng An toàn `Safety-Gatekeeper`** — chạy **`ORPO` (đúng Chat Template)** + **`GRPOTrainer` thật** với 3 hàm thưởng bất đối xứng, bọc ngoài bởi 4 tầng Guardrail (PII Masking mở rộng, Cross-Region Conflict, Hard Safety Override, Audit Log).

---

## 1. Giải thích theo Phương pháp Feynman

1. **Tại sao `ORPO` & `GRPO` bổ sung cho `DPO`?**
   - **`ORPO`** gộp luôn cả học có giám sát (SFT) và phân biệt Đúng/Sai (*Odds Ratio*) trong **1 bước duy nhất** — cực gọn nếu không muốn chia làm 2 giai đoạn SFT + DPO riêng.
   - **`GRPO` (`Group Relative Policy Optimization`):** Không cần cặp `Rejected` viết tay! Với mỗi câu hỏi, mô hình **tự sinh `G = 4` câu trả lời** rồi dùng **3 hàm code Python (`Verifiable Rewards`)** chấm trực tiếp — phạt gấp đôi (`-0.6`) lỗi bỏ sót văn bản hết hạn (*False Negative*).
2. **Bắt buộc đồng bộ Chat Template cho `ORPO` & `GRPO`:**
   - Phải gọi `tokenizer.apply_chat_template(..., enable_thinking=False)` cho trường `"prompt"` của `ORPO` và `GRPO`, nếu không mô hình sẽ học trên chuỗi thô không có thẻ `<|im_start|>`!

---

## 2. Thực hành `ORPO`, `GRPOTrainer` Thật & 4 Tầng Guardrail trên Colab

### Bước 1: Định nghĩa 3 Hàm Thưởng Kiểm chứng & Đo 🔴 TRƯỚC `GRPO`
```python
import json, re, torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer, ORPOConfig, ORPOTrainer

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
                score += 0.3 if pred_exp else -0.6
            else:
                score += 0.3 if not pred_exp else -0.2
        except Exception:
            score -= 0.5
        rewards.append(round(score, 2))
    return rewards

FastLanguageModel.for_inference(model_fast)
raw_before = ask_raft(model_fast, [POLICIES[3]], "TP.HCM - Phường Tân Tạo", "Xin tiền thuê nhà trọ năm 2024 cần giấy gì?")
r_before = verifiable_reward_fn([raw_before], ["QĐ-09/2024/UBND-HCM"], ["expired"])[0]
print(f"🔴 TRƯỚC ORPO+GRPO | Reward = {r_before}/1.00 | Raw: {raw_before[:120]}")
```

### Bước 2: Chạy `ORPOTrainer` (Đúng chuẩn Chat Template!)
```python
orpo_ds = Dataset.from_list([
    {
        "prompt": format_dpo_prompt(json.dumps([POLICIES[3]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Xin tiền thuê nhà trọ theo QĐ-09/2024?"),
        "chosen": json.dumps({"verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM nhưng status='expired' -> Bắt buộc bật cảnh báo đỏ.", "is_refusal": False, "expired_warning": True, "doc_code": "QĐ-09/2024/UBND-HCM", "required_documents": POLICIES[3]["required_docs"], "submission_place": "CẦN CÁN BỘ XÁC MINH LẠI"}, ensure_ascii=False),
        "rejected": json.dumps({"verification_reasoning": "Khớp QĐ-09/2024.", "is_refusal": False, "expired_warning": False, "doc_code": "QĐ-09/2024/UBND-HCM", "required_documents": ["Đơn tự viết"], "submission_place": "Nộp bình thường tại UBND Phường"}, ensure_ascii=False),
    },
    {
        "prompt": format_dpo_prompt(json.dumps([POLICIES[0]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Chạy xe ôm bị tai nạn cần chuẩn bị gì?"),
        "chosen": json.dumps({"verification_reasoning": "Khớp NQ-01/2026/HĐND-HCM đang còn hiệu lực.", "is_refusal": False, "expired_warning": False, "doc_code": "NQ-01/2026/HĐND-HCM", "required_documents": POLICIES[0]["required_docs"], "submission_place": POLICIES[0]["submission_place"]}, ensure_ascii=False),
        "rejected": json.dumps({"verification_reasoning": "Tự suy diễn.", "is_refusal": False, "expired_warning": True, "doc_code": "VB-GIA", "required_documents": ["Sổ đỏ"], "submission_place": "Bộ Tài chính"}, ensure_ascii=False),
    },
])

FastLanguageModel.for_training(model_fast)
ORPOTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=orpo_ds,
    args=ORPOConfig(output_dir="out_m6_orpo", beta=0.1, num_train_epochs=2, per_device_train_batch_size=1, learning_rate=5e-6, fp16=True, report_to="none"),
).train()
```

### Bước 3: Chạy `GRPOTrainer` Thật (Model tự sinh 4 phương án & chấm bằng `verifiable_reward_fn`)
```python
grpo_ds = Dataset.from_list([
    {"prompt": format_dpo_prompt(json.dumps([POLICIES[3]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Xin tiền thuê nhà trọ 2024 cần hồ sơ gì?"), "doc_code": "QĐ-09/2024/UBND-HCM", "status": "expired"},
    {"prompt": format_dpo_prompt(json.dumps([POLICIES[0]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Bán vé số bị tai nạn cần nộp gì?"), "doc_code": "NQ-01/2026/HĐND-HCM", "status": "active"},
])

grpo_trainer = GRPOTrainer(
    model=model_fast,
    processing_class=tokenizer,
    reward_funcs=verifiable_reward_fn,
    train_dataset=grpo_ds,
    args=GRPOConfig(
        output_dir="out_m6_grpo",
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=180,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-6,
        max_steps=4,
        fp16=True,
        report_to="none",
    ),
)
grpo_trainer.train()
```

### Bước 4: 🎯 Sản phẩm 4 Tầng `Safety-Gatekeeper` (Che PII + Chặn Lệch Địa bàn + Hậu kiểm `expired`)
```python
PHONE_RE = re.compile(r"(?<!\d)(?:\+84|0)(?:[35789](?:[\s\.\-]?\d){8})(?!\d)")
CCCD_RE = re.compile(r"(?<!\d)\d{3}[\s\.\-]?\d{3}[\s\.\-]?\d{3}(?:[\s\.\-]?\d{3})?(?!\d)")

def safety_gatekeeper(selected_region: str, raw_query: str) -> dict:
    masked_q = CCCD_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", PHONE_RE.sub("[ĐÃ_CHE_ĐỊNH_DANH]", raw_query))
    if not selected_region.strip():
        return {"blocked": True, "reason": "Thiếu địa bàn (Input Guardrail)", "masked_query": masked_q}
    for p in POLICIES:
        district = p["region"].split("-")[-1].strip().lower()
        if district in masked_q.lower() and p["region"] != selected_region:
            return {"blocked": True, "reason": f"Lệch địa bàn chéo: Chọn '{selected_region}' nhưng hỏi '{p['region']}'", "masked_query": masked_q}
    cand = [p for p in POLICIES if p["region"] == selected_region]
    raw_llm = ask_raft(model_fast, cand, selected_region, masked_q)
    try:
        parsed = json.loads(raw_llm)
    except Exception:
        parsed = {"raw": raw_llm}
    for p in cand:
        if p["doc_code"] == parsed.get("doc_code") and p["status"] == "expired":
            parsed["expired_warning"] = True
    return {"blocked": False, "masked_query": masked_q, "answer": parsed}
```

### Bước 5: 📊 Đo 🟢 SAU `ORPO + GRPO` & Chạy Thử 2 Ca Tấn Công Guardrail
```python
FastLanguageModel.for_inference(model_fast)
raw_after = ask_raft(model_fast, [POLICIES[3]], "TP.HCM - Phường Tân Tạo", "Xin tiền thuê nhà trọ năm 2024 cần giấy gì?")
r_after = verifiable_reward_fn([raw_after], ["QĐ-09/2024/UBND-HCM"], ["expired"])[0]
print(f"🏆 REWARD CA HẾT HẠN: Trước = {r_before}/1.00 ---> Sau ORPO+GRPO = {r_after}/1.00")
print("🛡️ Test Che PII + Expired:", safety_gatekeeper("TP.HCM - Phường Tân Tạo", "CCCD 079 099 001 234 SĐT 090.123.4567 xin tiền trọ 2024"))
print("🛡️ Test Lệch địa bàn chéo:", safety_gatekeeper("TP.HCM - Phường Tân Tạo", "Tôi ở Quận Liên Chiểu xin học nghề"))
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *So sánh DPO, ORPO và GRPO — khi nào chọn thuật toán nào?*
   - **Đáp:** **DPO** ổn định nhất khi có cặp `Chosen/Rejected` chất lượng và có KL constraint giữ nguyên năng lực gốc. **ORPO** gộp SFT + preference loss vào 1 bước, không cần Reference Model. **GRPO** mạnh nhất cho bài toán có đáp án chấm được bằng rule/code (**Verifiable Rewards** như JSON schema, mã văn bản, toán, code).
