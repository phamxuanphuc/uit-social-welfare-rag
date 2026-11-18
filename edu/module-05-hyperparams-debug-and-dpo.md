# Buổi 5 (Module 5): Siêu Tham số, Debug OOM/NaN, Early Stopping & Thực hành `DPO` (Phỏng vấn Hay Hỏi Nhất)

- **Mô hình thực hành:** `Qwen/Qwen3-1.7B` trên Google Colab `T4 GPU`.
- **🎯 Sản phẩm cuối buổi:** **Bộ Theo dõi & Căn chỉnh `DPO` (`Direct Preference Optimization`) Chuyên nghiệp** — biết đọc đồ thị `Train Loss` vs `Val Loss`, tự động dừng khi chạm đáy (`EarlyStoppingCallback`), biết xử lý OOM/NaN và nắm vững `DPO` (thuật toán căn chỉnh được hỏi nhiều nhất trong phỏng vấn LLM).

---

## 1. Ma trận Quyết định: Khi nào KHÔNG NÊN Fine-tune?

| Giải pháp | Dùng khi nào? | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **1. Prompt Engineering** | Logic đơn giản, thử nghiệm nhanh ngày đầu, model lớn (API). | Tốn $0 train, đổi luật trong 1 phút. | Prompt dài tốn tiền token, model nhỏ 1.7B dễ quên luật. |
| **2. RAG (Truy xuất)** | Dữ liệu thay đổi hàng ngày (văn bản hết hạn, giá cả, tin tức). | Cập nhật tri thức tức thì không cần train lại. | Không tự sửa được thói quen trả lời sai định dạng của model nhỏ. |
| **3. Fine-tune (`LoRA/DPO`)** | Cần ép văn phong/JSON cực chuẩn, hiểu từ lóng đặc thù, hoặc cần model nhỏ chạy CPU chi phí thấp. | Độ chính xác cao trên tác vụ hẹp, giảm chi phí vận hành. | Không dùng để nhồi tri thức mới thay đổi liên tục (phải kết hợp **RAG + Fine-tune**). |

---

## 2. Bộ Kỹ năng Chỉnh Siêu tham số & Debug Thực chiến trên T4

1. **Giảm VRAM khi gặp lỗi `CUDA Out Of Memory (OOM)`:**
   - Giảm `per_device_train_batch_size` xuống `1` hoặc `2`, và tăng `gradient_accumulation_steps` lên `4` hoặc `8` để giữ **Effective Batch Size** = `batch_size * grad_accum = 8` không đổi!
   - Bật `gradient_checkpointing=True` (hoặc `"unsloth"`): không giữ toàn bộ kết quả trung gian của 28 tầng trên VRAM mà tính lại khi cần $\rightarrow$ tốn thêm ~20% thời gian nhưng giảm tới **60% VRAM**.
   - Bật `packing=True` (gom nhiều câu ngắn vào chung 1 khối `max_seq_length` để không lãng phí token đệm `<pad>`).
2. **Đọc biểu đồ Loss & Xử lý `NaN`:**
   - **Underfitting:** `Train Loss` và `Val Loss` cùng cao, chưa đi xuống $\rightarrow$ tăng `num_train_epochs` hoặc tăng nhẹ `learning_rate`.
   - **Overfitting:** `Train Loss` giảm sát `0`, nhưng `Val Loss` quay đầu đi lên $\rightarrow$ bật `EarlyStoppingCallback`, giảm epoch hoặc tăng `weight_decay=0.01`, `neftune_noise_alpha=5`.
   - **Nổ Loss (`NaN`):** Thường do `learning_rate` quá cao hoặc `fp16` tràn số $\rightarrow$ giảm `lr`, đặt `warmup_ratio=0.1` (khởi động ấm 10% bước đầu), đặt `max_grad_norm=1.0` (cắt ngọn đạo hàm).
3. **`DPO` (`Direct Preference Optimization`) khác gì `ORPO`?**
   - **`DPO`** tối ưu dựa trên **2 mô hình**: Policy Model (đang học) và Reference Model (bản gốc cố định dùng để giữ model mới không đi quá xa). Trong `PEFT` (`LoRA`), ta **không cần nạp thêm mô hình thứ 2** — `DPOTrainer` tự tắt LoRA (`disable_adapter`) để lấy xác suất của Reference Model!

---

## 3. Thực hành `DPO` + `EarlyStoppingCallback` trên Google Colab

### Bước 1: Chuẩn bị Tập Preference (`Train` & `Val`) Đúng Chuẩn Chat Template
```python
from datasets import Dataset
from transformers import EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer

def format_dpo_prompt(context_str: str, question: str) -> str:
    msgs = [
        {"role": "system", "content": SYS_RAFT},
        {"role": "user", "content": f"<context>\n{context_str}\n</context>\n\nCÂU HỎI:\n{question}"},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

dpo_train = Dataset.from_list([
    {
        "prompt": format_dpo_prompt(json.dumps([POLICIES[0]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Chạy xe ôm bị tai nạn cần giấy gì?"),
        "chosen": json.dumps({"verification_reasoning": "Khớp NQ-01/2026/HĐND-HCM còn hạn.", "is_refusal": False, "expired_warning": False, "doc_code": "NQ-01/2026/HĐND-HCM", "required_documents": POLICIES[0]["required_docs"], "submission_place": POLICIES[0]["submission_place"]}, ensure_ascii=False),
        "rejected": json.dumps({"verification_reasoning": "Không rõ nguồn.", "is_refusal": False, "expired_warning": False, "doc_code": "VB-SAI", "required_documents": ["Hộ chiếu", "Sổ đỏ"], "submission_place": "UBND Quận 1"}, ensure_ascii=False),
    },
    {
        "prompt": format_dpo_prompt(json.dumps([POLICIES[3]], ensure_ascii=False), "Địa bàn: TP.HCM - Phường Tân Tạo. Xin tiền thuê nhà trọ năm 2024?"),
        "chosen": json.dumps({"verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM nhưng status='expired'.", "is_refusal": False, "expired_warning": True, "doc_code": "QĐ-09/2024/UBND-HCM", "required_documents": POLICIES[3]["required_docs"], "submission_place": "CẦN CÁN BỘ XÁC MINH LẠI"}, ensure_ascii=False),
        "rejected": json.dumps({"verification_reasoning": "Khớp QĐ-09/2024/UBND-HCM.", "is_refusal": False, "expired_warning": False, "doc_code": "QĐ-09/2024/UBND-HCM", "required_documents": ["Đơn viết tay"], "submission_place": "Nộp bình thường tại UBND Phường"}, ensure_ascii=False),
    },
])
dpo_val = dpo_train
```

### Bước 2: Chạy `DPOTrainer` với `Warmup`, `Gradient Accumulation` & `EarlyStopping`
```python
FastLanguageModel.for_training(model_fast)
dpo_args = DPOConfig(
    output_dir="out_m5_dpo",
    beta=0.1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=5e-6,
    warmup_ratio=0.1,
    max_grad_norm=1.0,
    num_train_epochs=2,
    eval_strategy="steps",
    eval_steps=2,
    save_strategy="steps",
    save_steps=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    report_to="none",
)

dpo_trainer = DPOTrainer(
    model=model_fast,
    ref_model=None,
    processing_class=tokenizer,
    train_dataset=dpo_train,
    eval_dataset=dpo_val,
    args=dpo_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)
dpo_trainer.train()
```
- **Lưu ý cực đắt giá cho phỏng vấn:** Truyền `ref_model=None` khi mô hình đang gắn LoRA (`PEFT`), `DPOTrainer` sẽ tự động dùng `disable_adapter()` làm Reference Model — **tiết kiệm 50% VRAM** so với việc nạp 2 model!

---

## 4. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Sự khác biệt giữa `batch_size=8` và `batch_size=2, gradient_accumulation_steps=4` là gì?*
   - **Đáp:** Cùng *Effective Batch Size = 8* (chất lượng bước cập nhật trọng số tương đương), nhưng `grad_accum=4` chỉ nạp 2 mẫu vào GPU mỗi lần nên **tốn VRAM ít hơn 4 lần**, đổi lại chạy chậm hơn một chút.
2. **Hỏi:** *Trong `DPO`, tham số `beta` (ví dụ `0.1`) có ý nghĩa gì?*
   - **Đáp:** `beta` kiểm soát lực kéo KL-Divergence giữ mô hình mới không trôi quá xa Reference Model. `beta` càng lớn thì mô hình càng bảo thủ bám sát bản gốc; `beta` nhỏ (`0.05–0.1`) cho phép đổi hành vi mạnh hơn.
3. **Hỏi:** *Khi nào nên dùng RAG, khi nào Fine-tune, khi nào dùng cả hai?*
   - **Đáp:** RAG cung cấp tri thức động/cập nhật liên tục; Fine-tune dạy kỹ năng, định dạng đầu ra (JSON/Tool Call) và khả năng từ chối/lọc nhiễu. Đồ án P01 kết hợp cả hai (**RAFT**).
