# Đề cương Thực chiến: Fine-tune `Qwen3-1.7B` & RAG Từ Số 0 đến Tự tin Đi làm / Phỏng vấn

- **Mô hình xuyên suốt:** `Qwen/Qwen3-1.7B` trên Google Colab T4 (15 GB VRAM) + `bge/e5` Embedding cho tầng Retrieval.
- **Triết lý đào tạo:**
  1. **100% Thực chiến có Sản phẩm & Số đo Trung thực:** Mỗi buổi đều có sản phẩm chạy được ngay và phép đo Trước/Sau trên **tập Test giữ riêng (0% rò rỉ dữ liệu)**.
  2. **Tách nhỏ $\rightarrow$ Gộp nhanh:** Buổi 0–3 tách từng ô siêu nhỏ (4–12 dòng); Buổi 4–11 gộp gọn theo chuẩn kỹ sư AI thực thụ.
  3. **Chuẩn Phỏng vấn & Production:** Bao quát đủ SFT (`LoRA`, `DoRA`, `rsLoRA`, `NEFTune`, `Unsloth`, `Liger`), Alignment (`DPO`, `ORPO`, `GRPO`), Dữ liệu (`RAFT`, `Synthetic Data`, `Dedup`), Agent (`Function Calling`, `Multi-turn`), `Embedding Fine-tuning`, Triển khai (`Merge 16-bit`, `llama-imatrix` + `GGUF`, `vLLM Multi-LoRA`, `DDP/FSDP`), và Đánh giá (`Sampling Stability`, `VMLU`, `Ragas`, `Model Card`).

---

## Lộ trình 12 Buổi Học Chi tiết trong Thư mục `edu/`

| Buổi | File tài liệu | Chủ đề & Kỹ thuật Trọng tâm | 🎯 Sản phẩm & Câu hỏi Phỏng vấn |
|---|---|---|---|
| **0** | [module-00-colab-t4-and-baseline.md](../edu/module-00-colab-t4-and-baseline.md) | Pinned env, `4-bit NF4`, `fp16` vs `bf16` trên T4, `enable_thinking=False` | Máy X-Quang `Softmax` & Đo Baseline gốc |
| **1** | [module-01-vivu-rag-sft-lora.md](../edu/module-01-vivu-rag-sft-lora.md) | `LoRA r=8`, `<context>`, **`Prompt Loss Masking` (`labels=-100`)** | Chatbot `ViVu-Mini` (đo trên Test giữ riêng) |
| **2** | [module-02-evidence-json-dora-rslora.md](../edu/module-02-evidence-json-dora-rslora.md) | `Verification-First JSON`, **`DoRA + rsLoRA`**, nạp model sạch chống nhiễm | API `Evidence-Guard` & `verify_grounding()` |
| **3** | [module-03-legal-rag-neftune-liger.md](../edu/module-03-legal-rag-neftune-liger.md) | `Unsloth` (import đầu tiên) + `Liger Kernel`, **`NEFTune`**, `train_on_responses_only` | `Legal-Translator` & Đo **Fairness Gap** |
| **4** | [module-04-raft-refusal-payload-filter.md](../edu/module-04-raft-refusal-payload-filter.md) | **`RAFT` (70/30)** & Code chia tập thật **`Scenario-Group Split 80/10/10`** | Khiên `RAFT-Shield` lọc 2 đoạn nhiễu + Từ chối |
| **5** | [module-05-hyperparams-debug-and-dpo.md](../edu/module-05-hyperparams-debug-and-dpo.md) | Khi nào Prompt/RAG/FT, `grad_accum`, `OOM/NaN`, `EarlyStopping`, **`DPOTrainer`** | Pipeline `DPO` (`ref_model=None`) + Debug |
| **6** | [module-06-safety-orpo-grpo-guardrails.md](../edu/module-06-safety-orpo-grpo-guardrails.md) | **`ORPO`** (đúng Chat Template), **`GRPOTrainer` thật** (phạt FN `-0.6`), 4-tier Guardrail | `Safety-Gatekeeper` (PII + Cross-Region + `expired`) |
| **7** | [module-07-synthetic-data-and-dedup.md](../edu/module-07-synthetic-data-and-dedup.md) | Sinh dữ liệu tổng hợp, Grounding Filter, **Dedup Cosine**, Audit rò rỉ | `Data-Factory` tạo hàng trăm mẫu sạch |
| **8** | [module-08-function-calling-and-multiturn.md](../edu/module-08-function-calling-and-multiturn.md) | **`<tools>` & `<tool_call>`** của Qwen3, Multi-turn Loss Masking | Agent `Qwen3-1.7B` gọi Tool + Chat 2 lượt |
| **9** | [module-09-finetune-embedding-and-reranker.md](../edu/module-09-finetune-embedding-and-reranker.md) | Bi-Encoder vs Cross-Encoder, **Hard Negatives**, `MNRL` cho tiếng Việt | Embedding tinh chỉnh tăng `Recall@1` RAG |
| **10** | [module-10-deployment-vllm-gguf-imatrix.md](../edu/module-10-deployment-vllm-gguf-imatrix.md) | Merge đúng chuẩn 16-bit, **`llama-imatrix` + `llama-quantize`**, `GGUF/AWQ/GPTQ`, `vLLM Multi-LoRA`, `DDP/FSDP` | Đóng gói `.gguf` thật & Đo `TTFT / tokens/s` |
| **11** | [module-11-capstone-p01-eval-and-portfolio.md](../edu/module-11-capstone-p01-eval-and-portfolio.md) | Đo 3 lần có Sampling (`do_sample=True`), Chống quên **`VMLU`**, `LLM-as-judge` bias, **Model Card & License** | Hồ sơ Portfolio & Bảng Target Benchmarks P01 |
