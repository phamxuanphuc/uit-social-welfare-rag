# Đề cương Thực chiến: Fine-tune `Qwen3-1.7B` & RAG — Từ số 0 đến tự tin đi làm / phỏng vấn

- **Mô hình xuyên suốt:** `Qwen/Qwen3-1.7B` trên Google Colab T4 (15 GB VRAM); tầng truy xuất dùng mô hình cùng họ `Qwen3-Embedding-0.6B` + `Qwen3-Reranker-0.6B`.
- **Tài liệu chi tiết, cách khởi động Colab, bộ dữ liệu và bộ công cụ `p01kit`:** xem [edu/README.md](../edu/README.md).
- **Triết lý đào tạo:**
  1. **Một tập test, một bộ chấm, một bảng điểm:** cả 16 buổi đo trên cùng 100 câu test giữ riêng (72 câu về văn bản chưa từng thấy), kèm khoảng tin cậy bootstrap và kiểm định McNemar — mọi con số so sánh được với nhau.
  2. **Mở hộp đen → Thực chiến:** Buổi 00–04 viết ô siêu nhỏ, tự tay làm masking/LoRA/vòng train; Buổi 05–15 gộp khối, tập trung thiết kế thí nghiệm và ra quyết định.
  3. **Code đã được kiểm chứng:** bộ thư viện ghim theo ràng buộc của `unsloth==2026.11.3` (`transformers==5.17.0`, `trl==1.15.0`); bộ `pytest` kiểm tra cả `p01kit` lẫn mọi ô code trong bài (cú pháp, API đã bị bỏ, phiên bản ghim, thứ tự import).
  4. **Chuẩn phỏng vấn & production:** SFT (LoRA, DoRA, rsLoRA, NEFTune, Unsloth), dữ liệu (RAFT, dữ liệu tổng hợp, dedup, chống rò rỉ), căn chỉnh (DPO, ORPO, GRPO), Qwen3 thinking mode, function calling, embedding/reranker, triển khai (merge FP16, GGUF + imatrix, JSON Schema decoding, vLLM multi-LoRA, DDP/FSDP), đánh giá (sampling stability, Global-MMLU, LLM-judge, red-team, Model Card), vận hành (API, cổng champion/challenger, kiểm thử tải, drift, nhãn trễ, KillSwitch, huấn luyện lại). Mỗi buổi có mục **🏢 Ở công ty thật** so bài lab với cách doanh nghiệp làm.

---

## Lộ trình 16 buổi trong thư mục `edu/`

| Buổi | File | Trọng tâm | 🎯 Sản phẩm |
|---|---|---|---|
| **00** | [module-00](../edu/module-00-colab-qwen3-anatomy-baseline.md) | Giải phẫu Qwen3, NF4, thinking on/off, sinh theo lô | Baseline zero-shot trên bảng điểm |
| **01** | [module-01](../edu/module-01-dataset-evaluator-prompt-baseline.md) | Dữ liệu RAFT, kiểm toán rò rỉ, bộ chấm, few-shot | Báo cáo dữ liệu + baseline few-shot |
| **02** | [module-02](../edu/module-02-first-lora-sft-manual-masking.md) | Loss masking tự viết, LoRA, QLoRA | Adapter SFT đầu tiên |
| **03** | [module-03](../edu/module-03-dora-rslora-controlled-ablation.md) | DoRA, rsLoRA, ablation có kiểm soát | Báo cáo ablation 4 biến thể |
| **04** | [module-04](../edu/module-04-unsloth-trl-neftune-fairness.md) | Unsloth, TRL, NEFTune, Fairness 2×2 | Bảng tốc độ + bảng fairness |
| **05** | [module-05](../edu/module-05-raft-distractors-refusal-ablation.md) | RAFT, văn bản nhiễu, học từ chối | Ma trận lỗi 3 cấu hình |
| **06** | [module-06](../edu/module-06-hyperparams-debugging-tracking.md) | Siêu tham số trên val, debug OOM/NaN | Mô hình chuẩn `sft_best` |
| **07** | [module-07](../edu/module-07-synthetic-data-factory.md) | Dữ liệu tổng hợp bằng Qwen3, phễu lọc | Data card + `sft_aug` |
| **08** | [module-08](../edu/module-08-preference-dpo-vs-orpo.md) | Khai thác lỗi on-policy, DPO vs ORPO | `sft_dpo`, `orpo_from_base` |
| **09** | [module-09](../edu/module-09-grpo-verifiable-rewards.md) | GRPO, phần thưởng kiểm chứng, reward hacking | `grpo` |
| **10** | [module-10](../edu/module-10-qwen3-thinking-mode.md) | Hybrid thinking, SFT trộn 2 chế độ | `mixed_think` + bộ định tuyến |
| **11** | [module-11](../edu/module-11-function-calling-agent.md) | Function calling, agent nhiều lượt | `agent` chạy end-to-end |
| **12** | [module-12](../edu/module-12-retrieval-embedding-reranker.md) | BM25, Qwen3-Embedding LoRA, Qwen3-Reranker | Retriever + RAG đầu-cuối |
| **13** | [module-13](../edu/module-13-merge-gguf-imatrix-serving.md) | Merge FP16, GGUF + imatrix, TTFT CPU | GGUF triển khai + báo cáo lượng tử |
| **14** | [module-14](../edu/module-14-capstone-eval-guardrails-release.md) | Guardrail, red-team, Global-MMLU, LLM-judge | Release Dossier + Model Card |
| **15** | [module-15](../edu/module-15-production-ops-monitoring-feedback.md) | API, champion/challenger, tải p95, drift, nhãn trễ + κ, KillSwitch | API chạy thật + báo cáo vận hành + runbook |
