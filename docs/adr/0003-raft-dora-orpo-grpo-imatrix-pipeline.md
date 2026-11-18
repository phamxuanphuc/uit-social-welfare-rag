# 0003. RAFT + DoRA/rsLoRA/NEFTune + ORPO/GRPO & imatrix GGUF Pipeline

- **Status:** superseded by [ADR 0004](0004-evidence-driven-finetune-pipeline.md)

Để mô hình `Qwen/Qwen3-1.7B` vừa bám sát ngữ cảnh tiếng Việt, vừa biết từ chối khi thiếu căn cứ và không bỏ sót cảnh báo văn bản hết hiệu lực (`expired`) trên GPU miễn phí Google Colab T4 (15 GB VRAM), chúng tôi chốt pipeline huấn luyện 2 giai đoạn kết hợp lượng tử hóa bảo toàn độ chính xác:

1. **Chuẩn bị dữ liệu RAFT & Chống Data Leakage:** Chia tập `80% Train / 10% Val / 10% Test` theo nhóm kịch bản gốc (`Scenario-Group Split`) để không rò rỉ biến thể câu hỏi sang tập Test. Mỗi mẫu trộn 1 đoạn đúng (Oracle) với 2 đoạn nhiễu (Distractors) đã xáo trộn vị trí (70% mẫu có đáp án, 30% mẫu toàn Distractors dạy từ chối) kèm trường `verification_reasoning` đứng đầu JSON.
2. **Giai đoạn 1 — SFT chính xác cao (`lr=2e-4`):** Sử dụng `Unsloth` + `Liger Kernel` + 4-bit NF4 Double Quantization với `DoRA` (`use_dora=True`), `rsLoRA` (`use_rslora=True`) và `NEFTune` (`neftune_noise_alpha=5`). Thiết kế cơ chế auto-fallback sang `AutoModelForCausalLM + LigerKernel + PEFT DoRA` nếu fast Triton kernel của Unsloth xung đột trên Colab T4.
3. **Giai đoạn 2 — Alignment giảm dần LR (`ORPO 5e-6` → `GRPO 1e-6`):** Chạy `ORPO` trên cặp `Chosen/Rejected` mà không tốn VRAM nạp Reference Model, sau đó chạy `GRPO` (`num_generations=4`) với 3 hàm thưởng kiểm chứng tự động (JSON Schema hợp lệ, đúng `doc_code` theo địa bàn, và phạt gấp đôi lỗi False Negative khi bỏ sót `expired_warning`).
4. **Đánh giá 4 mốc & Nén `imatrix Q4_K_M.gguf`:** Đánh giá qua 4 mốc (`Baseline BM25+Base` → `SFT+DoRA` → `ORPO+GRPO` → `Q4_K_M.gguf`) bằng Deterministic Metrics + `Ragas` `Faithfulness` (hỗ trợ Local/Free Judge) kèm kiểm tra độ ổn định qua 3 lần chạy, dùng chính tập chính sách + JSON chuẩn làm dữ liệu hiệu chuẩn `imatrix` trước khi đẩy lên Hugging Face Hub.
