# 0004. Evidence-Driven Fine-tune Pipeline (thay ADR 0003)

- **Status:** accepted
- **Supersedes:** [ADR 0003](0003-raft-dora-orpo-grpo-imatrix-pipeline.md)

## Bối cảnh

ADR 0003 bật cùng lúc DoRA + rsLoRA + NEFTune, chạy ORPO (`5e-6`) **sau** SFT rồi GRPO (`1e-6`). Đối chiếu với nghiên cứu 2025–2026 thấy ba vấn đề:

1. **Biến thể LoRA:** khi dò learning rate riêng cho từng phương pháp, LoRA thường, PiSSA, MiLoRA và DoRA cho kết quả gần như ngang nhau ([arXiv 2602.04998](https://arxiv.org/abs/2602.04998)). Một audit chạy nhiều seed ghép cặp thấy phần lớn mức lợi đã công bố của DoRA/rsLoRA/PiSSA không còn; nguyên nhân khả dĩ là LR hiệu dụng `lr·α/√r` ([AACL 2026 short 32](https://aclanthology.org/2026.aacl-short.32.pdf)). LoRA gần bằng full fine-tune khi gắn vào **mọi lớp, kể cả MLP** ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)). Với `r=16, α=32`, bật rsLoRA làm hệ số scale tăng 4 lần (2 → 8).
2. **ORPO sau SFT:** ORPO được thiết kế để thay luôn bước SFT (một giai đoạn, không cần mô hình tham chiếu). Chạy sau SFT với LR `5e-6` gần như không thay đổi mô hình. So sánh có kiểm soát 20 biến thể DPO không thấy biến thể nào hơn DPO thường có ý nghĩa thống kê ([arXiv 2603.19335](https://arxiv.org/abs/2603.19335)).
3. **GRPO:** LoRA cho RL cần LR cao hơn full fine-tune; `1e-6` quá thấp. Loss `grpo` gốc thiên lệch theo độ dài; TRL 1.15 đã mặc định `loss_type="dapo"`. Ground-GRPO cho thấy train phần thưởng theo 2 chặng (đúng + trích dẫn trước, từ chối sau) ổn định hơn gộp một lần ([arXiv 2506.15522](https://arxiv.org/abs/2506.15522)).

## Quyết định

Nguyên tắc: **mỗi kỹ thuật phải thắng cấu hình gốc trên tập val, 3 seed ghép cặp, McNemar p < 0,05 ở `strict_ok`, mới được giữ.** Tập test chỉ đo một lần cho mô hình cuối của mỗi mốc.

1. **Dữ liệu (giữ từ ADR 0003):** RAFT 70% có 1 oracle + 2 distractor, 30% toàn distractor dạy từ chối; chia `80/10/10` theo nhóm kịch bản; `verification_reasoning` là khóa đầu của JSON; kiểm tra rò rỉ train/test trước khi train.
2. **SFT gốc:** QLoRA 4-bit NF4 + double quant; **LoRA thường** `r=16, α=32` trên cả 7 lớp chiếu (`q,k,v,o,gate,up,down`); loss chỉ trên phần trả lời; `max_length=1280`; dò LR `{1e-4, 2e-4, 4e-4}` trên val. Unsloth là đường chính; fallback `AutoModelForCausalLM + PEFT` (Liger Kernel chỉ dùng ở nhánh fallback).
3. **Ablation SFT, mỗi lần đổi một yếu tố:** DoRA (`use_dora=True`); rsLoRA với α chỉnh để giữ nguyên hệ số scale (`r=16, α=8` → 8/√16 = 2); NEFTune `alpha=5`.
4. **Căn chỉnh sở thích:** đường chính **SFT → DPO** (`β=0.1`, LR `2e-5`, cặp *rejected* lấy on-policy từ lỗi thật của mô hình, ưu tiên bịa mã văn bản và bỏ sót hết hạn). Ablation: **ORPO một giai đoạn từ base** (`β=0.1`, LR `8e-5`). Chọn theo val.
5. **GRPO từ mô hình thắng ở bước 4:** `loss_type="dapo"` (mặc định TRL 1.15; ablation `dr_grpo`), `beta=0.0` (không nạp mô hình tham chiếu), `num_generations=4`, LR `5e-6` (thử thêm `1e-5`). Phần thưởng 2 chặng: (A) đúng định dạng + đúng `doc_code` theo địa bàn; (B) thêm phần thưởng từ chối và phạt bất đối xứng bỏ sót `expired_warning` (−2 so với −1 khi báo nhầm). Không dùng *dynamic sampling* của DAPO.
6. **Thử nghiệm tùy chọn — on-policy distillation:** `trl.DistillationTrainer` (`beta=1.0` = reverse KL), thầy Qwen3-8B 4-bit có ngữ cảnh, trò là mô hình sau bước 4. Chỉ chạy nếu vừa VRAM T4; so với GRPO ở cùng thời gian GPU ([On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)).
7. **Nén:** merge FP16 → GGUF `Q4_K_M` + imatrix hiệu chuẩn bằng tập chính sách và JSON chuẩn (giữ từ ADR 0003).
8. **Đánh giá 5 mốc:** `Baseline BM25+Base` → `SFT` → `DPO/ORPO` → `GRPO` → `Q4_K_M.gguf`. Chỉ số tất định + Ragas `Faithfulness`; bootstrap CI + McNemar; ngưỡng chốt trước giữ nguyên.
9. **Truy xuất:** trước khi chốt mô hình runtime ở ADR 0001, đo trên Zalo Legal (MRR@10, hit@k, độ trễ CPU, RAM): BM25 (`bm25s`), các mô hình của `fastembed`, `AITeamVN/Vietnamese_Embedding` + `AITeamVN/Vietnamese_Reranker`. Ứng viên phải xuất được ONNX và có giấy phép cho phép dùng; chọn theo số đo trong giới hạn RAM 2,5 GB.

## Hệ quả

- **Tốn nhiều lượt train hơn** (dò LR, 3 seed, ablation). Giảm bằng cách chạy ablation trên val với số epoch đã chốt, lưu checkpoint lên Drive để chạy tiếp được sau khi Colab ngắt.
- Mọi kỹ thuật trong pipeline cuối đều có số đo chứng minh; kỹ thuật không thắng thì bỏ.
- Có thêm một mốc đánh giá (tách căn chỉnh sở thích và GRPO) để biết công của từng bước.
