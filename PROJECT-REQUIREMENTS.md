# Yêu cầu Dự án: Trợ lý Tra cứu Quyền lợi An sinh (SDG 1.3 - P01)

- **Môn học:** Tư duy Trí tuệ nhân tạo (HK1 · 2026–2027)
- **Đích SDG:** `1.3` — Mở rộng độ phủ an sinh xã hội cho người nghèo và nhóm dễ bị tổn thương *(theo [Extended-Report-2026_Goal-1.pdf](document/Extended-Report-2026_Goal-1.pdf), tỷ lệ phủ toàn cầu năm 2023 mới đạt 52.4%, trợ cấp thất nghiệp chỉ 16.7%)*.
- **Tài liệu kiến trúc liên quan:** [CONTEXT.md](CONTEXT.md), [ADR 0001](docs/adr/0001-cpu-only-gguf-onnx-runtime.md), [ADR 0002](docs/adr/0002-dual-mode-rag-with-hard-safety-guardrails.md), [ADR 0004](docs/adr/0004-evidence-driven-finetune-pipeline.md) (thay ADR 0003), [Đề cương Fine-tune Thực chiến](document/qwen3-finetune-master-curriculum.md).
- **Mã nguồn mở tham chiếu:** `thanhdo881/qwen3-1.7b-vivu-travel-vn`, `duyet/vietnamese-legal-instruct`, `fisherman611/vietnamese-legal-chatbot`, `tangbamiinh/traffic-law-assistant`, `sameerahmed-cloud/verity-rag` (Grounding Check `[Context N]`), `Aryanupadhyay23/conversational-research-rag` (Qdrant keep-alive & nhãn *Grounded vs. Unverified*), `AcadifySolution/llm-security-proxy`, `lamhotsiagian/guardrail-system-rag` (4-layer Guardrail & `pytest` gate), `farhan-syah/unsloth-finetuning` (4 pha `train -> build -> benchmark -> push`), `anaalkmim/rag-eval` (Qdrant Payload filtering & unanswerable eval).

---

## 1. Kiến trúc Kỹ thuật & Hạ tầng $0

- **Backend & AI Runtime (Azure Container Apps — 2 CPU / 4 GiB RAM, Scale-to-Zero):** Python 3.11, FastAPI, Uvicorn, Pydantic v2, Loguru (Structured JSON Logging); `llama-cpp-python` (`Qwen3-1.7B-Q4_K_M.gguf` trên CPU), `fastembed` (ONNX Dense + BM25), `FlashRank` (ONNX Reranker). Mô hình embedding/reranker chốt sau khi đo trên Zalo Legal (MRR@10, độ trễ CPU, RAM) giữa BM25, `fastembed` và `AITeamVN/Vietnamese_Embedding` + `Vietnamese_Reranker` (bản ONNX, giấy phép phù hợp). Tuyệt đối không cài PyTorch/TensorFlow trong container thực thi để giữ RAM < 2.5 GB.
- **Vector DB & Audit Storage:** Qdrant Cloud (Free Tier) lọc kép `region IN [selected_region, "Toàn quốc"]`; SQLite/PostgreSQL + SQLAlchemy lưu `audit_logs`, `delayed_labels` (gắn `trace_id`) và `policy_approval_logs` (giữ file seed `data/mock_policies.json` bất biến).
- **Frontend:** 2 chế độ — Giao diện Người dân (chọn địa bàn, chat, thẻ trích dẫn, báo lỗi) và Bảng điều khiển Cán bộ (đổi trạng thái hiệu lực văn bản qua `PATCH /api/admin/policies/{doc_code}/status` xác thực `X-API-Key` + `X-Role: Officer`).
- **Dual-Mode & MLOps CI/CD:** Tự động chạy Local Hybrid Engine khi CI/test chưa nạp `.gguf`/Qdrant; GitHub Actions chạy `pytest` Quality Gate, build Docker `python:3.11-slim` đẩy lên GHCR, deploy Azure Container Apps kèm Revision Rollback, lưu **Release Dossier** và cảnh báo **Data/Concept Drift** kích hoạt Continuous Training (CT).

---

## 2. Pipeline Fine-tuning Qwen3-1.7B (Google Colab T4)

- **Dữ liệu (`data/mock_policies.json` + PDF SDG 1.3 + kho văn bản thật [`data/real/`](data/real/DATA_CARD.md): 2.937 văn bản an sinh từ vbpl.vn, có địa bàn và trạng thái hiệu lực):** Chia tập `80/10/10` theo nhóm kịch bản gốc (`Scenario-Group Split`) tránh Data Leakage; chuẩn bị dữ liệu RAFT (70% có 1 Oracle + 2 Distractors, 30% toàn Distractors dạy từ chối) với `verification_reasoning` làm key đầu tiên của JSON Schema.
- **Huấn luyện & Căn chỉnh:** Log lên `wandb`; mỗi kỹ thuật chỉ được giữ khi thắng cấu hình gốc trên val (3 seed ghép cặp, McNemar). SFT gốc: QLoRA 4-bit NF4 + **LoRA thường** `r=16, α=32` trên cả 7 lớp chiếu, dò LR `{1e-4, 2e-4, 4e-4}`, Unsloth (fallback `PEFT`); DoRA, rsLoRA (chỉnh α giữ nguyên hệ số scale) và NEFTune (`alpha=5`) là ablation. Căn chỉnh: `SFT → DPO` (`2e-5`, cặp rejected on-policy) là đường chính, `ORPO` một giai đoạn từ base (`8e-5`) là ablation; sau đó `GRPO` (`5e-6`, `loss_type="dapo"`, `beta=0`, `num_generations=4`) với phần thưởng 2 chặng: định dạng + đúng `doc_code` trước, rồi thêm phần thưởng từ chối và phạt gấp đôi lỗi bỏ sót `expired_warning`. Tùy chọn: on-policy distillation từ Qwen3-8B nếu vừa VRAM T4.
- **Đánh giá & Nén GGUF:** Đo qua 5 mốc (`Baseline` → `SFT` → `DPO/ORPO` → `GRPO` → `Q4_K_M.gguf + imatrix`) bằng Deterministic Metrics + `Ragas` (hỗ trợ Local/Free Judge) kèm kiểm tra độ ổn định qua 3 lần chạy.

---

## 3. Tiêu chuẩn 7 Trục Responsible AI & Ngưỡng Chốt Trước

- **Baseline đối chiếu:** BM25 thuần + `Qwen3-1.7B Base` (chưa chuẩn hóa từ đời thường).
- **Ngưỡng đạt chốt trước (Pre-defined Thresholds):**
  - `Faithfulness >= 0.85` · `Citation Precision >= 0.90` · `Refusal Accuracy >= 0.90` · `Expired Detection = 100%` · `Fairness Gap (Chuẩn vs. Đời thường) <= 10%`.
- **Quyền riêng tư:** Che CCCD (9–12 số) và SĐT di động kể cả khi viết cách bằng khoảng trắng/dấu chấm/gạch ngang thành `[ĐÃ_CHE_ĐỊNH_DANH]` trước khi ghi log/DB (không che nhầm số tiền hay hotline cố định); không gửi dữ liệu thô cho LLM bên thứ 3; hỗ trợ xóa theo `trace_id` / TTL 30 ngày; ghi rõ giới hạn không nhận diện được danh tính mô tả tự do.
- **Bảo mật & Chống chịu:** Từ chối ngay khi thiếu `region` (không gọi LLM); phát hiện lệch địa bàn chéo (**CrossRegionConflict**); khóa cứng `expired_warning` và `required_documents` từ metadata gốc chống Prompt Injection; ẩn `***REDACTED***` mọi API key trong log; tự động fallback về kho Local khi Qdrant mất kết nối.
- **An toàn & Giải thích 3 lớp:** Hiển thị (1) từ đời thường đã quy đổi, (2) thẻ trích dẫn nguyên văn (`doc_code`, `article`, `effective_date`, `version`), (3) nhãn mức độ chắc chắn `Cao / Trung bình / Thấp - Cần hỏi cán bộ` kèm **ResidualRisk** (chỉ hỗ trợ chuẩn bị hồ sơ, không thay thế quyết định hành chính). Kích hoạt **KillSwitch** tự động/bởi Officer khi lọt văn bản hết hạn `> 0%` hoặc tỷ lệ báo lỗi nhãn trễ `> 15%`.

---

## 4. Khung Quản trị NIST AI RMF & Bảng Rủi ro Ưu tiên

- **Phân vai (`GOVERN`):** `Data Steward` (dữ liệu & chống rò rỉ), `Evaluation & QA Lead` (Release Dossier & giám sát Drift), `Policy Approver & Safety Officer` (duyệt hiệu lực & nắm quyền `KillSwitch`).

| Tình huống | Bên bị ảnh hưởng | Khả năng & Hậu quả | Điều kiện đạt | Kết quả trước sửa | Biện pháp (`MANAGE`) | Kết quả thử lại | Người phụ trách & Quyết định |
|---|---|---|---|---|---|---|---|
| **R1:** Văn bản hết hiệu lực (`expired`) hoặc LLM bịa giấy tờ | Người lao động & cán bộ tiếp nhận | Cao · Bị trả hồ sơ, trễ đợt chi trả, mất niềm tin | Expired Detection = 100%; Refusal Acc >= 0.90 | Baseline dễ dùng nguồn cũ hoặc bịa thủ tục | Khóa cứng metadata gốc + RAFT 30% Refusal + GRPO 2 chặng phạt gấp đôi FN | Chờ đo sau khi chạy Notebook 02 & `pytest` | Safety Officer · Tiếp tục demo nếu đạt 100% cảnh báo |
| **R2:** Câu hỏi đời thường (xe ôm, vé số, phụ hồ) hiểu sai ý | Lao động phi chính thức ít tiếp xúc văn bản luật | Trung bình–Cao · Truy xuất sót quyền lợi hợp lệ | Fairness Gap (Chuẩn vs. Đời thường) <= 10% | BM25 thuần kém ở câu đời thường | Bảng `colloquial_mappings` + Hybrid Dense/BM25 + SFT LoRA (NEFTune nếu thắng ablation) | Chờ đo trên cặp câu hỏi song song tập Test | Evaluation & QA Lead · Mở rộng từ điển nếu lệch > 10% |
| **R3:** Nhập CCCD/SĐT vào chat hoặc chọn nhầm địa bàn chéo | Người dân gửi dữ liệu cá nhân | Trung bình · Lộ PII vào log hoặc nộp sai nơi tiếp nhận | 0% PII thô trong log/DB; 100% bắt lệch địa bàn | Prompt thô ghi nguyên văn vào log | Regex PII mở rộng + Check `CrossRegionConflict` + xóa theo `trace_id` | Chờ xác thực qua `pytest tests/test_ai.py` | Data Steward · Bắt buộc pass 100% Gate mới deploy |
