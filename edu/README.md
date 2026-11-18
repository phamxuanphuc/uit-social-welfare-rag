# Lộ trình Thực chiến Fine-tune `Qwen3-1.7B` — Đồ án P01 (SDG 1.3)

15 buổi, từ "chưa từng fine-tune" đến **tự tin đi phỏng vấn/đi làm** vị trí LLM / AI Engineer. Mọi buổi đều:
- **Thực hành trên `Qwen/Qwen3-1.7B`** (Google Colab T4 miễn phí). Hai buổi dùng thêm mô hình cùng họ Qwen3 cho tầng truy xuất (`Qwen3-Embedding-0.6B`, `Qwen3-Reranker-0.6B`).
- Có **sản phẩm chạy được** (adapter, pipeline, file GGUF, báo cáo) lưu trên Google Drive.
- **So sánh hiệu quả trên cùng 100 câu test giữ riêng**, cùng prompt, cùng bộ chấm. Kết quả ghi vào một **bảng điểm xuyên suốt** (`leaderboard.csv`), kèm khoảng tin cậy và kiểm định McNemar.
- Có **câu hỏi phỏng vấn kèm đáp án**, **bài tập** bắt buộc/nâng cao và **checklist**.

## Hai giai đoạn sư phạm
| Giai đoạn | Buổi | Cách viết code |
|---|---|---|
| 🔬 **Mở hộp đen** | 00 → 04 | Ô siêu nhỏ (≤ 10–12 dòng). Mỗi ô theo khung *Mục tiêu → Code → Giải thích → 📌 Kết quả nên thấy → ⚠️ Lỗi hay gặp*. Tự viết masking, LoRA, vòng train trước khi dùng thư viện cấp cao |
| 🧩 **Thực chiến** | 05 → 14 | Khối gộp theo chuẩn kỹ sư. Trọng tâm chuyển sang thiết kế thí nghiệm, đọc log, ra quyết định |

## Bản đồ 15 buổi
| Buổi | File | Kỹ thuật trọng tâm | 🎯 Sản phẩm | Phép so sánh chính |
|---|---|---|---|---|
| 00 | [module-00](module-00-colab-qwen3-anatomy-baseline.md) | Giải phẫu Qwen3, token tiếng Việt, NF4, thinking on/off, sinh theo lô | Máy X-quang token, bộ sinh theo lô, **baseline** | VRAM ước tính vs thực đo; batch 1 vs 8 |
| 01 | [module-01](module-01-dataset-evaluator-prompt-baseline.md) | Dữ liệu RAFT sinh bằng luật, chia nhóm, kiểm toán rò rỉ, bộ chấm 8 chỉ số, bootstrap, McNemar | Báo cáo kiểm toán dữ liệu, baseline few-shot | Zero-shot vs 2-shot vs 3-shot |
| 02 | [module-02](module-02-first-lora-sft-manual-masking.md) | **Loss masking tự viết**, LoRA tính tay 17.432.576 tham số, QLoRA, HF `Trainer` | Adapter `m02/lora_r16` | Base vs SFT (McNemar) |
| 03 | [module-03](module-03-dora-rslora-controlled-ablation.md) | DoRA, rsLoRA, **ablation có kiểm soát** | Báo cáo ablation 4 biến thể | LoRA vs DoRA vs LoRA-r64 vs rsLoRA-r64 |
| 04 | [module-04](module-04-unsloth-trl-neftune-fairness.md) | Unsloth, TRL `SFTTrainer`, bẫy `max_length=1024`, NEFTune, W&B | Adapter Unsloth, bảng tốc độ | HF vs Unsloth; **Fairness 2×2** |
| 05 | [module-05](module-05-raft-distractors-refusal-ablation.md) | RAFT: văn bản nhiễu khó, học từ chối, ngân sách bước cố định | Ma trận 6 loại lỗi | Oracle-only vs RAFT-0% vs RAFT-30% |
| 06 | [module-06](module-06-hyperparams-debugging-tracking.md) | Quét LR trên **val**, early stopping đúng, lab OOM/NaN/phân kỳ/overfit | **`m06/sft_best`** + sổ tay debug | 3 LR; test đo đúng 1 lần |
| 07 | [module-07](module-07-synthetic-data-factory.md) | Dữ liệu tổng hợp bằng chính Qwen3, phễu lọc, dedup, data card | `m07/sft_aug` + data card | sft_best vs sft_aug theo lát cắt |
| 08 | [module-08](module-08-preference-dpo-vs-orpo.md) | Khai thác lỗi **on-policy**, DPO (adapter `ref`), ORPO | `m08/sft_dpo`, `m08/orpo_from_base` | SFT vs SFT→DPO vs ORPO |
| 09 | [module-09](module-09-grpo-verifiable-rewards.md) | GRPO, 3 phần thưởng kiểm chứng, ràng buộc batch, reward hacking | `m09/grpo` | expired_recall / false_expired trước–sau |
| 10 | [module-10](module-10-qwen3-thinking-mode.md) | Hybrid thinking, sụp đổ suy luận, SFT trộn 2 chế độ, định tuyến | `m10/mixed_think` | think vs no-think: chính xác – token – độ trễ |
| 11 | [module-11](module-11-function-calling-agent.md) | Function calling Qwen3, masking nhiều lượt, agent end-to-end | `m11/agent` | Từ khóa (0,885) vs base vs agent |
| 12 | [module-12](module-12-retrieval-embedding-reranker.md) | BM25, Qwen3-Embedding + LoRA (MNRL, NO_DUPLICATES), Qwen3-Reranker, RAG đầu-cuối | `m12/qwen3_emb_lora` | 5 retriever; ngữ cảnh dựng sẵn vs truy xuất thật |
| 13 | [module-13](module-13-merge-gguf-imatrix-serving.md) | Merge FP16, llama.cpp `b12084`, **imatrix thật**, JSON Schema decoding, TTFT CPU | GGUF + `quant_report.json` | F16/Q8/Q4/Q3 ± imatrix; schema vs tự do |
| 14 | [module-14](module-14-capstone-eval-guardrails-release.md) | Sampling stability, guardrail 4 tầng, red-team, Global-MMLU vi, hiệu chuẩn LLM-judge | `release_dossier.json`, `MODEL_CARD.md` | Bảng tổng kết 7 mốc |

## Chuỗi checkpoint (buổi sau nạp sản phẩm của buổi trước)
```
M00 base ──► M01 few-shot (prompt)        M04 unsloth_sft ──► M05 RAFT ablation
M02 lora_r16 ──► M03 ablation                                     │
                                         M06 sft_best ◄───────────┘
                                          ├─► M07 sft_aug (nếu thắng thì thay sft_best)
                                          ├─► M08 sft_dpo / orpo_from_base ──► M09 grpo ──► M13 GGUF ──► M14 release
                                          ├─► M10 mixed_think (nhánh nghiên cứu)
                                          └─► M12 RAG đầu-cuối (dùng m12/qwen3_emb_lora)
M11 agent: adapter độc lập từ base
```
Mỗi buổi **nạp base sạch + adapter cụ thể** (`p01kit.modeling.load_model(adapter=...)`) — không dùng lại biến mô hình của buổi trước, tránh "nhiễm adapter".

## Khởi động Colab (đầu mỗi buổi)
**Lần đầu mỗi runtime** — clone repo, cài đúng phiên bản đã ghim ([requirements-colab.txt](requirements-colab.txt)), khởi động lại:
```python
import os, subprocess, base64
REPO, URL = "/content/uit-social-welfare-rag", "https://github.com/phamxuanphuc/uit-social-welfare-rag.git"
if not os.path.exists(REPO):
    env = dict(os.environ)
    try:  # repo private: thêm GITHUB_TOKEN vào Colab Secrets (🔑); repo public thì bỏ qua
        from google.colab import userdata
        basic = base64.b64encode(f"x-access-token:{userdata.get('GITHUB_TOKEN')}".encode()).decode()
        env |= {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}"}  # token chỉ nằm trong biến môi trường
    except Exception:
        pass
    r = subprocess.run(["git", "clone", "--depth", "1", URL, REPO], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"git clone lỗi:\n{r.stderr}"
```
```bash
!pip install -q -r /content/uit-social-welfare-rag/edu/requirements-colab.txt
```
```python
import os; os.kill(os.getpid(), 9)   # khởi động lại runtime
```
**Mỗi buổi:**
```python
import os, sys
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
```
*Cách B (không dùng GitHub):* tải thư mục `edu/` lên `MyDrive/p01/edu`, rồi `sys.path.insert(0, "/content/drive/MyDrive/p01/edu")`.

## Bộ dữ liệu P01 (tất định, sinh bằng `python -m p01kit.data`)
| Tập | Số câu | Nhóm kịch bản | Phải từ chối | Có văn bản hết hạn | Ghi chú |
|---|---|---|---|---|---|
| train | 268 | 46 | 33,2% | 6,7% | Không chứa 4 văn bản holdout (kể cả làm nhiễu) |
| val | 34 | 6 | 35,3% | 11,8% | Dùng chọn siêu tham số |
| test | 100 | 17 | 44% | 10% | 72 câu `unseen_doc`, 24 `seen_doc`, 4 `ood`; 50% câu đời thường |

- 23 văn bản chính sách **giả lập** ([data/policies.json](data/policies.json)), 11 chủ đề, 3 địa bàn + Toàn quốc, 5 văn bản hết hiệu lực.
- Mỗi câu có 3 đoạn ngữ cảnh; đáp án vàng JSON 8 khóa suy ra bằng luật `resolve()` → 0 nhãn mâu thuẫn.
- Độ dài (tokenizer Qwen3 thật): prompt ~836 token (tối đa 885), đáp án ~230 (tối đa 308) → `MAX_SEQ_LEN = 1280`.

## `p01kit` — bộ công cụ dùng chung (đã kiểm thử)
| Module | Vai trò |
|---|---|
| `data` | Kho văn bản, luật `resolve`, dựng tập RAFT, chia nhóm, holdout |
| `prompts` | **Nguồn sự thật duy nhất** cho prompt train/eval/deploy |
| `metrics`, `stats` | 8 chỉ số, cổng ngưỡng, bootstrap CI, McNemar |
| `sft` | Che nhãn prompt–completion và nhiều lượt (offset mapping), collator |
| `infer`, `modeling` | Sinh theo lô (đệm trái), nạp base/adapter chuẩn hóa, FP32 cho trọng số học |
| `rewards` | 3 hàm thưởng GRPO tương thích TRL |
| `guardrails`, `grounding` | Che PII (không che nhầm số tiền/hotline), lệch địa bàn, khóa metadata, nhật ký kiểm toán |
| `dedup`, `synthetic` | Khử trùng lặp n-gram ký tự, kiểm toán rò rỉ, lọc câu viết lại |
| `tools`, `retrieval`, `cot`, `llamacpp` | Agent/tool calling, BM25 + Recall/MRR, vết suy luận, client llama-server + JSON Schema |
| `scoreboard` | Bảng điểm `leaderboard.csv` trên Drive |

**Chạy kiểm thử trên máy cá nhân (CPU, không cần GPU):**
```bash
python -m venv .venv && .venv/bin/pip install -r edu/requirements-dev.txt && .venv/bin/python -m pytest edu/tests -q
```
Bộ test kiểm tra cả **nội dung bài học**: mọi ô Python phải đúng cú pháp, không dùng tên chưa import/định nghĩa (ghép các ô theo thứ tự như một notebook, kiểm bằng `pyflakes`), không ghép token vào URL, không dùng API đã bị bỏ ở bộ thư viện đã ghim (`warmup_ratio`, `torch_dtype=`, `max_prompt_length`...), lệnh `pip` phải ghim phiên bản, `unsloth` phải được import trước `transformers`/`trl`/`peft`, mọi `from p01kit... import` phải tồn tại.

## Quy ước ký hiệu trong bài
- 📌 **Kết quả nên thấy** — con số ghi "(tất định)" hoặc "(đã đo)" là số chính xác đã kiểm chứng ngoài GPU (thống kê dữ liệu, số tham số, độ dài token, BM25). Các chỉ số mô hình sau khi train **phụ thuộc lượt chạy của bạn**, nên bài chỉ nêu dấu hiệu/kiểm tra hợp lý, không bịa số.
- ⚠️ Lỗi hay gặp · 🧠 Điều cần ghi nhớ · 🎤 Hay bị hỏi khi phỏng vấn.

## Tài nguyên Hugging Face dùng trong lộ trình (đã kiểm tra tồn tại & giấy phép)
| Dùng ở | Tài nguyên | Giấy phép |
|---|---|---|
| Mọi buổi | `Qwen/Qwen3-1.7B` | Apache-2.0 |
| Buổi 04+ | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (tùy chọn) | Apache-2.0 |
| Buổi 12 | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, `AITeamVN/Vietnamese_Embedding`, `Qdrant/Qwen3-Embedding-0.6B-onnx` | Xem model card |
| Buổi 12 (bài tập) | `GreenNode/zalo-ai-legal-text-retrieval-vn` | MIT |
| Buổi 14 | `CohereLabs/Global-MMLU` (config `vi`) | Apache-2.0 |
| Buổi 11 (bài tập) | `phamhai/Vietnamese-Function-Calling-Test` | Apache-2.0 |
| Mở rộng | `duyet/vietnamese-legal-instruct`, `th1nhng0/vietnamese-legal-documents` | CC-BY-4.0 |

## Thời lượng & phần cứng
~55 giờ học (15 buổi × 3–4 giờ). Lượt train dài nhất: GRPO (Buổi 09) và biên dịch llama.cpp (Buổi 13). Tất cả chạy trên Colab T4 15 GB; nếu có GPU Ampere+ (L4/A100), đổi `fp16=True` thành `bf16=True`.
