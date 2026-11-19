# Lộ trình Thực chiến Fine-tune `Qwen3-1.7B` — Đồ án P01 (SDG 1.3)

16 buổi, từ "chưa từng fine-tune" đến **tự tin đi phỏng vấn/đi làm** vị trí LLM / AI Engineer. Mọi buổi đều:
- **Thực hành trên `Qwen/Qwen3-1.7B`** (Google Colab T4 miễn phí). Hai buổi dùng thêm mô hình cùng họ Qwen3 cho tầng truy xuất (`Qwen3-Embedding-0.6B`, `Qwen3-Reranker-0.6B`).
- Có **sản phẩm chạy được** (adapter, pipeline, file GGUF, báo cáo) lưu trên Google Drive.
- **So sánh hiệu quả trên cùng 100 câu test giữ riêng**, cùng prompt, cùng bộ chấm. Kết quả ghi vào một **bảng điểm xuyên suốt** (`leaderboard.csv`), kèm khoảng tin cậy và kiểm định McNemar.
- Có **câu hỏi phỏng vấn kèm đáp án**, **bài tập** bắt buộc/nâng cao và **checklist**.
- Có mục **🏢 Ở công ty thật** chỉ ra bài lab đã đơn giản hóa những gì so với dự án ở doanh nghiệp (dữ liệu thật, job có cấu hình, registry, CI, phân quyền, ký duyệt nhiều bên).

> ⚠️ **Dữ liệu GIẢ LẬP — không dùng để tra cứu thủ tục thật.** Toàn bộ văn bản chính sách trong [`data/policies.json`](data/policies.json) là **hư cấu** phục vụ học tập. Mã văn bản (ví dụ `NQ-01/2026/HĐND-HCM`), điều khoản, mức tiền, giấy tờ và nơi nộp đều do nhóm tự đặt. Tên địa bàn (TP.HCM, Phường Tân Tạo, Quận Liên Chiểu…) có thật nhưng **không có văn bản nào ở đây do cơ quan nhà nước ban hành**. Mô hình huấn luyện theo lộ trình này chỉ "biết" các văn bản hư cấu đó; muốn biết quyền lợi thật, hãy hỏi cán bộ UBND/BHXH nơi cư trú.

## Hai giai đoạn sư phạm
| Giai đoạn | Buổi | Cách viết code |
|---|---|---|
| 🔬 **Mở hộp đen** | 00 → 04 | Ô siêu nhỏ (≤ 10–12 dòng). Mỗi ô theo khung *Mục tiêu → Code → Giải thích → 📌 Kết quả nên thấy → ⚠️ Lỗi hay gặp*. Tự viết masking, LoRA, vòng train trước khi dùng thư viện cấp cao |
| 🧩 **Thực chiến** | 05 → 15 | Khối gộp theo chuẩn kỹ sư. Trọng tâm chuyển sang thiết kế thí nghiệm, đọc log, ra quyết định |

## Bản đồ 16 buổi
| Buổi | File | Kỹ thuật trọng tâm | 🎯 Sản phẩm | Phép so sánh chính |
|---|---|---|---|---|
| 00 | [module-00](module-00/module-00-colab-qwen3-anatomy-baseline.md) ([HTML](module-00/module-00-colab-qwen3-anatomy-baseline.html), [sơ đồ](module-00/module-00-diagram.html)) | Giải phẫu Qwen3, token tiếng Việt, NF4, thinking on/off, sinh theo lô | Máy X-quang token, bộ sinh theo lô, **baseline** | VRAM ước tính vs thực đo; batch 1 vs 8 |
| 01 | [module-01](module-01/module-01-dataset-evaluator-prompt-baseline.md) | Dữ liệu RAFT sinh bằng luật, chia nhóm, kiểm toán rò rỉ, bộ chấm 8 chỉ số, bootstrap, McNemar | Báo cáo kiểm toán dữ liệu, baseline few-shot | Zero-shot vs 2-shot vs 3-shot |
| 02 | [module-02](module-02/module-02-first-lora-sft-manual-masking.md) | **Loss masking tự viết**, LoRA tính tay 17.432.576 tham số, QLoRA, HF `Trainer` | Adapter `m02/lora_r16` | Base vs SFT (McNemar) |
| 03 | [module-03](module-03/module-03-dora-rslora-controlled-ablation.md) | DoRA, rsLoRA, **ablation có kiểm soát** | Báo cáo ablation 4 biến thể | LoRA vs DoRA vs LoRA-r64 vs rsLoRA-r64 |
| 04 | [module-04](module-04/module-04-unsloth-trl-neftune-fairness.md) | Unsloth, TRL `SFTTrainer`, bẫy `max_length=1024`, NEFTune, W&B | Adapter Unsloth, bảng tốc độ | HF vs Unsloth; **Fairness 2×2** |
| 05 | [module-05](module-05/module-05-raft-distractors-refusal-ablation.md) | RAFT: văn bản nhiễu khó, học từ chối, ngân sách bước cố định | Ma trận 6 loại lỗi | Oracle-only vs RAFT-0% vs RAFT-30% |
| 06 | [module-06](module-06/module-06-hyperparams-debugging-tracking.md) | Quét LR trên **val**, early stopping đúng, lab OOM/NaN/phân kỳ/overfit | **`m06/sft_best`** + sổ tay debug | 3 LR; test đo đúng 1 lần |
| 07 | [module-07](module-07/module-07-synthetic-data-factory.md) | Dữ liệu tổng hợp bằng chính Qwen3, phễu lọc, dedup, data card | `m07/sft_aug` + data card | sft_best vs sft_aug theo lát cắt |
| 08 | [module-08](module-08/module-08-preference-dpo-vs-orpo.md) | Khai thác lỗi **on-policy**, DPO (adapter `ref`), ORPO | `m08/sft_dpo`, `m08/orpo_from_base` | SFT vs SFT→DPO vs ORPO |
| 09 | [module-09](module-09/module-09-grpo-verifiable-rewards.md) | GRPO, 3 phần thưởng kiểm chứng, ràng buộc batch, reward hacking | `m09/grpo` | expired_recall / false_expired trước–sau |
| 10 | [module-10](module-10/module-10-qwen3-thinking-mode.md) | Hybrid thinking, sụp đổ suy luận, SFT trộn 2 chế độ, định tuyến | `m10/mixed_think` | think vs no-think: chính xác – token – độ trễ |
| 11 | [module-11](module-11/module-11-function-calling-agent.md) | Function calling Qwen3, masking nhiều lượt, agent end-to-end | `m11/agent` | Từ khóa (0,885) vs base vs agent |
| 12 | [module-12](module-12/module-12-retrieval-embedding-reranker.md) | BM25, Qwen3-Embedding + LoRA (MNRL, NO_DUPLICATES), Qwen3-Reranker, RAG đầu-cuối | `m12/qwen3_emb_lora` | 5 retriever; ngữ cảnh dựng sẵn vs truy xuất thật |
| 13 | [module-13](module-13/module-13-merge-gguf-imatrix-serving.md) | Merge FP16, llama.cpp `b12084`, **imatrix thật**, JSON Schema decoding, TTFT CPU | GGUF + `quant_report.json` | F16/Q8/Q4/Q3 ± imatrix; schema vs tự do |
| 14 | [module-14](module-14/module-14-capstone-eval-guardrails-release.md) | Sampling stability, guardrail 4 tầng, red-team, Global-MMLU vi, hiệu chuẩn LLM-judge | `release_dossier.json`, `MODEL_CARD.md` | Bảng tổng kết 7 mốc |
| 15 | [module-15](module-15/module-15-production-ops-monitoring-feedback.md) | **Vận hành sau phát hành**: API FastAPI, cổng champion/challenger, kiểm thử tải p95 & số replica, drift (PSI), nhãn trễ + Cohen's κ, chaos test KillSwitch, điều kiện huấn luyện lại | API chạy thật + `ops_report.json` + runbook | Champion vs challenger (sau guardrail); tuần 1 vs tuần 2 |

## Chuỗi checkpoint (buổi sau nạp sản phẩm của buổi trước)
```
M00 base ──► M01 few-shot (prompt)        M04 unsloth_sft ──► M05 RAFT ablation
M02 lora_r16 ──► M03 ablation                                     │
                                         M06 sft_best ◄───────────┘
                                          ├─► M07 sft_aug (nếu thắng thì thay sft_best)
                                          ├─► M08 sft_dpo / orpo_from_base ──► M09 grpo ──► M13 GGUF ──► M14 release ──► M15 vận hành
                                          ├─► M10 mixed_think (nhánh nghiên cứu)
                                          └─► M12 RAG đầu-cuối (dùng m12/qwen3_emb_lora)
M11 agent: adapter độc lập từ base
```
Mỗi buổi **nạp base sạch + adapter cụ thể** (`p01kit.modeling.load_model(adapter=...)`) — không dùng lại biến mô hình của buổi trước, tránh "nhiễm adapter".

## Khởi động Colab (đầu mỗi buổi)
**Lần đầu mỗi runtime** — clone repo public, cài đúng phiên bản đã ghim ([requirements-colab.txt](requirements-colab.txt)), khởi động lại:
```bash
!git clone --depth 1 https://github.com/phamxuanphuc/uit-social-welfare-rag.git /content/uit-social-welfare-rag
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
from p01kit.colab import check_env; check_env()   # cảnh báo nếu sai bản ghim / torch ngoài khoảng / không có GPU
```
*Cách B (không dùng GitHub):* tải thư mục `edu/` lên `MyDrive/p01/edu`, rồi `sys.path.insert(0, "/content/drive/MyDrive/p01/edu")`.

## Học trên Colab T4 miễn phí
Bản miễn phí **không có hạn mức GPU cố định**, phiên có thời gian tối đa (Colab ghi khoảng 12 giờ, thực tế thường ngắn hơn), bị ngắt khi bạn không tương tác một thời gian, và **không chạy nền** khi đóng tab. Mỗi phiên mới là một máy trắng. Lộ trình được thiết kế để chịu được điều đó:

**1. Đừng phí hạn mức GPU**
- Học xong hoặc chuyển sang đọc lý thuyết/viết báo cáo → giải phóng GPU ngay: *Runtime → Disconnect and delete runtime*, hoặc chạy `from google.colab import runtime; runtime.unassign()`.
- Phần **không cần GPU** chạy trên runtime CPU (*Runtime → Change runtime type → CPU*): Buổi 01 Ô 1–8 (kho văn bản, chia tập, kiểm toán rò rỉ, bộ chấm, chấm lại output Buổi 00) chỉ cần clone repo ở Ô 0a, không cần cài thư viện; `check_env()` sẽ báo "không thấy GPU" — bỏ qua. Bộ `pytest` chạy trên máy cá nhân.
- Giữ tab mở trong lúc train; thỉnh thoảng xem log để phiên không bị coi là rảnh.

**2. Bị ngắt giữa buổi → chạy lại ô khởi động rồi chạy lại ô đang dở.** Những phần sau tự nối tiếp, không làm lại việc đã xong:

| Buổi | Cơ chế | Lưu ở `p01_runs/` trên Drive |
|---|---|---|
| 03, 05, 06 (quét LR) | Kết quả từng biến thể lưu ngay khi xong (`@resumable`); chạy lại chỉ train biến thể còn thiếu | `m03/results/`, `m05/results/`, `m06/results/` |
| 06 (`sft_best`), 09 (GRPO) | Checkpoint định kỳ, `resume_point` train tiếp từ checkpoint đầy đủ mới nhất | `m06/ckpt_sft_best/`, `m09/ckpt/` (giữ 1 bản) |
| 08 | Mẫu on-policy (khối sinh lâu nhất) được đọc lại thay vì sinh lại | `m08_samples.json` |
| 13 | Điểm từng biến thể GGUF được lưu; file GGUF nằm ở ổ tạm nên phải tạo lại bằng Khối 2–5 | `m13/results/` |
| 13, 15 | Bản build llama.cpp (2 vCPU biên dịch rất lâu) được nén lên Drive sau lần đầu, các phiên sau chỉ giải nén | `cache/llama.cpp-b12084-sm75-bin.tgz` |
| Mọi buổi | Bảng điểm **thay** dòng cũ của cùng (buổi, biến thể) thay vì ghi trùng | `leaderboard.csv` |

Muốn chạy lại có chủ đích (đã sửa code): truyền `force=True` cho hàm có `@resumable`, hoặc xóa file kết quả tương ứng. Đừng dùng nó để "đo lại test cho đẹp".

**3. Chia buổi dài thành 2 lần ngồi**
- **Buổi 08:** lần 1 đến hết Khối 2 (mẫu đã lưu); lần 2 chạy lại Khối 1–3 (Khối 2 đọc file, Khối 3 dựng cặp trong vài giây) rồi DPO/ORPO.
- **Buổi 09:** GRPO tự lưu checkpoint mỗi 16 bước — dừng lúc nào cũng được.
- **Buổi 13:** làm Khối 2–6 **trong một lần ngồi** (bản gộp FP16 và các file GGUF ~12 GB nằm ở ổ tạm, không vừa Drive miễn phí); chỉ file `Q4_K_M-imat` được chép lên Drive ở Khối 8.

**4. Dung lượng Drive (15 GB miễn phí):** toàn bộ lộ trình dùng cỡ vài GB (khoảng 20 adapter LoRA ~70 MB, 1 file GGUF ~1,1 GB, checkpoint giữ tối đa 1–2 bản, bản build llama.cpp). Kiểm tra Drive còn trống ≥ 5 GB trước Buổi 13.

**5. Chạy xong một buổi (hoặc cả khóa) → gom kết quả để đối chiếu:**
```python
from p01kit.colab import export_results; export_results()
```
File zip vài MB ở `p01_runs/export/` trên Drive gồm bảng điểm, **output từng câu** của mọi lượt, các báo cáo JSON (`selection.json`, `quant_report.json`, `release_dossier.json`, `ops_report.json`…) và phiên bản thư viện đã cài — **không** có adapter, checkpoint hay GGUF. Gửi file này cho người phụ trách khóa để các buổi cần GPU có bảng "📌 Kết quả đã đo" thật như Buổi 01 và 12, và để so kết quả của bạn với lượt chạy tham chiếu.

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
| `scoreboard` | Bảng điểm `leaderboard.csv` trên Drive, **kèm output thô từng câu** của mỗi lượt (`record(..., outputs=...)`) |
| `compare` | **So sánh trước → sau** hai lượt bất kỳ bằng cùng bộ chấm: Δ + CI, McNemar, lát cắt, loại lỗi, ví dụ câu được sửa/bị làm hỏng; `journey` cho cả khóa |
| `vntext` | Tiếng Việt thật: NFC/NFD, hai kiểu đặt dấu (hoà/hòa), bỏ dấu, tách từ (pyvi/underthesea) |
| `runs`, `colab` | Chạy lại an toàn sau khi Colab ngắt (`@resumable`, `resume_point`), kiểm tra môi trường, lưu bản build llama.cpp |
| `ops`, `service`, `traffic` | Cổng champion/challenger, p95 & số replica, PSI drift, KillSwitch, Cohen's κ · API FastAPI bọc guardrail + nhật ký kiểm toán · mô phỏng traffic/báo lỗi (Buổi 15) |

**Chạy kiểm thử trên máy cá nhân (CPU, không cần GPU)** — CI chạy đúng bộ này trên mỗi push/PR ([`.github/workflows/edu-tests.yml`](../.github/workflows/edu-tests.yml)):
```bash
python -m venv .venv && .venv/bin/pip install -r edu/requirements-dev.txt && .venv/bin/python -m pytest edu/tests -q
```
Bộ test kiểm tra cả **nội dung bài học**: mọi ô Python phải đúng cú pháp, không dùng tên chưa import/định nghĩa (ghép các ô theo thứ tự như một notebook, kiểm bằng `pyflakes`), không ghép token vào URL, không dùng API đã bị bỏ ở bộ thư viện đã ghim (`warmup_ratio`, `torch_dtype=`, `max_prompt_length`...), lệnh `pip` phải ghim phiên bản, `unsloth` phải được import trước `transformers`/`trl`/`peft`, mọi `from p01kit... import` phải tồn tại.

## Từ khóa học đến đi làm
Bảng dưới nối mỗi việc một kỹ sư LLM/AI hay được giao ở doanh nghiệp Việt Nam với buổi học tương ứng và **sản phẩm bạn đưa được vào hồ sơ năng lực** (GitHub/báo cáo/phỏng vấn).

| Việc thường được giao khi đi làm | Học ở buổi | Bằng chứng trong hồ sơ |
|---|---|---|
| Đánh giá mô hình gốc trước khi đề xuất fine-tune | 00, 01 | Dòng baseline + báo cáo `M00 → M01` có p-value |
| Chuẩn bị và kiểm toán dữ liệu (rò rỉ, trùng lặp, nhãn sai) | 01, 05, 07 | Báo cáo kiểm toán rò rỉ, data card, phát hiện lỗi metadata văn bản thật |
| Fine-tune LoRA/QLoRA và giải thích được từng tham số | 02, 03, 04 | Adapter + báo cáo trước → sau từng câu, ablation có McNemar |
| Debug lượt train hỏng (OOM, NaN, overfit) | 06 | Sổ tay debug 4 lỗi tự gây và tự sửa |
| Căn chỉnh hành vi an toàn (từ chối, cảnh báo hết hạn) | 05, 08, 09 | Ma trận lỗi trước/sau, phân tích reward hacking |
| Xây RAG trên văn bản tiếng Việt thật | 12 | Bảng BM25/dense trên Zalo Legal (NFC, không dấu, tách từ), cảnh báo rò rỉ mô hình |
| Agent gọi API nội bộ | 11 | Agent end-to-end so với baseline từ khóa |
| Đưa mô hình lên CPU/chi phí thấp | 13 | `quant_report.json`, số TTFT/token/s, báo cáo cái giá của lượng tử hóa |
| Phát hành có trách nhiệm (guardrail, red-team, hồ sơ) | 14 | `release_dossier.json`, `MODEL_CARD.md`, bảng hành trình toàn khóa |
| Vận hành sau phát hành (API, tải, drift, phản hồi) | 15 | `ops_report.json`, runbook, cổng champion/challenger |
| Viết code có test và CI | Mọi buổi | `p01kit` + `pytest` + GitHub Actions của repo |

**Hồ sơ tối thiểu nên có khi đi phỏng vấn:** (1) repo GitHub có `p01kit`, test và CI xanh; (2) báo cáo một trang `M00 → M14` (sinh bằng `compare_runs`) kèm 3 ví dụ câu được sửa và 1 ví dụ còn sai, giải thích vì sao; (3) bảng thực chiến Zalo Legal bạn tự tái lập; (4) trả lời trôi chảy ≥ 5 câu phỏng vấn của mỗi buổi. Nhà tuyển dụng tin **số liệu bạn tự đo và giải thích được** hơn danh sách công nghệ.

## Quy ước ký hiệu trong bài
- 📌 **Kết quả nên thấy** — con số ghi "(tất định)" hoặc "(đã đo)" là số chính xác đã kiểm chứng ngoài GPU (thống kê dữ liệu, số tham số, độ dài token, BM25). Các chỉ số mô hình sau khi train **phụ thuộc lượt chạy của bạn**, nên bài chỉ nêu dấu hiệu/kiểm tra hợp lý, không bịa số.
- 📌 **Kết quả đã đo** — bảng số do chính đoạn code trong bài sinh ra trên dữ liệu thật (BM25 trên Zalo Legal, thống kê hiệu lực văn bản); chạy lại phải ra đúng như vậy.
- 🎯 **Mục tiêu buổi học** + ✅ **Tiêu chí hoàn thành** ở đầu mỗi buổi — đạt tiêu chí mới sang buổi sau. 🧒 **Giải thích kiểu Feynman** mở đầu phần lý thuyết: một phút, không công thức.
- 📊 **Trước → Sau** — ô `compare_runs` ở cuối mỗi buổi có thay đổi mô hình: so từng câu với mốc trước bằng cùng bộ chấm.
- ⚠️ Lỗi hay gặp · 🧠 Điều cần ghi nhớ · 🎤 Hay bị hỏi khi phỏng vấn.

## Tài nguyên Hugging Face dùng trong lộ trình (đã kiểm tra tồn tại & giấy phép)
| Dùng ở | Tài nguyên | Giấy phép |
|---|---|---|
| Mọi buổi | `Qwen/Qwen3-1.7B` | Apache-2.0 |
| Buổi 04+ | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (tùy chọn) | Apache-2.0 |
| Buổi 12 | `Qwen/Qwen3-Embedding-0.6B`, `Qwen/Qwen3-Reranker-0.6B`, `AITeamVN/Vietnamese_Embedding`, `Qdrant/Qwen3-Embedding-0.6B-onnx` | Xem model card |
| Buổi 12 (Khối 8) | `GreenNode/zalo-ai-legal-text-retrieval-vn` — 61.425 điều luật, 788 câu hỏi thật | MIT |
| Buổi 12 (Khối 8) | `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT, cần tách từ; đã train trên Zalo 2021) | Apache-2.0 |
| Buổi 01 (Ô 8b–8c) | `th1nhng0/vietnamese-legal-documents` — metadata hiệu lực 171.556 văn bản + quan hệ thay thế | CC-BY-4.0 |
| Buổi 14 | `CohereLabs/Global-MMLU` (config `vi`) | Apache-2.0 |
| Buổi 11 (bài tập) | `phamhai/Vietnamese-Function-Calling-Test` | Apache-2.0 |
| Mở rộng | `duyet/vietnamese-legal-instruct` (222 nghìn hội thoại hỏi–đáp pháp luật) | CC-BY-4.0 |

## Thời lượng & phần cứng
~59 giờ học (16 buổi × 3–4 giờ). Lượt train dài nhất: GRPO (Buổi 09) và biên dịch llama.cpp (Buổi 13). Tất cả chạy trên Colab T4 15 GB; nếu có GPU Ampere+ (L4/A100), đổi `fp16=True` thành `bf16=True`.

## Giấy phép
Code (`p01kit`, test, CI) theo **MIT** — [LICENSE](../LICENSE). Bài học, tài liệu và dữ liệu giả lập theo **CC BY 4.0** — [LICENSE-CONTENT.md](../LICENSE-CONTENT.md), kèm danh sách phần không thuộc giấy phép (báo cáo SDG của Liên Hợp Quốc, ghi chú bài giảng, mô hình/dữ liệu bên thứ ba tải lúc chạy).
