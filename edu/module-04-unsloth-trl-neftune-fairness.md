# Buổi 04 — Tăng tốc với `Unsloth` + `TRL SFTTrainer`, `NEFTune` & Ablation Fairness 2×2

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (50' lý thuyết · 2h40 lab) |
| **Đầu vào** | Số liệu M02 (thời gian, VRAM, chỉ số); cấu hình adapter đã chốt ở M03 (mặc định LoRA r16 α32) |
| **🎯 Sản phẩm** | (1) Adapter **`m04/unsloth_sft`** & **`m04/unsloth_neft5`**; (2) **Bảng tốc độ** HF-PEFT vs Unsloth; (3) **Bảng Fairness 2×2** (NEFTune × mở rộng truy vấn đời thường); (4) Dashboard **Weights & Biases** của các lượt train |
| **Chỉ số đo** | giây/bước, VRAM đỉnh · `acc_formal`, `acc_colloquial`, `fairness_gap` · McNemar |
| **Chế độ code** | 🔬 Ô nhỏ — **buổi cuối** của giai đoạn ô siêu nhỏ; từ Buổi 05 code được gộp thành khối lớn |

---

## 0. Bức tranh lớn

Ở công ty, GPU là tiền. Cùng một chất lượng, pipeline nhanh gấp đôi = chạy được gấp đôi số thí nghiệm. Buổi này bạn chuyển từ "tự viết mọi thứ" (Buổi 02) sang **công cụ chuẩn ngành** — nhưng với tư duy kiểm chứng: *nhanh hơn bao nhiêu, chất lượng có giữ nguyên không, và cái giá là gì?*

Đồng thời ta tấn công rủi ro **R2** trong bảng rủi ro đồ án: người lao động hỏi bằng ngôn ngữ đời thường ("té xe", "giấy chi", "đứa nhỏ") bị phục vụ kém hơn người hỏi bằng thuật ngữ chuẩn. Ngưỡng chốt trước: `fairness_gap ≤ 0,10`.

---

## 1. Lý thuyết

### 1.1. Unsloth tăng tốc bằng cách nào?
- **Kernel Triton viết tay** cho RoPE, RMSNorm, SwiGLU và phép nhân LoRA — gộp nhiều phép tính thành 1 lần đọc/ghi bộ nhớ.
- **Cross-entropy hợp nhất theo khối:** không bao giờ dựng nguyên tensor logits `batch × seq × 151.936` (với batch 4 × 1.200 token ở FP32 là **~2,9 GB** chỉ riêng logits!).
- **Gradient checkpointing kiểu "unsloth":** đẩy activations đã checkpoint sang RAM CPU bất đồng bộ.
- **Quy tắc bắt buộc:** `import unsloth` **trước** `transformers`/`trl`/`peft` (Unsloth vá các lớp lúc import). Đã import sai thứ tự → *Runtime → Restart session*.
- **Liger Kernel** là lựa chọn thay thế (cũng có fused linear cross-entropy, bật bằng `use_liger_kernel=True` trong `SFTConfig`). **Không bật cả hai** — Unsloth đã có kernel tương đương.

### 1.2. TRL `SFTTrainer` với định dạng prompt–completion
- Dataset gồm 2 cột chuỗi `prompt`, `completion` (ta tạo bằng `p01kit.prompts.to_prompt_completion`). TRL tự token hóa và tạo `completion_mask` → loss chỉ tính trên completion (`completion_only_loss=True`) — **đúng phép che nhãn bạn tự viết ở Buổi 02**.
- ⚠️ **Bẫy `max_length=1024`:** giá trị mặc định của `SFTConfig.max_length` là 1024. Mẫu P01 dài tới **1.181 token** → TRL **cắt phải**, tức cắt mất đuôi JSON và `<|im_end|>` mà **không báo lỗi**. Luôn đặt `max_length=p01kit.MAX_SEQ_LEN` (1280).

### 1.3. NEFTune — thêm nhiễu vào embedding khi train
- Mỗi bước, cộng nhiễu đều `U(−1, 1) × α/√(L·d)` vào embedding đầu vào (L = độ dài chuỗi, d = 2048). Chỉ khi **train**, tắt khi suy luận.
- Trực giác: mô hình không thể "học vẹt" chính xác chuỗi token của câu hỏi → buộc phải học **ý nghĩa** → khái quát tốt hơn với cách diễn đạt mới (đúng thứ ta cần cho câu đời thường).
- Bài báo gốc báo cáo cải thiện trên benchmark hội thoại; với tác vụ trích xuất có cấu trúc như P01, **chưa chắc** — vì vậy ta đo.

### 1.4. Thiết kế Fairness 2×2
| | Không mở rộng truy vấn | Mở rộng truy vấn (`ColloquialMapping`) |
|---|---|---|
| **SFT không NEFTune** | A | B |
| **SFT + NEFTune α=5** | C | D |

So A↔C cho biết tác dụng riêng của NEFTune; A↔B cho biết tác dụng riêng của mở rộng truy vấn; D cho biết cộng gộp. Bản cũ của giáo trình so "trước (không gì cả)" với "sau (cả hai)" → **không tách được nguyên nhân**.

---

## 2. Thực hành

### Ô 0 — Khởi động (lưu ý thứ tự import)
```python
import os, sys
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel          # ⚠️ PHẢI đứng trước transformers / trl / peft
import torch, time, json, p01kit
```

### Ô 1 (7 dòng) — Theo dõi thí nghiệm bằng Weights & Biases (tùy chọn)
```python
try:
    from google.colab import userdata
    os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")   # lưu trong Colab Secrets
    os.environ["WANDB_PROJECT"] = "p01-qwen3-1.7b"
    REPORT_TO = "wandb"
except Exception:
    REPORT_TO = "none"
print("report_to =", REPORT_TO)
```
- **Giải thích:** khóa API nằm trong Secrets, không bao giờ in ra hay ghi vào notebook. Không có khóa → vẫn chạy bình thường, chỉ không có dashboard.

### Ô 2 (8 dòng) — Nạp Qwen3-1.7B qua Unsloth
```python
model, tok = FastLanguageModel.from_pretrained(
    model_name=p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth")
model.print_trainable_parameters()
```
- 📌 **Kết quả nên thấy:** banner Unsloth ghi GPU Tesla T4, `Bfloat16 = FALSE`; `trainable params: 17,432,576` — **đúng bằng** Buổi 02.
- ⚠️ `lora_dropout=0` là điều kiện để Unsloth dùng kernel LoRA nhanh. Buổi 02 dùng 0,05 → đây là **yếu tố thứ hai** khác biệt khi so tốc độ; bài tập 1 yêu cầu bạn tách riêng nó.

### Ô 3 (6 dòng) — Dữ liệu prompt–completion
```python
from datasets import Dataset
from p01kit.data import load_split
from p01kit.prompts import to_prompt_completion
train, val, test = (load_split(s) for s in ("train", "val", "test"))
ds_train = Dataset.from_list([to_prompt_completion(tok, r) for r in train])
ds_val = Dataset.from_list([to_prompt_completion(tok, r) for r in val])
print(ds_train[0]["prompt"][-60:], "||", ds_train[0]["completion"][-40:])
```
- 📌 **Kết quả nên thấy:** prompt kết thúc bằng `...<|im_start|>assistant\n<think>\n\n</think>\n\n`, completion kết thúc bằng `...}<|im_end|>`.

### Ô 4 (11 dòng) — Cấu hình SFT (giữ siêu tham số như Buổi 02)
```python
from trl import SFTConfig, SFTTrainer
from p01kit.modeling import adapter_path
def sft_config(name, neftune=None):
    return SFTConfig(
        output_dir=str(adapter_path("m04", f"ckpt_{name}")), run_name=f"m04-{name}", seed=42,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, per_device_eval_batch_size=4,
        num_train_epochs=3, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05,
        optim="paged_adamw_8bit", fp16=True, bf16=False,
        max_length=p01kit.MAX_SEQ_LEN, completion_only_loss=True, packing=False,
        neftune_noise_alpha=neftune, logging_steps=5, eval_strategy="epoch", save_strategy="no",
        report_to=REPORT_TO)
```
- **Giải thích:** `packing=False` vì mẫu đã dài ~1.100 token (gần `max_length`) — packing chỉ có lợi khi mẫu ngắn.

### Ô 5 (8 dòng) — Kiểm tra mặt nạ do TRL tạo ra (đừng tin mù quáng!)
```python
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, eval_dataset=ds_val,
                     args=sft_config("unsloth_sft"))
ex = trainer.train_dataset[0]
n, k = len(ex["input_ids"]), sum(ex["completion_mask"])
print(f"{n} token, học {k} token ({k/n:.0%})")
print(tok.decode([t for t, m in zip(ex["input_ids"], ex["completion_mask"]) if m])[-80:])
```
- 📌 **Kết quả nên thấy:** tỉ lệ ~**20%** như Buổi 02; phần học kết thúc bằng `<|im_end|>`.
- ⚠️ Nếu báo `KeyError: 'completion_mask'`, in `trainer.train_dataset.column_names` — một số phiên bản Unsloth/TRL đổi tên cột (ví dụ `assistant_masks`). Đọc tên cột thật thay vì đoán là một kỹ năng debug quan trọng.
- 🧠 Thử đổi `max_length=1024` rồi chạy lại ô này với mẫu dài nhất: bạn sẽ thấy đuôi bị cắt — đó là bẫy mục 1.2.

### Ô 6 (6 dòng) — Train A (Unsloth, không NEFTune) & đo tốc độ
```python
torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter()
stats = trainer.train()
minutes = (time.perf_counter() - t0) / 60
peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print(f"{minutes:.1f} phút | {stats.metrics['train_runtime']/stats.global_step:.2f} s/bước | VRAM đỉnh {peak} GB")
model.save_pretrained(adapter_path("m04", "unsloth_sft")); tok.save_pretrained(adapter_path("m04", "unsloth_sft"))
```
- 📌 **Kết quả nên thấy:** cùng 51 bước như Buổi 02; s/bước và VRAM đỉnh **thấp hơn** M02. Ghi cả hai con số vào bảng mục 3.

### Ô 7 (8 dòng) — Đánh giá A trên test & ghi bảng điểm
```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
FastLanguageModel.for_inference(model)
resA = evaluate(model, tok, test, batch_size=8)
print(format_summary(resA["summary"]))
scoreboard.record("M04", "unsloth-sft", resA["summary"], latency_s=resA["summary"]["sec_per_row"], notes=f"{minutes:.1f}min peak{peak}GB")
```

### Ô 8 (9 dòng) — Train C (NEFTune α=5) trên base SẠCH
```python
del model, trainer; torch.cuda.empty_cache()
model, tok = FastLanguageModel.from_pretrained(model_name=p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, eval_dataset=ds_val, args=sft_config("unsloth_neft5", neftune=5))
trainer.train()
model.save_pretrained(adapter_path("m04", "unsloth_neft5")); FastLanguageModel.for_inference(model)
resC = evaluate(model, tok, test, batch_size=8)
scoreboard.record("M04", "unsloth-neft5", resC["summary"], latency_s=resC["summary"]["sec_per_row"])
```

### Ô 9 (10 dòng) — Bộ mở rộng truy vấn đời thường (`ColloquialMapping`)
```python
COLLOQUIAL_MAP = {
    "té xe": "tai nạn", "xe tông": "tai nạn", "bị thương": "tai nạn lao động",
    "đứa nhỏ": "con", "thẻ bảo hiểm": "thẻ BHYT", "nghỉ việc": "mất việc làm",
    "tiền phòng": "tiền thuê nhà trọ", "ngập": "thiên tai", "đóng học": "học phí",
    "giấy gì": "hồ sơ gồm giấy tờ gì", "giấy chi": "hồ sơ gồm giấy tờ gì", "nhiêu": "mức hỗ trợ bao nhiêu",
}
def expand(row):
    hits = [v for k, v in COLLOQUIAL_MAP.items() if k in row["question"].lower()]
    return {**row, "question": row["question"] + (f" (Thuật ngữ: {', '.join(dict.fromkeys(hits))})" if hits else "")}
test_x = [expand(r) for r in test]
print(sum(a["question"] != b["question"] for a, b in zip(test, test_x)), "câu được mở rộng")
```
- **Giải thích:** đây là phiên bản "luật" của bảng `colloquial_mappings` trong PROJECT-REQUIREMENTS. Ưu điểm: minh bạch, cán bộ sửa được. Nhược: phủ không hết cách nói.

### Ô 10 (9 dòng) — 📊 Ma trận Fairness 2×2
```python
from p01kit.modeling import free
resD = evaluate(model, tok, test_x, batch_size=8)                       # C + mở rộng
del model; free()
model, tok = FastLanguageModel.from_pretrained(model_name=str(adapter_path("m04", "unsloth_sft")),
                                               max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
FastLanguageModel.for_inference(model)
resB = evaluate(model, tok, test_x, batch_size=8)                       # A + mở rộng
for name, r in [("A sft", resA), ("B sft+map", resB), ("C neft", resC), ("D neft+map", resD)]:
    s = r["summary"]; print(f"{name:<12} formal={s['acc_formal']:.3f} colloq={s['acc_colloquial']:.3f} gap={s['fairness_gap']:.3f}")
```
- **Giải thích:** `FastLanguageModel.from_pretrained` nhận thẳng **thư mục adapter** — Unsloth tự đọc `adapter_config.json`, nạp base đúng phiên bản rồi gắn adapter.
- 📌 **Cách đọc:** so A↔C (NEFTune), A↔B (mở rộng), A↔D (kết hợp). Kiểm tra ý nghĩa bằng McNemar **trên riêng các câu đời thường**:
```python
from p01kit.stats import mcnemar_exact
col = [i for i, r in enumerate(test) if r["style"] == "colloquial"]
for name, r in [("NEFTune", resC), ("Map", resB), ("Cả hai", resD)]:
    print(name, mcnemar_exact([resA["scores"][i]["answer_ok"] for i in col], [r["scores"][i]["answer_ok"] for i in col]))
```

---

## 3. So sánh hiệu quả

**Bảng tốc độ (cùng 51 bước, cùng dữ liệu):**
| Pipeline | s/bước | Phút train | VRAM đỉnh | answer_acc | strict_acc |
|---|---|---|---|---|---|
| M02 · HF Trainer + PEFT (dropout 0,05) | | | | | |
| M04 · Unsloth + TRL (dropout 0) | | | | | |

**Bảng Fairness 2×2:**
| | Không mở rộng | Mở rộng truy vấn |
|---|---|---|
| SFT | A: gap = | B: gap = |
| SFT + NEFTune | C: gap = | D: gap = |

> 📌 **Kết luận cần viết:** Unsloth nhanh hơn bao nhiêu % với chất lượng có khác biệt thống kê không? Can thiệp nào giảm `fairness_gap` về ≤ 0,10? **Lưu ý:** tập test chỉ có 50 câu đời thường → khoảng tin cậy của `acc_colloquial` rất rộng; tránh kết luận quá mức.

---

## 🏢 Ở công ty thật

- **Chọn framework train** (Unsloth, TRL, Axolotl, LLaMA-Factory…) dựa trên phần cứng của cụm, khả năng chạy đa GPU, tần suất đổi API và rủi ro khóa phiên bản — tốc độ trên 1 GPU chỉ là một tiêu chí.
- **Theo dõi thí nghiệm tập trung** (W&B/MLflow) có phân quyền theo dự án; **không** ghi dữ liệu cá nhân hay output thô có PII vào log thí nghiệm.
- **Nhóm cần công bằng được định nghĩa cùng nghiệp vụ/pháp chế**, kèm mức chênh chấp nhận được. Ở P01 là giọng chuẩn vs đời thường; ngoài đời còn có vùng miền, người cao tuổi, người gõ không dấu (Buổi 15).

---

## 4. Bài tập

**Bắt buộc**
1. **Tách yếu tố dropout:** chạy lại Buổi 02 với `lora_dropout=0` (HF Trainer). Bây giờ chênh lệch tốc độ HF↔Unsloth chỉ còn do framework.
2. **Bẫy max_length:** đặt `max_length=1024`, đếm số mẫu train bị cắt mất `<|im_end|>` (gợi ý: kiểm tra token cuối của `input_ids` trong `trainer.train_dataset`). Train và đo `json_valid` — giải thích kết quả.
3. Mở dashboard W&B: so sánh đường `train/loss` và `eval/loss` của A và C. NEFTune làm train loss cao hơn hay thấp hơn? Vì sao đó **không** phải dấu hiệu xấu?

**Nâng cao**
4. Thay `p01kit.MODEL_ID` bằng `p01kit.UNSLOTH_MODEL_ID` (*dynamic 4-bit* của Unsloth: giữ một số lớp nhạy cảm ở 16-bit). So VRAM và chất lượng với NF4 thường.
5. Thử `neftune_noise_alpha ∈ {5, 10, 15}`. Vẽ `fairness_gap` theo α.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao phải import unsloth trước transformers?**
   A: Unsloth monkey-patch các lớp attention/MLP/norm và hàm loss khi được import; nếu transformers đã nạp các lớp gốc trước, bản vá không áp được hoặc áp không đầy đủ.
2. **Q: Fused / chunked cross-entropy tiết kiệm bộ nhớ thế nào?**
   A: Logits kích thước `batch × seq × vocab` (vocab 151.936) rất lớn; tính `lm_head` + cross-entropy theo từng khối và không lưu toàn bộ logits giúp giảm VRAM đáng kể, nhất là với vocab lớn như Qwen.
3. **Q: NEFTune hoạt động thế nào, khi nào nên dùng?**
   A: Cộng nhiễu đều có biên độ α/√(L·d) vào embedding khi train; có tác dụng điều chuẩn, giúp khái quát với cách diễn đạt mới. Dùng khi dữ liệu ít và đầu vào đa dạng; luôn kiểm chứng bằng val/test vì không phải tác vụ nào cũng lợi.
4. **Q: `completion_only_loss` trong TRL tương đương gì?**
   A: Tương đương che nhãn -100 cho prompt: TRL tạo `completion_mask` và chỉ tính loss trên token completion.
5. **Q: Làm sao đo và giảm thiên lệch giữa người hỏi chuẩn và đời thường?**
   A: Tạo cặp câu hỏi song song cùng ý định, đo chênh lệch độ chính xác (fairness gap) kèm CI; can thiệp bằng dữ liệu đa dạng giọng (Buổi 07), mở rộng truy vấn, tinh chỉnh retriever (Buổi 12); dùng thiết kế giai thừa để tách tác dụng từng can thiệp.
6. **Q: Giá trị mặc định nào của TRL có thể âm thầm làm hỏng dữ liệu tiếng Việt?**
   A: `max_length=1024` — tiếng Việt tốn token, prompt RAG dài → bị cắt mất phần đáp án và token kết thúc mà không báo lỗi.

---

## 6. Checklist cuối buổi
- [ ] Import đúng thứ tự; giải thích được 3 cơ chế tăng tốc của Unsloth.
- [ ] Tự kiểm tra `completion_mask` của TRL khớp với masking Buổi 02.
- [ ] Bảng tốc độ HF vs Unsloth và bảng Fairness 2×2 đã điền, có p-value.
- [ ] Adapter `m04/unsloth_sft`, `m04/unsloth_neft5` trên Drive.

## Tài liệu tham khảo
- Unsloth documentation — Qwen3 fine-tuning guide.
- Jain et al., *NEFTune: Noisy Embeddings Improve Instruction Finetuning* (2023).
- Hsu et al., *Liger Kernel: Efficient Triton Kernels for LLM Training* (2024).
- TRL documentation — `SFTTrainer`, dataset formats (prompt–completion).
