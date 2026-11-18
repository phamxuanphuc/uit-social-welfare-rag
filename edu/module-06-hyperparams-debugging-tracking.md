# Buổi 06 — Siêu tham số, Debug OOM/NaN/Phân kỳ & Chọn mô hình đúng cách

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (1h lý thuyết · 3h lab) |
| **Đầu vào** | Công thức dữ liệu chuẩn từ Buổi 05 (RAFT-30%); hàm `train_sft` của Buổi 05 |
| **🎯 Sản phẩm** | (1) **Bảng quét LR** chọn trên **val**; (2) **Sổ tay debug** gồm 4 lỗi tự tay gây ra và sửa; (3) Adapter chuẩn **`m06/sft_best`** — mô hình nền cho mọi buổi sau |
| **Chỉ số đo** | `eval_loss` theo bước · `grad_norm` · token/giây · VRAM đỉnh · 8 chỉ số P01 trên test (chỉ đo **1 lần** cho mô hình cuối) |
| **Chế độ code** | 🧩 Khối gộp |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Quét learning rate và chọn trên **val**, không bao giờ trên test.
2. Tự gây rồi sửa OOM, NaN, phân kỳ, overfit — biết đọc log để chẩn đoán.
3. Train `sft_best`, đo test **đúng một lần**, có báo cáo `base → sft_best` cho đồ án.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): `selection.json` ghi LR/epoch đã chọn kèm toàn bộ kết quả quét; test chỉ đo một lần; báo cáo `M00 → M06` đã lưu vào báo cáo đồ án; bạn sửa được ít nhất 3 lỗi trong phòng lab debug.

---

## 0. Bức tranh lớn

Câu hỏi phỏng vấn kinh điển: *"Loss của bạn bị NaN ở bước 30, bạn làm gì?"*. Người chưa từng gặp sẽ đoán; người đã tự tay gây lỗi và sửa sẽ trả lời theo **quy trình**. Buổi này biến bạn thành người thứ hai.

Và một nguyên tắc mà nhiều người vi phạm mà không biết: **chọn siêu tham số trên tập val, báo cáo trên tập test đúng một lần**. Nếu bạn chọn LR "đẹp nhất trên test", con số test đó không còn là ước lượng không chệch nữa.

---

## 1. Lý thuyết

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** Learning rate là **độ dài bước chân khi xuống dốc trong sương mù**: bước dài quá thì trượt ngã (phân kỳ, NaN), ngắn quá thì mãi không tới. Tập val là **bài thi thử** — thi bao nhiêu lần cũng được; tập test là **thi thật** — chỉ thi một lần, nhìn điểm rồi sửa bài là gian lận.

### 1.1. Thứ tự ưu tiên khi tinh chỉnh (LoRA, dữ liệu vài trăm mẫu)
| Ưu tiên | Siêu tham số | Dải hợp lý | Dấu hiệu sai |
|---|---|---|---|
| 1 | `learning_rate` | 5e-5 … 5e-4 | Quá cao: loss dao động/tăng vọt, `grad_norm` lớn. Quá thấp: loss giảm rất chậm |
| 2 | Số bước / epoch | 1–5 epoch | `eval_loss` đi lên trong khi `train_loss` tiếp tục giảm → overfit |
| 3 | Batch hiệu dụng | 8–32 | Quá nhỏ: nhiễu; quá lớn: ít bước cập nhật |
| 4 | `r`, `lora_alpha` | r 8–64 | Xem Buổi 03 |
| 5 | Dropout, weight decay, NEFTune | — | Điều chuẩn, chỉ tinh chỉnh khi đã overfit |
| 6 | Scheduler, warmup | cosine, 3–10% | Thiếu warmup + LR cao → nổ loss những bước đầu |

### 1.2. Gradient accumulation — tương đương nhưng không miễn phí
- `per_device_train_batch_size=4, gradient_accumulation_steps=4` ≈ batch 16 về mặt toán học, nhưng chỉ phần **activations** giảm (tỉ lệ với batch thực trên GPU); trọng số, trạng thái optimizer **không đổi**.
- Bẫy lịch sử (sửa ở transformers 4.46, 10/2024): khi các micro-batch có số token học khác nhau, chia trung bình loss theo từng micro-batch làm sai trọng số → loss khi dùng accumulation **khác** loss batch lớn. Bản hiện tại chuẩn hóa theo **tổng số token có nhãn** của cả bước. Biết chi tiết này là điểm cộng lớn khi phỏng vấn.

### 1.3. Bản đồ lỗi → nguyên nhân → cách sửa
| Triệu chứng | Nguyên nhân hay gặp | Cách sửa |
|---|---|---|
| `CUDA out of memory` | batch × seq quá lớn, tắt checkpointing, logits vocab lớn | Giảm batch + tăng accumulation; bật gradient checkpointing; dùng fused CE (Unsloth/Liger) |
| `loss = nan` ngay từ đầu | Mẫu không còn token có nhãn (prompt bị cắt hết), FP16 tràn | Kiểm tra độ dài trước khi train; dùng `max_length` đủ lớn; giảm LR |
| `loss` = 0,0 đáng ngờ | Nhãn rò rỉ đáp án vào prompt, hoặc mọi nhãn bị che | Soi lại masking (Buổi 02 Ô 3) |
| `loss` tăng vọt, `grad_norm` lớn | LR quá cao, thiếu warmup | Giảm LR, thêm warmup, `max_grad_norm=1.0` |
| `eval_loss` tăng, `train_loss` giảm | Overfit | Dừng sớm, ít epoch hơn, thêm dữ liệu đa dạng (Buổi 07) |
| Output lặp vô hạn | Thiếu `<|im_end|>` trong dữ liệu | Thêm eos vào completion |

---

## 2. Thực hành

### Khối 1 — Khởi động (dùng lại `train_sft` của Buổi 05 nhưng chấm trên VAL)
```python
import os, sys, json, time, math
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from transformers import EarlyStoppingCallback
from p01kit.data import load_split
from p01kit.prompts import to_prompt_completion
from p01kit.infer import evaluate
from p01kit.modeling import adapter_path, free
from p01kit import scoreboard
from p01kit.runs import resumable, resume_point
train, val, test = (load_split(s) for s in ("train", "val", "test"))
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def fresh_model(gc="unsloth"):
    m, t = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
    m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
                                         target_modules=TARGETS, use_gradient_checkpointing=gc)
    return m, t

def make_cfg(name, **over):
    base = dict(output_dir=str(adapter_path("m06", f"ckpt_{name}")), run_name=f"m06-{name}", seed=42,
                per_device_train_batch_size=4, gradient_accumulation_steps=4, num_train_epochs=3,
                learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05, optim="paged_adamw_8bit",
                fp16=True, max_length=p01kit.MAX_SEQ_LEN, completion_only_loss=True, logging_steps=1,
                eval_strategy="steps", eval_steps=17, save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
    base.update(over)
    return SFTConfig(**base)

from p01kit.modeling import load_tokenizer
tok = load_tokenizer()                       # chỉ cần tokenizer để dựng dataset, không nạp trọng số
ds_train = Dataset.from_list([to_prompt_completion(tok, r) for r in train])
ds_val = Dataset.from_list([to_prompt_completion(tok, r) for r in val])
```

### Khối 2 — Quét learning rate, chọn trên VAL
```python
@resumable("m06")                  # mỗi LR xong là lưu lên Drive → Colab ngắt thì chỉ chạy lại LR còn thiếu
def sweep_one(name, lr):
    model, tok = fresh_model()
    tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, eval_dataset=ds_val,
                    args=make_cfg(name, learning_rate=lr))
    out = tr.train()
    hist = tr.state.log_history
    FastLanguageModel.for_inference(model)
    v = evaluate(model, tok, val, batch_size=8)["summary"]
    stats = {"eval_loss": [round(h["eval_loss"], 4) for h in hist if "eval_loss" in h],
             "max_grad_norm": round(max(h.get("grad_norm", 0) for h in hist), 2),
             "val_strict": v["strict_acc"], "val_answer": v["answer_acc"],
             "tok_per_s": round(sum(len(x) for x in tr.train_dataset["input_ids"]) * 3 / out.metrics["train_runtime"])}
    del model, tr; free()
    return stats

sweep = {}
for lr in (5e-5, 2e-4, 5e-4):
    name = f"lr{lr:.0e}"
    sweep[name] = sweep_one(name, lr)
    print(name, sweep[name])
```
- **Giải thích:** `eval_loss` ghi ở các bước 17/34/51 (≈ hết epoch 1/2/3) → vừa chọn LR vừa chọn số epoch. `val_strict` trên 34 câu chỉ là tín hiệu **thô**; kết hợp với `eval_loss` (mượt hơn vì tính trên mọi token).
- 📌 **Kết quả nên thấy:** `5e-5` thường có `eval_loss` cao nhất (học chậm); `5e-4` có thể có `max_grad_norm` lớn và `eval_loss` đi lên ở epoch 3 (overfit/không ổn định). Chọn cấu hình có `eval_loss` thấp nhất **và** `val_strict` không kém.

### Khối 3 — Vẽ đường học
```python
import matplotlib.pyplot as plt
for k, v in sweep.items():
    plt.plot([17, 34, 51][:len(v["eval_loss"])], v["eval_loss"], marker="o", label=k)
plt.xlabel("bước"); plt.ylabel("eval_loss (val)"); plt.legend(); plt.title("Quét LR — chọn trên VAL"); plt.show()
BEST_LR = min(sweep, key=lambda k: min(sweep[k]["eval_loss"]))
BEST_EPOCHS = 1 + sweep[BEST_LR]["eval_loss"].index(min(sweep[BEST_LR]["eval_loss"]))
print("Chọn:", BEST_LR, "| số epoch tốt nhất:", BEST_EPOCHS)
```

---

### 🧪 Phòng lab debug — tự gây lỗi, tự sửa

#### Lab 1 — OOM (hết bộ nhớ)
```python
model, tok = fresh_model(gc=False)                       # tắt gradient checkpointing
try:
    SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train,
               args=make_cfg("oom", per_device_train_batch_size=16, gradient_accumulation_steps=1, max_steps=2,
                             eval_strategy="no")).train()
except torch.OutOfMemoryError as e:
    print("💥 OOM:", str(e)[:160])
del model; free()
```
- **Sửa:** quay về `per_device_train_batch_size=4, gradient_accumulation_steps=4` và `use_gradient_checkpointing="unsloth"` — batch hiệu dụng vẫn 16.
- 📌 Nếu runtime bị treo sau OOM: *Runtime → Restart session* rồi chạy lại Khối 1. Ghi lại thông báo lỗi vào sổ tay.
- 🧠 Ước lượng nhanh: activations ∝ `batch × seq × hidden × số_lớp`; tắt checkpointing giữ activations của cả 28 lớp.

#### Lab 2 — `loss = nan` vì mất nhãn
```python
model, tok = fresh_model()
bad_cfg = make_cfg("nan", max_length=800, max_steps=5, eval_strategy="no")   # prompt ~840 token > 800!
n_prompt = [len(tok(x, add_special_tokens=False)["input_ids"]) for x in ds_train["prompt"]]
print("Số mẫu prompt dài hơn max_length:", sum(n >= 800 for n in n_prompt), "/", len(n_prompt))
tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, args=bad_cfg)
tr.train()
print([h.get("loss") for h in tr.state.log_history if "loss" in h])
del model, tr; free()
```
- 📌 **Kết quả nên thấy:** gần như **toàn bộ** mẫu có prompt dài hơn 800 → sau khi cắt không còn token completion → loss `nan` hoặc `0.0` (một số phiên bản báo lỗi ngay — cũng là một "triệu chứng" cần ghi vào sổ tay).
- **Sửa:** kiểm tra độ dài **trước** khi train (đúng như `p01kit.sft.encode_prompt_completion` làm: báo lỗi khi prompt ≥ max_length) và đặt `max_length=p01kit.MAX_SEQ_LEN`.

#### Lab 3 — Phân kỳ vì LR quá cao
```python
model, tok = fresh_model()
tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train,
                args=make_cfg("diverge", learning_rate=5e-3, warmup_steps=0, max_steps=20, eval_strategy="no"))
tr.train()
for h in tr.state.log_history:
    if "loss" in h:
        print(f"bước {h['step']:>2}  loss={h['loss']:.3f}  grad_norm={h.get('grad_norm', float('nan')):.2f}")
del model, tr; free()
```
- 📌 **Kết quả nên thấy:** loss giảm vài bước rồi **dao động mạnh hoặc tăng**, `grad_norm` vọt cao. So với đường của `2e-4` ở Khối 2.
- **Sửa theo thứ tự:** giảm LR → thêm warmup → kiểm tra `max_grad_norm` (mặc định 1,0 đã có clipping).

#### Lab 4 — Overfit & dừng sớm đúng cách
```python
model, tok = fresh_model()
tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, eval_dataset=ds_val,
                args=make_cfg("overfit", learning_rate=5e-4, num_train_epochs=6, save_strategy="steps", save_steps=17,
                              save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss"),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
tr.train()
print("Dừng ở bước:", tr.state.global_step, "| checkpoint tốt nhất:", tr.state.best_model_checkpoint)
del model, tr; free()
```
- 📌 **Kết quả nên thấy:** quá trình dừng **trước** bước 102 (6 epoch) khi `eval_loss` không cải thiện 2 lần đánh giá liên tiếp; mô hình được nạp lại từ checkpoint tốt nhất.
- ⚠️ `load_best_model_at_end` bắt buộc `save_strategy` khớp `eval_strategy` (cùng "steps", cùng chu kỳ). Bản cũ của giáo trình dùng **chính tập train làm tập val** — early stopping khi đó vô nghĩa.

---

### Khối 4 — 🎯 Train mô hình chuẩn `m06/sft_best` và đo test MỘT lần
```python
lr = float(BEST_LR.replace("lr", ""))
model, tok = fresh_model()
tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds_train, eval_dataset=ds_val,
                args=make_cfg("sft_best", learning_rate=lr, num_train_epochs=BEST_EPOCHS, logging_steps=5,
                              save_strategy="steps", save_steps=17, save_total_limit=1))   # checkpoint mỗi epoch lên Drive
tr.train(resume_from_checkpoint=resume_point(adapter_path("m06", "ckpt_sft_best")))     # None ở lần đầu
model.save_pretrained(adapter_path("m06", "sft_best")); tok.save_pretrained(adapter_path("m06", "sft_best"))
json.dump({"lr": lr, "epochs": BEST_EPOCHS, "sweep": sweep}, open(adapter_path("m06", "sft_best") / "selection.json", "w"), indent=2)
FastLanguageModel.for_inference(model)
res = evaluate(model, tok, test, batch_size=8)
scoreboard.record("M06", "sft_best", res["summary"], latency_s=res["summary"]["sec_per_row"], notes=f"lr={lr} ep={BEST_EPOCHS}",
                  outputs=res["outputs"])
json.dump(res["outputs"], open(f"{os.environ['P01_RUNS']}/m06_sft_best_outputs.json", "w"), ensure_ascii=False)
scoreboard.show()
```
- 🧠 `selection.json` lưu **lý do** chọn cấu hình cạnh adapter — sau 3 tháng bạn (hoặc người review) vẫn truy được vì sao mô hình này ra đời. Đây là một phần của **Release Dossier** (Buổi 14).

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M00", "base-zero-shot"), ("M06", "sft_best"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** Đây là phép so **chuẩn** của cả khóa: mô hình gốc vs mô hình fine-tune tốt nhất, đo test đúng một lần. Lưu bảng này vào báo cáo đồ án — các buổi sau đều so với `M06/sft_best`.


| Cấu hình | eval_loss (bước 17/34/51) | max grad_norm | val_strict | token/s |
|---|---|---|---|---|
| lr 5e-5 | | | | |
| lr 2e-4 | | | | |
| lr 5e-4 | | | | |
| **Chọn** → `m06/sft_best` | | | test strict = | |

**Sổ tay debug (điền bằng lời của bạn):**
| Lab | Thông báo lỗi / triệu chứng thấy được | Nguyên nhân | Cách sửa đã kiểm chứng |
|---|---|---|---|
| OOM | | | |
| NaN | | | |
| Phân kỳ | | | |
| Overfit | | | |

---

## 🏢 Ở công ty thật

- **Quét siêu tham số chạy song song** bằng công cụ sweep (W&B Sweeps, Optuna, Ray Tune) với ngân sách GPU đặt trước và dừng sớm các lượt kém.
- **Kỷ luật "test đo đúng một lần" được thực thi bằng quy trình**, không bằng ý thức: tập test chỉ chạy qua job đánh giá chính thức, kết quả ghi vào hồ sơ phát hành.
- **Lỗi trên cụm GPU thường lẫn lỗi hạ tầng** (node hỏng, mất kết nối lưu trữ) → luôn lưu checkpoint định kỳ và kiểm tra khả năng resume trước lượt train dài.

---

## 4. Bài tập

**Bắt buộc**
1. **Grad accumulation tương đương:** so `train_loss` bước 1 của (batch 16, GA 1) và (batch 4, GA 4) với cùng seed (`max_steps=1`). Chênh lệch có nhỏ không? Nếu dùng batch 16 bị OOM, giải thích vì sao đó lại chính là lý do tồn tại của GA.
2. Thêm quét `lora_alpha ∈ {16, 32, 64}` ở LR đã chọn. Kết quả có phù hợp lý thuyết "α/r là hệ số nhân LR hiệu dụng" không?
3. Viết hàm `preflight(ds, tok, max_length)` kiểm tra trước khi train: (a) mẫu nào prompt ≥ max_length, (b) completion có kết thúc bằng `<|im_end|>`, (c) phân bố độ dài; chạy nó trên `ds_train`.

**Nâng cao**
4. Chạy lại sweep với **3 seed** (`random_state` và `seed`) cho LR tốt nhất. Độ lệch chuẩn của `val_strict` là bao nhiêu? Kết luận ở Khối 2 có còn vững?
5. Thử `lr_scheduler_type="constant_with_warmup"` vs `cosine`. Với số bước ít (~51), scheduler có quan trọng không?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Loss bị NaN, quy trình debug của bạn?**
   A: (1) Kiểm tra dữ liệu: mẫu có còn token có nhãn sau khi cắt không, có giá trị bất thường không; (2) giảm LR, thêm warmup, xác nhận gradient clipping; (3) với FP16, kiểm tra tràn số (chuyển BF16 nếu GPU hỗ trợ); (4) tái hiện trên 1 batch nhỏ để khoanh vùng.
2. **Q: `batch=8` vs `batch=2, grad_accum=4` khác gì?**
   A: Cùng batch hiệu dụng; GA giảm bộ nhớ **activations** (không giảm trọng số/optimizer), đổi lại chậm hơn do nhiều lượt forward/backward tuần tự. Thư viện hiện đại chuẩn hóa loss theo tổng token có nhãn để hai cách tương đương.
3. **Q: Vì sao không chọn siêu tham số trên tập test?**
   A: Test bị "dùng" để chọn sẽ cho ước lượng lạc quan (thiên lệch chọn lọc); chọn trên val, báo cáo test một lần cho mô hình cuối.
4. **Q: Early stopping cần cấu hình gì trong HF Trainer?**
   A: `eval_strategy` và `save_strategy` cùng chu kỳ, `load_best_model_at_end=True`, `metric_for_best_model` (thường `eval_loss`), callback `EarlyStoppingCallback(patience)`; val phải tách biệt train.
5. **Q: Dấu hiệu LR quá cao?**
   A: Loss dao động hoặc tăng vọt sau vài bước, `grad_norm` lớn bất thường, đôi khi NaN; mô hình sinh văn bản lặp/vô nghĩa.
6. **Q: Ưu tiên tinh chỉnh siêu tham số nào trước với LoRA?**
   A: Learning rate, rồi số bước/epoch (theo val loss), rồi batch hiệu dụng, sau đó mới rank/alpha và các tham số điều chuẩn.

---

## 6. Checklist cuối buổi
- [ ] Quét LR chọn trên val, có biểu đồ.
- [ ] Sổ tay debug 4 lỗi với thông báo lỗi thật.
- [ ] `m06/sft_best` + `selection.json` trên Drive; test đo đúng 1 lần.

## Tài liệu tham khảo
- Hugging Face blog: *Fixing Gradient Accumulation* (2024).
- Google Research, *Deep Learning Tuning Playbook*.
- Tài liệu Transformers: `EarlyStoppingCallback`, `TrainingArguments`.
