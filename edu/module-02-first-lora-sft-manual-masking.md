# Buổi 02 — SFT LoRA đầu tiên: Tự tay viết Loss Masking với `Trainer`

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (1h lý thuyết · 2h30 lab, trong đó ~10–15' chờ train) |
| **Đầu vào** | Bảng điểm có `M00` và `M01`; dữ liệu `p01_train.jsonl` / `p01_val.jsonl` |
| **🎯 Sản phẩm** | Adapter **`m02/lora_r16`** (~70 MB) lưu trên Drive + dòng bảng điểm **`M02/sft-lora-r16`** + bảng so sánh Trước/Sau có kiểm định McNemar |
| **Chỉ số đo** | Số tham số train được (khớp công thức) · train/val loss · 8 chỉ số P01 · thời gian & VRAM đỉnh |
| **Chế độ code** | 🔬 Ô siêu nhỏ — đây là buổi "mở hộp đen": không dùng thư viện fine-tune cấp cao |

---

## 0. Bức tranh lớn

Đây là buổi **quan trọng nhất về nền tảng**. Sau buổi này bạn phải giải thích được, không cần nhìn tài liệu:
- SFT tối ưu hàm mất mát gì, trên những token nào.
- LoRA thêm bao nhiêu tham số, khởi tạo thế nào, vì sao lúc đầu mô hình **chưa thay đổi gì**.
- Vì sao một mẫu dữ liệu 1.100 token chỉ có ~230 token thật sự được "học".

Từ Buổi 04, các thư viện (TRL, Unsloth) sẽ làm hộ những việc này — nhưng khi chúng hỏng (và chúng sẽ hỏng), chỉ người hiểu buổi này mới sửa được.

---

## 1. Lý thuyết

### 1.1. SFT = dự đoán token kế tiếp, nhưng chỉ chấm điểm phần trả lời
Với chuỗi `x₁ … x_T`, mất mát cross-entropy là `L = −(1/|S|) Σ_{t∈S} log p(x_t | x_<t)`, trong đó **S** là tập vị trí được tính điểm.
- **Không che (S = mọi vị trí):** mô hình phí sức học thuộc lòng **ngữ cảnh 840 token** và câu hỏi — những thứ lúc suy luận đã có sẵn.
- **Che prompt (S = vị trí của câu trả lời):** gán nhãn `-100` cho token prompt; `CrossEntropyLoss(ignore_index=-100)` bỏ qua chúng. 100% gradient dồn vào hành vi ta muốn: đọc–lọc–trả lời JSON.
- Ví von Feynman: giáo viên chỉ chấm **bài làm**, không chấm phần **chép lại đề**.

### 1.2. LoRA — dán "giấy ghi chú" lên lề sách
`W' = W₀ + (α/r) · B·A`, với `W₀ ∈ ℝ^{d_out×d_in}` **đóng băng** (ở đây đang là 4-bit), `A ∈ ℝ^{r×d_in}` khởi tạo ngẫu nhiên, `B ∈ ℝ^{d_out×r}` khởi tạo **bằng 0**.
- Vì `B = 0` nên **ngay sau khi gắn LoRA, mô hình y hệt bản gốc** — Ô 6 sẽ chứng minh bằng số.
- Số tham số thêm cho 1 lớp Linear = `r × (d_in + d_out)`.

**Tính tay cho Qwen3-1.7B, r = 16, gắn vào cả 7 lớp Linear mỗi khối:**
| Lớp | d_in → d_out | r·(d_in+d_out) |
|---|---|---|
| q_proj | 2048 → 2048 | 65.536 |
| k_proj, v_proj | 2048 → 1024 | 49.152 × 2 |
| o_proj | 2048 → 2048 | 65.536 |
| gate_proj, up_proj | 2048 → 6144 | 131.072 × 2 |
| down_proj | 6144 → 2048 | 131.072 |
| **Mỗi khối** | | **622.592** |
| **× 28 khối** | | **17.432.576 ≈ 1,01% của 1,72 tỷ** |

### 1.3. QLoRA = LoRA trên nền 4-bit
`prepare_model_for_kbit_training()` làm 3 việc: (1) đưa LayerNorm/RMSNorm về FP32 cho ổn định; (2) bật `enable_input_require_grads` để gradient chảy qua embedding đóng băng; (3) bật **gradient checkpointing** (không lưu activations của 28 khối, tính lại khi backward → tốn thêm ~30% thời gian, tiết kiệm phần lớn bộ nhớ activations).

### 1.4. Siêu tham số khởi điểm "an toàn" cho LoRA
| Tham số | Giá trị | Lý do |
|---|---|---|
| `learning_rate` | 2e-4 | LoRA cần LR cao hơn full fine-tune (~1e-5) vì chỉ học phần nhỏ, khởi tạo từ 0 |
| `lr_scheduler_type` | cosine + warmup 5% | Khởi động êm, giảm dần về cuối |
| batch hiệu dụng | 4 × 4 = 16 | `per_device_train_batch_size × gradient_accumulation_steps` |
| `num_train_epochs` | 3 | 268 mẫu / 16 ≈ 17 bước/epoch → **51 bước** |
| `optim` | `paged_adamw_8bit` | Trạng thái Adam lưu 8-bit, "phân trang" sang RAM khi VRAM đột biến (kỹ thuật của QLoRA) |
| `lora_alpha` | 32 (= 2r) | Hệ số `α/r = 2` — quy ước phổ biến |

---

## 2. Thực hành từng ô nhỏ

*(Chạy Ô 0b của Buổi 00 trước.)*

### Ô 1 (5 dòng) — Nạp dữ liệu, tokenizer, mô hình sạch
```python
import p01kit
from p01kit.data import load_split
from p01kit.modeling import load_model, load_tokenizer, set_seed, vram_gb
set_seed(42)
train, val, test = (load_split(s) for s in ("train", "val", "test"))
tok, model = load_tokenizer(), load_model()
print(len(train), len(val), len(test), vram_gb())
```

### Ô 2 (8 dòng) — Tự viết hàm che nhãn
```python
from p01kit.prompts import to_prompt_completion

def encode(row):
    ex = to_prompt_completion(tok, row)                 # prompt (đã áp template) + completion (JSON + <|im_end|>)
    p_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
    f_ids = tok(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]
    assert f_ids[:len(p_ids)] == p_ids, "ranh giới token lệch!"
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}
```
- **Giải thích:** `assert` kiểm tra BPE **không gộp token qua ranh giới** prompt|completion (nếu gộp, nhãn bị lệch 1 vị trí — lỗi im lặng rất khó phát hiện). Prompt của Qwen3 kết thúc bằng `\n\n` nên an toàn.
- ⚠️ Quên `<|im_end|>` ở cuối completion → mô hình không học cách dừng.

### Ô 3 (6 dòng) — Soi tận mắt nhãn đã che
```python
ex = encode(train[0])
n_learn = sum(l != -100 for l in ex["labels"])
print(f"Tổng {len(ex['labels'])} token, học {n_learn} token ({n_learn/len(ex['labels']):.0%})")
print("10 nhãn đầu:", ex["labels"][:10])
print("Phần được học:", tok.decode([t for t in ex["labels"] if t != -100])[:200], "...")
```
- 📌 **Kết quả nên thấy:** ~1.000–1.150 token, chỉ ~**20%** được học; 10 nhãn đầu toàn `-100`; phần được học bắt đầu bằng `{"verification_reasoning": ...` và kết thúc bằng `<|im_end|>`.

### Ô 4 (4 dòng) — Đối chiếu với bản tham chiếu trong p01kit
```python
from p01kit.sft import encode_prompt_completion
ref = encode_prompt_completion(tok, **to_prompt_completion(tok, train[0]))
print("Khớp bản tham chiếu:", ref == ex)
```
- 📌 `Khớp bản tham chiếu: True`. Nếu `False`, so từng trường để tìm chỗ sai.

### Ô 5 (6 dòng) — Dựng Dataset
```python
from datasets import Dataset
ds_train = Dataset.from_list([encode(r) for r in train])
ds_val = Dataset.from_list([encode(r) for r in val])
lens = [len(x) for x in ds_train["input_ids"]]
print(ds_train, "| dài nhất:", max(lens), "token")
```
- 📌 **Kết quả nên thấy:** 268 mẫu train, 34 mẫu val; dài nhất ≈ **1.181 token** (vì vậy `p01kit.MAX_SEQ_LEN = 1280`).

### Ô 6 (9 dòng) — Gắn LoRA & chứng minh "B = 0 nên chưa đổi gì"
```python
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
```
```python
probe = tok("Xe ôm bị tai nạn", return_tensors="pt").to(model.device)
with torch.no_grad():
    with model.disable_adapter():
        base_logits = model(**probe).logits
    lora_logits = model(**probe).logits
print("Chênh lệch lớn nhất:", (base_logits - lora_logits).abs().max().item())
```
- 📌 **Kết quả nên thấy:** `trainable params: 17,432,576 || all params: ... || trainable%: ~1.0` — **khớp đúng bảng tính tay ở mục 1.2**. Chênh lệch logits = `0.0`.
- 🧠 Nếu chênh lệch khác 0 trước khi train, bạn đang đo "trước fine-tune" trên mô hình **đã nhiễm** adapter cũ.

### Ô 7 (10 dòng) — Cấu hình huấn luyện
```python
from transformers import TrainingArguments
from p01kit.modeling import adapter_path
args = TrainingArguments(
    output_dir=str(adapter_path("m02", "ckpt")), seed=42,
    per_device_train_batch_size=4, gradient_accumulation_steps=4, per_device_eval_batch_size=4,
    num_train_epochs=3, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05,
    optim="paged_adamw_8bit", fp16=True,
    gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=5, eval_strategy="epoch", save_strategy="no",
    report_to="none", remove_unused_columns=False,
)
```
- **Giải thích các dòng đáng chú ý:** `warmup_steps=0.05` — transformers 5 cho phép số thực trong [0,1) nghĩa là **tỉ lệ** (thay `warmup_ratio` đã bị bỏ); `remove_unused_columns=False` vì dataset của ta đã đúng 3 cột; `use_reentrant=False` là kiểu checkpointing được PyTorch khuyến nghị.

### Ô 8 (8 dòng) — Train!
```python
import time
from transformers import Trainer
from p01kit.sft import PadCollator
torch.cuda.reset_peak_memory_stats()
trainer = Trainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val,
                  data_collator=PadCollator(tok.pad_token_id), processing_class=tok)
t0 = time.perf_counter(); trainer.train(); train_min = (time.perf_counter() - t0) / 60
print(f"Train {train_min:.1f} phút | VRAM đỉnh {vram_gb()['peak']} GB")
```
- 📌 **Kết quả nên thấy:** 51 bước; `loss` giảm nhanh trong ~15 bước đầu rồi chậm dần; `eval_loss` sau mỗi epoch **giảm** (nếu epoch 3 tăng lại là dấu hiệu overfit — Buổi 06). VRAM đỉnh nằm thoải mái dưới 15 GB.
- ⚠️ `loss = nan` ngay bước đầu → thường do mọi nhãn bị `-100` (prompt dài hơn max_length) hoặc FP16 tràn số. Ô 3 đã giúp bạn loại trừ nguyên nhân thứ nhất.

### Ô 9 (5 dòng) — Lưu adapter lên Drive
```python
out_dir = adapter_path("m02", "lora_r16")
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
import os; print(out_dir, sorted(os.listdir(out_dir)))
print(f"{os.path.getsize(out_dir / 'adapter_model.safetensors')/1e6:.0f} MB")
```
- 📌 **Kết quả nên thấy:** `adapter_config.json`, `adapter_model.safetensors` ≈ **70 MB** (17,4 triệu tham số × 4 byte FP32). So với 3,44 GB của mô hình đầy đủ: nhỏ hơn ~50 lần.

### Ô 10 (7 dòng) — 📊 Đo SAU fine-tune & ghi bảng điểm
```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
model.eval(); model.config.use_cache = True     # bật lại KV-cache (đã tắt khi train với checkpointing)
res2 = evaluate(model, tok, test, batch_size=8, max_new_tokens=p01kit.MAX_NEW_TOKENS)
print(format_summary(res2["summary"]))
scoreboard.record("M02", "sft-lora-r16", res2["summary"], latency_s=res2["summary"]["sec_per_row"], notes=f"{train_min:.1f} min",
                  outputs=res2["outputs"])
```
- 📌 **Kiểm tra hợp lý:** `json_valid` và `schema_ok` nên **gần 1,0**. Nếu < 0,95 → nghi ngờ thiếu `<|im_end|>` hoặc masking sai. `use_cache=True` rất quan trọng: quên bật thì sinh chậm hơn nhiều lần.

### Ô 11 (8 dòng) — Trước vs Sau trên cùng câu hỏi + McNemar
```python
import json, os
from p01kit.metrics import score_one
from p01kit.stats import mcnemar_exact
outs0 = json.load(open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json"))
before = [score_one(o, r)["answer_ok"] for o, r in zip(outs0, test)]
after = [s["answer_ok"] for s in res2["scores"]]
print("McNemar base → SFT:", mcnemar_exact(before, after))
json.dump(res2["outputs"], open(f"{os.environ['P01_RUNS']}/m02_outputs.json", "w"), ensure_ascii=False)
```

### Ô 12 (8 dòng) — Soi 3 ca tiêu biểu
```python
cases = {"hết hạn": next(i for i, r in enumerate(test) if r["is_expired"]),
         "từ chối": next(i for i, r in enumerate(test) if r["is_unanswerable"] and r["topic"]),
         "đời thường, văn bản mới": next(i for i, r in enumerate(test) if r["style"] == "colloquial" and r["slice"] == "unseen_doc" and not r["is_unanswerable"])}
for name, i in cases.items():
    print(f"\n=== {name}: {test[i]['question']}")
    print("🔴 Trước:", outs0[i][:220].replace("\n", " "))
    print("🟢 Sau  :", res2["outputs"][i][:220].replace("\n", " "))
```

### Ô 13 (6 dòng) — Kiểm tra adapter đã lưu tái tạo đúng kết quả
```python
from p01kit.modeling import free
free("model", "trainer", scope=globals())
model = load_model(adapter=adapter_path("m02", "lora_r16"))
chk = evaluate(model, tok, test[:16], batch_size=8)
print("Khớp 16 output đầu:", chk["outputs"] == res2["outputs"][:16])
```
- 📌 `True`. Nếu `False` (hiếm), thường do khác biệt nhỏ về số học FP16 giữa batch khác nhau — chấp nhận được nếu điểm số giống nhau. Thói quen "nạp lại để kiểm tra" giúp bạn không bao giờ nộp nhầm checkpoint.

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M00", "base-zero-shot"), ("M02", "sft-lora-r16"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** Đây là **lần fine-tune đầu tiên** của bạn. Kỳ vọng hợp lý: `json_valid` gần 1,0 và nhóm lỗi `❌ no_json`/`le_ra_tu_choi` giảm mạnh; nếu `làm hỏng` > 10 câu, xem lại masking (Ô 3) trước khi kết luận.


```python
scoreboard.show()
```
| module | variant | json_valid | refusal_acc | citation_precision | expired_recall | strict_acc | fairness_gap | acc_unseen_doc |
|---|---|---|---|---|---|---|---|---|
| M00 | base-zero-shot | | | | | | | |
| M01 | base-3-shot | | | | | | | |
| M02 | sft-lora-r16 | | | | | | | |

**Câu hỏi thảo luận sau khi điền số:**
1. SFT cải thiện mạnh nhất ở chỉ số nào? Chỉ số nào cải thiện ít? (gợi ý: `expired_recall` chỉ có 18 mẫu train — dữ liệu hiếm).
2. `acc_unseen_doc` so với `acc_seen_doc` (xem `res2["summary"]`) chênh bao nhiêu? Nếu chênh lớn → mô hình đang **nhớ văn bản** hơn là **đọc văn bản**.
3. Prompt SFT dài 836 token, few-shot 3.523 token: với chất lượng tương đương, chọn phương án nào cho CPU production?

---

## 🏢 Ở công ty thật

- **Huấn luyện chạy như một job có cấu hình** (script trong repo + file YAML/CLI, chạy trên cụm GPU hoặc dịch vụ train), không chạy tay từng ô notebook. Notebook dùng để khám phá; thứ được phát hành phải tái lập được bằng một lệnh.
- **Mỗi lượt train ghi đủ "dấu vết"**: commit code, phiên bản dữ liệu, seed, siêu tham số, phiên bản thư viện — thiếu một thứ là không tái lập được.
- **Adapter vào model registry** (repo private trên Hugging Face Hub có tag, MLflow, W&B Artifacts) thay vì một thư mục Drive; người khác lấy đúng phiên bản bằng tên + phiên bản.
- **Kiểm tra masking (Ô 3) thường thành test tự động trong CI**, vì lỗi chat template/masking là lỗi "im lặng": loss vẫn giảm, mô hình vẫn tệ.

---

## 4. Bài tập

**Bắt buộc**
1. **Thí nghiệm "không che nhãn":** sửa `encode` để `labels = f_ids` (học cả prompt). Train lại cùng cấu hình (`m02/lora_nomask`), so sánh `strict_acc` và thời gian hội tụ. Giải thích kết quả bằng mục 1.1.
2. **Chỉ gắn LoRA vào attention** (`q,k,v,o`), r=16. Tính tay số tham số trước (đáp án: 14.336 × 16 × 28 = 6.422.528), rồi train và so sánh.
3. Vẽ `trainer.state.log_history` (loss theo bước) bằng matplotlib; đánh dấu các điểm `eval_loss`.

**Nâng cao**
4. Thử `lora_dropout=0` và `0.1`. Với 268 mẫu, dropout có giúp chống overfit không? Kiểm định bằng McNemar.
5. Đo VRAM đỉnh khi **tắt** gradient checkpointing (`gradient_checkpointing=False`, giảm `per_device_train_batch_size` nếu OOM). Kết luận đánh đổi thời gian–bộ nhớ.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Prompt loss masking là gì? Khi nào KHÔNG nên che prompt?**
   A: Gán nhãn -100 cho token prompt để loss chỉ tính trên câu trả lời. Không che khi làm *continued pretraining* trên văn bản thô (muốn học phân phối toàn văn bản), hoặc khi prompt rất ngắn và dữ liệu rất ít (một số nghiên cứu cho thấy học cả prompt có tác dụng điều chuẩn nhẹ).
2. **Q: Vì sao LoRA khởi tạo B = 0?**
   A: Để `ΔW = BA = 0` lúc đầu — mô hình xuất phát đúng từ trọng số gốc, huấn luyện ổn định. A khởi tạo ngẫu nhiên để gradient của B khác 0.
3. **Q: Tính số tham số LoRA r=16 cho q_proj 2048→2048.**
   A: `16 × (2048 + 2048) = 65.536`.
4. **Q: `prepare_model_for_kbit_training` làm gì?**
   A: Ép các lớp norm về FP32, bật gradient cho đầu vào embedding, bật gradient checkpointing — để train ổn định trên trọng số 4-bit đóng băng.
5. **Q: Vì sao LR của LoRA (~2e-4) cao hơn full fine-tune (~1e-5)?**
   A: Số tham số học ít, B bắt đầu từ 0, và cập nhật bị nhân hệ số α/r — cần bước lớn hơn để dịch chuyển hành vi trong số bước hạn chế.
6. **Q: Sau khi train với gradient checkpointing, vì sao sinh văn bản chậm?**
   A: Checkpointing yêu cầu `use_cache=False`; phải bật lại `model.config.use_cache=True` (hoặc gọi `model.eval()` với Unsloth `for_inference`) để dùng KV-cache khi generate.
7. **Q: Adapter LoRA ~70 MB, mô hình 3,4 GB. Lợi ích vận hành là gì?**
   A: Lưu/phiên bản hóa/rollback rẻ; phục vụ nhiều adapter trên một mô hình gốc (multi-LoRA, Buổi 13); chia sẻ không cần phân phối lại trọng số gốc.

---

## 6. Checklist cuối buổi
- [ ] Tự viết `encode()` với assert ranh giới token và giải thích được từng dòng.
- [ ] `print_trainable_parameters` ra đúng **17.432.576**.
- [ ] Chứng minh logits trước/sau gắn LoRA bằng nhau.
- [ ] Adapter `m02/lora_r16` trên Drive; bảng điểm có `M02/sft-lora-r16` + kết quả McNemar.

## Tài liệu tham khảo
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021).
- Dettmers et al., *QLoRA* (2023) — paged optimizers, NF4.
- Tài liệu PEFT: `LoraConfig`, `prepare_model_for_kbit_training`.
