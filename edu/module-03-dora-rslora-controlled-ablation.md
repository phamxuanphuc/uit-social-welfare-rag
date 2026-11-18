# Buổi 03 — LoRA vs DoRA vs rsLoRA: Thí nghiệm có Kiểm soát (Ablation)

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (50' lý thuyết · 2h40 lab, trong đó ~50' chờ 3 lượt train) |
| **Đầu vào** | Adapter `m02/lora_r16`, file `m02_outputs.json`, bảng điểm có `M02` |
| **🎯 Sản phẩm** | **Báo cáo ablation 4 biến thể** (LoRA r16 · DoRA r16 · LoRA r64 · rsLoRA r64) gồm: tham số, thời gian, VRAM, 8 chỉ số, p-value — đúng định dạng một mục "Experiments" trong tài liệu kỹ thuật |
| **Chỉ số đo** | `scaling` thực tế trong mô hình · số tham số · phút/epoch · VRAM đỉnh · `strict_acc` · McNemar so với M02 |
| **Chế độ code** | 🔬 Ô nhỏ (≤ 12 dòng); bắt đầu đóng gói lặp lại thành hàm |

---

## 0. Bức tranh lớn

Nhà tuyển dụng không hỏi "DoRA là gì" để nghe định nghĩa — họ hỏi để xem bạn **có biết cách chứng minh** một kỹ thuật mới tốt hơn hay không. Câu trả lời của kỹ sư giỏi luôn có dạng: *"Tôi giữ nguyên dữ liệu, seed, số bước, LR; chỉ đổi đúng một yếu tố; đo trên cùng tập test; khác biệt X điểm, p = Y; chi phí thời gian tăng Z%."* Buổi này luyện đúng kỹ năng đó.

---

## 1. Lý thuyết

### 1.1. DoRA — tách "độ lớn" và "hướng"
- Mọi cột trọng số có thể viết `W = m · V/‖V‖`: **m** là độ lớn (vô hướng/cột), **V/‖V‖** là hướng (vector đơn vị).
- Phân tích của tác giả DoRA: full fine-tuning thường đổi hướng nhiều mà đổi độ lớn ít (hoặc ngược lại), còn LoRA buộc hai thứ thay đổi **cùng chiều** → kém linh hoạt.
- DoRA: `W' = m' · (W₀ + s·BA) / ‖W₀ + s·BA‖`, với **m'** là vector học được (khởi tạo bằng ‖W₀‖), LoRA chỉ lo phần **hướng**.
- **Chi phí:** thêm 1 tham số cho mỗi chiều đầu ra — với Qwen3-1.7B là `(2048+1024+1024+2048+6144+6144+2048) × 28 = 573.440` tham số; **chậm hơn** vì mỗi bước phải tính chuẩn của ma trận trọng số (với nền 4-bit còn phải giải nén để tính).

### 1.2. rsLoRA — sửa hệ số tỉ lệ khi rank lớn
- LoRA chuẩn nhân `ΔW` với `s = α/r`. Kalajdzievski (2023) chứng minh với `s = α/r`, khi r tăng thì **gradient và cập nhật co lại** → tăng rank không giúp mô hình học thêm.
- rsLoRA dùng `s = α/√r` → độ lớn cập nhật **ổn định theo r** → rank cao mới phát huy tác dụng.
- ⚠️ **Bẫy thực chiến:** bật `use_rslora=True` mà **giữ nguyên α** thì hệ số scale đổi ngay:

| Cấu hình | Công thức | `s` |
|---|---|---|
| LoRA r=16, α=32 (Buổi 02) | 32/16 | **2,0** |
| rsLoRA r=16, α=32 | 32/√16 | **8,0** ← gấp 4 lần! |
| LoRA r=64, α=16 | 16/64 | **0,25** |
| rsLoRA r=64, α=16 | 16/√64 | **2,0** |

→ Đổi sang rsLoRA mà không chỉnh α/LR là **thay đổi 2 yếu tố cùng lúc** — thí nghiệm không còn kiểm soát.

### 1.3. Thiết kế thí nghiệm của buổi này
| Mã | Biến thể | Câu hỏi nghiên cứu |
|---|---|---|
| V1 | LoRA r16 α32 (lấy lại từ M02) | Mốc so sánh |
| V2 | **DoRA** r16 α32 | Tách độ lớn/hướng có giúp không, tốn thêm bao nhiêu thời gian? |
| V3 | LoRA r64 **α16** (s=0,25) | Tăng rank theo cách "ngây thơ" giữ α cố định |
| V4 | **rsLoRA** r64 α16 (s=2,0) | rsLoRA có cứu được rank cao không? |

Giữ nguyên: dữ liệu, seed 42, 3 epoch, LR 2e-4, batch hiệu dụng 16, target = 7 lớp Linear, bộ chấm.

---

## 2. Thực hành

*(Chạy Ô 0b của Buổi 00 trước.)*

### Ô 1 (8 dòng) — Chuẩn bị dữ liệu (dùng lại kỹ năng Buổi 02)
```python
import json, os, time, torch, p01kit
from datasets import Dataset
from p01kit.data import load_split
from p01kit.prompts import to_prompt_completion
from p01kit.sft import encode_prompt_completion, PadCollator
from p01kit.modeling import load_model, load_tokenizer, set_seed, vram_gb, adapter_path, count_trainable, free
train, val, test = (load_split(s) for s in ("train", "val", "test"))
tok = load_tokenizer()
enc = lambda rows: Dataset.from_list([encode_prompt_completion(tok, **to_prompt_completion(tok, r)) for r in rows])
ds_train, ds_val = enc(train), enc(val)
```

### Ô 2 (8 dòng) — Dự đoán số tham số TRƯỚC khi chạy
```python
PER_R = 1_089_536                # = 622.592 / 16 : tham số LoRA mỗi đơn vị rank (7 lớp × 28 khối)
DORA_MAG = 573_440               # vector độ lớn của DoRA
expected = {"V2_dora_r16": 16 * PER_R + DORA_MAG, "V3_lora_r64": 64 * PER_R, "V4_rslora_r64": 64 * PER_R}
for k, v in expected.items():
    print(f"{k:<15} dự đoán {v:>12,} tham số")
```
- 📌 **Kết quả nên thấy:** V2 = **18.006.016**, V3 = V4 = **69.730.304**. Ô 5 sẽ đối chiếu với số thật.

### Ô 3 (10 dòng) — Cấu hình huấn luyện dùng chung
```python
from transformers import TrainingArguments
def make_args(name: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(adapter_path("m03", f"ckpt_{name}")), seed=42,
        per_device_train_batch_size=4, gradient_accumulation_steps=4, per_device_eval_batch_size=4,
        num_train_epochs=3, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05,
        optim="paged_adamw_8bit", fp16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5,
        eval_strategy="epoch", save_strategy="no", report_to="none", remove_unused_columns=False)
```

### Ô 4 (12 dòng) — Hàm chạy MỘT biến thể (nạp base sạch mỗi lần!)
```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer
from p01kit.infer import evaluate
from p01kit import scoreboard
ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def build(name, **lora_kw):
    set_seed(42)
    m = prepare_model_for_kbit_training(load_model(), use_gradient_checkpointing=True)
    m = get_peft_model(m, LoraConfig(task_type="CAUSAL_LM", bias="none", lora_dropout=0.05,
                                     target_modules=ALL_LINEAR, **lora_kw))
    return m
```
- 🧠 **Kỷ luật quan trọng nhất:** mỗi biến thể nạp **base sạch** — không bao giờ gắn adapter mới lên mô hình đã train ở biến thể trước.

### Ô 5 (12 dòng) — Train + đo + lưu + ghi bảng điểm
```python
def run(name, **lora_kw):
    m = build(name, **lora_kw)
    n = count_trainable(m)["trainable"]
    scale = m.base_model.model.model.layers[0].self_attn.q_proj.scaling["default"]
    tr = Trainer(model=m, args=make_args(name), train_dataset=ds_train, eval_dataset=ds_val,
                 data_collator=PadCollator(tok.pad_token_id), processing_class=tok)
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); tr.train()
    minutes, peak = (time.perf_counter() - t0) / 60, vram_gb()["peak"]
    m.eval(); m.config.use_cache = True
    res = evaluate(m, tok, test, batch_size=8)
    m.save_pretrained(adapter_path("m03", name))
    scoreboard.record("M03", name, res["summary"], latency_s=res["summary"]["sec_per_row"], notes=f"{minutes:.1f}min peak{peak}GB")
    eval_loss = [h["eval_loss"] for h in tr.state.log_history if "eval_loss" in h]
    del m, tr; free()
    return {"name": name, "params": n, "scale": scale, "minutes": round(minutes, 1), "peak_gb": peak,
            "eval_loss": eval_loss, "summary": res["summary"], "answer_ok": [s["answer_ok"] for s in res["scores"]]}
```
- **Giải thích:** đọc `scaling["default"]` trực tiếp từ một lớp `q_proj` để **xác minh bằng số** hệ số `s` của mục 1.2 — không tin tài liệu, tin đo đạc.

### Ô 6 (3 dòng) — V2: DoRA
```python
results = {}
results["V2_dora_r16"] = run("V2_dora_r16", r=16, lora_alpha=32, use_dora=True)
print({k: v for k, v in results["V2_dora_r16"].items() if k not in ("summary", "answer_ok")})
```
- 📌 **Kết quả nên thấy:** `params = 18,006,016` (khớp Ô 2), `scale = 2.0`; thời gian train **dài hơn rõ rệt** so với M02.

### Ô 7 (4 dòng) — V3: LoRA r64 (α giữ 16) và V4: rsLoRA r64
```python
results["V3_lora_r64"] = run("V3_lora_r64", r=64, lora_alpha=16)
results["V4_rslora_r64"] = run("V4_rslora_r64", r=64, lora_alpha=16, use_rslora=True)
for k in ("V3_lora_r64", "V4_rslora_r64"):
    print(k, results[k]["params"], "scale =", results[k]["scale"], "eval_loss =", results[k]["eval_loss"])
```
- 📌 **Kết quả nên thấy:** cả hai `params = 69,730,304`; `scale` lần lượt **0.25** và **2.0**. Đường `eval_loss` của V3 thường **giảm chậm hơn** V4 — đúng dự đoán của rsLoRA.

### Ô 8 (12 dòng) — 📊 Bảng ablation hoàn chỉnh + McNemar so với V1
```python
from p01kit.metrics import score_one
from p01kit.stats import mcnemar_exact
outs_v1 = json.load(open(f"{os.environ['P01_RUNS']}/m02_outputs.json"))
v1_ok = [score_one(o, r)["answer_ok"] for o, r in zip(outs_v1, test)]
print(f"{'biến thể':<15}{'params':>12}{'s':>6}{'phút':>7}{'VRAM':>6}{'answer':>8}{'strict':>8}{'unseen':>8}  p(vs V1)")
for k, v in results.items():
    s = v["summary"]
    p = mcnemar_exact(v1_ok, v["answer_ok"])["p_value"]
    print(f"{k:<15}{v['params']:>12,}{v['scale']:>6}{v['minutes']:>7}{v['peak_gb']:>6}"
          f"{s['answer_acc']:>8.3f}{s['strict_acc']:>8.3f}{s['acc_unseen_doc']:>8.3f}  {p:.3f}")
```
- **Cách đọc kết quả (quan trọng hơn con số):**
  - Nếu p ≥ 0,05 cho mọi biến thể → **không có bằng chứng** kỹ thuật nào tốt hơn LoRA r16 trên bài toán này → chọn phương án **rẻ nhất** (LoRA r16). Đây là kết luận hoàn toàn hợp lệ và rất "chuyên nghiệp".
  - Nếu V4 tốt hơn V3 rõ rệt → minh chứng thực nghiệm cho rsLoRA.
  - Ghi **chi phí**: DoRA tăng bao nhiêu % thời gian cho bao nhiêu điểm?

### Ô 9 (6 dòng) — Soi đường loss (overfit hay underfit?)
```python
import matplotlib.pyplot as plt
for k, v in results.items():
    plt.plot(range(1, len(v["eval_loss"]) + 1), v["eval_loss"], marker="o", label=k)
plt.xlabel("epoch"); plt.ylabel("eval_loss"); plt.legend(); plt.title("Val loss theo epoch"); plt.show()
```

---

## 3. So sánh hiệu quả — Mẫu báo cáo ablation (điền số đo của bạn)

| Biến thể | Tham số train | s | Phút train | VRAM đỉnh | answer_acc | strict_acc | acc_unseen_doc | p vs V1 |
|---|---|---|---|---|---|---|---|---|
| V1 LoRA r16 α32 | 17.432.576 | 2,0 | | | | | | — |
| V2 DoRA r16 α32 | 18.006.016 | 2,0 | | | | | | |
| V3 LoRA r64 α16 | 69.730.304 | 0,25 | | | | | | |
| V4 rsLoRA r64 α16 | 69.730.304 | 2,0 | | | | | | |

**Kết luận mẫu (viết 3–5 câu):** *"Trên 268 mẫu P01, [biến thể] cho strict_acc cao nhất nhưng khác biệt so với LoRA r16 [có/không] có ý nghĩa (p=...). DoRA tăng thời gian train ...%. Chọn [cấu hình] cho các buổi sau vì ..."*

> 📌 **Quyết định cho lộ trình:** các buổi sau mặc định dùng cấu hình **tốt nhất có ý nghĩa thống kê**, nếu không có thì dùng **LoRA r16 α32** (rẻ nhất). Ghi lựa chọn vào vở.

---

## 4. Bài tập

**Bắt buộc**
1. **rsLoRA r16 α32** (s = 8,0): chạy `run("V5_rslora_r16_a32", r=16, lora_alpha=32, use_rslora=True)`. So với V1, chuyện gì xảy ra với loss những bước đầu? Giải thích bằng bảng 1.2.
2. Viết đoạn "Experiments" 150 chữ cho báo cáo đồ án, có bảng và p-value.

**Nâng cao — Thí nghiệm "thứ tự khóa JSON"**
3. Giả thuyết của ADR 0002: đặt `verification_reasoning` **đầu tiên** giúp mô hình đối chiếu trước khi kết luận. Kiểm chứng:
   ```python
   REASON_LAST = ("is_refusal", "expired_warning", "doc_code", "article", "benefit",
                  "required_documents", "submission_place", "verification_reasoning")
   def reorder(row):
       return {**row, "gold": {k: row["gold"][k] for k in REASON_LAST}}
   ds_train_last = enc([reorder(r) for r in train])
   ```
   Train V1 trên `ds_train_last` (nhớ đổi `SYSTEM_PROMPT` liệt kê khóa theo thứ tự mới, truyền `system_prompt=` vào cả `to_prompt_completion` và `evaluate`). So sánh `answer_acc` (không phụ thuộc thứ tự khóa) bằng McNemar. Kết quả ủng hộ hay bác bỏ ADR?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: DoRA khác LoRA thế nào? Chi phí?**
   A: DoRA phân rã W thành độ lớn m (học riêng) và hướng (LoRA cập nhật), cho phép đổi hướng/độ lớn độc lập, gần hành vi full fine-tune hơn. Chi phí: thêm tham số bằng số chiều đầu ra và chậm hơn do tính chuẩn trọng số mỗi bước.
2. **Q: Vì sao cần rsLoRA? Khi nào nó quan trọng?**
   A: Với scale α/r, cập nhật co lại khi r lớn nên rank cao không giúp; α/√r giữ độ lớn ổn định. Quan trọng khi r ≥ 64; ở r nhỏ khác biệt chủ yếu là thay đổi hệ số tương đương với chỉnh α.
3. **Q: Bật `use_rslora` cho cấu hình r=16, α=32 đang chạy tốt thì sao?**
   A: Scale nhảy từ 2 lên 8 (×√r = ×4) — tương đương tăng LR hiệu dụng; có thể mất ổn định. Phải chỉnh lại α hoặc LR.
4. **Q: Thế nào là một ablation đúng?**
   A: Thay đổi đúng một yếu tố, giữ nguyên dữ liệu/seed/ngân sách bước/bộ chấm; báo cáo cả chất lượng, chi phí và ý nghĩa thống kê; nếu có thể chạy nhiều seed.
5. **Q: Kết quả không khác biệt có ý nghĩa — báo cáo thế nào?**
   A: Nói rõ "không có bằng chứng cải thiện" kèm p-value và kích thước mẫu; chọn phương án rẻ nhất; đề xuất cần bao nhiêu mẫu test để phát hiện mức chênh mong muốn.
6. **Q: Tăng rank LoRA có luôn tốt hơn?**
   A: Không. Với dữ liệu ít, rank cao dễ overfit và tốn bộ nhớ; với scale α/r còn học chậm hơn. Rank nên chọn theo độ phức tạp của thay đổi hành vi và kiểm chứng bằng val.

---

## 6. Checklist cuối buổi
- [ ] Dự đoán đúng số tham số 3 biến thể trước khi chạy.
- [ ] Đọc được `scaling` thực tế từ mô hình và giải thích bảng 1.2.
- [ ] Bảng ablation 4 dòng có thời gian, VRAM, chỉ số và p-value.
- [ ] Đã chốt cấu hình adapter mặc định cho các buổi sau.

## Tài liệu tham khảo
- Liu et al., *DoRA: Weight-Decomposed Low-Rank Adaptation* (ICML 2024).
- Kalajdzievski, *A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA* (2023).
- Tài liệu PEFT: `use_dora`, `use_rslora`.
