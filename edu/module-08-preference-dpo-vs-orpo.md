# Buổi 08 — Căn chỉnh theo Sở thích: Khai thác lỗi On-policy, `DPO` vs `ORPO`

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (1h lý thuyết · 3h lab, ~60' chờ sinh mẫu + train) |
| **Đầu vào** | Mô hình nền đã chốt (`m06/sft_best` hoặc `m07/sft_aug`) |
| **🎯 Sản phẩm** | (1) **Bộ cặp sở thích** khai thác từ chính lỗi của mô hình (on-policy) + thống kê loại lỗi; (2) Adapter **`m08/sft_dpo`** (SFT → DPO) và **`m08/orpo_from_base`** (ORPO một bước); (3) Bảng so sánh 4 mô hình: chất lượng · chi phí · VRAM |
| **Chỉ số đo** | `rewards/margins`, `rewards/accuracies` · `refusal_acc`, `expired_recall`, `strict_acc` · McNemar · phút train |
| **Chế độ code** | 🧩 Khối gộp |

---

## 0. Bức tranh lớn

SFT dạy mô hình **bắt chước** câu trả lời đúng, nhưng không bao giờ cho nó thấy **câu trả lời sai trông thế nào**. Căn chỉnh theo sở thích (preference alignment) bổ sung đúng mảnh đó: *"giữa A và B, hãy thích A hơn"*. Với P01, B chính là những lỗi nguy hiểm: trả lời khi phải từ chối, trích nhầm văn bản nhiễu, quên cảnh báo hết hạn.

Điểm khác biệt của buổi này so với giáo trình phổ thông: cặp sở thích **không viết tay** mà được **khai thác từ lỗi thật** của mô hình SFT (on-policy) — đúng cách các đội làm sản phẩm thực hiện.

---

## 1. Lý thuyết

### 1.1. DPO — Direct Preference Optimization
Mất mát cho cặp (prompt x, chosen y_w, rejected y_l):

`L_DPO = −log σ( β · [ (log π_θ(y_w|x) − log π_ref(y_w|x)) − (log π_θ(y_l|x) − log π_ref(y_l|x)) ] )`

- Đại lượng trong ngoặc vuông là **"phần thưởng ngầm"** của chosen trừ rejected. DPO tăng khoảng cách đó.
- **π_ref** (mô hình tham chiếu, đóng băng) giữ π_θ không trôi xa — vai trò như KL trong RLHF. **β** lớn → bảo thủ (bám ref), β nhỏ → thay đổi mạnh.
- **Với LoRA, không cần nạp 2 mô hình:** TRL 1.x tạo thêm một adapter tên `ref` là **bản sao adapter SFT** làm π_ref. Lưu ý: reference đúng phải là **mô hình SFT**, không phải base — các phiên bản TRL cũ dùng `disable_adapter()` (tức base) khi bạn nạp sẵn adapter, đó là một khác biệt tinh tế đáng nhắc khi phỏng vấn.

### 1.2. ORPO — Odds Ratio Preference Optimization
`L_ORPO = L_SFT(y_w) + λ · ( −log σ( log [odds(y_w|x) / odds(y_l|x)] ) )`, với `odds(y|x) = p/(1−p)`.
- **Một giai đoạn duy nhất** từ base: vừa học bắt chước chosen (SFT) vừa phạt rejected — **không cần mô hình tham chiếu**.
- Ưu: rẻ, đơn giản. Nhược: không có "neo" reference nên có thể trôi khỏi phân phối gốc nhiều hơn; λ (tham số `beta` trong TRL) cần tinh chỉnh.

### 1.3. On-policy vs off-policy
- **Off-policy:** rejected do người viết hoặc mô hình khác sinh → có thể là những lỗi mô hình của bạn **không bao giờ mắc** → học vô ích.
- **On-policy:** lấy mẫu từ chính mô hình hiện tại → rejected là lỗi **thật** của nó → tín hiệu học hiệu quả nhất. Khi mô hình làm đúng mọi mẫu, ta bổ sung **âm bản tổng hợp có kiểm soát** (đảo cờ hết hạn, thay mã văn bản bằng văn bản nhiễu trong ngữ cảnh) — vẫn bám luật nghiệp vụ.

### 1.4. Theo dõi DPO đúng cách
| Log | Ý nghĩa | Dấu hiệu tốt |
|---|---|---|
| `rewards/accuracies` | % cặp có reward(chosen) > reward(rejected) | Tăng về gần 1 |
| `rewards/margins` | Khoảng cách trung bình | Tăng đều, không bùng nổ |
| `logps/chosen` | Log-xác suất của chosen | ⚠️ Có thể **giảm** dù margin tăng (*likelihood displacement*) — DPO chỉ cần rejected giảm nhanh hơn. Nếu chosen giảm mạnh, mô hình có thể sinh lệch → giảm LR/tăng β |

---

## 2. Thực hành

### Khối 1 — Khởi động (pipeline HF + PEFT + TRL, không dùng Unsloth ở buổi này)
```python
import os, sys, json, time, random
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
import torch, p01kit
from collections import Counter
from datasets import Dataset
from p01kit.data import load_split, load_corpus
from p01kit.prompts import render_prompt, completion_text
from p01kit.metrics import score_one, extract_json
from p01kit.infer import generate, evaluate
from p01kit.modeling import load_model, load_tokenizer, adapter_path, free, vram_gb
from p01kit import scoreboard
corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
START = adapter_path("m06", "sft_best")          # đổi thành adapter_path("m07", "sft_aug") nếu Buổi 07 thắng
tok = load_tokenizer()
```
- **Vì sao không dùng Unsloth ở đây?** Để thấy rõ cơ chế adapter `ref` của TRL và tránh phụ thuộc vào lớp vá. Buổi 09 sẽ dùng lại Unsloth cho GRPO (cần sinh nhanh).

### Khối 2 — Khai thác lỗi on-policy: lấy 4 mẫu/câu từ mô hình SFT
```python
K = 4
def sample_k(model, rows, seed):
    prompts = [render_prompt(tok, r) for r in rows for _ in range(K)]
    outs = generate(model, tok, prompts, batch_size=16, do_sample=True, temperature=0.9, top_p=0.95, seed=seed)
    return [outs[i * K:(i + 1) * K] for i in range(len(rows))]
SAMPLES = f"{os.environ['P01_RUNS']}/m08_samples.json"
if os.path.exists(SAMPLES):                       # đã sinh ở phiên trước (Colab ngắt sau khối này) → dùng lại
    cached = json.load(open(SAMPLES)); samples_train, samples_val = cached["train"], cached["val"]; policy = None
else:
    policy = load_model(adapter=START)
    t0 = time.perf_counter()
    samples_train, samples_val = sample_k(policy, train, 1), sample_k(policy, val, 2)
    print(f"Sinh {K * (len(train) + len(val))} mẫu trong {(time.perf_counter() - t0) / 60:.1f} phút")
    json.dump({"train": samples_train, "val": samples_val}, open(SAMPLES, "w"), ensure_ascii=False)
```
- **Giải thích:** T=0,9 cao hơn mức chuẩn để **lộ ra** lỗi mà greedy che giấu. Lưu mẫu thô để tái sử dụng (Buổi 09 cần danh sách "câu khó").
- 💡 **Colab miễn phí:** đây là khối sinh lâu nhất buổi (4 mẫu × 302 câu). Mẫu được ghi lên Drive ngay khi xong; nếu runtime bị ngắt ở Khối 4–6, chạy lại khối này sẽ đọc file thay vì sinh lại. Xóa `m08_samples.json` nếu bạn đổi `START`.

### Khối 3 — Dựng cặp (chosen, rejected) với thứ tự ưu tiên lỗi nguy hiểm
```python
SEVERITY = ["tu_choi_thieu", "bia_hoac_nham", "bo_sot_het_han", "tu_choi_thua", "sai_chi_tiet", "json_hong"]

def error_type(out, r):
    o, _ = extract_json(out)
    g = r["gold"]
    if o is None: return "json_hong"
    if g["is_refusal"] and not o.get("is_refusal"): return "tu_choi_thieu"
    if not g["is_refusal"] and o.get("is_refusal"): return "tu_choi_thua"
    if not g["is_refusal"] and o.get("doc_code") != g["doc_code"]: return "bia_hoac_nham"
    if g["expired_warning"] and not o.get("expired_warning"): return "bo_sot_het_han"
    return None if score_one(out, r)["strict_ok"] else "sai_chi_tiet"

def synthetic_negative(r):
    g, ctx = r["gold"], [corpus.by_id(i) for i in r["context_ids"]]
    if g["expired_warning"]:
        return dict(g, expired_warning=False), "synth_bo_sot_het_han"
    wrong = next((p for p in ctx if p.doc_code != g["doc_code"]), None)
    if wrong is None:
        return None, None
    return dict(g, is_refusal=False, doc_code=wrong.doc_code, article=wrong.article, benefit=wrong.benefit,
                required_documents=list(wrong.required_documents), submission_place=wrong.submission_place), "synth_chon_nham_nhieu"

def build_pairs(rows, samples):
    pairs, stats = [], Counter()
    for r, outs in zip(rows, samples):
        errs = [(SEVERITY.index(e), e, o) for o in outs if (e := error_type(o, r))]
        if errs:
            _, e, rej = min(errs)
            stats["on_policy:" + e] += 1
        else:
            neg, e = synthetic_negative(r)
            if neg is None:
                stats["bo_qua"] += 1; continue
            rej = json.dumps(neg, ensure_ascii=False); stats[e] += 1
        pairs.append({"prompt": render_prompt(tok, r), "chosen": completion_text(r) + tok.eos_token,
                      "rejected": rej.strip() + tok.eos_token, "row_id": r["id"]})
    return pairs, stats

pairs_train, st_train = build_pairs(train, samples_train)
pairs_val, _ = build_pairs(val, samples_val)
print(len(pairs_train), "cặp train |", len(pairs_val), "cặp val"); print(st_train.most_common())
hard_ids = [r["id"] for r, outs in zip(train, samples_train) if any(error_type(o, r) for o in outs)]
json.dump(hard_ids, open(f"{os.environ['P01_RUNS']}/m08_hard_ids.json", "w"))
print("Câu mô hình còn sai ít nhất 1/4 lần:", len(hard_ids))
del policy; free()
```
- 📌 **Kết quả nên thấy:** phân bố loại lỗi on-policy của **mô hình bạn** — đây là bức tranh trung thực nhất về điểm yếu còn lại sau SFT. File `m08_hard_ids.json` (các câu mô hình còn sai ít nhất 1/4 lần) dùng cho GRPO ở Buổi 09.
- ⚠️ Rejected sinh bởi mô hình **không** có `<|im_end|>` (đã bị `skip_special_tokens`), ta thêm lại — chosen và rejected phải kết thúc **cùng kiểu** để mô hình không học "mẹo" độ dài/token kết thúc.

### Khối 4 — Train DPO từ mô hình SFT
```python
from trl import DPOConfig, DPOTrainer
policy = load_model(adapter=START, trainable=True)
cfg = DPOConfig(output_dir=str(adapter_path("m08", "ckpt_dpo")), seed=42, beta=0.1,
                learning_rate=2e-5, lr_scheduler_type="cosine", warmup_steps=0.1,
                per_device_train_batch_size=1, gradient_accumulation_steps=8, per_device_eval_batch_size=1,
                num_train_epochs=2, max_length=p01kit.MAX_SEQ_LEN, fp16=True, optim="paged_adamw_8bit",
                gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                logging_steps=2, eval_strategy="epoch", save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
dpo = DPOTrainer(model=policy, ref_model=None, args=cfg, processing_class=tok,
                 train_dataset=Dataset.from_list(pairs_train), eval_dataset=Dataset.from_list(pairs_val))
print("Adapter đang có:", list(policy.peft_config.keys()))      # mong đợi: 'default' và 'ref'
torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); dpo.train()
dpo_min, dpo_peak = (time.perf_counter() - t0) / 60, vram_gb()["peak"]
logs = [h for h in dpo.state.log_history if "rewards/margins" in h]
for h in logs[::max(1, len(logs) // 6)]:
    print(f"bước {h['step']:>3}  acc={h['rewards/accuracies']:.2f}  margin={h['rewards/margins']:.2f}  logp_chosen={h['logps/chosen']:.1f}")
policy.save_pretrained(adapter_path("m08", "sft_dpo"), selected_adapters=["default"])
```
- 📌 **Kết quả nên thấy:** danh sách adapter có `ref`; `rewards/accuracies` tăng dần; `margin` dương và tăng. Theo dõi `logp_chosen` (mục 1.4).
- ⚠️ Nếu gặp `ValueError: Attempting to unscale FP16 gradients` → trọng số LoRA đang ở FP16; `load_model(..., trainable=True)` của p01kit đã ép về FP32 — kiểm tra bạn có gọi đúng hàm.

### Khối 5 — Đánh giá SFT→DPO
```python
policy.eval(); policy.config.use_cache = True
res_dpo = evaluate(policy, tok, test, batch_size=8)
scoreboard.record("M08", "sft_dpo", res_dpo["summary"], latency_s=res_dpo["summary"]["sec_per_row"], notes=f"{dpo_min:.1f}min peak{dpo_peak}GB",
                  outputs=res_dpo["outputs"])
del policy, dpo; free()
```

### Khối 6 — ORPO một bước từ base (không SFT trước)
```python
from peft import LoraConfig
from trl.experimental.orpo import ORPOConfig, ORPOTrainer
base = load_model(trainable=True)                         # base sạch 4-bit, đã prepare cho k-bit training
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
ocfg = ORPOConfig(output_dir=str(adapter_path("m08", "ckpt_orpo")), seed=42, beta=0.1, learning_rate=8e-5,
                  lr_scheduler_type="cosine", warmup_steps=0.05, per_device_train_batch_size=1, gradient_accumulation_steps=8,
                  num_train_epochs=3, max_length=p01kit.MAX_SEQ_LEN, fp16=True, optim="paged_adamw_8bit",
                  gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
                  logging_steps=5, save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
orpo = ORPOTrainer(model=base, args=ocfg, train_dataset=Dataset.from_list(pairs_train), processing_class=tok, peft_config=lcfg)
torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); orpo.train()
orpo_min, orpo_peak = (time.perf_counter() - t0) / 60, vram_gb()["peak"]
orpo.model.save_pretrained(adapter_path("m08", "orpo_from_base"))
orpo.model.eval(); orpo.model.config.use_cache = True
res_orpo = evaluate(orpo.model, tok, test, batch_size=8)
scoreboard.record("M08", "orpo_from_base", res_orpo["summary"], latency_s=res_orpo["summary"]["sec_per_row"], notes=f"{orpo_min:.1f}min peak{orpo_peak}GB",
                  outputs=res_orpo["outputs"])
del base, orpo; free()
```
- **Giải thích:** `beta=0.1` của ORPOConfig chính là λ trong công thức mục 1.2. LR 8e-5 cao hơn DPO vì ORPO **đồng thời làm SFT từ đầu** (LoRA khởi tạo từ 0).
- ⚠️ Cặp sở thích được khai thác từ lỗi của mô hình **SFT**, nên với ORPO-từ-base chúng là dữ liệu *off-policy*. Ghi nhận điều này khi diễn giải kết quả.

### Khối 7 — 📊 So sánh 4 mô hình
```python
from p01kit.stats import mcnemar_exact
sft_out = json.load(open(f"{os.environ['P01_RUNS']}/m06_sft_best_outputs.json"))
sft_scores = [score_one(o, r) for o, r in zip(sft_out, test)]
rows = {"sft_best": sft_scores, "sft_dpo": res_dpo["scores"], "orpo_from_base": res_orpo["scores"]}
keys = ("refusal_acc", "citation_precision", "expired_recall", "false_expired_rate", "strict_acc", "acc_unseen_doc")
from p01kit.metrics import summarize
for name, sc in rows.items():
    s = summarize(sc, with_ci=False)
    p = "—" if name == "sft_best" else f"{mcnemar_exact([x['strict_ok'] for x in sft_scores], [x['strict_ok'] for x in sc])['p_value']:.3f}"
    print(f"{name:<15}" + "  ".join(f"{k}={s[k]}" for k in keys) + f"  p(strict vs SFT)={p}")
scoreboard.show()
```

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M06", "sft_best"), ("M08", "sft_dpo"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** DPO học từ chính lỗi on-policy của `sft_best` → các loại lỗi đã dùng làm cặp *rejected* phải giảm. Đổi thành `("M08", "orpo_from_base")` để so ORPO.


| Mô hình | Giai đoạn | Phút train | VRAM đỉnh | refusal_acc | expired_recall | false_expired | strict_acc | acc_unseen_doc | p vs SFT |
|---|---|---|---|---|---|---|---|---|---|
| base (M00) | — | 0 | | | | | | | |
| sft_best (M06) | SFT | | | | | | | | — |
| **sft_dpo** | SFT → DPO | (M06 +) | | | | | | | |
| **orpo_from_base** | ORPO 1 bước | | | | | | | | |

**Thảo luận:** DPO sửa được loại lỗi nào trong thống kê Khối 3? ORPO một bước có đuổi kịp hai bước SFT→DPO không, với chi phí bao nhiêu? Kết quả này giúp bạn trả lời câu hỏi phỏng vấn "khi nào chọn ORPO thay vì SFT+DPO" bằng **số liệu của chính mình**.

---

## 🏢 Ở công ty thật

- **Cặp ưu tiên ngoài đời đến từ người**: người gán nhãn so hai câu trả lời, chuyên gia sửa câu trả lời, hoặc phản hồi thích/không thích của người dùng. Vì tốn kém, đội thường khai thác lỗi on-policy như buổi này để giảm số cặp cần người duyệt.
- **Hướng dẫn chấm cặp được viết thành văn bản** (đúng văn bản > đủ giấy tờ > văn phong) và đo đồng thuận giữa người chấm, giống nhãn lỗi ở Buổi 15.
- **Mỗi vòng căn chỉnh phải chạy lại toàn bộ bộ đánh giá + red-team**, vì căn chỉnh cho một tiêu chí có thể làm hỏng tiêu chí khác.

---

## 4. Bài tập

**Bắt buộc**
1. **Quét β ∈ {0,05; 0,1; 0,3}** cho DPO. Vẽ `strict_acc` và `logps/chosen` cuối cùng theo β.
2. **Chỉ dùng âm bản tổng hợp** (bỏ on-policy) rồi train DPO. On-policy có thật sự tốt hơn?
3. Kiểm tra *likelihood displacement*: so log-xác suất của chosen trên tập val **trước và sau** DPO (gợi ý: dùng `dpo.evaluate()` hoặc tự tính `logps`).

**Nâng cao**
4. Đọc docstring `DPOConfig.loss_type` của bản TRL đã ghim để liệt kê các biến thể hỗ trợ (ví dụ `ipo`, `hinge`...), chọn 1 biến thể và so sánh với `sigmoid` mặc định. (Lưu ý: tham số `rpo_alpha` của các bản TRL cũ **không còn** trong TRL 1.15 — luôn kiểm tra API thật thay vì chép từ blog.)
5. Chạy **KTO** (`trl.KTOTrainer`) với dữ liệu nhị phân (đúng/sai từng mẫu, không cần ghép cặp). Khi nào KTO tiện hơn DPO trong sản phẩm thật (gợi ý: nút 👍/👎 của người dùng = `DelayedLabel`)?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: DPO hoạt động thế nào, vai trò của β?**
   A: Tối ưu trực tiếp xác suất ưa chosen hơn rejected so với mô hình tham chiếu, qua hàm sigmoid của hiệu log-ratio; β điều khiển độ bám reference (lớn = bảo thủ).
2. **Q: Với LoRA, DPO cần bộ nhớ cho 2 mô hình không?**
   A: Không. Reference lấy từ chính mô hình gốc với adapter đóng băng (TRL 1.x tạo adapter `ref` sao chép adapter SFT; cách cũ là tắt adapter). Chỉ có thêm một bộ trọng số LoRA nhỏ.
3. **Q: ORPO khác DPO?**
   A: ORPO gộp SFT và phạt rejected qua odds ratio trong một giai đoạn, không cần reference model; DPO thường chạy sau SFT và cần reference.
4. **Q: Cặp sở thích nên lấy từ đâu?**
   A: Tốt nhất là on-policy — lấy mẫu từ chính mô hình hiện tại và chấm bằng bộ chấm/người; bổ sung âm bản tổng hợp có kiểm soát cho lỗi hiếm; tránh rejected quá dễ (học không được gì).
5. **Q: `rewards/margins` tăng nhưng chất lượng giảm — vì sao?**
   A: Có thể do likelihood displacement (cả chosen lẫn rejected giảm xác suất), overfit vào cặp, hoặc reward hacking theo đặc điểm bề mặt (độ dài). Kiểm tra logps/chosen, đánh giá trên tập ngoài, tăng β/giảm LR.
6. **Q: Khi nào dùng DPO, khi nào GRPO?**
   A: DPO khi có cặp sở thích (người hoặc mô hình chấm); GRPO khi có hàm thưởng kiểm chứng được tự động và muốn mô hình tự khám phá bằng cách sinh nhiều mẫu (Buổi 09).

---

## 6. Checklist cuối buổi
- [ ] Thống kê loại lỗi on-policy của mô hình SFT.
- [ ] `m08/sft_dpo` và `m08/orpo_from_base` trên Drive; `m08_hard_ids.json` sẵn sàng cho Buổi 09.
- [ ] Bảng 4 mô hình có chi phí và p-value; đã chọn mô hình cho GRPO.

## Tài liệu tham khảo
- Rafailov et al., *Direct Preference Optimization* (NeurIPS 2023).
- Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model* (2024).
- Razin et al., *Unintentional Unalignment: Likelihood Displacement in DPO* (2024).
- Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization* (2024).
