# Buổi 09 — Học tăng cường với `GRPO` & Phần thưởng Kiểm chứng được (Verifiable Rewards)

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (1h lý thuyết · 3h lab — lượt train dài nhất khóa) |
| **Đầu vào** | Mô hình thắng ở Buổi 08 (`m08/sft_dpo` hoặc `m06/sft_best`); `m08_hard_ids.json`; `m08_samples.json` |
| **🎯 Sản phẩm** | (1) **3 hàm thưởng** đã kiểm thử (`p01kit/rewards.py`); (2) Adapter **`m09/grpo`**; (3) Báo cáo **đường thưởng** + phân tích *reward hacking*; (4) So sánh `expired_recall` / `false_expired_rate` trước–sau |
| **Chỉ số đo** | `reward` & từng thành phần · `frac_reward_zero_std` · độ dài completion · 8 chỉ số P01 · McNemar |
| **Chế độ code** | 🧩 Khối gộp |

---

## 0. Bức tranh lớn

DPO cần cặp đúng/sai có sẵn. Nhưng P01 có một tài sản quý: **đáp án kiểm chứng được bằng code** (đúng mã văn bản? có bật cờ hết hạn? JSON hợp lệ?). Khi có thứ đó, ta để mô hình **tự sinh nhiều câu trả lời, tự chấm bằng code, và học từ chênh lệch** — chính là cách DeepSeek-R1 và Qwen3 được huấn luyện suy luận (RLVR — Reinforcement Learning with Verifiable Rewards).

Mục tiêu nghiệp vụ cụ thể: đẩy **`expired_recall` → 100%** (ngưỡng chốt trước) mà **không** tăng báo động giả, nhờ phần thưởng **bất đối xứng**: bỏ sót văn bản hết hạn bị phạt gấp đôi báo nhầm.

---

## 1. Lý thuyết

### 1.1. GRPO trong 4 bước
Với mỗi prompt *x*:
1. Sinh **G = 4** câu trả lời `y₁…y₄` (có sampling).
2. Chấm thưởng `rᵢ = Σ_k w_k · R_k(yᵢ)` bằng 3 hàm code.
3. **Lợi thế tương đối trong nhóm:** `Aᵢ = (rᵢ − mean(r)) / std(r)` → câu tốt hơn trung bình nhóm được đẩy lên, kém hơn bị đẩy xuống. **Không cần mô hình critic/value** như PPO → tiết kiệm bộ nhớ một nửa.
4. Cập nhật theo mục tiêu PPO-clip trên từng token: `min(ρ·A, clip(ρ, 1±ε)·A)`, với ρ là tỉ lệ xác suất mới/cũ.

### 1.2. Những điều TRL 1.15 làm mặc định (đọc kỹ — hay bị hỏi)
| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `beta` | **0,0** | Không có KL với reference → **không cần nạp reference model**. Nghiên cứu gần đây (DAPO, Dr. GRPO) cho thấy KL thường không cần cho RLVR |
| `loss_type` | `"dapo"` | Chuẩn hóa loss theo tổng token của cả batch — tránh thiên vị câu dài/ngắn |
| `scale_rewards` | `"group"` | Chia cho độ lệch chuẩn trong nhóm |
| `num_generations` | 8 | Ta dùng 4 cho vừa T4 |
| **Ràng buộc batch** | | `generation_batch_size = per_device_train_batch_size × steps_per_generation` phải **chia hết cho** `num_generations`. Bản cũ của giáo trình đặt batch 1 với 4 generation → **lỗi ngay khi khởi tạo** |

### 1.3. Nhóm "không có tín hiệu"
Nếu cả 4 câu trả lời của một prompt đều đúng (hoặc đều sai như nhau) → `std = 0` → lợi thế = 0 → **prompt đó không dạy được gì**. Log `frac_reward_zero_std` cho biết tỉ lệ này. Vì vậy ta **chọn prompt khó** (những câu mô hình còn sai ở Buổi 08) thay vì toàn bộ tập train.

### 1.4. Ba hàm thưởng của P01 (`p01kit/rewards.py`)
| Hàm | Thưởng | Trọng số |
|---|---|---|
| `reward_format` | +1 đúng schema 8 khóa đúng thứ tự · 0 JSON sai schema · −1 không phải JSON | 0,5 |
| `reward_citation` | +1 quyết định đúng & mã văn bản đúng · −1 sai/bịa · −0,5 từ chối đúng nhưng vẫn ghi mã | 1,0 |
| `reward_expired` | Hết hạn: +1 bật cờ / **−2 bỏ sót** · Còn hạn: +0,5 / −1 báo nhầm · Từ chối: 0 | 1,0 |

### 1.5. Reward hacking — mô hình "lách luật"
Mô hình tối ưu **đúng thứ bạn chấm**, không phải thứ bạn muốn. Ví dụ có thể xảy ra: luôn bật `expired_warning=true` (né −2 nhưng dính −1 ở câu còn hạn — nếu tỉ lệ hết hạn trong prompt quá cao, chiến lược này vẫn "lời"); luôn từ chối khi không chắc. Phòng tránh: thưởng bất đối xứng **có phạt cả hai chiều**, cân bằng tỉ lệ prompt, và **luôn đánh giá bằng bộ chấm đầy đủ** (không chỉ nhìn reward).

---

## 2. Thực hành

### Khối 1 — Khởi động & kiểm thử hàm thưởng TRƯỚC khi train
```python
import os, sys, json, random, time
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from collections import Counter
from datasets import Dataset
from p01kit.data import load_split
from p01kit.prompts import build_messages
from p01kit.rewards import REWARD_FUNCS, reward_format, reward_citation, reward_expired
from p01kit.modeling import adapter_path, free
from p01kit.runs import resume_point
train, val, test = (load_split(s) for s in ("train", "val", "test"))
START = adapter_path("m08", "sft_dpo")       # hoặc adapter_path("m06", "sft_best") — theo kết luận Buổi 08

samples = json.load(open(f"{os.environ['P01_RUNS']}/m08_samples.json"))["train"]
flat = [(o, r) for r, outs in zip(train, samples) for o in outs]
outs, golds = [o for o, _ in flat], [json.dumps(r["gold"], ensure_ascii=False) for _, r in flat]
for f in REWARD_FUNCS:
    print(f"{f.__name__:<16}", Counter(f(outs, gold=golds)).most_common())
perfect = [json.dumps(r["gold"], ensure_ascii=False) for _, r in flat]
print("Đáp án hoàn hảo →", {f.__name__: Counter(f(perfect, gold=golds)) for f in REWARD_FUNCS})
```
- **Giải thích:** chấm thử hàm thưởng trên **mẫu thật** của mô hình (từ Buổi 08) và trên **đáp án hoàn hảo**. Nếu mẫu thật gần như toàn điểm tối đa → GRPO sẽ không có tín hiệu (mục 1.3).
- 📌 **Kết quả nên thấy:** đáp án hoàn hảo chỉ nhận giá trị tối đa của từng hàm (format 1,0; citation 1,0; expired ∈ {1,0; 0,5; 0,0}); mẫu thật có phân bố hỗn hợp, đặc biệt `reward_expired` có vài giá trị −2,0.

### Khối 2 — Chọn 128 prompt "khó" có cân bằng
```python
hard = set(json.load(open(f"{os.environ['P01_RUNS']}/m08_hard_ids.json")))
rng = random.Random(42)
pick = [r for r in train if r["id"] in hard] + [r for r in train if r["is_expired"] and r["id"] not in hard]
rest = [r for r in train if r not in pick]; rng.shuffle(rest)
pick = (pick + rest)[:128]
rl_rows = [{"prompt": build_messages(r), "gold": json.dumps(r["gold"], ensure_ascii=False)} for r in pick]
print(len(pick), "prompt | hết hạn:", sum(r["is_expired"] for r in pick), "| từ chối:", sum(r["is_unanswerable"] for r in pick),
      "| câu khó:", sum(r["id"] in hard for r in pick))
ds_rl = Dataset.from_list(rl_rows)
```
- **Giải thích:** prompt ở dạng **hội thoại** (list message) — TRL tự áp chat template; ta truyền `enable_thinking=False` qua `chat_template_kwargs` ở Khối 3. Cột `gold` được TRL chuyển nguyên vào hàm thưởng qua `**kwargs`.
- ⚠️ Kiểm tra tỉ lệ hết hạn trong `pick`: nếu quá cao (> 40%), chiến lược "luôn bật cờ" có thể có lợi → nguy cơ reward hacking (mục 1.5).

### Khối 3 — Nạp mô hình khởi đầu & cấu hình GRPO
```python
from trl import GRPOConfig, GRPOTrainer
model, tok = FastLanguageModel.from_pretrained(str(START), max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
FastLanguageModel.for_training(model)
cfg = GRPOConfig(
    output_dir=str(adapter_path("m09", "ckpt")), seed=42,
    learning_rate=5e-6, lr_scheduler_type="cosine", warmup_steps=0.1, weight_decay=0.1, optim="paged_adamw_8bit",
    per_device_train_batch_size=4, gradient_accumulation_steps=2, num_generations=4,   # 4×2 = 8 chia hết cho 4 ✔
    max_completion_length=p01kit.MAX_NEW_TOKENS, temperature=0.9, max_steps=64,
    reward_weights=[0.5, 1.0, 1.0], beta=0.0, mask_truncated_completions=True,
    chat_template_kwargs={"enable_thinking": False}, log_completions=True,
    logging_steps=1, save_strategy="steps", save_steps=16, save_total_limit=1,     # checkpoint lên Drive mỗi 16 bước
    fp16=True, report_to=os.environ.get("REPORT_TO", "none"))
print("generation_batch_size =", cfg.generation_batch_size, "→", cfg.generation_batch_size // cfg.num_generations, "prompt/lần sinh")
```
- 📌 **Kết quả nên thấy:** `generation_batch_size = 8 → 2 prompt/lần sinh`; 64 bước × 2 prompt = **128 prompt** = đúng 1 lượt qua `ds_rl`.
- **Giải thích:** `mask_truncated_completions=True` loại khỏi loss các câu bị cắt vì chạm `max_completion_length` — tránh dạy mô hình rằng JSON dở dang cũng được thưởng.

### Khối 4 — Train GRPO & theo dõi
```python
trainer = GRPOTrainer(model=model, processing_class=tok, reward_funcs=REWARD_FUNCS, args=cfg, train_dataset=ds_rl)
t0 = time.perf_counter(); trainer.train(resume_from_checkpoint=resume_point(cfg.output_dir)); grpo_min = (time.perf_counter() - t0) / 60
model.save_pretrained(adapter_path("m09", "grpo")); tok.save_pretrained(adapter_path("m09", "grpo"))
hist = [h for h in trainer.state.log_history if "reward" in h]
keys = [k for k in hist[0] if k.startswith("rewards/") and k.endswith("/mean")] + ["reward", "frac_reward_zero_std"]
for h in hist[::8]:
    print(f"bước {h['step']:>2}  " + "  ".join(f"{k.split('/')[1] if '/' in k else k}={h.get(k, float('nan')):.2f}" for k in keys))
print(f"GRPO: {grpo_min:.1f} phút")
```
- 📌 **Đọc log:** `reward` trung bình nên **tăng dần** (có nhiễu); `rewards/reward_expired/mean` là thứ ta quan tâm nhất; `frac_reward_zero_std` cao (> 0,7) nghĩa là phần lớn nhóm không mang tín hiệu → tăng `temperature` hoặc chọn prompt khó hơn.
- ⚠️ Tên khóa log có thể khác nhẹ giữa các phiên bản — in `hist[0].keys()` để xem tên thật.
- 💡 **Colab miễn phí:** GRPO là lượt train dài nhất khóa. Checkpoint (adapter + trạng thái optimizer, cỡ 100–200 MB, chỉ giữ 1 bản) được ghi vào `p01_runs/m09/ckpt` mỗi 16 bước. Bị ngắt → chạy lại Khối 1–4: `resume_point` tìm checkpoint mới nhất và train tiếp từ đó; `grpo_min` khi đó chỉ tính phần chạy tiếp.

### Khối 5 — Vẽ đường thưởng
```python
import matplotlib.pyplot as plt
steps = [h["step"] for h in hist]
for k in keys[:-1]:
    plt.plot(steps, [h.get(k) for h in hist], label=k)
plt.xlabel("bước"); plt.legend(); plt.title("GRPO — phần thưởng theo bước"); plt.show()
plt.plot(steps, [h.get("completions/mean_length", h.get("completion_length")) for h in hist]); plt.title("Độ dài completion"); plt.show()
```
- 🧠 **Độ dài completion** tăng/giảm bất thường là tín hiệu reward hacking sớm (ví dụ mô hình rút ngắn `verification_reasoning` vì không có thưởng nào cho nó).

### Khối 6 — 📊 Đánh giá & so sánh với mô hình khởi đầu
```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit.stats import mcnemar_exact
from p01kit import scoreboard
FastLanguageModel.for_inference(model)
res_g = evaluate(model, tok, test, batch_size=8)
scoreboard.record("M09", "grpo", res_g["summary"], latency_s=res_g["summary"]["sec_per_row"], notes=f"start={START.name} {grpo_min:.0f}min",
                  outputs=res_g["outputs"])
del model, trainer; free()
m0, t0k = FastLanguageModel.from_pretrained(str(START), max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
FastLanguageModel.for_inference(m0)
res_s = evaluate(m0, t0k, test, batch_size=8)
for k in ("refusal_acc", "citation_precision", "expired_recall", "false_expired_rate", "strict_acc", "acc_unseen_doc"):
    print(f"{k:<20} {res_s['summary'][k]} → {res_g['summary'][k]}")
print("McNemar strict:", mcnemar_exact([s["strict_ok"] for s in res_s["scores"]], [s["strict_ok"] for s in res_g["scores"]]))
```

### Khối 7 — Truy vết reward hacking trên test
```python
from p01kit.metrics import extract_json
always_warn = sum(bool((extract_json(o)[0] or {}).get("expired_warning")) for o in res_g["outputs"])
always_ref = sum(bool((extract_json(o)[0] or {}).get("is_refusal")) for o in res_g["outputs"])
print(f"Bật cờ hết hạn {always_warn}/100 (thật: {sum(r['is_expired'] for r in test)}) | Từ chối {always_ref}/100 (thật: {sum(r['is_unanswerable'] for r in test)})")
reason_len = [len((extract_json(o)[0] or {}).get("verification_reasoning", "")) for o in res_g["outputs"]]
print("Độ dài verification_reasoning trung bình:", sum(reason_len) / len(reason_len))
```
- **Cách đọc:** số lần bật cờ phải **gần** số câu thật sự hết hạn (10). Nếu tăng vọt → mô hình đang "đánh bừa an toàn". Nếu `verification_reasoning` ngắn dần → lý do kiểm chứng bị bỏ qua — cân nhắc thêm hàm thưởng thứ 4 (bài tập).

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M08", "sft_dpo"), ("M09", "grpo"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** Nếu `START` của bạn là `sft_best`, đổi bản *trước* thành `("M06", "sft_best")`. Mục tiêu nghiệp vụ là nhóm lỗi `khong_canh_bao_het_han` về 0 mà `false_expired_rate` không tăng — đọc ví dụ bị làm hỏng để bắt reward hacking.


| Chỉ số | Mô hình khởi đầu | Sau GRPO | Ghi chú |
|---|---|---|---|
| expired_recall | | | Mục tiêu = 1,00 |
| false_expired_rate | | | Không được tăng |
| refusal_acc | | | |
| citation_precision | | | |
| strict_acc | | | p = |
| Độ dài reasoning | | | Dấu hiệu hacking |
| Phút train | — | | |

---

## 🏢 Ở công ty thật

- **RL chỉ dùng khi đáng**: tốn GPU, khó tái lập, dễ bị lách luật. Đội thường chỉ làm khi có phần thưởng kiểm chứng được bằng code và SFT/DPO đã bão hòa.
- **Hàm thưởng là code sản phẩm**: được review, có test (`tests/test_training_utils.py`) và có phiên bản — đổi hàm thưởng là đổi mục tiêu tối ưu, phải ghi vào hồ sơ phát hành.
- **Theo dõi reward hacking bằng bộ đánh giá độc lập** và đọc mẫu ngẫu nhiên định kỳ trong lúc train, không chỉ nhìn đường thưởng đi lên.

---

## 4. Bài tập

**Bắt buộc**
1. **Bỏ bất đối xứng:** sửa `reward_expired` để bỏ sót chỉ bị −1. Train lại và so `expired_recall`. Bất đối xứng có thật sự cần thiết?
2. **Hàm thưởng thứ 4 — grounding:** viết `reward_grounded(completions, context, **kw)` trả +0,5 nếu mọi giấy tờ/nơi nộp có trong ngữ cảnh (dùng `p01kit.grounding.check_grounding`), −1 nếu bịa. Thêm cột `context` vào `ds_rl` và train lại.
3. Đặt `beta=0.04` (bật KL, cần thêm bộ nhớ cho reference). So sánh độ ổn định đường thưởng và chất lượng.

**Nâng cao**
4. Tăng `num_generations` lên 8 (và `per_device_train_batch_size=8`). `frac_reward_zero_std` thay đổi thế nào? Đánh đổi bộ nhớ ra sao?
5. Đọc mục "Dr. GRPO" (Liu et al., 2025) về thiên lệch độ dài trong chuẩn hóa loss, liên hệ với `loss_type="dapo"`.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: GRPO khác PPO thế nào?**
   A: GRPO bỏ mô hình value/critic, ước lượng lợi thế bằng cách so phần thưởng của nhiều mẫu cùng prompt (chuẩn hóa theo nhóm); tiết kiệm bộ nhớ và đơn giản hơn, phù hợp phần thưởng kiểm chứng được.
2. **Q: "Verifiable reward" là gì, ví dụ?**
   A: Phần thưởng tính được bằng chương trình, không cần mô hình reward học từ người: đáp án toán đúng, code qua test, JSON hợp lệ, mã văn bản khớp nhãn, cờ hết hạn đúng.
3. **Q: Vì sao có prompt không đóng góp gradient trong GRPO?**
   A: Khi mọi mẫu trong nhóm có cùng phần thưởng, độ lệch chuẩn bằng 0 nên lợi thế bằng 0. Cần prompt đủ khó (tỉ lệ đúng trung gian) và đủ đa dạng khi sampling.
4. **Q: Thiết kế phần thưởng chống reward hacking thế nào?**
   A: Phạt cả hai chiều lỗi (bỏ sót và báo nhầm), cân bằng phân bố prompt, kết hợp nhiều hàm thưởng có trọng số, theo dõi độ dài và các thống kê hành vi, và luôn đánh giá bằng bộ chấm độc lập.
5. **Q: Khi nào cần KL penalty (β > 0)?**
   A: Khi phần thưởng có thể bị khai thác (reward model học được) hoặc cần giữ phong cách/năng lực chung gần reference; với phần thưởng kiểm chứng chặt, nhiều công trình bỏ KL để tiết kiệm bộ nhớ và cho phép khám phá.
6. **Q: Ràng buộc batch của GRPOTrainer?**
   A: Số completion sinh mỗi lần (`generation_batch_size`) phải chia hết cho `num_generations` để mỗi prompt có đủ G mẫu trong cùng nhóm.

---

## 6. Checklist cuối buổi
- [ ] Đã kiểm thử 3 hàm thưởng trên mẫu thật và đáp án hoàn hảo trước khi train.
- [ ] Hiểu và giải thích được `generation_batch_size`, `frac_reward_zero_std`, `beta=0`.
- [ ] `m09/grpo` trên Drive; bảng so sánh có kiểm tra reward hacking.

## Tài liệu tham khảo
- Shao et al., *DeepSeekMath* (2024) — giới thiệu GRPO.
- DeepSeek-AI, *DeepSeek-R1* (2025) — RL với phần thưởng kiểm chứng.
- Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* (2025).
- Liu et al., *Understanding R1-Zero-Like Training: A Critical Perspective* (Dr. GRPO, 2025).
- Tài liệu TRL — `GRPOTrainer`, `GRPOConfig`.
