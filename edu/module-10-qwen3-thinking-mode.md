# Buổi 10 — Chế độ Thinking của Qwen3: Fine-tune mà không làm "mất trí" suy luận

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (50' lý thuyết · 2h40 lab) |
| **Đầu vào** | `m06/sft_best` (SFT chỉ dữ liệu non-thinking); `p01kit/cot.py` |
| **🎯 Sản phẩm** | (1) **Phép thử "sụp đổ thinking"** trên mô hình SFT; (2) Adapter **`m10/mixed_think`** train trộn 2 chế độ; (3) **Bảng đánh đổi** độ chính xác – số token – độ trễ của thinking vs non-thinking; (4) Đề xuất **bộ định tuyến** khi nào nên bật thinking |
| **Chỉ số đo** | % output có suy luận thật · số token sinh/câu · giây/câu · `answer_acc`, `acc_unseen_doc`, `expired_recall` ở cả 2 chế độ |
| **Chế độ code** | 🧩 Khối gộp |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Đo thinking vs non-thinking theo ba trục: độ chính xác, số token, độ trễ.
2. Phát hiện "sụp đổ thinking" sau khi SFT chỉ với dữ liệu non-thinking.
3. Train mô hình trộn 2 chế độ và dựng bộ định tuyến "chỉ nghĩ khi cần".

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): Bảng 3 mô hình × 2 chế độ có số token và giây/câu; quyết định bật/tắt thinking cho P01 kèm số độ trễ, so với ngân sách độ trễ CPU của Buổi 13.

---

## 0. Bức tranh lớn

Qwen3 là mô hình **hybrid thinking**: cùng một bộ trọng số vừa trả lời nhanh (non-thinking) vừa suy luận dài trong `<think>…</think>` (thinking). Đây là điểm khác biệt lớn so với các mô hình cũ và là chủ đề phỏng vấn "nóng" khi làm việc với Qwen3.

Rủi ro thực tế: từ Buổi 02 đến 09 ta chỉ train dữ liệu non-thinking (khối think rỗng). Mô hình có thể **quên cách suy luận** — khi bật thinking, nó trả về khối nghĩ rỗng hoặc suy luận lộn xộn. Buổi này **đo** hiện tượng đó và học cách **giữ** cả hai năng lực.

---

## 1. Lý thuyết

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** Thinking giống **nháp ra giấy trước khi trả lời**: đúng hơn ở câu khó nhưng tốn giấy và thời gian. Bộ định tuyến là **chỉ lấy giấy nháp khi thấy câu khó**.

### 1.1. Qwen3 được huấn luyện 2 chế độ thế nào (theo technical report)
1. *Long-CoT cold start* — SFT trên chuỗi suy luận dài.
2. *Reasoning RL* — RL với phần thưởng kiểm chứng (toán, code).
3. **Thinking mode fusion** — SFT trộn dữ liệu có suy luận và không suy luận, dạy công tắc `enable_thinking` và lệnh mềm `/think`, `/no_think`.
4. *General RL* — căn chỉnh tổng quát.

Bản 1.7B nhận năng lực này qua **chưng cất (distillation)** từ mô hình lớn hơn. → Năng lực thinking của 1.7B là thứ "mượn được", dễ mất khi fine-tune lệch.

### 1.2. Công tắc ở mức token
| Cách bật/tắt | Cơ chế |
|---|---|
| `enable_thinking=False` (template) | Chèn sẵn `<think>\n\n</think>\n\n` → mô hình bỏ qua suy luận |
| `enable_thinking=True` (mặc định) | Prompt dừng ở `<|im_start|>assistant\n` → mô hình tự mở `<think>` |
| `/think`, `/no_think` trong tin nhắn user | Công tắc mềm theo từng lượt (khi `enable_thinking=True`) |

Ở lượt hội thoại **cũ**, template **xóa nội dung think** của assistant khỏi lịch sử → suy luận không chiếm ngữ cảnh các lượt sau (đã thấy ở Buổi 00, Ô 4).

### 1.3. Giữ năng lực thinking khi fine-tune
- Nguyên tắc: **dữ liệu train phải có cả hai chế độ**, như giai đoạn 3 ở trên. Hướng dẫn thực hành của Unsloth cho Qwen3 gợi ý trộn phần lớn mẫu có suy luận với một phần mẫu không suy luận; tỉ lệ cụ thể là **siêu tham số** — buổi này dùng 50/50 và để bạn thử tỉ lệ khác.
- Vết suy luận (reasoning trace) lấy từ đâu? (a) mô hình lớn hơn (chưng cất), (b) chính mô hình rồi lọc theo đáp án đúng (*rejection sampling*), (c) **sinh bằng luật** như P01: duyệt từng văn bản theo 3 câu hỏi *đúng chủ đề? đúng địa bàn? còn hiệu lực?* (`p01kit.cot.reasoning_trace`). Cách (c) không có ảo giác nhưng "cứng", cách (b) tự nhiên hơn — bài tập nâng cao.

### 1.4. Khi nào thinking đáng tiền ở production?
Trên CPU (ADR 0001), số token sinh **quyết định độ trễ**. Nếu thinking sinh thêm vài trăm token, độ trễ tăng tương ứng. Thinking chỉ đáng bật khi nó **cải thiện có ý nghĩa** ở nhóm câu khó (ví dụ `unseen_doc`, câu có văn bản hết hạn) → ý tưởng **bộ định tuyến**: mặc định non-thinking, chỉ bật thinking khi mức chắc chắn thấp.

---

## 2. Thực hành

### Khối 1 — Khởi động & hàm đo số token suy luận
```python
import os, sys, json, re, time
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from p01kit.data import load_split, load_corpus
from p01kit.cot import reasoning_trace, to_mixed_example
from p01kit.infer import evaluate
from p01kit.modeling import adapter_path, free
from p01kit import scoreboard
corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)

def think_stats(outputs, tok):
    lens = []
    for o in outputs:
        m = THINK_RE.search(o)
        body = m.group(1).strip() if m else ""
        lens.append(len(tok(body, add_special_tokens=False)["input_ids"]) if body else 0)
    return {"pct_co_suy_luan": round(sum(l > 0 for l in lens) / len(lens), 3), "token_suy_luan_TB": round(sum(lens) / len(lens), 1)}

def run_both_modes(model, tok, label, rows=test):
    out = {}
    for think in (False, True):
        r = evaluate(model, tok, rows, enable_thinking=think, batch_size=4 if think else 8,
                     max_new_tokens=1024 if think else p01kit.MAX_NEW_TOKENS)
        s = r["summary"]; s.update(think_stats(r["outputs"], tok))
        scoreboard.record("M10", f"{label}-{'think' if think else 'nothink'}", s, latency_s=s["sec_per_row"], outputs=r["outputs"])
        out[think] = r
        print(f"{label:<14} think={think!s:<5} answer={s['answer_acc']:.3f} unseen={s['acc_unseen_doc']} "
              f"expired={s['expired_recall']} sec/câu={s['sec_per_row']} {think_stats(r['outputs'], tok)}")
    return out
```
- **Giải thích:** `max_new_tokens=1024` khi thinking vì khối suy luận có thể dài; batch nhỏ hơn để vừa KV-cache.

### Khối 2 — Phép thử "sụp đổ thinking": base vs mô hình SFT non-thinking
```python
base, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(base)
res_base = run_both_modes(base, tok, "base")
del base; free()
sft, tok = FastLanguageModel.from_pretrained(str(adapter_path("m06", "sft_best")), max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(sft)
res_sft = run_both_modes(sft, tok, "sft_best")
print("\nVí dụ output thinking của sft_best:\n", res_sft[True]["outputs"][0][:600])
del sft; free()
```
- 📌 **Kết quả cần quan sát:** với **base**, `pct_co_suy_luan` khi think=True gần 1,0 và số token suy luận lớn. Với **sft_best**, kiểm tra xem `pct_co_suy_luan` có **tụt** không, và khối nghĩ (nếu có) có còn mạch lạc không. Đó là bằng chứng (hoặc phản chứng) cho hiện tượng sụp đổ.

### Khối 3 — Dựng dữ liệu trộn 2 chế độ (50/50, vết suy luận sinh bằng luật)
```python
r0 = next(r for r in train if r["is_expired"])
print("Vết suy luận mẫu:\n" + reasoning_trace(r0, corpus))
from p01kit.modeling import load_tokenizer
tok = load_tokenizer()
mixed = [to_mixed_example(tok, r, think=(i % 2 == 0), corpus=corpus) for i, r in enumerate(train)]
lens = [len(tok(m["prompt"] + m["completion"], add_special_tokens=False)["input_ids"]) for m in mixed]
print("Mẫu think:", sum(m["mode"] == "think" for m in mixed), "| no_think:", sum(m["mode"] == "no_think" for m in mixed), "| dài nhất:", max(lens))
ds_mixed = Dataset.from_list([{"prompt": m["prompt"], "completion": m["completion"]} for m in mixed])
```
- 📌 **Kết quả nên thấy (đã đo):** 134 / 134 mẫu; vết suy luận liệt kê đúng 3 văn bản với kết luận "dùng được/loại"; mẫu think dài trung bình **1.268**, tối đa **1.388** token (mẫu no-think tối đa 1.177) → vượt `MAX_SEQ_LEN=1280`, nên buổi này đặt `max_length = MAX_SEQ_LEN + 256 = 1536`. Bỏ qua bước đo này là TRL sẽ âm thầm cắt mất JSON cuối của các mẫu think (bẫy Buổi 04).

### Khối 4 — Train `m10/mixed_think` (cùng ngân sách bước với sft_best)
```python
sel = json.load(open(adapter_path("m06", "sft_best") / "selection.json"))
model, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN + 256, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
cfg = SFTConfig(output_dir=str(adapter_path("m10", "ckpt")), seed=42, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                num_train_epochs=sel["epochs"], learning_rate=sel["lr"], lr_scheduler_type="cosine", warmup_steps=0.05,
                optim="paged_adamw_8bit", fp16=True, max_length=p01kit.MAX_SEQ_LEN + 256, completion_only_loss=True,
                logging_steps=5, save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
SFTTrainer(model=model, processing_class=tok, train_dataset=ds_mixed, args=cfg).train()
model.save_pretrained(adapter_path("m10", "mixed_think")); tok.save_pretrained(adapter_path("m10", "mixed_think"))
FastLanguageModel.for_inference(model)
res_mix = run_both_modes(model, tok, "mixed_think")
```

### Khối 5 — Bảng đánh đổi & McNemar giữa 2 chế độ của cùng mô hình
```python
from p01kit.stats import mcnemar_exact
a, b = res_mix[False]["scores"], res_mix[True]["scores"]
for name, idx in [("tất cả", range(100)), ("văn bản mới", [i for i, r in enumerate(test) if r["slice"] == "unseen_doc"]),
                  ("hết hạn", [i for i, r in enumerate(test) if r["is_expired"]]), ("từ chối", [i for i, r in enumerate(test) if r["is_unanswerable"]])]:
    idx = list(idx)
    p = mcnemar_exact([a[i]["answer_ok"] for i in idx], [b[i]["answer_ok"] for i in idx])["p_value"]
    print(f"{name:<12} n={len(idx):>3} nothink={sum(a[i]['answer_ok'] for i in idx)/len(idx):.3f} "
          f"think={sum(b[i]['answer_ok'] for i in idx)/len(idx):.3f} p={p:.3f}")
```

### Khối 6 — Bộ định tuyến đơn giản: chỉ "nghĩ" khi không chắc
```python
from p01kit.metrics import extract_json
def needs_thinking(output_nothink: str) -> bool:
    o, err = extract_json(output_nothink)
    return o is None or o.get("is_refusal") or o.get("expired_warning")   # ca rủi ro → nghĩ lại cho chắc
route = [needs_thinking(o) for o in res_mix[False]["outputs"]]
final = [res_mix[True]["scores"][i] if route[i] else res_mix[False]["scores"][i] for i in range(100)]
from p01kit.metrics import summarize
s_route = summarize(final, with_ci=False)
cost = (res_mix[False]["summary"]["sec_per_row"] * 100 + res_mix[True]["summary"]["sec_per_row"] * sum(route)) / 100
print(f"Định tuyến: bật thinking cho {sum(route)}/100 câu | answer_acc={s_route['answer_acc']:.3f} | ước tính {cost:.2f} s/câu")
```
- **Giải thích:** mô phỏng hệ thống 2 tầng — trả lời nhanh trước, chỉ gọi thinking cho ca rủi ro (từ chối, hết hạn, JSON hỏng). Chi phí ước tính = tất cả câu chạy non-thinking + phần câu được định tuyến chạy thêm thinking.

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M10", "sft_best-nothink"), ("M10", "mixed_think-think"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** So mô hình non-thinking tốt nhất với mô hình trộn 2 chế độ khi bật thinking: cải thiện (nếu có) thường ở `slice=unseen_doc`; nhớ cân với số token và độ trễ ở bảng phía trên.


| Mô hình · chế độ | % có suy luận | Token suy luận TB | s/câu | answer_acc | acc_unseen_doc | expired_recall |
|---|---|---|---|---|---|---|
| base · nothink | — | 0 | | | | |
| base · think | | | | | | |
| sft_best · nothink | — | 0 | | | | |
| sft_best · think | | | | | | |
| mixed_think · nothink | — | 0 | | | | |
| mixed_think · think | | | | | | |
| mixed_think · **định tuyến** | | | | | — | — |

> 📌 **Kết luận cần viết:** (1) Có hiện tượng sụp đổ thinking ở `sft_best` không? (2) Huấn luyện trộn có giữ được thinking mà không làm giảm chế độ non-thinking? (3) Với ràng buộc CPU, chế độ nào/định tuyến nào bạn chọn cho production?

---

## 🏢 Ở công ty thật

- **Bật thinking là đánh đổi chi phí & độ trễ lấy độ chính xác.** Quyết định dựa trên SLO và ngân sách (Buổi 15), thường bằng một bộ định tuyến như Khối 6 chứ không bật cho mọi request.
- **Vết suy luận cũng là dữ liệu người dùng**: áp cùng chính sách che PII và thời hạn lưu như câu trả lời; nhiều sản phẩm không hiển thị vết suy luận cho người dùng cuối.

---

## 4. Bài tập

**Bắt buộc**
1. Đổi tỉ lệ trộn thành 75% think / 25% no-think. So sánh với 50/50 ở cả hai chế độ.
2. Thử công tắc mềm: thêm `/no_think` vào cuối câu hỏi khi `enable_thinking=True`. Mô hình `mixed_think` có tôn trọng không? (Gợi ý: tạo `rows` mới với câu hỏi đã nối thêm.)
3. Đọc 5 output thinking sai của `mixed_think`: suy luận sai ở bước nào (chủ đề? địa bàn? hiệu lực?)

**Nâng cao — Rejection sampling thay cho vết suy luận bằng luật**
4. Dùng **base** ở chế độ thinking sinh 4 mẫu/câu train, giữ mẫu có JSON cuối **đúng hoàn toàn** (`strict_ok`), dùng phần `<think>` của nó làm vết suy luận. So sánh mô hình train bằng vết "tự nhiên" này với vết bằng luật. Lưu ý chi phí sinh.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: "Hybrid thinking" của Qwen3 là gì, bật/tắt thế nào?**
   A: Một mô hình hỗ trợ cả suy luận dài trong `<think>` và trả lời trực tiếp. Tắt bằng `enable_thinking=False` (template chèn khối think rỗng) hoặc lệnh mềm `/no_think` theo lượt.
2. **Q: Fine-tune chỉ dữ liệu non-thinking gây rủi ro gì?**
   A: Mô hình có thể mất/giảm khả năng suy luận khi bật thinking (khối nghĩ rỗng, lộn xộn) vì phân phối train chỉ có khối think rỗng. Cách giữ: trộn dữ liệu có suy luận và không suy luận.
3. **Q: Lấy vết suy luận để SFT từ đâu?**
   A: Chưng cất từ mô hình lớn; rejection sampling từ chính mô hình (giữ mẫu cho đáp án đúng); sinh bằng luật/chương trình khi bài toán có cấu trúc. Luôn lọc theo tính đúng của đáp án cuối.
4. **Q: Vì sao chat template xóa nội dung think ở các lượt trước?**
   A: Để không lãng phí ngữ cảnh và tránh mô hình bị ảnh hưởng bởi suy luận cũ; chỉ câu trả lời cuối của mỗi lượt được giữ trong lịch sử.
5. **Q: Khi nào nên bật thinking ở production?**
   A: Khi đo được cải thiện có ý nghĩa ở nhóm câu khó và ngân sách độ trễ cho phép; thường dùng định tuyến — mặc định nhanh, bật thinking cho ca rủi ro hoặc độ tin cậy thấp.
6. **Q: Tham số sampling khuyến nghị cho 2 chế độ của Qwen3?**
   A: Thinking: T=0,6, top_p=0,95, top_k=20 (tránh greedy vì dễ lặp); non-thinking: T=0,7, top_p=0,8, top_k=20.

---

## 6. Checklist cuối buổi
- [ ] Bảng 6 dòng (3 mô hình × 2 chế độ) có số token suy luận và độ trễ.
- [ ] `m10/mixed_think` trên Drive.
- [ ] Đề xuất định tuyến có số liệu chi phí–chất lượng.

## Tài liệu tham khảo
- Qwen Team, *Qwen3 Technical Report* (2025) — thinking mode fusion, chưng cất cho mô hình nhỏ.
- Unsloth documentation — *Qwen3: How to Run & Fine-tune* (giữ năng lực reasoning khi fine-tune).
- Zelikman et al., *STaR: Bootstrapping Reasoning With Reasoning* (2022) — rejection sampling vết suy luận.
