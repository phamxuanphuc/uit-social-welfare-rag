# Buổi 05 — RAFT thực chiến: Văn bản nhiễu, Học cách Từ chối & Ablation theo ngân sách bước

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (45' lý thuyết · 2h45 lab, ~35' chờ train) |
| **Đầu vào** | Adapter `m04/unsloth_sft` (= RAFT 30%) và số liệu của nó |
| **🎯 Sản phẩm** | **Báo cáo RAFT 3 cấu hình** (chỉ văn bản đúng · RAFT không từ chối · RAFT 30% từ chối) + **ma trận nhầm lẫn lỗi** (chọn nhầm văn bản nhiễu, bịa văn bản ngoài ngữ cảnh, từ chối thừa, từ chối thiếu) |
| **Chỉ số đo** | `answer_acc` trên câu trả lời được · `refusal_acc` · tỉ lệ *over-refusal* / *under-refusal* · tỉ lệ chọn nhầm văn bản nhiễu |
| **Chế độ code** | 🧩 **Khối gộp** — học viên đã thành thạo từng bước, giờ tập trung vào thiết kế thí nghiệm |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Dựng dữ liệu RAFT với văn bản nhiễu **khó** và một tỉ lệ câu phải từ chối.
2. So 3 cấu hình với **cùng ngân sách bước** để so sánh công bằng.
3. Phân loại lỗi thành ma trận thay vì chỉ nhìn một con số tổng.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): 3 cấu hình trên bảng điểm với cùng `max_steps=51`; báo cáo `A → C` cho thấy nhóm lỗi `le_ra_tu_choi` thay đổi thế nào; giải thích được đánh đổi "từ chối nhầm".

---

## 0. Bức tranh lớn

Trong production, retriever **không bao giờ** trả về đúng 1 văn bản hoàn hảo. Nó trả về top-k, trong đó có văn bản *gần đúng* — cùng chủ đề nhưng khác địa bàn, hoặc bản cũ đã hết hạn. Một mô hình chỉ được train trên ngữ cảnh "sạch" sẽ **tin mọi thứ trong ngữ cảnh** và trích nhầm. Đó chính là rủi ro **R1** (dùng nguồn hết hạn, bịa thủ tục) trong bảng rủi ro của đồ án.

RAFT (Zhang et al., 2024) giải quyết bằng 2 cơ chế: (1) **trộn văn bản nhiễu** vào ngữ cảnh khi train; (2) **rút văn bản đúng** khỏi một phần mẫu để mô hình học từ chối. Buổi này đo riêng từng cơ chế.

---

## 1. Lý thuyết

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** RAFT giống **bài đọc hiểu có cả đoạn văn gây nhiễu**, và có câu hỏi "không có trong bài" để học sinh học cách nói "không biết" thay vì bịa. Ngân sách bước cố định giống **cho mọi thí sinh cùng một thời gian làm bài**.

### 1.1. Ba cấu hình so sánh
| Mã | Dữ liệu train | Ngữ cảnh | Có câu phải từ chối? |
|---|---|---|---|
| **A · Oracle-only** | Chỉ câu trả lời được | **1** văn bản đúng | Không |
| **B · RAFT-0%** | Chỉ câu trả lời được | 1 đúng + 2 nhiễu | Không |
| **C · RAFT-30%** | Như M04 | 1 đúng + 2 nhiễu (70%) / 3 nhiễu (30%) + câu ngoài phạm vi | **Có** |

Tất cả **test trên cùng tập test có nhiễu** (đúng điều kiện production).

### 1.2. Kiểm soát ngân sách tính toán
A có ít mẫu hơn C (bỏ câu từ chối). Nếu train "3 epoch" cho cả 3, C sẽ được **nhiều bước cập nhật hơn** → so sánh không công bằng. Ta cố định **`max_steps = 51`** (bằng M02/M04) cho mọi cấu hình: cùng số lần cập nhật, cùng số mẫu nhìn thấy.

### 1.3. Phân loại lỗi — quan trọng hơn một con số tổng
| Loại lỗi | Định nghĩa | Hậu quả thực tế |
|---|---|---|
| **Chọn nhầm nhiễu** | Trích một văn bản **có trong ngữ cảnh** nhưng sai | Người dân nộp sai nơi, sai giấy tờ |
| **Bịa văn bản** | Trích mã **không có** trong ngữ cảnh | Mất niềm tin, rủi ro pháp lý |
| **Từ chối thiếu** (under-refusal) | Không có căn cứ mà vẫn trả lời | Nguy hiểm nhất |
| **Từ chối thừa** (over-refusal) | Có căn cứ mà lại từ chối | Phiền, đẩy việc cho cán bộ |
| **Bỏ sót hết hạn** | Văn bản đúng đã hết hạn nhưng không cảnh báo | Bị trả hồ sơ, trễ chi trả (R1) |

### 1.4. Tỉ lệ từ chối bao nhiêu là đủ?
- Quá ít → mô hình quen "đoán mò" (under-refusal).
- Quá nhiều (>50%) → mô hình "lười", từ chối cả câu có đáp án (over-refusal).
- RAFT gốc khảo sát tỉ lệ có văn bản đúng P ∈ {0%…100%} và thấy giá trị tối ưu phụ thuộc bộ dữ liệu → **phải đo**, không có số thần kỳ. Bài tập 1 cho bạn đo thêm 50%.

---

## 2. Thực hành

### Khối 1 — Khởi động & hàm train dùng chung (Unsloth)
```python
import os, sys, json, time
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from p01kit.data import load_split, load_corpus, build_dataset, render_context, gold_answer
from p01kit.prompts import to_prompt_completion
from p01kit.infer import evaluate
from p01kit.modeling import adapter_path, free
from p01kit import scoreboard
from p01kit.runs import resumable

corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
# os.environ["REPORT_TO"] = "wandb"   # bật nếu đã cấu hình WANDB_API_KEY như Buổi 04
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def fresh_model():
    m, t = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
    m = FastLanguageModel.get_peft_model(m, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
                                         target_modules=TARGETS, use_gradient_checkpointing="unsloth")
    return m, t

@resumable("m05")                 # mỗi cấu hình xong là lưu kết quả lên Drive; chạy lại ô chỉ train phần còn thiếu
def train_sft(name, rows, module="m05", max_steps=51, lr=2e-4, neftune=None, eval_rows=val):
    model, tok = fresh_model()
    ds = Dataset.from_list([to_prompt_completion(tok, r) for r in rows])
    dv = Dataset.from_list([to_prompt_completion(tok, r) for r in eval_rows])
    cfg = SFTConfig(output_dir=str(adapter_path(module, f"ckpt_{name}")), run_name=f"{module}-{name}", seed=42,
                    per_device_train_batch_size=4, gradient_accumulation_steps=4, max_steps=max_steps,
                    learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=0.05, optim="paged_adamw_8bit",
                    fp16=True, max_length=p01kit.MAX_SEQ_LEN, completion_only_loss=True, neftune_noise_alpha=neftune,
                    logging_steps=5, eval_strategy="steps", eval_steps=17, save_strategy="no",
                    report_to=os.environ.get("REPORT_TO", "none"))
    tr = SFTTrainer(model=model, processing_class=tok, train_dataset=ds, eval_dataset=dv, args=cfg)
    t0 = time.perf_counter(); tr.train(); minutes = (time.perf_counter() - t0) / 60
    model.save_pretrained(adapter_path(module, name)); tok.save_pretrained(adapter_path(module, name))
    FastLanguageModel.for_inference(model)
    res = evaluate(model, tok, test, batch_size=8)
    scoreboard.record(module.upper(), name, res["summary"], latency_s=res["summary"]["sec_per_row"],
                      notes=f"n_train={len(rows)} steps={max_steps} {minutes:.1f}min", outputs=res["outputs"])
    del model, tr; free()
    return res
```
- **Giải thích những điểm mới:** `max_steps=51` thay `num_train_epochs` (mục 1.2); `eval_steps=17` ≈ mỗi "epoch" của tập 268 mẫu; mỗi cấu hình **nạp base sạch** qua `fresh_model()`; adapter + bảng điểm tự động lưu.

### Khối 2 — Dựng dữ liệu cho A và B
```python
def oracle_only(rows):
    """Chỉ giữ câu trả lời được; ngữ cảnh = đúng 1 văn bản gốc; nhãn tính lại bằng luật."""
    out = []
    for r in rows:
        if r["policy_id"] is None or r["is_unanswerable"]:
            continue
        ctx = [corpus.by_id(r["policy_id"])]
        out.append({**r, "context_ids": [r["policy_id"]], "context": render_context([r["policy_id"]], corpus),
                    "gold": gold_answer(r["topic"], r["region_code"], ctx, corpus), "is_unanswerable": False})
    return out

rows_A = oracle_only(train)
rows_B = [r for r in build_dataset(corpus, unanswerable_ratio=0.0)["train"] if r["policy_id"] is not None]
rows_C = train
for n, rows in [("A oracle-only", rows_A), ("B raft-0%", rows_B), ("C raft-30%", rows_C)]:
    print(f"{n:<15} n={len(rows):>3} | từ chối={sum(r['gold']['is_refusal'] for r in rows):>3} | "
          f"TB văn bản/ngữ cảnh={sum(len(r['context_ids']) for r in rows)/len(rows):.1f}")
```
- 📌 **Kết quả nên thấy (tất định):** A = **179** mẫu, 0 từ chối, 1,0 văn bản/ngữ cảnh; B = **252** mẫu, 0 từ chối, 3,0; C = **268** mẫu, **89** từ chối, 3,0. (Bài tập 1: RAFT-50% cho 268 mẫu, 137 từ chối.)
- 🧠 `build_dataset(unanswerable_ratio=0.0)` dùng **cùng seed và cùng cách chia nhóm** → B có đúng các nhóm kịch bản như C, chỉ khác là không rút văn bản đúng.

### Khối 3 — Train A và B (C đã có từ Buổi 04, train lại cho đúng `max_steps`)
```python
results = {}
results["A_oracle_only"] = train_sft("A_oracle_only", rows_A)
results["B_raft_0"] = train_sft("B_raft_0", rows_B)
results["C_raft_30"] = train_sft("C_raft_30", rows_C)
```

### Khối 4 — Ma trận lỗi chi tiết
```python
from p01kit.metrics import extract_json

def error_profile(res, rows=test):
    c = dict(dung=0, chon_nham_nhieu=0, bia_van_ban=0, tu_choi_thieu=0, tu_choi_thua=0, bo_sot_het_han=0, json_hong=0)
    for out, r in zip(res["outputs"], rows):
        o, _ = extract_json(out)
        g = r["gold"]
        if o is None:
            c["json_hong"] += 1; continue
        ref, code = bool(o.get("is_refusal")), o.get("doc_code")
        ctx_codes = {corpus.by_id(i).doc_code for i in r["context_ids"]}
        if g["is_refusal"] and not ref:
            c["tu_choi_thieu"] += 1
        elif not g["is_refusal"] and ref:
            c["tu_choi_thua"] += 1
        elif not g["is_refusal"] and code != g["doc_code"]:
            c["chon_nham_nhieu" if code in ctx_codes else "bia_van_ban"] += 1
        elif g["expired_warning"] and not o.get("expired_warning"):
            c["bo_sot_het_han"] += 1
        else:
            c["dung"] += 1
    return c

for k, res in results.items():
    s = res["summary"]
    print(f"{k:<15} answer={s['answer_acc']:.3f} refusal={s['refusal_acc']:.3f} expired={s['expired_recall']}  {error_profile(res)}")
```
- 📌 **Cách đọc (giả thuyết cần kiểm chứng bằng số của bạn):**
  - A thường có **từ chối thiếu** cao nhất (chưa từng thấy câu phải từ chối) và **chọn nhầm nhiễu** nhiều (chưa từng thấy nhiễu).
  - B giảm *chọn nhầm nhiễu* nhưng vẫn *từ chối thiếu*.
  - C cân bằng nhất — nếu C có *từ chối thừa* tăng, đó là cái giá phải đo.

### Khối 5 — Kiểm định & phân tích theo lát cắt
```python
from p01kit.stats import mcnemar_exact
ok = {k: [s["answer_ok"] for s in r["scores"]] for k, r in results.items()}
print("A vs C:", mcnemar_exact(ok["A_oracle_only"], ok["C_raft_30"]))
print("B vs C:", mcnemar_exact(ok["B_raft_0"], ok["C_raft_30"]))
for k, r in results.items():
    s = r["summary"]
    print(f"{k:<15} seen={s['acc_seen_doc']} unseen={s['acc_unseen_doc']} ood={s['acc_ood']}")
```
- 🧠 Lát cắt `ood` chỉ có 4 câu → đừng kết luận từ nó; lát `unseen_doc` (72 câu) là chỉ số đáng tin nhất về khả năng **đọc** văn bản mới.

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M05", "A_oracle_only"), ("M05", "C_raft_30"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** RAFT phải làm giảm `le_ra_tu_choi` và câu `❌` bịa mã văn bản, có thể đổi lại vài câu `tu_choi_nham`. Đọc các ví dụ bị làm hỏng: mô hình có từ chối quá tay không?


| Cấu hình | n train | answer_acc | refusal_acc | expired_recall | Chọn nhầm nhiễu | Bịa | Từ chối thiếu | Từ chối thừa | p vs C |
|---|---|---|---|---|---|---|---|---|---|
| A oracle-only | | | | | | | | | |
| B RAFT-0% | | | | | | | | | |
| C RAFT-30% | 268 | | | | | | | | — |

> 📌 **Quyết định cho lộ trình:** cấu hình thắng (thường là C) trở thành **công thức dữ liệu chuẩn**. Buổi 06 sẽ tối ưu siêu tham số trên công thức này.

---

## 🏢 Ở công ty thật

- **Văn bản nhiễu "khó" lấy từ chính retriever đang chạy** (các kết quả top-k sai trong log) thay vì sinh bằng luật — đó đúng là phân phối lỗi mô hình sẽ gặp.
- **Tỉ lệ từ chối là quyết định sản phẩm**: từ chối nhiều → người dân bỏ đi hoặc dồn về quầy; từ chối ít → rủi ro hướng dẫn sai. Ngưỡng được chốt cùng chủ sản phẩm dựa trên chi phí của từng loại lỗi.
- **Ma trận lỗi được theo dõi xuyên phiên bản**, mỗi loại lỗi nguy hiểm (bịa văn bản, bỏ sót hết hạn) có người phụ trách.

---

## 4. Bài tập

**Bắt buộc**
1. **RAFT-50%:** `rows_D = build_dataset(corpus, unanswerable_ratio=0.5)["train"]`, train `D_raft_50`. Vẽ `refusal_acc` và `answer_acc trên câu trả lời được` theo tỉ lệ {0, 30, 50}%. Tìm điểm "đầu gối".
2. **Nhiễu ngẫu nhiên vs nhiễu khó:** viết hàm thay 2 văn bản nhiễu bằng 2 văn bản **ngẫu nhiên** (khác chủ đề, khác địa bàn) rồi train. Nhiễu khó có thật sự cần thiết không?
3. Với mỗi lỗi *bịa văn bản* của C, in câu hỏi và output. Có mẫu chung nào không (ví dụ toàn câu đời thường)?

**Nâng cao**
4. **Vị trí văn bản đúng:** đo `answer_acc` theo vị trí văn bản đúng trong ngữ cảnh (1, 2, 3) — trong tập test có lần lượt 24 / 21 / 11 câu. Có hiện tượng *lost-in-the-middle* không? Nếu có, đề xuất cách tăng cường dữ liệu.
5. Đọc mục "Chain-of-thought" của bài báo RAFT. So sánh với `verification_reasoning` của P01 — điểm giống và khác?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: RAFT khác SFT thường trên dữ liệu RAG thế nào?**
   A: RAFT trộn văn bản nhiễu vào ngữ cảnh train và rút văn bản đúng ở một phần mẫu, dạy mô hình vừa lọc nhiễu vừa từ chối khi thiếu căn cứ; SFT thường chỉ đưa ngữ cảnh đúng nên mô hình tin mọi thứ trong ngữ cảnh.
2. **Q: Tại sao phải cố định số bước khi so các tập dữ liệu khác kích thước?**
   A: Để so sánh công bằng về ngân sách tính toán; nếu cố định epoch, tập lớn hơn được nhiều cập nhật hơn và hiệu quả bị lẫn với lượng tính toán.
3. **Q: Over-refusal và under-refusal, cái nào nguy hiểm hơn trong P01?**
   A: Under-refusal (trả lời khi không có căn cứ) vì dẫn người dân làm sai thủ tục; over-refusal gây phiền nhưng còn đường chuyển cán bộ. Vì vậy ngưỡng và thưởng (Buổi 09) phải bất đối xứng.
4. **Q: "Hard negative" trong ngữ cảnh RAFT là gì?**
   A: Văn bản rất giống văn bản đúng (cùng chủ đề khác địa bàn, bản cũ hết hạn) — buộc mô hình kiểm tra metadata thay vì khớp từ khóa.
5. **Q: Làm sao biết mô hình đang "nhớ" văn bản thay vì "đọc"?**
   A: So độ chính xác trên văn bản đã thấy khi train với văn bản bị giấu hoàn toàn (lát `unseen_doc`); chênh lệch lớn là dấu hiệu ghi nhớ.

---

## 6. Checklist cuối buổi
- [ ] 3 cấu hình cùng `max_steps=51`, cùng tập test có nhiễu.
- [ ] Ma trận lỗi 6 loại cho từng cấu hình.
- [ ] Kết luận có p-value; đã chọn công thức dữ liệu chuẩn.

## Tài liệu tham khảo
- Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG* (2024).
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts* (2023).
- Rajpurkar et al., *Know What You Don't Know: Unanswerable Questions for SQuAD* (2018) — nguồn cảm hứng của UIT-ViQuAD 2.0.
