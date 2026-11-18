# Buổi 07 — Nhà máy Dữ liệu Tổng hợp: Sinh, Lọc, Khử trùng lặp, Chống rò rỉ & Đo lợi ích thật

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (50' lý thuyết · 2h40 lab) |
| **Đầu vào** | `m06/sft_best` + `selection.json` (LR, số epoch đã chọn) |
| **🎯 Sản phẩm** | (1) **Pipeline `Data-Factory`** sinh câu hỏi đa giọng bằng chính `Qwen3-1.7B`; (2) **Phễu dữ liệu** có số liệu từng tầng + *data card*; (3) Adapter **`m07/sft_aug`** và kết luận "dữ liệu tổng hợp có đáng không" |
| **Chỉ số đo** | Tỉ lệ giữ lại qua từng bộ lọc · lý do loại · số cặp nghi rò rỉ · `acc_colloquial`, `fairness_gap`, `acc_unseen_doc` · McNemar |
| **Chế độ code** | 🧩 Khối gộp |

---

## 0. Bức tranh lớn

Ở doanh nghiệp, câu hỏi "lấy đâu ra dữ liệu?" quan trọng hơn "dùng thuật toán nào?". Dữ liệu thật thường **ít, đắt, có PII**. Dữ liệu tổng hợp là lối ra — nhưng cũng là nguồn lỗi lớn nhất: nhãn bịa, câu lặp lại, và **rò rỉ sang tập test** khiến điểm đẹp ảo.

Thiết kế an toàn của buổi này: **mô hình chỉ viết lại câu hỏi** theo nhiều giọng (bác xe ôm, cô bán vé số, thanh niên gõ không dấu...), còn **ngữ cảnh và đáp án vàng giữ nguyên** từ luật nghiệp vụ → không thể có nhãn sai. Bộ lọc chặn các câu viết lại làm **đổi nghĩa** (thêm địa danh, thêm số tiền, tự thêm mã văn bản).

---

## 1. Lý thuyết

### 1.1. Các kiểu dữ liệu tổng hợp
| Kiểu | Ví dụ | Rủi ro chính |
|---|---|---|
| **Viết lại đầu vào** (paraphrase) — *buổi này* | Diễn đạt câu hỏi theo persona | Đổi nghĩa ngầm |
| Sinh cả hỏi lẫn đáp (Self-Instruct / Evol-Instruct) | Teacher tự ra đề, tự giải | **Nhãn ảo giác** lan sang student |
| Chưng cất (distillation) | Student học output/logits của teacher lớn | Giấy phép, kế thừa thiên lệch của teacher |
| Sinh từ luật/khuôn mẫu | `make_question()` của p01kit | Thiếu đa dạng, "giọng máy" |

### 1.2. Phễu chất lượng 4 tầng
1. **Parse:** đúng định dạng yêu cầu (mỗi câu một dòng, bắt đầu `- `).
2. **Giữ nghĩa (luật):** không thêm số, không thêm địa danh, không tự thêm mã văn bản (`p01kit.synthetic.filter_paraphrase`).
3. **Khử trùng lặp gần:** cosine n-gram ký tự ≥ 0,85 với câu gốc hoặc câu đã giữ → loại (chống "học vẹt" và sụp đổ đa dạng).
4. **Kiểm toán rò rỉ:** so với tập test ở **mức đầu vào** (Buổi 01) — dữ liệu sinh từ train không được "đụng" test.

### 1.3. Đo lợi ích đúng cách
- Thêm dữ liệu làm **tăng số bước** train → so sánh với `sft_best` lẫn lộn 2 yếu tố (dữ liệu mới + nhiều bước hơn). Bài tập 1 yêu cầu bản **cùng số bước** để tách riêng.
- Chỉ số mục tiêu: `acc_colloquial` và `fairness_gap` (rủi ro R2), đồng thời **không được giảm** `acc_formal` và `refusal_acc`.

### 1.4. Vì sao dùng chính Qwen3-1.7B làm teacher (không cần GPT/Claude)?
- Nhiệm vụ viết lại câu ngắn là việc **dễ** — mô hình nhỏ làm tốt; dữ liệu không rời khỏi hạ tầng của bạn (quyền riêng tư); giấy phép Apache-2.0 của Qwen3 cho phép dùng output để huấn luyện.
- Chế độ **non-thinking + sampling T=0,8** đủ đa dạng mà nhanh; thinking chỉ đáng dùng cho bộ lọc "LLM-as-judge" (bài tập nâng cao).

---

## 2. Thực hành

### Khối 1 — Khởi động, nạp teacher (base sạch, không adapter)
```python
import os, sys, json, random
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from collections import Counter
from p01kit.data import load_split, load_corpus
from p01kit.synthetic import PERSONAS, paraphrase_messages, parse_paraphrases, filter_paraphrase
from p01kit.dedup import near_dedup, leakage_audit
from p01kit.infer import generate
from p01kit.modeling import adapter_path, free
corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
teacher, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(teacher)
```

### Khối 2 — Sinh 3 cách diễn đạt cho mỗi câu train (mỗi câu một persona)
```python
src = [r for r in train if r["topic"]]                 # bỏ câu ngoài phạm vi
prompts = [tok.apply_chat_template(paraphrase_messages(r["question"], PERSONAS[i % len(PERSONAS)], n=3),
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for i, r in enumerate(src)]
raw = generate(teacher, tok, prompts, max_new_tokens=200, batch_size=16, do_sample=True, temperature=0.8, top_p=0.9, seed=42)
json.dump(raw, open(f"{os.environ['P01_RUNS']}/m07_raw_generations.json", "w"), ensure_ascii=False)
print(len(src), "câu nguồn →", len(raw), "lượt sinh. Ví dụ:\n", raw[0][:400])
```
- **Giải thích:** lưu output thô (`m07_raw_generations.json`) để có thể **lọc lại** với luật mới mà không phải sinh lại — tiết kiệm GPU và tái lập được.
- 📌 **Kết quả nên thấy:** 252 câu nguồn; mỗi output gồm vài dòng `- ...`. Một số output sai định dạng — đó là việc của tầng 1.

### Khối 3 — Phễu lọc: parse → giữ nghĩa
```python
funnel, reasons, cands = Counter(), Counter(), []
for r, out in zip(src, raw):
    paras = parse_paraphrases(out)
    funnel["sinh_ra"] += 3
    funnel["parse_ok"] += len(paras)
    for p in paras:
        ok, why = filter_paraphrase(r["question"], p, corpus)
        if ok:
            cands.append((r, p))
        else:
            reasons[why.split(" ")[0] + " " + why.split(" ")[1]] += 1
funnel["giu_nghia"] = len(cands)
print(dict(funnel)); print("Lý do loại:", reasons.most_common())
for r, p in random.Random(0).sample(cands, 5):
    print(f"• {r['question']}\n  → {p}")
```
- **Việc cần làm:** đọc 20 câu ngẫu nhiên đã qua lọc và **tự chấm** bao nhiêu câu thật sự giữ nghĩa. Đây là "precision của bộ lọc" — số liệu bắt buộc trong data card.

### Khối 4 — Khử trùng lặp gần & kiểm toán rò rỉ
```python
orig_q = [r["question"] for r in train]
texts = orig_q + [p for _, p in cands]
kept, dropped = near_dedup(texts, threshold=0.85)
kept_c = [cands[i - len(orig_q)] for i in kept if i >= len(orig_q)]
funnel["sau_dedup"] = len(kept_c)
print("Loại vì trùng gần:", len(cands) - len(kept_c))

full = lambda r, q: f"{r['region']} | {q} | {sorted(r['context_ids'])}"
L_q = leakage_audit([p for _, p in kept_c], [r["question"] for r in test], threshold=0.85)
L_in = leakage_audit([full(r, p) for r, p in kept_c], [full(r, r["question"]) for r in test], threshold=0.95)
print(f"Nghi rò rỉ — mức câu hỏi: {len({i for i,_,_ in L_q})} câu test | mức đầu vào: {len({i for i,_,_ in L_in})} câu test")
```
- 📌 **Kết quả nên thấy:** mức đầu vào = **0** (ngữ cảnh của train và test không bao giờ trùng vì chia theo nhóm). Mức câu hỏi có thể > 0 — nhắc lại bài học Buổi 01: không xóa vội, xem từng cặp.

### Khối 5 — Ghép tập train tăng cường + data card
```python
aug = [{**r, "id": f"{r['id']}::syn{k}", "question": p, "synthetic": True} for k, (r, p) in enumerate(kept_c)]
train_aug = train + aug
card = {"nguon": "p01_train.jsonl", "teacher": p01kit.MODEL_ID, "sinh": "non-thinking, T=0.8, top_p=0.9, seed=42",
        "personas": PERSONAS, "pheu": dict(funnel), "ly_do_loai": dict(reasons),
        "kich_thuoc": {"goc": len(train), "tong_hop": len(aug), "tong": len(train_aug)},
        "kiem_toan_ro_ri_muc_dau_vao": len({i for i, _, _ in L_in})}
os.makedirs(adapter_path("m07", "data"), exist_ok=True)
json.dump(card, open(adapter_path("m07", "data") / "data_card.json", "w"), ensure_ascii=False, indent=2)
with open(adapter_path("m07", "data") / "train_aug.jsonl", "w") as f:
    for r in train_aug: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(json.dumps(card["pheu"], ensure_ascii=False), card["kich_thuoc"])
del teacher; free()
```

### Khối 6 — Train `m07/sft_aug` với đúng cấu hình đã chọn ở Buổi 06
```python
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from p01kit.prompts import to_prompt_completion
from p01kit.infer import evaluate
from p01kit import scoreboard
sel = json.load(open(adapter_path("m06", "sft_best") / "selection.json"))
model, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
cfg = SFTConfig(output_dir=str(adapter_path("m07", "ckpt")), seed=42, per_device_train_batch_size=4, gradient_accumulation_steps=4,
                num_train_epochs=sel["epochs"], learning_rate=sel["lr"], lr_scheduler_type="cosine", warmup_steps=0.05,
                optim="paged_adamw_8bit", fp16=True, max_length=p01kit.MAX_SEQ_LEN, completion_only_loss=True,
                logging_steps=5, save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
tr = SFTTrainer(model=model, processing_class=tok, args=cfg,
                train_dataset=Dataset.from_list([to_prompt_completion(tok, r) for r in train_aug]))
tr.train()
model.save_pretrained(adapter_path("m07", "sft_aug")); tok.save_pretrained(adapter_path("m07", "sft_aug"))
FastLanguageModel.for_inference(model)
res = evaluate(model, tok, test, batch_size=8)
scoreboard.record("M07", "sft_aug", res["summary"], latency_s=res["summary"]["sec_per_row"], notes=f"n={len(train_aug)} steps={tr.state.global_step}")
```

### Khối 7 — So với `sft_best`: lợi ở đâu, mất ở đâu?
```python
from p01kit.metrics import score_one
from p01kit.stats import mcnemar_exact
base_out = json.load(open(f"{os.environ['P01_RUNS']}/m06_sft_best_outputs.json"))
b = [score_one(o, r) for o, r in zip(base_out, test)]
a = res["scores"]
for name, idx in [("tất cả", range(len(test))),
                  ("đời thường", [i for i, r in enumerate(test) if r["style"] == "colloquial"]),
                  ("chuẩn", [i for i, r in enumerate(test) if r["style"] == "formal"]),
                  ("văn bản mới", [i for i, r in enumerate(test) if r["slice"] == "unseen_doc"])]:
    idx = list(idx)
    acc_b = sum(b[i]["answer_ok"] for i in idx) / len(idx); acc_a = sum(a[i]["answer_ok"] for i in idx) / len(idx)
    p = mcnemar_exact([b[i]["answer_ok"] for i in idx], [a[i]["answer_ok"] for i in idx])["p_value"]
    print(f"{name:<12} n={len(idx):>3}  sft_best={acc_b:.3f} → sft_aug={acc_a:.3f}  p={p:.3f}")
print("fairness_gap:", res["summary"]["fairness_gap"])
```

---

## 3. So sánh hiệu quả

**Phễu dữ liệu (điền số của bạn):**
| Tầng | Số câu | Tỉ lệ giữ |
|---|---|---|
| Sinh ra (252 × 3) | 756 | 100% |
| Parse đúng định dạng | | |
| Giữ nghĩa (luật) | | |
| Sau khử trùng lặp | | |
| **Precision bộ lọc** (tự chấm 20 câu) | | |

**Tác động lên mô hình:**
| | acc tất cả | acc đời thường | acc chuẩn | acc văn bản mới | fairness_gap |
|---|---|---|---|---|---|
| `m06/sft_best` | | | | | |
| `m07/sft_aug` | | | | | |
| p-value | | | | | — |

> 📌 **Quyết định:** nếu `sft_aug` không tốt hơn có ý nghĩa ở nhóm đời thường **và** không làm giảm nhóm chuẩn → giữ `sft_best` làm mô hình nền. Ghi lý do vào `selection.json` của bạn.

---

## 🏢 Ở công ty thật

- **Kiểm tra điều khoản trước khi sinh dữ liệu bằng mô hình bên thứ 3**: điều khoản của nhiều nhà cung cấp API hạn chế dùng output để phát triển mô hình cạnh tranh — một lý do P01 dùng chính Qwen3 làm teacher.
- **Data card là yêu cầu khi bàn giao hoặc kiểm toán**: nguồn, quy trình lọc, tỉ lệ loại bỏ, giới hạn, giấy phép.
- **Người kiểm tra mẫu ngẫu nhiên** mỗi đợt dữ liệu tổng hợp (ví dụ 50 mẫu) và ghi tỉ lệ lỗi trước khi cho vào train; phễu lọc tự động không thay được bước này.

---

## 4. Bài tập

**Bắt buộc**
1. **Cùng số bước:** train lại `sft_aug` với `max_steps` = số bước của `sft_best`. Lợi ích còn lại có phải do dữ liệu đa dạng hơn hay chỉ do train lâu hơn?
2. Thêm persona "người dân tộc thiểu số nói tiếng Việt chưa thạo, câu ngắn, sai ngữ pháp". Bộ lọc giữ được bao nhiêu? Precision ra sao?
3. Viết bộ lọc thứ 5: loại câu viết lại **mất ý định** (ví dụ câu gốc hỏi "nộp ở đâu" mà câu mới hỏi "được bao nhiêu"). Gợi ý: dùng từ khóa ý định đơn giản; đo precision trước/sau.

**Nâng cao**
4. **LLM-as-judge làm bộ lọc:** dùng Qwen3-1.7B **thinking** hỏi "Câu B có cùng ý định và hoàn cảnh với câu A không? Trả lời CÓ hoặc KHÔNG." Đo mức đồng thuận với nhãn tự chấm của bạn (Cohen's kappa). Nó có đáng chi phí token thinking không?
5. Thay `near_dedup` bằng MinHash-LSH (`datasketch`) và so tốc độ trên 10.000 câu nhân bản.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Rủi ro lớn nhất khi dùng LLM sinh dữ liệu huấn luyện?**
   A: Nhãn ảo giác lan sang mô hình học; thiếu đa dạng/trùng lặp gây học vẹt; rò rỉ sang test (sinh từ tài liệu dùng để test); thiên lệch và vấn đề giấy phép của teacher.
2. **Q: Làm sao giữ nhãn đúng khi dùng dữ liệu tổng hợp?**
   A: Tách phần sinh (đầu vào) khỏi phần nhãn (suy ra từ nguồn sự thật/luật); kiểm chứng tự động bằng bộ lọc grounding/schema; lấy mẫu kiểm tra thủ công để đo precision bộ lọc.
3. **Q: Khử trùng lặp gần làm thế nào ở quy mô lớn?**
   A: MinHash + LSH trên shingle n-gram (ước lượng Jaccard, gần tuyến tính), hoặc embedding + ANN (Faiss); quy mô nhỏ dùng cosine n-gram O(n²).
4. **Q: Vì sao "1.000 mẫu sạch tốt hơn 50.000 mẫu bẩn" (LIMA)?**
   A: Hành vi căn chỉnh chủ yếu học từ định dạng/phong cách nhất quán; nhãn mâu thuẫn làm gradient triệt tiêu và dạy mô hình đoán bừa; dữ liệu ít nhưng sạch đủ "mở khóa" năng lực sẵn có.
5. **Q: Làm sao chứng minh dữ liệu tổng hợp thật sự giúp?**
   A: Ablation có kiểm soát số bước, đo trên lát cắt mục tiêu (đời thường) lẫn lát cắt có thể bị hại (chuẩn, từ chối), kiểm định thống kê, và kiểm toán rò rỉ.

---

## 6. Checklist cuối buổi
- [ ] Phễu 4 tầng có số liệu, có precision bộ lọc tự chấm.
- [ ] `data_card.json` + `train_aug.jsonl` trên Drive.
- [ ] Kết luận có p-value theo từng lát cắt; đã quyết định mô hình nền.

## Tài liệu tham khảo
- Wang et al., *Self-Instruct* (2022); Xu et al., *WizardLM / Evol-Instruct* (2023).
- Zhou et al., *LIMA: Less Is More for Alignment* (2023).
- Lee et al., *Deduplicating Training Data Makes Language Models Better* (2021).
- Shumailov et al., *The Curse of Recursion: Training on Generated Data Makes Models Forget* (2023).
