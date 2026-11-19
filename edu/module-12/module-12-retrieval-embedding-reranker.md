# Buổi 12 — Tầng Truy xuất: BM25 vs Embedding, Fine-tune `Qwen3-Embedding` & Rerank bằng `Qwen3-Reranker`

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (1h lý thuyết · 3h lab) |
| **Đầu vào** | `p01kit/retrieval.py` (BM25, Recall@k, MRR); `m06/sft_best` cho phần RAG đầu-cuối |
| **🎯 Sản phẩm** | (1) **Bảng so sánh 5 retriever** (BM25 · Vietnamese_Embedding · Qwen3-Embedding gốc · Qwen3-Embedding fine-tune · + Qwen3-Reranker); (2) Adapter **`m12/qwen3_emb_lora`**; (3) **Đo RAG đầu-cuối thật**: retriever + Qwen3-1.7B, so với ngữ cảnh "lý tưởng" của các buổi trước |
| **Chỉ số đo** | Recall@1/3, MRR (tổng & theo giọng chuẩn/đời thường) · độ chính xác hệ thống đầu-cuối · độ trễ |
| **Chế độ code** | 🧩 Khối gộp |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. So BM25, dense retriever và reranker trên cùng câu hỏi.
2. Fine-tune embedding bằng MNRL + hard negatives.
3. Đo RAG đầu-cuối: "cái giá của truy xuất" so với ngữ cảnh dựng sẵn.
4. Xử lý tiếng Việt **thật** trên 61.425 điều luật Zalo: NFC, câu không dấu, tách từ.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): Tái lập **đúng** bảng BM25 của Khối 8 (tất định); bảng 5 retriever trên dữ liệu P01; giải thích được rủi ro rò rỉ khi đánh giá mô hình đã được train trên Zalo.

---

## 0. Bức tranh lớn

Từ Buổi 01 đến 11, ngữ cảnh của mỗi câu hỏi được **dựng sẵn** (luôn chứa văn bản đúng nếu câu trả lời được). Production không như vậy: retriever có thể **trượt**. Trong RAG thực tế, rất nhiều lỗi đến từ tầng truy xuất chứ không phải LLM — và LLM giỏi đến đâu cũng không trả lời đúng được khi văn bản đúng không có trong ngữ cảnh.

Số liệu đã đo trên tập test P01 (56 câu trả lời được) với BM25 có lọc kép địa bàn:

| BM25 | Recall@1 | Recall@3 | MRR |
|---|---|---|---|
| Tất cả | **0,839** | 0,946 | 0,894 |
| Câu chuẩn | **1,000** | | |
| Câu đời thường | **0,609** | | |

→ **Fairness Gap 39 điểm ngay ở tầng truy xuất!** Người hỏi bằng thuật ngữ được phục vụ hoàn hảo, người hỏi "té xe, đứa nhỏ, giấy chi" thì trượt gần một nửa. Đây là vấn đề trung tâm của buổi học.

---

## 1. Lý thuyết

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** BM25 là **tìm sách bằng cách đếm từ trùng** với câu hỏi. Dense retriever **tìm theo ý nghĩa**: hai câu khác chữ nhưng cùng ý vẫn nằm gần nhau. Reranker là **thủ thư đọc kỹ 5 cuốn đầu** rồi xếp lại. Với tiếng Việt, máy so từng mã ký tự — "hỗ trợ" và "ho tro" là hai từ khác hẳn nhau.

### 1.1. Bi-encoder vs Cross-encoder
| | Bi-encoder (embedding) | Cross-encoder (reranker) |
|---|---|---|
| Cách tính | Mã hóa câu hỏi và văn bản **riêng rẽ** → cosine | Đưa **cặp (câu hỏi, văn bản)** vào cùng Transformer |
| Tốc độ | Văn bản mã hóa trước, tìm bằng ANN (HNSW ~ O(log N)) | Phải chạy mô hình cho **từng cặp** → chỉ dùng cho top-k |
| Chất lượng | Tốt, kém tinh tế với chi tiết | Tốt nhất, hiểu tương tác từng token |
| Trong P01 | `Qwen3-Embedding-0.6B` | `Qwen3-Reranker-0.6B` |

### 1.2. Họ mô hình Qwen3 cho truy xuất
- **Qwen3-Embedding-0.6B:** 28 lớp, vector **1.024 chiều**, ngữ cảnh 32K, hỗ trợ **instruction** cho câu truy vấn (`Instruct: …\nQuery: …`); văn bản **không** cần instruction. Trong sentence-transformers: `encode_query()` / `encode_document()` tự thêm đúng prompt.
- **Qwen3-Reranker-0.6B:** thực chất là LLM chấm "yes/no": điểm = `P("yes")` ở vị trí cuối của prompt dạng `<Instruct> <Query> <Document>`.
- Cùng họ với Qwen3-1.7B → cùng tokenizer, cùng "cảm nhận" tiếng Việt; có sẵn bản ONNX tương thích `fastembed` (`Qdrant/Qwen3-Embedding-0.6B-onnx`) — khớp ADR 0001 (CPU-only).

### 1.3. Fine-tune embedding: MultipleNegativesRankingLoss + hard negatives
- Mỗi batch gồm các bộ `(anchor=câu hỏi, positive=văn bản đúng, negative=văn bản nhiễu khó)`. MNRL coi **mọi positive/negative khác trong batch** cũng là âm bản (in-batch negatives) → batch càng lớn càng nhiều âm bản.
- ⚠️ **Bẫy trùng lặp:** nếu 2 câu hỏi cùng positive nằm chung batch, positive của câu này bị coi là **âm bản** của câu kia → tín hiệu sai. Giải pháp: `batch_sampler=BatchSamplers.NO_DUPLICATES`. (Giáo trình cũ nhân bản dữ liệu `* 4` rồi trộn ngẫu nhiên — đúng loại lỗi này.)
- **Prompt nhất quán:** câu hỏi lúc train phải có cùng instruction như lúc suy luận → dùng `prompts={"anchor": <query prompt>}` trong training args.
- Fine-tune **LoRA** cho mô hình embedding (`model.add_adapter(...)`) để vừa T4 và giữ năng lực gốc.

### 1.4. Đo RAG đầu-cuối đúng cách
Độ chính xác hệ thống = mô hình trích **đúng văn bản người dân thật sự cần** (`resolve(topic, region, toàn_bộ_kho)`), hoặc từ chối khi kho không có. Nếu retriever trượt mà LLM từ chối → **an toàn nhưng không hữu ích**; nếu LLM trích nhầm văn bản nhiễu → **nguy hiểm**. Ta đo cả hai.

---

## 2. Thực hành

### Khối 1 — Khởi động, dữ liệu truy xuất
```python
import os, sys, json, time
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
import torch, numpy as np, p01kit
from p01kit.data import load_split, load_corpus, resolve, HOLDOUT_POLICIES
from p01kit.retrieval import passage, region_filter, bm25_retrieve, recall_at_k, mrr
from p01kit.modeling import adapter_path, free
corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
q_test = [r for r in test if not r["is_unanswerable"]]          # 56 câu có văn bản đúng trong kho
gold = [r["policy_id"] for r in q_test]
PASSAGES = {p.id: passage(p, corpus) for p in corpus.policies}

def report(name, ranked, rows=q_test, gold=gold):
    out = {"R@1": recall_at_k(ranked, gold, 1), "R@3": recall_at_k(ranked, gold, 3), "MRR": mrr(ranked, gold)}
    for st in ("formal", "colloquial"):
        idx = [i for i, r in enumerate(rows) if r["style"] == st]
        out[f"R@1_{st}"] = recall_at_k([ranked[i] for i in idx], [gold[i] for i in idx], 1)
    print(f"{name:<28}" + "  ".join(f"{k}={v:.3f}" for k, v in out.items()))
    return out

results = {"BM25": report("BM25", [[pid for pid, _ in bm25_retrieve(r["question"], r["region_code"], corpus, k=5)] for r in q_test])}
```
- 📌 **Kết quả nên thấy (tất định):** `R@1=0.839 R@3=0.946 MRR=0.894 R@1_formal=1.000 R@1_colloquial=0.609`.

### Khối 2 — Hàm truy xuất dày (dense) có lọc kép địa bàn
```python
from sentence_transformers import SentenceTransformer

def dense_ranker(model, use_prompt: bool):
    ids = [p.id for p in corpus.policies]
    enc_doc = model.encode_document if use_prompt else model.encode
    enc_q = model.encode_query if use_prompt else model.encode
    D = enc_doc([PASSAGES[i] for i in ids], normalize_embeddings=True, batch_size=16)
    def rank(rows, k=5):
        Q = enc_q([r["question"] for r in rows], normalize_embeddings=True, batch_size=32)
        out = []
        for q, r in zip(Q, rows):
            allowed = {p.id for p in region_filter(corpus.policies, r["region_code"])}
            scores = sorted(((float(q @ D[j]), ids[j]) for j in range(len(ids)) if ids[j] in allowed), reverse=True)
            out.append([pid for _, pid in scores[:k]])
        return out
    return rank
```
- **Giải thích:** lọc địa bàn **trước** khi xếp hạng (đúng thiết kế Qdrant payload filter `region IN [đã chọn, Toàn quốc]` của đồ án). Kho P01 nhỏ (23 văn bản) nên tính cosine trực tiếp; kho lớn dùng ANN.

### Khối 3 — Hai mô hình embedding chưa fine-tune
```python
vi = SentenceTransformer("AITeamVN/Vietnamese_Embedding", device="cuda")
results["Vietnamese_Embedding"] = report("Vietnamese_Embedding (bge-m3)", dense_ranker(vi, use_prompt=False)(q_test))
del vi; free()
qe = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
print("Prompt truy vấn của Qwen3-Embedding:", repr(qe.prompts.get("query")))
results["Qwen3-Emb gốc"] = report("Qwen3-Embedding-0.6B gốc", dense_ranker(qe, use_prompt=True)(q_test))
```
- **Giải thích:** `AITeamVN/Vietnamese_Embedding` là bge-m3 tinh chỉnh tiếng Việt (~568M tham số) — baseline mạnh **chuyên tiếng Việt**; Qwen3-Embedding là đa ngôn ngữ. So sánh cho bạn biết đi theo hướng "chuyên ngôn ngữ" hay "cùng họ mô hình" lợi hơn.
- 📌 Quan sát đặc biệt cột `R@1_colloquial`: embedding có thu hẹp khoảng cách 39 điểm của BM25 không?

### Khối 4 — Dữ liệu huấn luyện bộ ba (anchor, positive, hard negative)
```python
from datasets import Dataset
triples = []
for r in train:
    if r["is_unanswerable"] or r["policy_id"] is None:
        continue
    pos = corpus.by_id(r["policy_id"])
    negs = [corpus.by_id(i) for i in r["context_ids"] if i != r["policy_id"]]
    hard = sorted(negs, key=lambda p: (p.topic != pos.topic, p.region != pos.region))[0]   # ưu tiên cùng chủ đề khác địa bàn
    triples.append({"anchor": r["question"], "positive": PASSAGES[pos.id], "negative": PASSAGES[hard.id]})
assert not any(h in t["positive"] + t["negative"] for t in triples for h in [PASSAGES[x] for x in HOLDOUT_POLICIES])
ds_tri = Dataset.from_list(triples)
print(len(triples), "bộ ba | ví dụ:", triples[0]["anchor"], "→ âm bản:", triples[0]["negative"][:80])
```
- 📌 **Kết quả nên thấy:** 179 bộ ba; `assert` xác nhận văn bản holdout (lát `unseen_doc`) **không** xuất hiện trong dữ liệu train embedding.

### Khối 5 — Fine-tune LoRA cho Qwen3-Embedding với MNRL + NO_DUPLICATES
```python
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
qe.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
args = SentenceTransformerTrainingArguments(
    output_dir=str(adapter_path("m12", "ckpt")), seed=42, num_train_epochs=3, per_device_train_batch_size=32,
    learning_rate=1e-4, warmup_steps=0.1, lr_scheduler_type="cosine", fp16=True,
    batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"anchor": qe.prompts["query"]},
    logging_steps=5, save_strategy="no", report_to=os.environ.get("REPORT_TO", "none"))
trainer = SentenceTransformerTrainer(model=qe, args=args, train_dataset=ds_tri, loss=losses.MultipleNegativesRankingLoss(qe))
t0 = time.perf_counter(); trainer.train(); print(f"{(time.perf_counter() - t0) / 60:.1f} phút")
qe.save_pretrained(str(adapter_path("m12", "qwen3_emb_lora")))
results["Qwen3-Emb fine-tune"] = report("Qwen3-Embedding + LoRA", dense_ranker(qe, use_prompt=True)(q_test))
```
- **Giải thích:** nạp mặc định FP32 (0,6B × 4 byte ≈ 2,4 GB) và train với `fp16=True` (autocast) → trọng số LoRA ở FP32, tránh lỗi unscale FP16. `prompts={"anchor": ...}` đảm bảo câu hỏi lúc train có **cùng instruction** như `encode_query` lúc suy luận.
- 📌 Kiểm tra riêng lát cắt **văn bản mới** (thuộc `HOLDOUT_POLICIES`) để biết mô hình học "cách khớp" hay chỉ nhớ 19 văn bản đã thấy:
```python
ranked_ft = dense_ranker(qe, use_prompt=True)(q_test)
idx_u = [i for i, r in enumerate(q_test) if r["slice"] == "unseen_doc"]
print("R@1 văn bản mới:", recall_at_k([ranked_ft[i] for i in idx_u], [gold[i] for i in idx_u], 1), f"(n={len(idx_u)})")
```

### Khối 6 — Rerank top-5 bằng Qwen3-Reranker-0.6B (cài đặt theo model card chính thức)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
rr_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side="left")
rr = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", dtype=torch.float16, device_map="cuda").eval()
PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
TASK = "Cho câu hỏi của người lao động về chính sách an sinh, xác định văn bản có trả lời đúng câu hỏi không"
YES, NO = rr_tok.convert_tokens_to_ids("yes"), rr_tok.convert_tokens_to_ids("no")

@torch.inference_mode()
def rerank_scores(query, docs):
    texts = [PREFIX + f"<Instruct>: {TASK}\n<Query>: {query}\n<Document>: {d}" + SUFFIX for d in docs]
    enc = rr_tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    logits = rr(**enc).logits[:, -1, :]
    return torch.log_softmax(torch.stack([logits[:, NO], logits[:, YES]], dim=1).float(), dim=1)[:, 1].exp().tolist()

t0 = time.perf_counter()
reranked = [[pid for _, pid in sorted(zip(rerank_scores(r["question"], [PASSAGES[p] for p in cand]), cand), reverse=True)]
            for r, cand in zip(q_test, ranked_ft)]
results["+ Qwen3-Reranker"] = report("Qwen3-Emb FT + Reranker", reranked)
print(f"Rerank: {(time.perf_counter() - t0) / len(q_test) * 1000:.0f} ms/câu (5 văn bản)")
```
- **Giải thích:** điểm = `P(yes)` sau softmax trên 2 logit {no, yes} tại token cuối. Instruction viết **bằng tiếng Việt, theo tác vụ** — Qwen khuyến nghị viết instruction riêng cho từng tác vụ.
- 📌 Rerank chỉ đổi thứ tự trong top-5 → `R@3`, `R@5` không đổi; kỳ vọng `R@1` và `MRR` tăng.

### Khối 7 — 🎯 RAG đầu-cuối thật: retriever → ngữ cảnh → Qwen3-1.7B (`sft_best`)
```python
from p01kit.data import render_context, gold_answer
from p01kit.infer import evaluate
from p01kit.metrics import extract_json
from p01kit.modeling import load_model, load_tokenizer
del rr, qe, trainer; free()
rank_all = dense_ranker(SentenceTransformer(str(adapter_path("m12", "qwen3_emb_lora")), device="cuda"), True)(test, k=3)
free()

def to_retrieved_row(r, top3):
    ctx = [corpus.by_id(i) for i in top3]
    return {**r, "context_ids": top3, "context": render_context(top3, corpus), "gold": gold_answer(r["topic"], r["region_code"], ctx, corpus)}

rows_rag = [to_retrieved_row(r, t) for r, t in zip(test, rank_all)]
tok, llm = load_tokenizer(), load_model(adapter=adapter_path("m06", "sft_best"))
res_rag = evaluate(llm, tok, rows_rag, batch_size=8)

need = [resolve(r["topic"], r["region_code"], corpus.policies) for r in test]       # văn bản người dân THẬT SỰ cần
def system_ok(out, n):
    o, _ = extract_json(out)
    if o is None: return False
    return bool(o.get("is_refusal")) if n is None else o.get("doc_code") == n.doc_code
sys_acc = sum(system_ok(o, n) for o, n in zip(res_rag["outputs"], need)) / len(test)
unsafe = sum((o := extract_json(out)[0]) is not None and not o.get("is_refusal") and n is not None and o.get("doc_code") != n.doc_code
             for out, n in zip(res_rag["outputs"], need))
print(f"Hệ thống đầu-cuối: đúng {sys_acc:.3f} | trích nhầm văn bản (nguy hiểm): {unsafe} câu")
```
- **Giải thích 2 con số:** `res_rag["summary"]` đo LLM có **đọc đúng ngữ cảnh được truy xuất** không (nhãn tính lại theo ngữ cảnh thật). `sys_acc` đo người dân **có nhận được văn bản họ cần** không — chỉ số kinh doanh thật.
- 📌 So `sys_acc` với `answer_acc` của `sft_best` trên ngữ cảnh dựng sẵn (Buổi 06): khoảng chênh chính là **cái giá của truy xuất**.

---

### Khối 8 — 🇻🇳 Thực chiến dữ liệu thật: Zalo AI Legal Retrieval (61.425 điều luật, 788 câu hỏi thật)
Các khối trên dùng kho 23 văn bản giả lập. Đi làm, bạn sẽ gặp kho **hàng chục nghìn văn bản thật** và câu hỏi do người thật viết. Khối này dùng bộ [`GreenNode/zalo-ai-legal-text-retrieval-vn`](https://huggingface.co/datasets/GreenNode/zalo-ai-legal-text-retrieval-vn) (giấy phép MIT, từ cuộc thi Zalo AI 2021) — **chỉ cần CPU**, chạy được trên runtime CPU để tiết kiệm hạn mức GPU.

```bash
!pip install -q bm25s==0.3.16 pyvi==0.1.1
```
```python
import unicodedata
from collections import Counter
import pandas as pd, bm25s
from huggingface_hub import hf_hub_download
from p01kit.vntext import normalize, segment, syllables, has_nfd, tone_style
from p01kit.retrieval import hit_at_k, mrr_at_k
ZALO = "GreenNode/zalo-ai-legal-text-retrieval-vn"
get = lambda f: pd.read_parquet(hf_hub_download(ZALO, f, repo_type="dataset"))
z_corpus, z_queries, z_qrels = (get(f"{x}/test-00000-of-00001.parquet") for x in ("corpus", "queries", "qrels"))
z_gold = z_qrels.groupby("query-id")["corpus-id"].apply(set).to_dict()
print(len(z_queries), "dòng câu hỏi,", z_queries["id"].nunique(), "id khác nhau")      # bộ công khai cũng có dòng trùng!
z_queries = z_queries[z_queries["id"].isin(z_gold)].drop_duplicates("id").reset_index(drop=True)
z_docs = (z_corpus["title"].fillna("") + ". " + z_corpus["text"].fillna("")).tolist()
z_ids, z_qs = z_corpus["id"].tolist(), z_queries["text"].tolist()
z_gs = [z_gold[i] for i in z_queries["id"]]
tones = Counter()
for d in z_docs:
    tones.update(tone_style(d))
print("Văn bản có NFD:", sum(map(has_nfd, z_docs)), "| số từ oa/oe/uy theo kiểu đặt dấu:", dict(tones))
```
```python
def bm25_eval(doc_tok, q_tok):
    ret = bm25s.BM25(); ret.index(doc_tok, show_progress=False)
    res, _ = ret.retrieve(q_tok, k=100, show_progress=False)
    ranked = [[z_ids[j] for j in row] for row in res]
    return {f"hit@{k}": round(hit_at_k(ranked, z_gs, k), 3) for k in (1, 10, 100)} | {"mrr@10": round(mrr_at_k(ranked, z_gs, 10), 3)}
low = lambda t: [w.lower() for w in syllables(t)]
nfc_docs = [low(normalize(d, tone=None)) for d in z_docs]
nfd_qs = [unicodedata.normalize("NFD", x) for x in z_qs]                      # mô phỏng câu dán từ PDF/macOS
no_accent_qs = [low(normalize(x, diacritics=False)) for x in z_qs]          # mô phỏng người gõ không dấu
real = {"A. âm tiết + NFC": bm25_eval(nfc_docs, [low(normalize(x, tone=None)) for x in z_qs]),
        "B. câu hỏi NFD, KHÔNG chuẩn hóa": bm25_eval(nfc_docs, [low(x) for x in nfd_qs]),
        "C. câu hỏi NFD + NFC": bm25_eval(nfc_docs, [low(normalize(x, tone=None)) for x in nfd_qs]),
        "D. + thống nhất dấu thanh": bm25_eval([low(normalize(d)) for d in z_docs], [low(normalize(x)) for x in z_qs]),
        "E. câu hỏi không dấu, kho có dấu": bm25_eval(nfc_docs, no_accent_qs),
        "F. bỏ dấu cả hai phía": bm25_eval([low(normalize(d, diacritics=False)) for d in z_docs], no_accent_qs)}
seg_docs = [[w.lower() for w in segment(normalize(d))] for d in z_docs]       # pyvi: ~1,5 phút trên CPU laptop
seg_qs = [[w.lower() for w in segment(normalize(x))] for x in z_qs]
real["G. tách từ (pyvi)"] = bm25_eval(seg_docs, seg_qs)
real["H. âm tiết + từ"] = bm25_eval([a + b for a, b in zip([low(normalize(d)) for d in z_docs], seg_docs)],
                                     [low(normalize(x)) + s for x, s in zip(z_qs, seg_qs)])
for k, v in real.items():
    print(f"{k:<34} {v}")
```
- 📌 **Kết quả đã đo** (tất định — BM25 không có ngẫu nhiên; `bm25s==0.3.16`, `pyvi==0.1.1`, Python 3.12, CPU; chính đoạn code trên đã được chạy trên dữ liệu thật ngày 18/11/2026 và in ra đúng các số này). Bạn chạy lại phải ra **đúng** như vậy:

| Cách xử lý | hit@1 | hit@10 | hit@100 | MRR@10 |
|---|---|---|---|---|
| A. âm tiết + NFC (mốc) | 0,435 | 0,822 | 0,964 | 0,570 |
| B. câu hỏi NFD, **không** chuẩn hóa | 0,001 | **0,015** | 0,166 | 0,004 |
| C. câu hỏi NFD + NFC | 0,435 | 0,822 | 0,964 | 0,570 |
| D. + thống nhất dấu thanh | 0,437 | 0,824 | 0,964 | 0,570 |
| E. câu hỏi **không dấu**, kho có dấu | 0,013 | **0,082** | 0,265 | 0,029 |
| F. bỏ dấu cả hai phía | 0,359 | 0,765 | 0,928 | 0,491 |
| G. tách từ (pyvi) | 0,448 | **0,858** | 0,972 | 0,582 |
| H. âm tiết + từ | **0,471** | 0,858 | 0,973 | **0,604** |

- 🧒 **Feynman:** máy không "nhìn" chữ như người — nó so **từng mã số**. "ộ" dựng sẵn và "o + ^ + ." là hai dãy mã khác nhau dù trông y hệt (dòng B: gần như **không tìm được gì**). Một dòng `unicodedata.normalize("NFC", ...)` sửa hết (dòng C). Tương tự, "ho tro" và "hỗ trợ" là hai từ khác hẳn với máy (dòng E).
- **Đọc bảng như một kỹ sư:**
  - Kho Zalo **không có** chuỗi NFD nào, nhưng dữ liệu của công ty bạn (PDF, Word cũ, macOS) thì có thể có → **luôn NFC ở cả lúc nạp kho lẫn lúc nhận câu hỏi**. Chi phí 1 dòng code, rủi ro nếu quên: mất ~98% kết quả.
  - Kho đặt dấu **kiểu cũ** áp đảo (trong 61.425 văn bản đo được 112.506 từ kiểu cũ "hòa, thủy" so với 6.514 từ kiểu mới "hoà, thuỷ"; 1.184 văn bản trộn cả hai) — vì câu hỏi cũng chủ yếu kiểu cũ nên thống nhất dấu thanh chỉ thêm +0,002 hit@10 (dòng D). Với dữ liệu trộn nhiều hơn, lợi ích sẽ lớn hơn: **đo trên dữ liệu của bạn**, đừng đoán.
  - Người dùng gõ không dấu là chuyện hằng ngày. Bỏ dấu cả hai phía (F) cứu từ 0,082 lên 0,765, nhưng mất 5,7 điểm so với có dấu vì "bảo/bão/báo" trùng nhau → hệ thống thật thường **đánh chỉ mục cả hai dạng** hoặc phục hồi dấu cho câu hỏi.
  - Tách từ (G) +3,6 điểm hit@10 vì "bảo_hiểm" là **một** đơn vị nghĩa; ghép âm tiết + từ (H) cho hit@1 và MRR tốt nhất — đó là lý do PhoBERT và các bi-encoder tiếng Việt đòi đầu vào đã tách từ.
- ⚠️ **Bộ công khai cũng bẩn:** tập câu hỏi có 818 dòng nhưng chỉ 788 câu khác nhau. Không khử trùng thì mẫu số sai và câu trùng được tính hai lần.

```python
from sentence_transformers import SentenceTransformer
bkai = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder", device="cuda")
bkai.max_seq_length = 256
D = bkai.encode([" ".join(s) for s in seg_docs], batch_size=128, normalize_embeddings=True, show_progress_bar=True)
Q = bkai.encode([" ".join(s) for s in seg_qs], batch_size=128, normalize_embeddings=True)
top = (Q @ D.T).argsort(axis=1)[:, ::-1][:, :100]
ranked = [[z_ids[j] for j in row] for row in top]
real["I. vietnamese-bi-encoder (dense)"] = {f"hit@{k}": round(hit_at_k(ranked, z_gs, k), 3) for k in (1, 10, 100)} | {"mrr@10": round(mrr_at_k(ranked, z_gs, 10), 3)}
print(real["I. vietnamese-bi-encoder (dense)"])
```
- **Giải thích:** mô hình PhoBERT nhận **từ đã tách** (`bảo_hiểm xã_hội`) — ta dùng lại `seg_docs` của dòng G. Cần GPU (mã hóa 61.425 đoạn); ghi số của bạn vào bảng trên.
- ⚠️ **Đọc model card trước khi tin con số.** Model card của `vietnamese-bi-encoder` ghi mô hình được huấn luyện trên **80% tập train của Zalo Legal 2021**. Nếu câu hỏi đánh giá trùng với dữ liệu huấn luyện của mô hình, điểm cao là do **rò rỉ** chứ không phải do giỏi. Trước khi kết luận "dense thắng BM25", kiểm tra xem tập bạn đánh giá có nằm trong dữ liệu huấn luyện của mô hình không — đó là câu hỏi phỏng vấn rất hay gặp.

## 3. So sánh hiệu quả

| Retriever | R@1 | R@3 | MRR | R@1 chuẩn | R@1 đời thường | Ghi chú |
|---|---|---|---|---|---|---|
| BM25 | 0,839 | 0,946 | 0,894 | 1,000 | 0,609 | Đã đo |
| Vietnamese_Embedding | | | | | | |
| Qwen3-Embedding gốc | | | | | | |
| Qwen3-Embedding + LoRA | | | | | | R@1 văn bản mới = |
| + Qwen3-Reranker | | | | | | ms/câu = |

| Hệ thống | Độ chính xác đầu-cuối | Trích nhầm (nguy hiểm) |
|---|---|---|
| sft_best + ngữ cảnh dựng sẵn (Buổi 06) | | |
| sft_best + retriever tốt nhất | | |

---

## 🏢 Ở công ty thật

- **Kho văn bản thật thay đổi liên tục**: cần pipeline nạp → tách đoạn → embed lại, xử lý PDF scan (OCR), quản lý phiên bản chỉ mục, và đồng bộ trạng thái hiệu lực (ApprovalAudit) với vector DB.
- **Đánh giá retriever trên câu hỏi thật** lấy từ log (đã che PII) và được gán nhãn văn bản đúng, cập nhật định kỳ — tập câu sinh bằng luật chỉ là điểm khởi đầu.
- **Hybrid và reranker được chọn theo ngân sách độ trễ**: reranker cross-encoder thường là bước tốn thời gian nhất trong chuỗi truy xuất.

---

## 4. Bài tập

**Bắt buộc**
1. **Hybrid search:** kết hợp BM25 và dense bằng *Reciprocal Rank Fusion* (`score = Σ 1/(60 + rank)`). RRF có vượt từng thành phần không, đặc biệt ở câu đời thường?
2. **Thiếu NO_DUPLICATES:** train lại với `BatchSamplers.BATCH_SAMPLER` (mặc định) trên dữ liệu đã nhân bản 4 lần (`triples * 4`). Recall thay đổi thế nào? Giải thích bằng mục 1.3.
3. **Baseline bkai:** cài `pyvi==0.1.1`, tách từ câu hỏi và văn bản bằng `ViTokenizer.tokenize` rồi đo `bkai-foundation-models/vietnamese-bi-encoder`. Nếu **không** tách từ thì sao? (Bài học: luôn đọc model card về tiền xử lý.)

**Nâng cao**
4. **Xuất ONNX cho production CPU:** gộp LoRA (`qe[0].auto_model.merge_and_unload()`), lưu mô hình, cài `sentence-transformers[onnx]==6.2.0`, nạp với `SentenceTransformer(path, backend="onnx")`, đo độ trễ CPU và kiểm tra Recall không đổi. Đối chiếu với `Qdrant/Qwen3-Embedding-0.6B-onnx` trong `fastembed`.
5. Đánh giá Qwen3-Embedding gốc vs fine-tune trên một phần `GreenNode/zalo-ai-legal-text-retrieval-vn` (MIT) — fine-tune hẹp cho P01 có làm giảm khả năng truy xuất văn bản luật tổng quát?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao RAG thường dùng 2 tầng embedding rồi reranker?**
   A: Bi-encoder mã hóa văn bản trước, tìm nhanh trên kho lớn (ANN ~ O(log N)) nhưng thiếu tương tác token; cross-encoder chính xác hơn nhưng phải chạy cho từng cặp nên chỉ áp dụng cho top-k nhỏ.
2. **Q: Hard negative là gì? Vì sao cần NO_DUPLICATES với MNRL?**
   A: Văn bản sai nhưng rất giống văn bản đúng (cùng chủ đề khác địa bàn). Với MNRL, các mẫu khác trong batch là âm bản; nếu trùng positive trong batch, positive thật bị coi là âm bản → tín hiệu sai.
3. **Q: Qwen3-Embedding dùng instruction thế nào?**
   A: Câu truy vấn được thêm `Instruct: <mô tả tác vụ>\nQuery: `; văn bản không cần. Phải dùng cùng prompt khi train và khi suy luận.
4. **Q: Làm sao đo một hệ RAG đúng cách?**
   A: Đo riêng tầng truy xuất (Recall@k, MRR), riêng tầng sinh với ngữ cảnh cố định (faithfulness, citation), và đầu-cuối so với nhu cầu thật của người dùng; tách các lỗi "an toàn nhưng vô ích" và "nguy hiểm".
5. **Q: BM25 tốt hoàn hảo với câu chuẩn nhưng kém với câu đời thường — giải pháp?**
   A: Embedding ngữ nghĩa, hybrid search (RRF), fine-tune embedding với cặp đời thường–văn bản, mở rộng truy vấn bằng từ điển đời thường, và đo Fairness Gap theo từng giọng.
6. **Q: Reranker kiểu LLM (yes/no) tính điểm thế nào?**
   A: Ghép instruction, truy vấn, văn bản vào prompt; lấy logit của token "yes" và "no" tại vị trí cuối, softmax giữa hai token, dùng P(yes) làm điểm liên quan.

---

## 6. Checklist cuối buổi
- [ ] Bảng 5 retriever có tách theo giọng chuẩn/đời thường.
- [ ] `m12/qwen3_emb_lora` trên Drive; đã kiểm tra lát văn bản mới.
- [ ] Đo được độ chính xác đầu-cuối và số ca trích nhầm.

## Tài liệu tham khảo
- Qwen Team, *Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models* (2025).
- Henderson et al., *Efficient Natural Language Response Suggestion for Smart Reply* (in-batch negatives, 2017).
- Cormack et al., *Reciprocal Rank Fusion* (2009).
- Tài liệu sentence-transformers v5/v6 — `SentenceTransformerTrainer`, `BatchSamplers`, `encode_query`.
