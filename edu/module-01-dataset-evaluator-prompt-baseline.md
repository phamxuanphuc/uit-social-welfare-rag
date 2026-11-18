# Buổi 01 — Bộ dữ liệu RAFT, Bộ chấm điểm & Giới hạn của Prompt Engineering

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3 giờ (1h lý thuyết · 2h lab) |
| **Đầu vào** | Ô 0a/0b của [Buổi 00](module-00-colab-qwen3-anatomy-baseline.md); dòng `M00/base-zero-shot` trong bảng điểm |
| **🎯 Sản phẩm** | (1) **Báo cáo kiểm toán dữ liệu** (rò rỉ, phân bố, holdout); (2) **Bộ chấm P01** hiểu tường tận từng chỉ số; (3) 2 dòng bảng điểm **few-shot** + kết luận McNemar "prompt có đủ không?" |
| **Chỉ số đo** | Số câu trùng giữa train/test ở 2 cấp độ · 8 chỉ số P01 · p-value McNemar · chi phí token của few-shot |
| **Chế độ code** | 🔬 Ô siêu nhỏ, giải thích từng dòng |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Giải thích được cách dữ liệu RAFT được sinh và vì sao phải **chia theo nhóm kịch bản** để chống rò rỉ.
2. Dùng bộ chấm 8 chỉ số và đọc đúng khoảng tin cậy của tập test 100 câu.
3. Đọc số liệu **thật** về hiệu lực văn bản pháp luật Việt Nam (171.556 văn bản) và chỉ ra lỗi trong metadata cào tự động.
4. Đo few-shot và dùng McNemar để biết cải thiện có thật hay chỉ là nhiễu.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): Đáp án hoàn hảo được chấm 1,0 ở mọi chỉ số; tái lập đúng số 171.556 văn bản / 52,9% hết hiệu lực toàn bộ; báo cáo `M00 → M01/base-3-shot` có p-value.

---

## 0. Bức tranh lớn — "80% công việc là dữ liệu và đánh giá"

Kỹ sư fine-tune giỏi không phải người gõ `.train()` nhanh nhất, mà là người **trả lời được 3 câu hỏi** trước khi train:
1. Dữ liệu của tôi có **nhãn đúng** không? (nhãn sai → mô hình học sai một cách rất tự tin)
2. Tập test có **rò rỉ** sang train không? (rò rỉ → điểm đẹp ảo, ra production sập)
3. Bộ chấm có **đo đúng thứ người dùng quan tâm** không? (đo sai → tối ưu sai hướng)

Buổi này trả lời cả 3 câu, rồi thử giải bài toán **chỉ bằng prompt** (few-shot). Nếu prompt đã đủ tốt, ta **không cần fine-tune** — một kết luận được đánh giá rất cao khi phỏng vấn.

---

## 1. Lý thuyết

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** Chia tập theo nhóm giống **ra đề thi không được có câu chỉ đổi vài chữ so với bài tập về nhà** — nếu không, điểm cao chỉ chứng tỏ học thuộc. Khoảng tin cậy là câu "điểm 70 ± 9": thi có 100 câu thì chênh 3 điểm chưa nói lên gì. McNemar chỉ đếm **những câu hai thí sinh làm khác nhau** — câu cả hai cùng đúng hay cùng sai không cho thông tin.

### 1.1. Bộ dữ liệu P01 được thiết kế thế nào (đọc [`p01kit/data.py`](p01kit/data.py))
- **Kho văn bản giả lập:** 23 văn bản ([`data/policies.json`](data/policies.json)) thuộc 11 chủ đề an sinh, 3 địa bàn + cấp Toàn quốc; 5 văn bản **đã hết hiệu lực**, trong đó 2 văn bản đã có bản mới thay thế.
  - ⚠️ Mã văn bản, điều khoản, mức tiền và giấy tờ đều **hư cấu** (xem khóa `_disclaimer` trong file), chỉ tên địa bàn là có thật. Khi chia sẻ output của mô hình (ảnh chụp, demo), luôn ghi kèm *"dữ liệu giả lập"* để không ai nhầm với quy định thật.
- **Kịch bản (scenario)** = (văn bản, ý định hỏi) với 3 ý định: *hồ sơ gồm gì*, *nộp ở đâu*, *mức hỗ trợ*. Mỗi kịch bản sinh **6 biến thể câu hỏi**: 3 giọng chuẩn (formal) + 3 giọng đời thường (colloquial) — ghép cặp theo `pair_id` để đo **Fairness Gap**.
- **RAFT (Retrieval-Augmented Fine-Tuning):** mỗi câu hỏi kèm **3 đoạn ngữ cảnh**. ~70% câu có 1 văn bản đúng + 2 văn bản nhiễu *khó* (cùng chủ đề khác địa bàn, bản cũ hết hạn, cùng địa bàn khác chủ đề). ~30% câu **bị rút văn bản đúng** → chỉ còn nhiễu → mô hình phải **từ chối**.
- **Nhãn suy ra bằng luật, không viết tay:** hàm `resolve()` chọn văn bản đúng theo thứ tự *đúng chủ đề & địa bàn ∈ {đã chọn, Toàn quốc}* → *còn hiệu lực trước* → *địa phương trước Toàn quốc*. Nhờ vậy **0 nhãn mâu thuẫn**, và có bộ test tự động kiểm chứng điều đó (`tests/test_data.py`).

### 1.2. Chia tập chống rò rỉ — 2 lát cắt test
| Lát cắt | Cách tạo | Đo năng lực gì |
|---|---|---|
| `seen_doc` | Chia **theo nhóm kịch bản** 80/10/10: mọi biến thể của 1 kịch bản nằm trọn trong 1 tập | Hiểu câu hỏi mới về văn bản đã gặp |
| `unseen_doc` | 4 văn bản bị **giấu hoàn toàn** khỏi train/val (kể cả vai trò văn bản nhiễu) | **Đọc hiểu văn bản mới** — năng lực thật của RAG |
| `ood` | Câu hỏi ngoài phạm vi (kết hôn, visa, sổ đỏ...) | Từ chối đúng chỗ |

### 1.3. Rò rỉ (leakage) phải đo ở mức **(đầu vào → nhãn)**, không phải mức câu hỏi
Hai văn bản cùng chủ đề ở 2 địa bàn có thể sinh **cùng một câu hỏi nguyên văn** ("Hồ sơ đề nghị hỗ trợ đào tạo nghề... gồm giấy tờ nào?") nhưng **đáp án khác nhau** (khác địa bàn, khác văn bản). Đó **không** phải rò rỉ — đó là đúng bài toán thực tế. Ô 6 sẽ cho bạn thấy con số cụ thể.

### 1.4. Bộ chấm P01 — 8 chỉ số tất định
| Chỉ số | Định nghĩa | Ngưỡng |
|---|---|---|
| `json_valid` | Tỉ lệ output parse được thành JSON object | — |
| `schema_ok` | Đúng 8 khóa, **đúng thứ tự** (verification_reasoning đứng đầu) | — |
| `refusal_acc` | Quyết định trả lời/từ chối đúng, trên **mọi** câu | ≥ 0,90 |
| `citation_precision` | Trong các câu mô hình **có trích** mã văn bản, tỉ lệ trích đúng | ≥ 0,90 |
| `answer_acc` | Đúng hoàn toàn quyết định chính (từ chối đúng, hoặc trích đúng văn bản) | — |
| `expired_recall` | Trong các câu có văn bản đúng đã hết hạn, tỉ lệ bật `expired_warning` | = 1,00 |
| `docs_f1` | F1 tập giấy tờ (so khớp sau chuẩn hóa) | — |
| `strict_acc` | Đúng **tất cả**: văn bản, cảnh báo, giấy tờ, nơi nộp | — |
| `fairness_gap` | \|answer_acc(chuẩn) − answer_acc(đời thường)\| | ≤ 0,10 |

### 1.5. Thống kê trung thực với tập test nhỏ
- **Bootstrap CI:** lấy mẫu lại có hoàn lại 2.000 lần → khoảng chứa 95% giá trị trung bình. Với n=100 và p≈0,8, khoảng rộng ≈ ±8 điểm %.
- **McNemar (so 2 mô hình trên cùng câu hỏi):** chỉ những câu **bất đồng** (A đúng–B sai = *b*, A sai–B đúng = *c*) mang thông tin. Nếu *b* và *c* gần nhau → chưa thể kết luận mô hình nào tốt hơn.

---

## 2. Thực hành từng ô nhỏ

*(Chạy Ô 0b của Buổi 00 trước.)*

### Ô 1 (6 dòng) — Kho văn bản
```python
from collections import Counter
from p01kit.data import load_corpus
corpus = load_corpus()
print(len(corpus.policies), "văn bản |", len(corpus.topics), "chủ đề |", corpus.regions)
print(Counter(p.status for p in corpus.policies))
print(Counter(corpus.region_name(p.region) for p in corpus.policies))
```
- 📌 **Kết quả nên thấy:** `23 văn bản | 11 chủ đề`, `Counter({'active': 18, 'expired': 5})`.

### Ô 2 (7 dòng) — Luật chọn văn bản đúng
```python
from p01kit.data import resolve
old, new = corpus.by_id("DN-TAINAN-2023"), corpus.by_id("DN-TAINAN-2026")
print(resolve("tai_nan", "DN", [old, new]).doc_code)    # có cả 2 bản
print(resolve("tai_nan", "DN", [old]).doc_code)         # chỉ còn bản cũ
print(resolve("tai_nan", "HCM", [new]))                  # lệch địa bàn
print(resolve("vay_von", "CT", corpus.policies).doc_code) # chỉ có văn bản Toàn quốc
```
- 📌 **Kết quả nên thấy:** `NQ-11/2026/HĐND-ĐN` · `QĐ-04/2023/UBND-ĐN` (dùng được nhưng **phải cảnh báo hết hạn**) · `None` (phải từ chối) · `QĐ-45/2026/QĐ-TTg`.
- 🧠 Đây là **luật nghiệp vụ** mà mô hình phải học. Viết nó ra code trước giúp bạn (1) sinh nhãn không lỗi, (2) có guardrail hậu kiểm ở Buổi 14.

### Ô 3 (8 dòng) — Soi 4 loại câu hỏi
```python
from p01kit.data import load_split
train, val, test = (load_split(s) for s in ("train", "val", "test"))
pick = {"trả lời được": lambda r: not r["is_unanswerable"] and not r["is_expired"],
        "hết hạn": lambda r: r["is_expired"],
        "từ chối (rút văn bản)": lambda r: r["is_unanswerable"] and r["topic"],
        "ngoài phạm vi": lambda r: r["slice"] == "ood"}
for name, f in pick.items():
    r = next(x for x in train if f(x))
    print(f"\n=== {name} === [{r['region']}] {r['question']}\n→ {r['gold']['verification_reasoning']}")
```
- **Việc cần làm:** với câu "hết hạn", in thêm `r["context"]` và tự kiểm tra lý do trong `verification_reasoning` có khớp ngữ cảnh không.

### Ô 4 (4 dòng) — Thống kê 3 tập
```python
from p01kit.data import dataset_stats
for name, rows in [("train", train), ("val", val), ("test", test)]:
    print(name, dataset_stats(rows))
```
- 📌 **Kết quả nên thấy (tất định):**
  - `train`: n=268, 46 nhóm, unanswerable 0,332, expired 0,067
  - `val`: n=34, 6 nhóm
  - `test`: n=100, 17 nhóm, unanswerable 0,44, expired 0,10, `unseen_doc`=72, `seen_doc`=24, `ood`=4

### Ô 5 (6 dòng) — Kiểm chứng không trùng nhóm & holdout
```python
from p01kit.data import HOLDOUT_POLICIES
g = {n: {r["group_id"] for r in rows} for n, rows in [("train", train), ("val", val), ("test", test)]}
print("train∩test:", g["train"] & g["test"], "| train∩val:", g["train"] & g["val"])
leak_ctx = [r["id"] for r in train + val if set(r["context_ids"]) & set(HOLDOUT_POLICIES)]
print("Văn bản holdout xuất hiện trong train/val:", leak_ctx)
```
- 📌 **Kết quả nên thấy:** hai tập rỗng `set()` và danh sách rỗng `[]`.

### Ô 6 (9 dòng) — Kiểm toán rò rỉ: mức câu hỏi vs mức đầu vào
```python
from p01kit.dedup import leakage_audit
tq, eq = [r["question"] for r in train], [r["question"] for r in test]
print("Câu test trùng NGUYÊN VĂN câu train:", sum(q in set(tq) for q in eq))
L1 = leakage_audit(tq, eq, threshold=0.85)
print("Mức câu hỏi  @0.85 → số câu test bị gắn cờ:", len({i for i, _, _ in L1}))
full = lambda r: f"{r['region']} | {r['question']} | {sorted(r['context_ids'])}"
L2 = leakage_audit([full(r) for r in train], [full(r) for r in test], threshold=0.95)
print("Mức đầu vào  @0.95 → số câu test bị gắn cờ:", len({i for i, _, _ in L2}))
```
- 📌 **Kết quả nên thấy (tất định):** **18** câu trùng nguyên văn · **30** câu bị gắn cờ ở mức câu hỏi · **0** câu ở mức đầu vào đầy đủ.
- 🧠 **Bài học:** nếu chỉ nhìn câu hỏi, bạn sẽ hoảng hốt xóa 30% tập test — và vô tình **xóa mất chính những ca khó nhất** (cùng câu hỏi, khác địa bàn → khác đáp án). Rò rỉ thật là khi **cùng đầu vào → cùng nhãn** xuất hiện ở cả hai tập.

### Ô 7 (8 dòng) — Kiểm tra bộ chấm bằng đáp án hoàn hảo và đáp án "cố tình sai"
```python
import json
from p01kit.metrics import score_one, summarize, format_summary
perfect = [score_one(json.dumps(r["gold"], ensure_ascii=False), r) for r in test]
print(format_summary(summarize(perfect, with_ci=False)))
r = next(x for x in test if x["is_expired"])
bad = dict(r["gold"], expired_warning=False)
print(score_one(json.dumps(bad, ensure_ascii=False), r))
```
- 📌 **Kết quả nên thấy:** mọi chỉ số = `1.000`, `fairness_gap = 0.000`; ca cố tình sai có `citation_ok=True` nhưng `expired_ok=False, strict_ok=False`.
- 🧠 **"Test bộ chấm trước khi tin bộ chấm"** — kỹ sư chuyên nghiệp luôn chạy *sanity check* này. Một bộ chấm sai làm lệch hướng toàn bộ dự án.

### Ô 8 (8 dòng) — Chấm lại output Buổi 00 **không cần GPU**
```python
import os
from p01kit.metrics import gate
outs0 = json.load(open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json"))
s0 = summarize([score_one(o, r) for o, r in zip(outs0, test)])
print(format_summary(s0))
for k, (v, ok) in gate(s0).items():
    print(f"{k:<20} {v} {'✅' if ok else '❌'}")
```
- **Giải thích:** lưu output thô giúp bạn **chấm lại miễn phí** mỗi khi sửa bộ chấm — không phải sinh lại.
- 📌 **Kết quả nên thấy:** đúng các số của Buổi 00; phần lớn ngưỡng còn ❌.

### Ô 8b (8 dòng) — 🇻🇳 Dữ liệu thật: văn bản pháp luật Việt Nam hết hiệu lực nhiều đến đâu?
**Mục tiêu:** trả lời bằng số thật câu hỏi "vì sao P01 phải có `expired_warning`?" — dùng metadata 171.556 văn bản từ [`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) (CC-BY-4.0, ~15 MB, chạy trên CPU).
```python
import pandas as pd
from huggingface_hub import hf_hub_download
VB = "th1nhng0/vietnamese-legal-documents"
meta = pd.read_parquet(hf_hub_download(VB, "data/metadata.parquet", repo_type="dataset"),
                       columns=["id", "title", "so_ky_hieu", "loai_van_ban", "ngay_co_hieu_luc", "ngay_het_hieu_luc", "tinh_trang_hieu_luc"])
print(len(meta), "văn bản")
print(meta["tinh_trang_hieu_luc"].value_counts(normalize=True).round(3).head(4).to_dict())
print("Riêng Luật:", meta.loc[meta["loai_van_ban"] == "Luật", "tinh_trang_hieu_luc"].value_counts().to_dict())
```
- 📌 **Kết quả đã đo:** `171556 văn bản`; **52,9% "Hết hiệu lực toàn bộ"**, 41,7% "Còn hiệu lực", 3,1% "Hết hiệu lực một phần", 1,9% "Chưa xác định". Riêng 610 Luật: 271 hết hiệu lực toàn bộ, 154 hết một phần, 179 còn hiệu lực, 6 chưa có hiệu lực.
- 🧒 **Feynman:** kho luật giống một thư viện mà **hơn nửa số sách đã bị thay bằng bản mới** nhưng vẫn nằm trên kệ. Người tra cứu lấy nhầm sách cũ là chuyện rất dễ xảy ra — vì vậy P01 không xóa văn bản hết hạn mà **gắn nhãn đỏ** (`expired_warning`).

### Ô 8c (9 dòng) — Chuỗi thay thế Luật Bảo hiểm xã hội & đừng tin metadata mù quáng
```python
rel = pd.read_parquet(hf_hub_download(VB, "data/relationships.parquet", repo_type="dataset"))
title = dict(zip(meta["id"].astype(str), meta["title"]))
bhxh = meta[(meta["loai_van_ban"] == "Luật") & meta["title"].str.contains("Bảo hiểm xã hội", case=False, na=False)]
print(bhxh[["so_ky_hieu", "ngay_co_hieu_luc", "ngay_het_hieu_luc", "tinh_trang_hieu_luc"]].to_string(index=False))
het = rel[(rel["relationship"] == "Văn bản quy định hết hiệu lực") & rel["doc_id"].astype(str).isin(set(bhxh["id"].astype(str)))]
for d, o in het[["doc_id", "other_doc_id"]].astype(str).drop_duplicates().itertuples(index=False):
    print(f"{title.get(d)}  →  hết hiệu lực do  →  {title.get(o)}")
```
- 📌 **Kết quả đã đo:** 3 luật BHXH nối tiếp nhau: **71/2006/QH11** (01/01/2007 → 01/01/2016) → **58/2014/QH13** (01/01/2016 → 01/07/2025) → **41/2024/QH15** (từ 01/07/2025). Đây đúng là tình huống `expired` + "đã có văn bản mới thay thế" mà bộ dữ liệu P01 mô phỏng.
- ⚠️ **Dữ liệu cào tự động có lỗi — và bạn phải bắt được.** Cũng bộ dữ liệu này ghi Luật 41/2024/QH15 là *"Hết hiệu lực toàn bộ"* từ 01/01/2026 do Luật Việc làm 74/2025/QH15, và ghi Luật 71/2006 hết hiệu lực do một **Nghị định** (23/2021/NĐ-CP). Cả hai rất đáng ngờ: một nghị định đứng **dưới** luật trong thứ bậc văn bản nên không thể làm một luật hết hiệu lực, và một luật việc làm khó có thể bãi bỏ toàn bộ luật BHXH vừa có hiệu lực 6 tháng (nhiều khả năng là "hết hiệu lực **một phần**"). Trước khi dùng nhãn hiệu lực để huấn luyện hay trả lời người dân, **đối chiếu nguồn chính thức** (Cơ sở dữ liệu quốc gia về văn bản pháp luật, Công báo). Đó là lý do P01 bắt **cán bộ phê duyệt** mọi thay đổi trạng thái hiệu lực (`ApprovalAudit`) thay vì tin nhãn tự động.

### Ô 9 (7 dòng) — Nạp mô hình (chuẩn hóa bằng `p01kit.modeling`)
```python
import p01kit
from p01kit.modeling import load_model, load_tokenizer, vram_gb
tok = load_tokenizer()
model = load_model()          # base sạch, 4-bit NF4 — giống Ô 4 Buổi 00
print(vram_gb())
```
- **Giải thích:** từ buổi này ta dùng hàm nạp chuẩn để mọi buổi nạp mô hình **giống hệt nhau** (tránh lệch cấu hình ngầm).

### Ô 10 (9 dòng) — Few-shot: tốn bao nhiêu token?
```python
from p01kit.prompts import pick_fewshot, render_prompt
shots2, shots3 = pick_fewshot(train, 2), pick_fewshot(train, 3)
for name, fs in [("zero-shot", None), ("2-shot", shots2), ("3-shot", shots3)]:
    n = [len(tok(render_prompt(tok, r, fewshot=fs), add_special_tokens=False)["input_ids"]) for r in test]
    print(f"{name:<10} trung bình {sum(n)/len(n):.0f} token/prompt, tối đa {max(n)}")
```
- 📌 **Kết quả nên thấy (đã đo):** zero-shot **836** · 2-shot **2.570** · 3-shot **3.523** token.
- 🧠 Few-shot làm prompt dài gấp **3–4 lần** → trên CPU production (ADR 0001) độ trễ prefill tăng tương ứng. Đây là **chi phí ẩn** của prompt engineering mà fine-tune loại bỏ được.

### Ô 11 (7 dòng) — 📊 Đo few-shot (2-shot và 3-shot)
```python
from p01kit.infer import evaluate
from p01kit import scoreboard
results = {}
for name, fs in [("2-shot", shots2), ("3-shot", shots3)]:
    results[name] = evaluate(model, tok, test, fewshot=fs, batch_size=4, max_new_tokens=p01kit.MAX_NEW_TOKENS)
    print(f"\n### {name}\n" + format_summary(results[name]["summary"]))
    scoreboard.record("M01", f"base-{name}", results[name]["summary"], latency_s=results[name]["summary"]["sec_per_row"],
                      outputs=results[name]["outputs"])
```
- **Giải thích:** `batch_size=4` vì prompt dài gấp 3–4 lần → KV-cache lớn hơn; batch 8 có thể hết VRAM.
- 📌 **Kết quả nên thấy:** `json_valid`/`schema_ok` thường **tăng rõ** so với zero-shot (mô hình bắt chước định dạng ví dụ); `expired_recall` và `refusal_acc` thường cải thiện ít hơn — luật nghiệp vụ khó học qua 2–3 ví dụ.

### Ô 12 (6 dòng) — Kiểm định McNemar: few-shot có thật sự tốt hơn?
```python
from p01kit.stats import mcnemar_exact
zero = [score_one(o, r)["answer_ok"] for o, r in zip(outs0, test)]
for name in ("2-shot", "3-shot"):
    few = [s["answer_ok"] for s in results[name]["scores"]]
    print(name, mcnemar_exact(zero, few))
```
- **Giải thích:** `b` = số câu zero-shot đúng mà few-shot sai; `c` = ngược lại. `p_value < 0.05` ⇒ khác biệt có ý nghĩa thống kê. Nếu `b + c` rất nhỏ, hai mô hình gần như trả lời giống nhau.

### Ô 13 (4 dòng) — Bảng điểm & quyết định
```python
scoreboard.show()
```
- **Việc cần làm — viết 5 dòng kết luận vào vở:** (1) few-shot cải thiện chỉ số nào, có ý nghĩa thống kê không; (2) còn ngưỡng nào ❌; (3) chi phí token; (4) **quyết định**: có cần fine-tune không và vì sao.

---

## 3. So sánh hiệu quả — Ma trận quyết định Prompt vs RAG vs Fine-tune

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M00", "base-zero-shot"), ("M01", "base-3-shot"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** Few-shot chỉ đổi **prompt**, không đổi trọng số: thường sửa được lỗi định dạng (`json_valid`) nhưng ít sửa lỗi nghiệp vụ (`le_ra_tu_choi`, `khong_canh_bao_het_han`) — đó là lý do cần fine-tune ở Buổi 02.


| Giải pháp | Khi nào đủ? | Chi phí | Điểm yếu với P01 |
|---|---|---|---|
| **Prompt / few-shot** | Mô hình đã có năng lực, chỉ cần định dạng | 0 train, nhưng prompt dài 3–4 lần | Luật nghiệp vụ (hết hạn, lệch địa bàn) khó nhớ qua vài ví dụ; mô hình 1,7B dễ "quên luật" |
| **RAG** | Tri thức thay đổi thường xuyên | Hạ tầng truy xuất | Không tự sửa thói quen bám nhiễu, không tự biết từ chối |
| **Fine-tune (LoRA)** | Cần hành vi ổn định, định dạng chặt, mô hình nhỏ chạy CPU | GPU vài chục phút | Không dùng để nhồi tri thức thay đổi hàng ngày |
| **RAG + Fine-tune (RAFT)** | Cần cả tri thức mới **và** hành vi đọc–lọc–từ chối | Cả hai | **Lựa chọn của P01** |

---

## 🏢 Ở công ty thật

- **Dữ liệu thật không sạch như tập sinh bằng luật.** Nó đến từ log, biểu mẫu, email; cần **hướng dẫn gán nhãn** viết thành văn bản, nhiều người gán độc lập và đo đồng thuận (Cohen's κ — Buổi 15) trước khi tin nhãn.
- **Tập test được đóng băng và có phiên bản** (hash như Release Dossier, DVC, hoặc revision của dataset trên Hub). Ai đổi tập test phải ghi lý do; số liệu trước/sau khi đổi không so trực tiếp được.
- **Bộ chấm là code sản phẩm**: được review và có unit test (`tests/test_metrics.py`). Sửa bộ chấm = phải chấm lại toàn bộ bảng điểm lịch sử.
- **Ma trận Prompt vs RAG vs Fine-tune** thường được viết thành một trang *design doc* để trưởng nhóm/quản lý duyệt **trước** khi tốn GPU.

---

## 4. Bài tập

**Bắt buộc**
1. **Prompt engineering có kiểm soát:** viết `SYSTEM_PROMPT_V2` (ví dụ thêm checklist 3 bước, thêm ví dụ mẫu JSON rỗng). Đo zero-shot với `evaluate(..., system_prompt=SYSTEM_PROMPT_V2)`, ghi bảng điểm, chạy McNemar với bản gốc. Chỉ giữ thay đổi nếu p < 0,05.
2. **Tính độ rộng CI:** với `answer_acc` của 3-shot, tính độ rộng `95%CI`. Cần bao nhiêu câu test để độ rộng còn ±3 điểm %? (gợi ý: độ rộng tỉ lệ với 1/√n).
3. Dùng `p01kit.grounding.hallucination_rate` đo tỉ lệ **bịa giấy tờ/nơi nộp** của zero-shot và 3-shot.

**Nâng cao**
4. Sinh lại dữ liệu với `build_dataset(seed=7)` (không ghi đè file). Phân bố có ổn định không? Nhóm test thay đổi thế nào? Vì sao **không bao giờ** được chọn seed "đẹp nhất" theo điểm test?
5. Thêm một văn bản mới vào bản sao của `policies.json` (ví dụ hỗ trợ người khuyết tật tại Cần Thơ), dựng lại dữ liệu, và chạy `pytest edu/tests` để chứng minh nhãn vẫn nhất quán.

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao chia ngẫu nhiên theo dòng là sai trong bài toán có câu hỏi diễn đạt lại?**
   A: Các biến thể của cùng kịch bản lọt vào cả train và test → mô hình "nhớ" thay vì "hiểu", điểm test cao ảo. Phải chia theo nhóm (GroupKFold / scenario split).
2. **Q: Câu hỏi test trùng nguyên văn câu train có phải luôn là rò rỉ?**
   A: Không. Rò rỉ là khi **cùng đầu vào đầy đủ → cùng nhãn**. Cùng câu hỏi nhưng khác địa bàn/ngữ cảnh → khác nhãn là ca kiểm tra năng lực đọc ngữ cảnh, phải giữ.
3. **Q: Tại sao cần lát cắt `unseen_doc`?**
   A: Đo năng lực **đọc hiểu văn bản chưa gặp** — điều kiện thật khi kho văn bản được cập nhật. Mô hình có thể đạt điểm cao ở `seen_doc` chỉ nhờ nhớ nội dung văn bản.
4. **Q: Làm sao biết mô hình B tốt hơn A khi test chỉ có 100 câu?**
   A: Dùng kiểm định ghép cặp (McNemar) trên cùng câu hỏi, báo cáo khoảng tin cậy bootstrap; không kết luận từ chênh lệch trung bình đơn thuần.
5. **Q: Nhược điểm của few-shot khi triển khai trên CPU?**
   A: Prompt dài gấp nhiều lần → thời gian prefill (TTFT) và RAM KV-cache tăng tương ứng; fine-tune "nướng" hành vi vào trọng số nên prompt ngắn.
6. **Q: Tại sao nên sinh nhãn bằng luật khi có thể?**
   A: Nhất quán tuyệt đối, kiểm thử được tự động, tái tạo được khi dữ liệu nguồn thay đổi; dành LLM/con người cho phần không mã hóa được (diễn đạt tự nhiên).

---

## 6. Checklist cuối buổi
- [ ] Giải thích được luật `resolve()` và 3 lát cắt test.
- [ ] Hiểu vì sao 18 câu trùng nguyên văn **không** phải rò rỉ.
- [ ] Đã chạy sanity check bộ chấm (đáp án hoàn hảo = 1,0).
- [ ] Bảng điểm có `M01/base-2-shot`, `M01/base-3-shot` và kết luận McNemar.

## Tài liệu tham khảo
- Zhang et al., *RAFT: Adapting Language Model to Domain Specific RAG* (2024).
- Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science*.
- Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms* (McNemar).
