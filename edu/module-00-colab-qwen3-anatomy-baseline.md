# Buổi 00 — Giải phẫu `Qwen3-1.7B` trên Colab T4 & Đo Baseline Trung thực

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3 giờ (45' lý thuyết · 2h15' lab) |
| **Đầu vào** | Tài khoản Google (Colab T4 miễn phí), repo đồ án đã clone được (xem [README](README.md)) |
| **🎯 Sản phẩm** | (1) **Máy X-quang** soi xác suất token kế tiếp; (2) **Bộ sinh theo lô** nhanh gấp nhiều lần hỏi từng câu; (3) **Dòng đầu tiên của bảng điểm** — điểm Baseline zero-shot trên 100 câu test |
| **Chỉ số đo** | VRAM thực tế vs. ước tính · số token tiếng Việt vs. tiếng Anh · thời gian sinh thinking vs. non-thinking · 8 chỉ số P01 |
| **Chế độ code** | 🔬 **Ô siêu nhỏ** (≤ 10 dòng), mỗi ô có *Mục tiêu → Code → Giải thích → 📌 Kết quả nên thấy → ⚠️ Lỗi hay gặp* |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Tính tay số tham số (≈1,72 tỷ) và bộ nhớ FP16/NF4 của Qwen3-1.7B, rồi đối chiếu với VRAM đo thật.
2. Đo được thinking vs non-thinking bằng số token và thời gian, không đoán.
3. Viết bộ sinh theo lô (đệm trái) và đo mức tăng tốc so với hỏi từng câu.
4. Có điểm **baseline** của mô hình gốc trên 100 câu test — mốc "trước" cho mọi phép so sánh sau này.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): `check_env()` báo ✅; bảng điểm có dòng `M00/base-zero-shot` và file `p01_runs/outputs/M00__base-zero-shot.json` tồn tại; bạn giải thích được vì sao VRAM đo khác ước tính.

---

## 0. Bức tranh lớn — Vì sao doanh nghiệp bắt đầu bằng Baseline?

Ở công ty, câu đầu tiên sếp hỏi khi bạn đề xuất fine-tune luôn là: **"Mô hình gốc đang được bao nhiêu điểm?"**. Không có baseline thì không chứng minh được fine-tune có ích, và không biết dừng ở đâu là đủ. Buổi này bạn sẽ:

1. Hiểu **Qwen3-1.7B là gì bên trong** (tokenizer, chat template, chế độ thinking, bộ nhớ).
2. Viết công cụ sinh văn bản **theo lô** (batched generation) — kỹ năng tối ưu bắt buộc khi chạy đánh giá trên hàng trăm câu.
3. Đo baseline trên **chính tập test sẽ dùng xuyên suốt 16 buổi**, ghi vào bảng điểm `leaderboard.csv`.

> 💡 **Nguyên tắc vàng của cả lộ trình:** mọi buổi đều đo trên **cùng 100 câu test** (`edu/data/p01_test.jsonl`), cùng prompt, cùng bộ chấm. Nhờ vậy các con số giữa các buổi **so sánh được với nhau**.

---

## 1. Lý thuyết (Feynman) — 4 điều phải hiểu trước khi chạm vào GPU

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** Mô hình ngôn ngữ là **máy đoán chữ tiếp theo**. "1,7 tỷ tham số" là 1,7 tỷ núm vặn đã được vặn sẵn trong lúc tiền huấn luyện. NF4 nén mỗi núm từ 16 bit xuống 4 bit — như nén ảnh: nhẹ đi 4 lần, mờ đi một chút. **Baseline** là cân sức khỏe trước khi tập gym: không có số "trước" thì không chứng minh được tập có tác dụng.

### 1.1. `Qwen/Qwen3-1.7B` không phải "mô hình base"
- Trên Hugging Face có **2 bản**: `Qwen/Qwen3-1.7B-Base` (chỉ tiền huấn luyện, biết đoán chữ nhưng không biết hội thoại) và `Qwen/Qwen3-1.7B` (đã **post-train**: SFT + RL, biết chat, biết gọi tool, có 2 chế độ thinking/non-thinking).
- Lộ trình này fine-tune **bản post-train** vì dữ liệu của ta ít (vài trăm mẫu): kế thừa khả năng làm theo chỉ dẫn rẻ hơn rất nhiều so với dạy lại từ bản Base.
- 🎤 *Phỏng vấn hay hỏi đúng chỗ này* — gọi nhầm bản post-train là "base model" là bị trừ điểm.

### 1.2. Cấu hình thật của Qwen3-1.7B (đọc từ `config.json`)
| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_hidden_layers` | 28 | 28 khối Transformer chồng lên nhau |
| `hidden_size` | 2048 | Mỗi token là vector 2048 chiều |
| `intermediate_size` | 6144 | Lớp MLP phình 3 lần rồi co lại |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | **GQA**: 2 đầu Query dùng chung 1 cặp Key/Value → tiết kiệm KV-cache 50% |
| `head_dim` | 128 | 16 × 128 = 2048 |
| `vocab_size` | 151.936 | Kích thước ma trận embedding (đã đệm); tokenizer thật có 151.669 token |
| `tie_word_embeddings` | `true` | Ma trận đầu vào và `lm_head` **dùng chung** → bớt 311 triệu tham số |

**Đếm tham số bằng tay** (bài tập nhẩm hay gặp khi phỏng vấn):
- Mỗi lớp: Attention = `2048×2048 (q) + 2×2048×1024 (k,v) + 2048×2048 (o)` = 12,58 triệu; MLP = `3 × 2048×6144` = 37,75 triệu → **50,3 triệu/lớp**.
- 28 lớp → **1,409 tỷ** (non-embedding) + embedding `151.936 × 2048` = **0,311 tỷ** → tổng **≈ 1,72 tỷ**.

### 1.3. Bộ nhớ: FP16 vs 4-bit NF4
- FP16: 1,72 tỷ × 2 byte = **3,44 GB**.
- NF4 (4 bit): các lớp Linear còn 1,409 tỷ × 0,5 byte ≈ 0,70 GB + hằng số lượng tử ≈ 0,02 GB; **embedding không bị lượng tử** (giữ 16-bit) = 0,62 GB → **≈ 1,35 GB** (≈ 1,26 GiB).
- Khi train còn cần: gradient + optimizer của LoRA (nhỏ, ~0,3 GB) và **activations** (lớn nhất, tỉ lệ với `batch × seq_len`) → đó là lý do ta cần gradient checkpointing ở các buổi sau.

### 1.4. T4 không có BF16 — dùng FP16
- T4 là kiến trúc **Turing (compute capability 7.5)**. BF16 cần phần cứng **Ampere (8.0) trở lên** (A100, L4, RTX 30xx).
- `TrainingArguments(bf16=True)` trên T4 sẽ **báo lỗi ngay khi khởi tạo**. Ta dùng `fp16=True` + `bnb_4bit_compute_dtype=torch.float16`.
- FP16 có dải số hẹp (tối đa ~65.504) nên dễ tràn số → `Trainer` tự dùng **loss scaling** động. Buổi 06 sẽ cho bạn tự tay gây ra và sửa lỗi NaN.

### 1.5. Chế độ Thinking của Qwen3
- Mặc định Qwen3 "nghĩ" trong khối `<think>...</think>` trước khi trả lời → chính xác hơn ở bài toán khó nhưng **dài và chậm**.
- `enable_thinking=False` trong `apply_chat_template` sẽ chèn sẵn khối rỗng `<think>\n\n</think>\n\n` vào prompt → mô hình trả lời thẳng.
- Từ Buổi 00 đến 09 ta dùng non-thinking (JSON ngắn, độ trễ thấp cho CPU). **Buổi 10** quay lại thinking và đo xem fine-tune có làm "hỏng" khả năng suy luận không.

---

## 2. Thực hành từng ô nhỏ

### Ô 0a — Cài thư viện (chạy 1 lần cho mỗi runtime)
**Mục tiêu:** đưa code + dữ liệu đồ án vào Colab và cài **đúng phiên bản đã ghim**.
```python
import os, subprocess, base64
REPO, URL = "/content/uit-social-welfare-rag", "https://github.com/phamxuanphuc/uit-social-welfare-rag.git"
if not os.path.exists(REPO):
    env = dict(os.environ)
    try:  # repo private: thêm GITHUB_TOKEN vào Colab Secrets (🔑); repo public thì bỏ qua
        from google.colab import userdata
        basic = base64.b64encode(f"x-access-token:{userdata.get('GITHUB_TOKEN')}".encode()).decode()
        env |= {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}"}  # token chỉ nằm trong biến môi trường
    except Exception:
        pass
    r = subprocess.run(["git", "clone", "--depth", "1", URL, REPO], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"git clone lỗi:\n{r.stderr}"
```
```bash
!pip install -q -r /content/uit-social-welfare-rag/edu/requirements-colab.txt
```
```python
import os; os.kill(os.getpid(), 9)  # khởi động lại runtime để nạp thư viện mới (Colab sẽ tự kết nối lại)
```
- **Giải thích:** token lấy từ *Colab Secrets* nên **không bao giờ xuất hiện trong notebook** (không lộ khi chia sẻ). Token được đưa cho `git` qua **biến môi trường** (header `Authorization`), **không ghép vào URL**: nếu ghép vào URL, khi clone lỗi thì thông báo `CalledProcessError` sẽ in nguyên URL chứa token ra output của ô, và token còn bị lưu lại trong `.git/config`. Ghim phiên bản (`transformers==5.17.0`, `trl==1.15.0`...) để notebook chạy lại được sau 6 tháng — các thư viện này đổi tên tham số rất nhanh (ví dụ `warmup_ratio` đã bị bỏ ở transformers 5).
- ⚠️ **Lỗi hay gặp:** quên khởi động lại runtime → `import` vẫn dùng bản `transformers` cũ có sẵn trên Colab, sinh lỗi khó hiểu ở các buổi sau.

### Ô 0b — Khởi động phiên (chạy đầu MỖI buổi)
```python
import os, sys
REPO = "/content/uit-social-welfare-rag"
sys.path.insert(0, f"{REPO}/edu")
from google.colab import drive
drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"  # adapter + bảng điểm sống sót khi Colab ngắt
import p01kit; print("p01kit", p01kit.__version__, "| model:", p01kit.MODEL_ID)
from p01kit.colab import check_env; check_env()   # đúng bản ghim? torch trong khoảng unsloth hỗ trợ? có GPU?
```
- 📌 **Kết quả nên thấy:** `p01kit 2.0.0 | model: Qwen/Qwen3-1.7B`, dòng `GPU: {'name': 'Tesla T4', ...}` và `✅ Môi trường khớp bản ghim`.
- 💡 Học trên **Colab miễn phí**: đọc mục *Học trên Colab T4 miễn phí* trong [README](README.md) trước buổi đầu — cách giải phóng GPU, phần nào chạy trên CPU, và bị ngắt giữa buổi thì làm gì.
- ⚠️ Thấy `⚠️ transformers: đang là ..., bản ghim là 5.17.0` → bạn quên khởi động lại runtime sau Ô 0a. Thấy `torch ...: ngoài khoảng unsloth hỗ trợ` → Colab vừa đổi bản `torch` mặc định; báo cho nhóm để cập nhật bộ ghim, đừng tự nâng/hạ từng gói.

### Ô 1 (4 dòng) — Kiểm tra GPU
```python
import torch
print(torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Hỗ trợ BF16?", torch.cuda.is_bf16_supported())
```
- 📌 **Kết quả nên thấy:** `Tesla T4` · `(7, 5)` · `Hỗ trợ BF16? False`.
- ⚠️ Nếu không thấy GPU: *Runtime → Change runtime type → T4 GPU*.

### Ô 2 (6 dòng) — Tokenizer: tiếng Việt "đắt" hơn tiếng Anh
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(p01kit.MODEL_ID)
for s in ["Tôi chạy xe ôm ở Phường Tân Tạo bị té xe gãy tay",
          "I drive a motorbike taxi in Tan Tao ward and broke my arm"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(len(s.split()), "từ ->", len(ids), "token:", [tok.decode([i]) for i in ids])
```
- **Giải thích:** Qwen3 dùng **byte-level BPE**. Âm tiết tiếng Việt có dấu ít gặp trong dữ liệu tiền huấn luyện nên bị cắt vụn.
- 📌 **Kết quả nên thấy (đã đo):** câu Việt 13 từ → **19 token** (`' ô', 'm'`, `' Ph', 'ường'`, `' g', 'ã', 'y'`...); câu Anh 13 từ → **14 token**.
- 🧠 **Hệ quả thực tế:** prompt RAG của P01 dài trung bình **836 token**, tối đa 885 — chi phí bộ nhớ và độ trễ lớn hơn bản tiếng Anh tương đương khoảng 35%.

### Ô 3 (5 dòng) — Token đặc biệt quyết định mọi thứ
```python
for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"]:
    print(f"{t:>14} -> id {tok.convert_tokens_to_ids(t)}")
print("eos =", tok.eos_token, "| pad =", tok.pad_token, "| len(tok) =", len(tok))
```
- 📌 **Kết quả nên thấy:** `<|im_start|>`=151644, `<|im_end|>`=151645, `<|endoftext|>`=151643, `<think>`=151667, `</think>`=151668, `<tool_call>`=151657; `eos = <|im_end|>`, `pad = <|endoftext|>`, `len(tok) = 151669`.
- **Giải thích:** `<|im_end|>` là token **kết thúc lượt trả lời**. Nếu dữ liệu train thiếu token này, mô hình sau fine-tune sẽ **không biết dừng** và nói lan man đến hết `max_new_tokens` — lỗi số 1 của người mới.

### Ô 4 (6 dòng) — Chat template: thinking bật vs tắt
```python
msgs = [{"role": "system", "content": "Bạn là trợ lý an sinh."},
        {"role": "user", "content": "Xe ôm bị tai nạn cần giấy gì?"}]
for think in (True, False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    print(f"--- enable_thinking={think} ---\n{text!r}\n")
```
- 📌 **Kết quả nên thấy:** hai chuỗi giống nhau, chỉ khác phần đuôi:
  - `think=True` → kết thúc bằng `'<|im_start|>assistant\n'` (mô hình tự quyết định nghĩ).
  - `think=False` → kết thúc bằng `'<|im_start|>assistant\n<think>\n\n</think>\n\n'` (khối nghĩ rỗng được "điền sẵn").
- 🧠 **Ghi nhớ cho các buổi train:** prompt lúc train và lúc suy luận **phải giống hệt nhau đến từng ký tự**. `p01kit.prompts.render_prompt()` là nguồn sự thật duy nhất cho việc này.

### Ô 5 (6 dòng) — Ước tính bộ nhớ trước khi nạp
```python
L, H, I, Q, KV, V = 28, 2048, 6144, 2048, 1024, 151_936
per_layer = H*Q + 2*H*KV + Q*H + 3*H*I
non_emb, emb = L * per_layer, V * H
print(f"Tham số: {(non_emb + emb)/1e9:.3f} tỷ | FP16: {(non_emb + emb)*2/1e9:.2f} GB")
print(f"NF4 ước tính: {(non_emb*0.5 + non_emb*0.127/8 + emb*2)/1e9:.2f} GB")
```
- 📌 **Kết quả nên thấy:** `Tham số: 1.721 tỷ | FP16: 3.44 GB` và `NF4 ước tính: 1.35 GB`.
- **Giải thích:** `0.127/8` là chi phí hằng số lượng tử sau *double quantization* (~0,127 bit/tham số theo paper QLoRA). Embedding nhân 2 byte vì bitsandbytes **không** lượng tử hóa lớp `Embedding`.

### Ô 6 (8 dòng) — Nạp mô hình 4-bit NF4 và đối chiếu ước tính
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(p01kit.MODEL_ID, quantization_config=bnb,
                                             device_map="auto", dtype=torch.float16)
model.eval()
print(f"Footprint: {model.get_memory_footprint()/1e9:.2f} GB | VRAM đang cấp: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```
- **Giải thích từng tham số:** `nf4` = NormalFloat 4-bit (tối ưu cho trọng số phân phối chuẩn); `double_quant` = lượng tử luôn cả hằng số scale; `compute_dtype=float16` = khi nhân ma trận, trọng số được **giải nén tạm** về FP16; `dtype=float16` cho các lớp không lượng tử (embedding, norm).
- 📌 **Kết quả nên thấy:** footprint ≈ **1,3–1,5 GB** (khớp ô 5). Nếu ra ~3,4 GB nghĩa là `quantization_config` chưa có hiệu lực.
- ⚠️ Transformers 5 dùng `dtype=`; tham số cũ `torch_dtype=` sẽ cảnh báo deprecated.

### Ô 7 (9 dòng) — 🎯 Sản phẩm 1: Máy X-quang token kế tiếp
```python
@torch.inference_mode()
def xray(text: str, k: int = 5, temperature: float = 1.0):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1].float()          # logit của vị trí CUỐI = dự đoán token kế tiếp
    probs = torch.softmax(logits / temperature, dim=-1)
    for p, i in zip(*probs.topk(k)):
        print(f"{tok.decode([i.item()])!r:>16}  {p.item():6.1%}")

xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại")
```
- **Thí nghiệm nhỏ:** chạy lại với `temperature=0.3` và `temperature=1.5`. Nhiệt độ thấp làm phân phối **nhọn** (gần greedy), nhiệt độ cao làm **phẳng** (đa dạng nhưng dễ bịa).
- 📌 **Kết quả nên thấy:** top-5 là các mảnh như `' UBND'`, `' Ủy'`, `' cơ'`, `' phường'`... (cụ thể tùy mô hình); tổng xác suất top-5 giảm rõ khi tăng temperature.
- 🧠 Đây chính là "bộ não" bên dưới mọi thứ: fine-tune = **dịch chuyển phân phối này** về phía câu trả lời ta muốn.

### Ô 8 (10 dòng) — Thinking vs Non-thinking: đo, đừng đoán
```python
import time
def chat(question: str, think: bool, max_new_tokens: int = 1024):
    m = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(temperature=0.6, top_p=0.95) if think else dict(temperature=0.7, top_p=0.8)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, top_k=20, **kw)
    new = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True), len(new), time.perf_counter() - t0
```
```python
for think in (True, False):
    ans, n, s = chat("Một năm nhuận có bao nhiêu ngày? Giải thích ngắn.", think)
    print(f"think={think}: {n} token, {s:.1f}s, {n/s:.1f} tok/s\n{ans[:300]}\n")
```
- **Giải thích:** tham số sampling theo **khuyến nghị chính thức của Qwen3** — thinking: `T=0.6, top_p=0.95, top_k=20`; non-thinking: `T=0.7, top_p=0.8, top_k=20`. Qwen khuyến cáo **không dùng greedy cho chế độ thinking** (dễ lặp vô hạn).
- 📌 **Kết quả nên thấy:** thinking sinh **nhiều token hơn hẳn** (thường gấp vài lần) với tốc độ tok/s tương đương → tổng thời gian dài hơn tương ứng. Ghi lại số đo thật của bạn vào vở.

### Ô 9 (7 dòng) — Vì sao sinh theo lô phải đệm TRÁI?
```python
tok.padding_side = "left"
batch = tok(["Xin chào", "Tôi chạy xe ôm ở Tân Tạo bị té xe"], return_tensors="pt",
            padding=True, add_special_tokens=False)
print(batch["input_ids"])
print(batch["attention_mask"])
```
- **Giải thích:** mô hình decoder-only sinh token **nối vào cuối chuỗi**. Nếu đệm phải, câu ngắn sẽ có `<pad>` nằm giữa prompt và phần sinh → kết quả sai. Đệm trái giữ mọi prompt "thẳng hàng" ở mép phải.
- 📌 **Kết quả nên thấy:** hàng 1 bắt đầu bằng các số `151643` (pad) và `attention_mask` tương ứng là `0`.
- ⚠️ Khi **train** thì ngược lại: đệm phải (xem `p01kit.sft.PadCollator`). Lẫn lộn hai chiều này là lỗi rất phổ biến.

### Ô 10 (10 dòng) — 🎯 Sản phẩm 2: Bộ sinh theo lô & đo tốc độ tăng
```python
@torch.inference_mode()
def generate_batch(prompts, max_new_tokens=128, batch_size=8):
    tok.padding_side, outs = "left", []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i+batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs
```
```python
qs = [tok.apply_chat_template([{"role": "user", "content": f"Viết 1 câu về số {i}."}], tokenize=False,
                              add_generation_prompt=True, enable_thinking=False) for i in range(8)]
for bs in (1, 8):
    t0 = time.perf_counter(); generate_batch(qs, batch_size=bs); print(f"batch_size={bs}: {time.perf_counter()-t0:.1f}s")
```
- **Giải thích:** `temperature=None, top_p=None, top_k=None` tắt các giá trị sampling mặc định trong `generation_config` của Qwen3 (mặc định `do_sample=True, T=0.6`) để greedy thật sự tất định và không bị cảnh báo.
- 📌 **Kết quả nên thấy:** `batch_size=8` nhanh hơn `batch_size=1` **nhiều lần** (GPU xử lý 8 chuỗi song song trong mỗi bước). Đây chính là `p01kit.infer.generate` mà các buổi sau dùng lại.

### Ô 11 (8 dòng) — Làm quen tập test P01
```python
from p01kit.data import load_split, dataset_stats
test = load_split("test")
print(dataset_stats(test))
r = test[0]
print("Địa bàn:", r["region"], "\nCâu hỏi:", r["question"])
print("Ngữ cảnh:\n", r["context"][:600], "...")
print("Đáp án vàng:", r["gold"])
```
- 📌 **Kết quả nên thấy:** `n=100`, `unanswerable_ratio=0.44`, `expired_ratio=0.1`, `colloquial_ratio=0.5`, `slices={'seen_doc': 24, 'unseen_doc': 72, 'ood': 4}`. Buổi 01 sẽ mổ xẻ kỹ cấu trúc này.

### Ô 12 (8 dòng) — 📊 Sản phẩm 3: Đo Baseline zero-shot & ghi bảng điểm
```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
res0 = evaluate(model, tok, test, batch_size=8, max_new_tokens=p01kit.MAX_NEW_TOKENS)
print(format_summary(res0["summary"]))
print("Thời gian:", res0["summary"]["seconds"], "giây")
scoreboard.record("M00", "base-zero-shot", res0["summary"], latency_s=res0["summary"]["sec_per_row"], outputs=res0["outputs"])
```
- **Giải thích:** `evaluate()` = `render_prompt` (prompt chuẩn P01) → `generate` theo lô (greedy) → `score_one` từng câu → `summarize` kèm **khoảng tin cậy 95% bootstrap**.
- 📌 **Kết quả nên thấy:** một bảng 8 chỉ số kèm `95%CI` và `n`. Kiểm tra hợp lý: `json_valid` phải > 0 (nếu bằng 0, kiểm tra lại `enable_thinking=False` và `max_new_tokens`). Đừng ngạc nhiên nếu `expired_recall` và `refusal_acc` thấp — đó là lý do ta fine-tune.
- ⚠️ Khoảng tin cậy rộng (ví dụ ±10 điểm %) là **bình thường** với 100 câu. Hai mô hình chênh 3 điểm thường **chưa** khác nhau thật — Buổi 01 dạy kiểm định McNemar.

### Ô 13 (5 dòng) — Phân tích lỗi (error analysis)
```python
import json
from p01kit.infer import show_errors
show_errors(res0, test, k=5)
with open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json", "w") as f:
    json.dump(res0["outputs"], f, ensure_ascii=False)
```
- **Việc cần làm:** đọc 5 ca sai và **phân loại bằng tay**: (a) sai định dạng JSON, (b) không từ chối khi phải từ chối, (c) chọn nhầm văn bản lệch địa bàn, (d) bỏ sót cảnh báo hết hạn, (e) bịa giấy tờ. Ghi tỉ lệ vào vở — đây là "bản đồ" cho các buổi sau.

---

## 3. So sánh hiệu quả — bảng điểm sau Buổi 00

Mở bảng điểm bất cứ lúc nào:
```python
scoreboard.show()
```

| module | variant | json_valid | refusal_acc | citation_precision | expired_recall | strict_acc | fairness_gap |
|---|---|---|---|---|---|---|---|
| M00 | base-zero-shot | *(số đo của bạn)* | | | | | |

> 📏 **Đọc bảng điểm thế nào?** Mỗi chỉ số tương ứng một ngưỡng chốt trước trong [PROJECT-REQUIREMENTS.md](../PROJECT-REQUIREMENTS.md): `citation_precision ≥ 0.90`, `refusal_acc ≥ 0.90`, `expired_recall = 1.0`, `fairness_gap ≤ 0.10`. Gọi `p01kit.metrics.gate(summary)` để xem đạt/không đạt.

---

## 🏢 Ở công ty thật

- **Baseline thật là hệ thống đang có**, không chỉ base model: quy trình tra cứu thủ công, trang FAQ, bộ luật từ khóa, hoặc một API mô hình lớn. Fine-tune chỉ đáng làm khi thắng các phương án đó về chất lượng **và** chi phí/độ trễ.
- **Ràng buộc deploy được chốt trước khi chọn mô hình.** Ở P01 là CPU 2 vCPU / 4 GiB (ADR 0001), nên 1,7B được chọn trước khi đo bất cứ thứ gì. Ở công ty, đội ước tính chi phí suy luận (token/request × request/ngày) ngay từ đầu.
- **Giấy phép mô hình** (Apache-2.0 với Qwen3) được bộ phận pháp chế kiểm tra trước khi dùng cho sản phẩm.
- **Phân tích lỗi trên mẫu traffic thật** (đã che PII), không chỉ trên tập test tự dựng: câu hỏi thật thường ngắn hơn, sai chính tả, không dấu (Buổi 15).

---

## 4. Bài tập

**Bắt buộc**
1. Đo `tok/s` của `generate_batch` với `batch_size ∈ {1, 4, 8, 16}` trên 32 prompt; vẽ đồ thị. Ở `batch_size` nào tốc độ bão hòa? Vì sao (gợi ý: KV-cache và bộ nhớ)?
2. Tính tay số tham số nếu Qwen3-1.7B **không** tie embedding. VRAM FP16 tăng bao nhiêu?
3. Chạy baseline với **thinking bật** trên 20 câu test đầu (`evaluate(..., enable_thinking=True, max_new_tokens=1536, batch_size=4)`). So sánh `answer_acc` và thời gian với non-thinking. Ghi vào bảng điểm với `variant="base-thinking-20"`.

**Nâng cao**
4. So sánh footprint khi nạp **8-bit** (`load_in_8bit=True`) và FP16 (không lượng tử). Chênh lệch có khớp lý thuyết không?
5. Viết hàm `xray_chat(question)` soi top-5 token **đầu tiên của câu trả lời** sau khi áp chat template với `enable_thinking=False`. Token đầu tiên có phải `{` không? Điều đó nói gì về khả năng tuân thủ JSON?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Qwen3-1.7B có bao nhiêu tham số và vì sao có chỗ ghi 1.4B?**
   A: Tổng ≈ 1,72 tỷ; 1,41 tỷ là phần *non-embedding*. Embedding 0,31 tỷ được tie với `lm_head`.
2. **Q: Vì sao train QLoRA trên T4 phải dùng FP16 chứ không BF16?**
   A: T4 (Turing, sm_75) không có đơn vị tính BF16; BF16 cần Ampere (sm_80+). FP16 dải hẹp nên cần loss scaling, đổi lại vẫn tăng tốc được bằng Tensor Core của T4.
3. **Q: GQA là gì và lợi gì khi suy luận?**
   A: Nhiều đầu Query dùng chung một cặp Key/Value (Qwen3-1.7B: 16 Q / 8 KV). KV-cache nhỏ đi 2 lần → phục vụ được context dài hơn, batch lớn hơn.
4. **Q: Vì sao sinh theo lô phải đệm trái còn train lại đệm phải?**
   A: Sinh token nối vào cuối chuỗi nên mọi prompt phải kết thúc thẳng hàng (đệm trái). Khi train, vị trí pad được che bằng `attention_mask` và nhãn `-100`, đệm phải đơn giản và an toàn.
5. **Q: `enable_thinking=False` hoạt động thế nào ở mức token?**
   A: Chat template chèn sẵn `<think>\n\n</think>\n\n` sau `<|im_start|>assistant\n`, mô hình coi như đã nghĩ xong và trả lời ngay.
6. **Q: Vì sao phải đo baseline trước khi fine-tune?**
   A: Để có mốc so sánh, phát hiện bài toán có thể giải bằng prompt/RAG (không cần fine-tune), và để đặt tiêu chí dừng. Baseline phải đo trên **cùng** tập test, prompt, bộ chấm với các phiên bản sau.
7. **Q: Tại sao embedding không bị lượng tử 4-bit trong bitsandbytes?**
   A: bitsandbytes chỉ thay các lớp `nn.Linear` bằng `Linear4bit`; embedding là bảng tra cứu, lượng tử nó gây sai số lớn cho token hiếm mà lợi bộ nhớ không nhiều.

---

## 6. Checklist cuối buổi
- [ ] Giải thích được 2 bản `Qwen3-1.7B` vs `Qwen3-1.7B-Base`.
- [ ] Tự tính được 1,72 tỷ tham số và 1,35 GB NF4.
- [ ] Viết được `generate_batch` với đệm trái và đo được tốc độ tăng.
- [ ] Bảng điểm có dòng `M00 / base-zero-shot`; đã phân loại 5 lỗi đầu tiên.

## Tài liệu tham khảo
- Qwen3 Technical Report & model card `Qwen/Qwen3-1.7B` (mục *Best Practices* về tham số sampling).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double quantization).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.
