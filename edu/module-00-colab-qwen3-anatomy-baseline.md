# Buổi 00 — Làm quen `Qwen3-1.7B` trên Colab T4 & Đo Baseline Trung thực

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3 giờ (45' lý thuyết · 2h15' lab). Có thể chia **2 lần ngồi**: lần 1 làm Phần A–C, lần 2 làm Phần D–F |
| **Cần biết trước** | Python cơ bản (biến, vòng lặp, hàm), biết mở Google Colab. **Không cần** biết Transformer hay PyTorch — buổi này giải thích từ đầu |
| **Đầu vào** | Tài khoản Google (Colab T4 miễn phí), repo đồ án clone được (xem [README](README.md)) |
| **🎯 Sản phẩm** | (1) **Máy X-quang** xem mô hình đoán chữ kế tiếp; (2) **Bộ sinh theo lô** nhanh gấp nhiều lần hỏi từng câu; (3) **Dòng đầu tiên của bảng điểm** — điểm baseline trên 100 câu test |
| **Chế độ code** | 🔬 **Ô siêu nhỏ** (≤ 10 dòng), mỗi ô có *Mục tiêu → Code → Giải thích → 📌 Kết quả nên thấy → ⚠️ Lỗi hay gặp* |

### 🎯 Mục tiêu buổi học — sau buổi này bạn làm được
1. Giải thích bằng lời của mình: mô hình ngôn ngữ **sinh chữ như thế nào** (token → xác suất → chọn token → lặp lại).
2. Nhẩm được Qwen3-1.7B nặng bao nhiêu GB, và kiểm tra con số đó bằng đo thật.
3. Bật/tắt chế độ **thinking** và đo xem nó tốn thêm bao nhiêu token, bao nhiêu giây.
4. Chạy mô hình trên **nhiều câu cùng lúc** (sinh theo lô) và hiểu vì sao phải đệm bên trái.
5. Có điểm **baseline** của mô hình gốc trên 100 câu test — mốc "trước" cho mọi phép so sánh của 15 buổi sau.

**✅ Tiêu chí hoàn thành** (tự kiểm trước khi sang buổi sau): `check_env()` báo ✅; bảng điểm có dòng `M00/base-zero-shot`; file `p01_runs/outputs/M00__base-zero-shot.json` và `p01_runs/m00_base_outputs.json` tồn tại; bạn giải thích được cho một người bạn vì sao mô hình 1,7 tỷ tham số chỉ chiếm khoảng 1,4 GB VRAM.

### 🗺️ Bản đồ buổi học
| Phần | Câu hỏi trả lời | Ô |
|---|---|---|
| **A. Chuẩn bị** | Máy đã sẵn sàng chưa? | 0a, 0b, 1 |
| **B. Mô hình nhìn chữ thế nào** | Câu tiếng Việt bị cắt thành gì? | 2, 3 |
| **C. Máy đoán chữ** | Mô hình nặng bao nhiêu? Nó "nghĩ" ra chữ kế tiếp ra sao? | 4, 5, 6, 7 |
| **D. Nói chuyện với mô hình** | Chat template và thinking là gì? | 8, 9 |
| **E. Chạy nhanh** | Làm sao hỏi 100 câu mà không chờ cả giờ? | 10, 11 |
| **F. Baseline** | Mô hình gốc được bao nhiêu điểm? | 12, 13, 14 |

---

## 0. Bức tranh lớn — Vì sao bắt đầu bằng Baseline?

Ở công ty, khi bạn đề xuất fine-tune, câu đầu tiên sếp hỏi luôn là: **"Mô hình gốc đang được bao nhiêu điểm?"**. Không có baseline thì không chứng minh được fine-tune có ích, cũng không biết dừng ở đâu là đủ.

> 💡 **Nguyên tắc vàng của cả lộ trình:** mọi buổi đều đo trên **cùng 100 câu test** (`edu/data/p01_test.jsonl`), cùng prompt, cùng bộ chấm. Nhờ vậy con số giữa các buổi **so sánh được với nhau** — giống cân sức khỏe luôn bằng cùng một cái cân.

---

## 📖 Từ điển nhanh (đọc lướt trước, quay lại tra khi gặp)

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| **Token** | Mảnh chữ nhỏ nhất mô hình đọc được. Một từ có thể là 1 token (`' bị'`) hoặc nhiều token (`' Ph'` + `'ường'`) |
| **Tokenizer** | Bộ cắt câu thành token và đổi mỗi token thành một con số (id) |
| **Tham số (parameter)** | Một con số bên trong mô hình, đã được "vặn" sẵn khi huấn luyện. Qwen3-1.7B có khoảng 1,72 tỷ con số như vậy |
| **Logits** | Điểm số thô mô hình chấm cho **từng** token trong từ vựng (151.936 điểm) ở vị trí kế tiếp |
| **Softmax** | Phép đổi điểm số thành xác suất (cộng lại = 100%) |
| **Temperature** | Núm chỉnh "độ liều": thấp → chọn token chắc nhất; cao → chọn đa dạng hơn, dễ bịa hơn |
| **Greedy / Sampling** | Greedy = luôn chọn token xác suất cao nhất (tất định). Sampling = bốc thăm theo xác suất (`top_k`, `top_p` giới hạn vùng được bốc) |
| **FP16 / BF16** | Hai cách lưu số thực bằng 16 bit (2 byte). Card T4 chỉ chạy nhanh được FP16 |
| **Lượng tử hóa (quantization), NF4** | Nén mỗi tham số từ 16 bit xuống 4 bit để mô hình nhẹ đi ~4 lần, đổi lại sai số nhỏ |
| **VRAM** | Bộ nhớ của GPU. T4 có khoảng 15 GB |
| **Chat template** | Khuôn mẫu đặt câu hỏi vào đúng định dạng hội thoại mà mô hình đã được dạy |
| **Thinking mode** | Chế độ Qwen3 "nghĩ nháp" trong `<think>...</think>` trước khi trả lời |
| **Batch / sinh theo lô** | Đưa nhiều câu vào GPU cùng lúc thay vì từng câu một |
| **Padding (đệm)** | Thêm token rỗng để các câu trong cùng lô dài bằng nhau |
| **Baseline / zero-shot** | Điểm của mô hình gốc khi chỉ được đưa câu hỏi, không có ví dụ mẫu, chưa fine-tune |

---

## 1. Lý thuyết — 5 điều phải hiểu hôm nay

> 🧒 **Giải thích kiểu Feynman (đọc trước, 1 phút):** Mô hình ngôn ngữ là **máy đoán chữ tiếp theo** — giống tính năng gợi ý từ trên bàn phím điện thoại, nhưng giỏi hơn rất nhiều. "1,7 tỷ tham số" là 1,7 tỷ núm vặn đã được vặn sẵn trong lúc huấn luyện. NF4 nén mỗi núm từ 16 bit xuống 4 bit — như nén ảnh: nhẹ đi 4 lần, mờ đi một chút. **Baseline** là cân sức khỏe trước khi tập gym: không có số "trước" thì không chứng minh được tập có tác dụng.

### 1.1. Mô hình sinh chữ như thế nào?
Mọi thứ trong khóa học — fine-tune, RAG, agent — đều xây trên vòng lặp đơn giản này:

```text
"Xe ôm bị tai nạn cần nộp hồ sơ tại"
        │  ① tokenizer cắt câu thành token, mỗi token là một con số (id)
        ▼
[' Xe', ' ô', 'm', ' bị', ' tai', ' nạn', ...]
        │  ② mô hình (28 lớp xếp chồng) đọc cả chuỗi
        ▼
151.936 điểm số (logits) — một điểm cho mỗi token trong từ vựng
        │  ③ softmax đổi điểm số thành xác suất
        ▼
' UBND' 31% · ' phường' 12% · ' cơ' 8% · ...      (số minh họa)
        │  ④ chọn 1 token, nối vào cuối chuỗi
        ▼
quay lại ② — lặp cho đến khi sinh ra <|im_end|> hoặc hết max_new_tokens
```

- Mô hình **chỉ đoán 1 token mỗi lần**. Câu trả lời 200 token = 200 vòng lặp.
- **Fine-tune** = chỉnh các núm vặn để xác suất ở bước ③ nghiêng về câu trả lời ta muốn. Ô 5 sẽ cho bạn tận mắt nhìn bước ③.

### 1.2. Token: tiếng Việt "đắt" hơn tiếng Anh
- Tokenizer học cách cắt chữ từ dữ liệu huấn luyện — chủ yếu là tiếng Anh và tiếng Trung. Âm tiết tiếng Việt có dấu ít gặp hơn nên hay bị **cắt vụn** (`ôm` → `' ô'` + `'m'`).
- Nhiều token hơn = tốn bộ nhớ hơn và chậm hơn. Ô 2 sẽ đo con số cụ thể.

### 1.3. Mô hình nặng bao nhiêu? — quy tắc nhẩm
| Cách lưu | Mỗi tham số chiếm | Qwen3-1.7B (≈ 1,72 tỷ tham số) |
|---|---|---|
| FP16 | 2 byte | 1,72 × 2 ≈ **3,44 GB** |
| NF4 (4 bit) | 0,5 byte | ≈ **1,35 GB** (không phải 0,86 GB — xem giải thích bên dưới) |

- **Quy tắc nhẩm:** *số tỷ tham số × 2 = số GB ở FP16*. Mô hình 7B → ~14 GB, không vừa T4 nếu không nén.
- NF4 không nhẹ đúng 4 lần vì **bảng embedding** (bảng tra token → vector, chiếm 0,31 tỷ tham số) **không bị nén**, vẫn giữ 2 byte/tham số.
- Khi train còn cần thêm bộ nhớ cho phần đang học và kết quả trung gian — các buổi sau mới bàn.

### 1.4. `Qwen/Qwen3-1.7B` không phải "mô hình base"
- Trên Hugging Face có **2 bản**: `Qwen/Qwen3-1.7B-Base` (chỉ học đoán chữ, chưa biết hội thoại) và `Qwen/Qwen3-1.7B` (đã được dạy thêm — *post-train* — để biết chat, gọi tool, có chế độ thinking).
- Ta dùng **bản đã post-train** vì dữ liệu của mình ít (vài trăm câu): kế thừa khả năng làm theo chỉ dẫn rẻ hơn nhiều so với dạy lại từ bản Base.
- 🎤 *Phỏng vấn hay hỏi đúng chỗ này* — gọi nhầm bản post-train là "base model" là bị trừ điểm.

### 1.5. Thinking mode và card T4
- **Thinking:** mặc định Qwen3 viết nháp trong `<think>...</think>` rồi mới trả lời → chính xác hơn ở bài khó nhưng **dài và chậm**. Từ Buổi 00 đến 09 ta **tắt** thinking (cần JSON ngắn, nhanh). Buổi 10 quay lại chủ đề này.
- **T4 không có BF16:** T4 là GPU đời cũ (Turing). Ở Colab miễn phí, **luôn dùng FP16** (`fp16=True`, `bnb_4bit_compute_dtype=torch.float16`). Viết `bf16=True` sẽ báo lỗi ngay.

<details>
<summary>📚 <b>Đọc thêm — nâng cao, có thể bỏ qua lần đầu</b> (đếm tham số bằng tay, GQA, NF4 chi tiết, FP16 tràn số)</summary>

**Cấu hình thật của Qwen3-1.7B** (đọc từ `config.json`)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_hidden_layers` | 28 | 28 khối Transformer chồng lên nhau |
| `hidden_size` | 2048 | Mỗi token được biểu diễn bằng vector 2048 chiều |
| `intermediate_size` | 6144 | Lớp MLP phình lên 6144 chiều rồi co lại 2048 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | **GQA**: 2 đầu Query dùng chung 1 cặp Key/Value → KV-cache nhỏ đi một nửa |
| `head_dim` | 128 | 16 đầu × 128 = 2048 |
| `vocab_size` | 151.936 | Số hàng của bảng embedding (đã đệm); tokenizer thật có 151.669 token |
| `tie_word_embeddings` | `true` | Bảng embedding đầu vào và lớp `lm_head` đầu ra **dùng chung** → bớt 311 triệu tham số |

**Đếm tham số từng bước** (bỏ qua các lớp chuẩn hóa vì rất nhỏ):
1. Chiều của Key/Value = 8 đầu KV × 128 = **1024** (nhỏ hơn Query 2048 vì GQA).
2. Attention mỗi lớp: `q 2048×2048` + `k 2048×1024` + `v 2048×1024` + `o 2048×2048` = **12,58 triệu**.
3. MLP mỗi lớp có 3 ma trận (gate, up, down), mỗi cái `2048×6144` → **37,75 triệu**.
4. Một lớp ≈ **50,3 triệu**; 28 lớp → **1,409 tỷ** (gọi là phần *non-embedding*).
5. Embedding `151.936 × 2048` = **0,311 tỷ**. Tổng ≈ **1,720 tỷ**.

**NF4 chi tiết:** các lớp Linear còn 1,409 tỷ × 0,5 byte ≈ 0,70 GB; hằng số lượng tử sau *double quantization* thêm ~0,127 bit/tham số (paper QLoRA) ≈ 0,02 GB; embedding giữ 16-bit = 0,62 GB → **≈ 1,35 GB** (≈ 1,26 GiB). bitsandbytes chỉ thay các lớp `nn.Linear`, không đụng tới `Embedding`.

**FP16 dễ tràn số:** FP16 chỉ biểu diễn được tới khoảng 65.504, nên khi train `Trainer` tự nhân/chia loss theo hệ số (*loss scaling* động) để tránh tràn. BF16 có dải rộng như FP32 nhưng cần GPU Ampere (compute capability 8.0) trở lên. Buổi 06 sẽ cho bạn tự gây ra và sửa lỗi NaN.

</details>

---

## 2. Thực hành từng ô nhỏ

## Phần A — Chuẩn bị

### Ô 0a — Lấy code & cài thư viện (chạy 1 lần cho mỗi runtime)
**Mục tiêu:** đưa code + dữ liệu đồ án vào Colab và cài **đúng phiên bản đã ghim**.

> 🧭 **Người mới:** 3 ô dưới là "thủ tục". Bạn chỉ cần chạy lần lượt, chưa cần hiểu từng dòng. Nếu repo để **private**, làm trước 3 bước: bấm biểu tượng 🔑 (*Secrets*) ở thanh bên trái Colab → *Add new secret* → tên `GITHUB_TOKEN`, giá trị là token GitHub của bạn, bật *Notebook access*.

```python
import os, subprocess, base64
REPO, URL = "/content/uit-social-welfare-rag", "https://github.com/phamxuanphuc/uit-social-welfare-rag.git"
if not os.path.exists(REPO):
    env = dict(os.environ)
    try:  # repo private: đọc GITHUB_TOKEN từ Colab Secrets (🔑); repo public thì bỏ qua
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
- **Giải thích:** ô 1 tải repo về `/content`, ô 2 cài thư viện theo danh sách ghim sẵn, ô 3 khởi động lại Python để dùng thư viện vừa cài. Thông báo "session crashed" sau ô 3 là **bình thường**.
- **Vì sao ô 1 viết phức tạp vậy?** Để token **không bao giờ lộ**: token lấy từ *Secrets* (không nằm trong notebook) và đưa cho `git` qua biến môi trường, **không ghép vào URL**. Nếu ghép vào URL, khi clone lỗi thông báo sẽ in nguyên URL chứa token ra màn hình, và token còn bị lưu trong `.git/config`.
- **Vì sao ghim phiên bản** (`transformers==5.17.0`, `trl==1.15.0`...)? Để notebook vẫn chạy được sau 6 tháng — các thư viện này đổi tên tham số rất nhanh.
- ⚠️ **Lỗi hay gặp:** quên chạy ô 3 → Python vẫn dùng bản `transformers` cũ có sẵn trên Colab, sinh lỗi khó hiểu ở các buổi sau.

### Ô 0b — Khởi động phiên (chạy đầu MỖI buổi)
**Mục tiêu:** nối Google Drive (nơi lưu kết quả để không mất khi Colab ngắt) và kiểm tra môi trường.
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
- **Giải thích:** `sys.path.insert` cho Python biết chỗ tìm thư viện `p01kit` của đồ án. `P01_RUNS` là thư mục trên Drive chứa mọi kết quả của 16 buổi.
- 📌 **Kết quả nên thấy:** `p01kit 2.0.0 | model: Qwen/Qwen3-1.7B`, dòng `GPU: {'name': 'Tesla T4', ...}` và `✅ Môi trường khớp bản ghim`.
- 💡 Đọc mục *Học trên Colab T4 miễn phí* trong [README](README.md) trước buổi đầu: cách giải phóng GPU, phần nào chạy được trên CPU, bị ngắt giữa buổi thì làm gì.
- ⚠️ Thấy `⚠️ transformers: đang là ..., bản ghim là 5.17.0` → bạn quên khởi động lại runtime sau Ô 0a. Thấy `torch ...: ngoài khoảng unsloth hỗ trợ` → Colab vừa đổi bản `torch` mặc định; báo cho nhóm để cập nhật bộ ghim, đừng tự nâng/hạ từng gói.

### Ô 1 (4 dòng) — Kiểm tra GPU
**Mục tiêu:** chắc chắn đang có GPU T4 và biết nó **không** hỗ trợ BF16.
```python
import torch
print(torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Hỗ trợ BF16?", torch.cuda.is_bf16_supported())
```
- **Giải thích:** *compute capability* là "đời" của GPU. T4 là đời 7.5; BF16 cần đời 8.0 trở lên.
- 📌 **Kết quả nên thấy:** `Tesla T4` · `(7, 5)` · `Hỗ trợ BF16? False`.
- ⚠️ Báo lỗi `no CUDA GPUs` → *Runtime → Change runtime type → T4 GPU*, rồi chạy lại Ô 0b.

## Phần B — Mô hình nhìn chữ thế nào

### Ô 2 (6 dòng) — Tokenizer: tiếng Việt "đắt" hơn tiếng Anh
**Mục tiêu:** tận mắt thấy một câu bị cắt thành những mảnh nào.
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(p01kit.MODEL_ID)
for s in ["Tôi chạy xe ôm ở Phường Tân Tạo bị té xe gãy tay",
          "I drive a motorbike taxi in Tan Tao ward and broke my arm"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(len(s.split()), "từ ->", len(ids), "token:", [tok.decode([i]) for i in ids])
```
- **Giải thích:** `tok(s)` đổi câu thành danh sách id; `tok.decode([i])` đổi ngược từng id về mảnh chữ để ta nhìn. Dấu cách ở đầu mảnh (`' bị'`) nghĩa là mảnh đó bắt đầu một từ mới.
- 📌 **Kết quả nên thấy (đã đo):** câu Việt 13 từ → **19 token** (`' ô', 'm'`, `' Ph', 'ường'`, `' g', 'ã', 'y'`...); câu Anh 13 từ → **14 token**.
- 🧠 **Hệ quả thực tế:** prompt RAG của P01 dài trung bình **836 token**, tối đa 885 — tốn bộ nhớ và chậm hơn bản tiếng Anh tương đương khoảng 35%.

### Ô 3 (5 dòng) — Token đặc biệt quyết định mọi thứ
**Mục tiêu:** biết các "dấu câu" riêng của Qwen3 — mô hình dựa vào chúng để biết đâu là lượt người dùng, đâu là lúc dừng.
```python
for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"]:
    print(f"{t:>14} -> id {tok.convert_tokens_to_ids(t)}")
print("eos =", tok.eos_token, "| pad =", tok.pad_token, "| len(tok) =", len(tok))
```
- 📌 **Kết quả nên thấy:** `<|im_start|>`=151644, `<|im_end|>`=151645, `<|endoftext|>`=151643, `<think>`=151667, `</think>`=151668, `<tool_call>`=151657; `eos = <|im_end|>`, `pad = <|endoftext|>`, `len(tok) = 151669`.
- **Giải thích:** `<|im_start|>` mở một lượt nói, `<|im_end|>` (`eos`) **kết thúc lượt trả lời**. `pad` là token rỗng dùng để đệm (Ô 10).
- 🧠 Nếu dữ liệu train thiếu `<|im_end|>`, mô hình sau fine-tune sẽ **không biết dừng** và nói lan man đến hết `max_new_tokens` — lỗi số 1 của người mới.

## Phần C — Máy đoán chữ

### Ô 4 (8 dòng) — Nạp mô hình ở dạng nén 4-bit NF4
**Mục tiêu:** đưa mô hình lên GPU ở dạng nén và xem nó chiếm bao nhiêu bộ nhớ.
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(p01kit.MODEL_ID, quantization_config=bnb,
                                             device_map="auto", dtype=torch.float16)
model.eval()
print(f"Footprint: {model.get_memory_footprint()/1e9:.2f} GB | VRAM đang cấp: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```
- **Giải thích từng tham số:**
  - `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`: nén trọng số xuống 4 bit theo kiểu NormalFloat (hợp với trọng số phân phối hình chuông).
  - `bnb_4bit_use_double_quant=True`: nén luôn cả các hằng số dùng để giải nén → tiết kiệm thêm một chút.
  - `bnb_4bit_compute_dtype=torch.float16`: lúc tính toán, trọng số được **giải nén tạm** về FP16.
  - `dtype=torch.float16`: các phần không bị nén (embedding, lớp chuẩn hóa) giữ ở FP16.
  - `model.eval()`: chuyển sang chế độ dùng (không train).
- 📌 **Kết quả nên thấy:** footprint ≈ **1,3–1,5 GB**. Ô 6 sẽ giải thích vì sao ra con số này.
- ⚠️ Ra ~3,4 GB nghĩa là `quantization_config` chưa có hiệu lực. Transformers 5 dùng `dtype=`; tham số cũ có tiền tố `torch_` sẽ báo cảnh báo deprecated.

### Ô 5 (9 dòng) — 🎯 Sản phẩm 1: Máy X-quang token kế tiếp
**Mục tiêu:** nhìn tận mắt bước ③ của sơ đồ mục 1.1 — mô hình đang phân vân giữa những token nào.
```python
@torch.inference_mode()
def xray(text: str, k: int = 5, temperature: float = 1.0):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1].float()          # điểm số tại vị trí CUỐI = dự đoán token kế tiếp
    probs = torch.softmax(logits / temperature, dim=-1)  # đổi điểm số thành xác suất
    for p, i in zip(*probs.topk(k)):                     # lấy k token xác suất cao nhất
        print(f"{tok.decode([i.item()])!r:>16}  {p.item():6.1%}")

xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại")
```
- **Giải thích từng dòng:**
  - `@torch.inference_mode()`: báo PyTorch "chỉ dùng, không học" → nhanh hơn, tốn ít bộ nhớ hơn.
  - `model(**enc).logits` có dạng `[số câu, số token, 151.936]`. `[0, -1]` = câu đầu tiên, token **cuối cùng** — vì ta muốn biết chữ *sau* câu này.
  - Chia cho `temperature` trước softmax: chia cho số nhỏ làm các điểm số cách xa nhau hơn → phân phối "nhọn" hơn.
- 📌 **Kết quả nên thấy:** top-5 là các mảnh như `' UBND'`, `' Ủy'`, `' cơ'`, `' phường'`... (cụ thể tùy mô hình), xác suất giảm dần.
- 🧠 Đây chính là "bộ não" bên dưới mọi thứ: fine-tune = **dịch chuyển phân phối này** về phía câu trả lời ta muốn.

### Ô 6 (3 dòng) — Thí nghiệm: núm vặn temperature
**Mục tiêu:** thấy temperature thay đổi phân phối xác suất thế nào.
```python
for T in (0.3, 1.0, 1.5):
    print(f"--- temperature={T} ---")
    xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại", temperature=T)
```
- 📌 **Kết quả nên thấy:** cùng thứ tự token nhưng khác xác suất: `T=0.3` → token đứng đầu chiếm gần hết (gần như greedy); `T=1.5` → các xác suất sát nhau hơn, **tổng top-5 giảm rõ** (mô hình đa dạng hơn nhưng dễ bịa hơn).
- 🧠 Với bài toán cần câu trả lời chính xác như P01, ta dùng **greedy** (chọn token cao nhất) khi đánh giá — kết quả tất định, chạy lại ra y hệt.

### Ô 7 (7 dòng) — Nhẩm bộ nhớ và đối chiếu với đo thật
**Mục tiêu:** tự tính số tham số và dung lượng, rồi so với con số Ô 4 đã đo.
```python
L, H, I, KV, V = 28, 2048, 6144, 1024, 151_936   # số lớp, chiều ẩn, chiều MLP, chiều K/V (8 đầu × 128), số token
attn, mlp = H*H + 2*H*KV + H*H, 3*H*I             # attention (q, k+v, o) và MLP (gate, up, down) mỗi lớp
non_emb, emb = L * (attn + mlp), V * H            # 28 lớp + bảng embedding
print(f"Tham số: {(non_emb + emb)/1e9:.3f} tỷ | FP16: {(non_emb + emb)*2/1e9:.2f} GB")
print(f"NF4 ước tính: {(non_emb*0.5 + non_emb*0.127/8 + emb*2)/1e9:.2f} GB")
print(f"Đo thật (Ô 4): {model.get_memory_footprint()/1e9:.2f} GB")
```
- **Giải thích dòng NF4:** `non_emb*0.5` = các lớp Linear, 4 bit = 0,5 byte; `non_emb*0.127/8` = hằng số giải nén (~0,127 bit mỗi tham số); `emb*2` = embedding **không** bị nén, vẫn 2 byte. Muốn hiểu từng phép nhân, mở mục *📚 Đọc thêm* ở phần lý thuyết.
- 📌 **Kết quả nên thấy:** `Tham số: 1.720 tỷ | FP16: 3.44 GB`, `NF4 ước tính: 1.35 GB`, và số đo thật nằm trong khoảng 1,3–1,5 GB.
- 🧠 Ước tính lệch đo thật một chút là bình thường: thư viện còn giữ thêm vài lớp nhỏ (chuẩn hóa) và bộ đệm. Kỹ sư giỏi **ước tính trước khi chạy** để biết mô hình có vừa GPU không.

## Phần D — Nói chuyện với mô hình

### Ô 8 (6 dòng) — Chat template: thinking bật vs tắt
**Mục tiêu:** xem câu hỏi thật sự được gửi vào mô hình trông như thế nào.
```python
msgs = [{"role": "system", "content": "Bạn là trợ lý an sinh."},
        {"role": "user", "content": "Xe ôm bị tai nạn cần giấy gì?"}]
for think in (True, False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    print(f"--- enable_thinking={think} ---\n{text!r}\n")
```
- **Giải thích:** `apply_chat_template` bọc từng tin nhắn bằng `<|im_start|>vai_trò ... <|im_end|>`. `add_generation_prompt=True` thêm `<|im_start|>assistant\n` ở cuối — "đến lượt mô hình nói".
- 📌 **Kết quả nên thấy:** hai chuỗi giống nhau, chỉ khác phần đuôi:
  - `think=True` → kết thúc bằng `'<|im_start|>assistant\n'` (mô hình tự quyết định có nghĩ hay không).
  - `think=False` → kết thúc bằng `'<|im_start|>assistant\n<think>\n\n</think>\n\n'` — khối nghĩ **rỗng** được điền sẵn, mô hình coi như đã nghĩ xong và trả lời luôn.
- 🧠 **Ghi nhớ cho các buổi train:** prompt lúc train và lúc dùng **phải giống hệt nhau đến từng ký tự**. `p01kit.prompts.render_prompt()` là nơi duy nhất tạo prompt cho cả khóa.

### Ô 9 (10 + 3 dòng) — Thinking vs Non-thinking: đo, đừng đoán
**Mục tiêu:** đo thinking tốn thêm bao nhiêu token và bao nhiêu giây.
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
- **Giải thích:**
  - `model.generate` chạy vòng lặp ①–④ của mục 1.1 cho tới khi gặp `<|im_end|>` hoặc đủ `max_new_tokens`.
  - `out[0, enc["input_ids"].shape[1]:]` cắt bỏ phần prompt, chỉ giữ phần mô hình mới sinh.
  - Tham số sampling theo **khuyến nghị chính thức của Qwen3**: thinking `T=0.6, top_p=0.95, top_k=20`; non-thinking `T=0.7, top_p=0.8, top_k=20`. Qwen khuyên **không dùng greedy cho thinking** (dễ lặp vô hạn).
- 📌 **Kết quả nên thấy:** thinking sinh **nhiều token hơn hẳn** (thường gấp vài lần) với tốc độ tok/s tương đương → tổng thời gian dài hơn tương ứng. Ghi số đo thật của bạn vào vở.

## Phần E — Chạy nhanh trên nhiều câu

### Ô 10 (5 dòng) — Vì sao sinh theo lô phải đệm TRÁI?
**Mục tiêu:** hiểu cách xếp nhiều câu dài ngắn khác nhau vào cùng một lô.
```python
tok.padding_side = "left"
batch = tok(["Xin chào", "Tôi chạy xe ôm ở Tân Tạo bị té xe"], return_tensors="pt",
            padding=True, add_special_tokens=False)
print(batch["input_ids"])
print(batch["attention_mask"])
```
- 🧒 **Hình dung:** hai câu xếp thành hai hàng, mô hình **viết tiếp vào mép phải** của mỗi hàng. Câu ngắn phải được đệm ở bên **trái** để mép phải của hai hàng thẳng nhau. Nếu đệm bên phải, câu ngắn sẽ có token rỗng chen giữa câu hỏi và phần trả lời → kết quả sai.
- **Giải thích:** `attention_mask` đánh dấu `1` = token thật, `0` = token đệm (mô hình bỏ qua).
- 📌 **Kết quả nên thấy:** hàng 1 bắt đầu bằng vài số `151643` (pad) và `attention_mask` tương ứng là `0`.
- ⚠️ Khi **train** thì ngược lại: đệm phải (xem `p01kit.sft.PadCollator`). Lẫn lộn hai chiều này là lỗi rất phổ biến.

### Ô 11 (10 + 4 dòng) — 🎯 Sản phẩm 2: Bộ sinh theo lô & đo tốc độ tăng
**Mục tiêu:** viết hàm hỏi nhiều câu cùng lúc và đo xem nhanh hơn hỏi từng câu bao nhiêu.
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
- **Giải thích:**
  - Vòng `for` cắt danh sách prompt thành từng lô `batch_size` câu.
  - `do_sample=False` = greedy. `temperature=None, top_p=None, top_k=None` tắt các giá trị sampling mặc định của Qwen3 (mặc định `do_sample=True, T=0.6`) để greedy thật sự tất định và không bị cảnh báo.
  - `batch_decode` đổi cả lô id về chữ một lần.
- 📌 **Kết quả nên thấy:** `batch_size=8` nhanh hơn `batch_size=1` **nhiều lần** — GPU xử lý 8 câu song song trong mỗi bước. Hàm này chính là `p01kit.infer.generate` mà các buổi sau dùng lại.

## Phần F — Baseline

### Ô 12 (7 dòng) — Làm quen tập test P01
**Mục tiêu:** xem một câu test gồm những gì trước khi chấm điểm.
```python
from p01kit.data import load_split, dataset_stats
test = load_split("test")
print(dataset_stats(test))
r = test[0]
print("Địa bàn:", r["region"], "\nCâu hỏi:", r["question"])
print("Ngữ cảnh:\n", r["context"][:600], "...")
print("Đáp án vàng:", r["gold"])
```
- **Giải thích:** mỗi câu test có **câu hỏi** của người dân, **ngữ cảnh** (3 đoạn văn bản chính sách — có thể có đoạn gây nhiễu) và **đáp án vàng** (JSON đúng mà ta mong mô hình trả ra). Có câu mà ngữ cảnh không chứa văn bản phù hợp → mô hình phải **từ chối**.
- 📌 **Kết quả nên thấy:** `n=100`, `unanswerable_ratio=0.44` (44% câu phải từ chối), `expired_ratio=0.1`, `colloquial_ratio=0.5` (một nửa là giọng đời thường), `slices={'seen_doc': 24, 'unseen_doc': 72, 'ood': 4}`. Buổi 01 sẽ mổ xẻ kỹ cấu trúc này.
- ⚠️ Văn bản chính sách trong bộ dữ liệu là **giả lập** — không dùng để tra cứu thủ tục thật.

### Ô 13 (7 dòng) — 📊 Sản phẩm 3: Đo Baseline zero-shot & ghi bảng điểm
**Mục tiêu:** chấm mô hình gốc trên 100 câu test và lưu kết quả làm mốc.
```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
res0 = evaluate(model, tok, test, batch_size=8, max_new_tokens=p01kit.MAX_NEW_TOKENS)
print(format_summary(res0["summary"]))
print("Thời gian:", res0["summary"]["seconds"], "giây")
scoreboard.record("M00", "base-zero-shot", res0["summary"], latency_s=res0["summary"]["sec_per_row"], outputs=res0["outputs"])
```
- **Giải thích:** `evaluate()` làm 4 việc: tạo prompt chuẩn P01 (`render_prompt`) → sinh theo lô bằng greedy (như Ô 11) → chấm từng câu (`score_one`) → tổng hợp điểm kèm **khoảng tin cậy 95%**. `scoreboard.record` ghi một dòng vào `leaderboard.csv` trên Drive và lưu output thô từng câu.
- **Hôm nay chỉ cần đọc 2 chỉ số** (6 chỉ số còn lại học ở Buổi 01):

  | Chỉ số | Nghĩa | Đọc thế nào |
  |---|---|---|
  | `json_valid` | Tỉ lệ câu trả lời đúng định dạng JSON | Thấp → mô hình chưa quen định dạng P01 |
  | `answer_acc` | Tỉ lệ quyết định đúng: từ chối đúng lúc, hoặc trích đúng văn bản | Đây là "điểm tổng" dễ hiểu nhất |

- **Khoảng tin cậy `95%CI` là gì?** Ví dụ `answer_acc 0.60 [0.50, 0.70]`: với chỉ 100 câu, điểm "thật" của mô hình có thể nằm đâu đó từ 0,50 đến 0,70. Hai mô hình chênh nhau 3 điểm thường **chưa** khác nhau thật — Buổi 01 dạy cách kiểm định.
- 📌 **Kết quả nên thấy:** bảng 8 chỉ số kèm `95%CI` và `n`. Kiểm tra hợp lý: `json_valid` phải > 0. Đừng ngạc nhiên nếu điểm từ chối và cảnh báo hết hạn thấp — đó chính là lý do ta fine-tune.
- ⚠️ `json_valid = 0` → kiểm tra lại có tắt thinking không (`evaluate` mặc định đã tắt) và `max_new_tokens` có đủ lớn không.

### Ô 14 (5 dòng) — Phân tích lỗi (error analysis)
**Mục tiêu:** đọc tận mắt các câu mô hình làm sai — đây là "bản đồ" cho các buổi sau.
```python
import json
from p01kit.infer import show_errors
show_errors(res0, test, k=5)
with open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json", "w") as f:
    json.dump(res0["outputs"], f, ensure_ascii=False)
```
- **Giải thích:** `show_errors` in 5 câu sai, mỗi câu gồm: câu hỏi, output của mô hình, và tóm tắt đáp án vàng (`gold_refusal` = có phải từ chối không, `gold_doc` = văn bản đúng). Dòng cuối lưu output thô ra Drive để **Buổi 01 chấm lại miễn phí**, không phải sinh lại.
- **Việc cần làm:** xếp 5 ca sai vào các nhóm dưới đây và ghi tỉ lệ vào vở:

  | Nhóm lỗi | Dấu hiệu |
  |---|---|
  | (a) Sai định dạng JSON | Output không phải JSON, thiếu khóa, thêm chữ ngoài JSON |
  | (b) Không từ chối khi phải từ chối | Ngữ cảnh không có văn bản phù hợp nhưng mô hình vẫn trả lời |
  | (c) Chọn nhầm văn bản lệch địa bàn | Người hỏi ở Đà Nẵng, mô hình trích văn bản TP.HCM |
  | (d) Bỏ sót cảnh báo hết hạn | Văn bản đã hết hiệu lực nhưng `expired_warning` là `false` |
  | (e) Bịa giấy tờ | Liệt kê giấy tờ không có trong ngữ cảnh |

---

## 3. So sánh hiệu quả — bảng điểm sau Buổi 00

Mở bảng điểm bất cứ lúc nào:
```python
scoreboard.show()
```

| module | variant | json_valid | refusal_acc | citation_precision | expired_recall | strict_acc | fairness_gap |
|---|---|---|---|---|---|---|---|
| M00 | base-zero-shot | *(số đo của bạn)* | | | | | |

> 📏 **Đọc bảng điểm thế nào?** Mỗi chỉ số có một ngưỡng chốt trước trong [PROJECT-REQUIREMENTS.md](../PROJECT-REQUIREMENTS.md): `citation_precision ≥ 0.90`, `refusal_acc ≥ 0.90`, `expired_recall = 1.0`, `fairness_gap ≤ 0.10`. Gọi `p01kit.metrics.gate(summary)` để xem đạt/không đạt. Sau buổi này **phần lớn ngưỡng chưa đạt** là bình thường — cả khóa học là hành trình đưa chúng về ✅.

---

## 🏢 Ở công ty thật

- **Baseline thật là hệ thống đang có**, không chỉ base model: quy trình tra cứu thủ công, trang FAQ, bộ luật từ khóa, hoặc một API mô hình lớn. Fine-tune chỉ đáng làm khi thắng các phương án đó về chất lượng **và** chi phí/độ trễ.
- **Ràng buộc triển khai được chốt trước khi chọn mô hình.** Ở P01 là CPU 2 vCPU / 4 GiB (ADR 0001), nên 1,7B được chọn trước khi đo bất cứ thứ gì. Ở công ty, đội ước tính chi phí suy luận (token/request × request/ngày) ngay từ đầu.
- **Giấy phép mô hình** (Apache-2.0 với Qwen3) được bộ phận pháp chế kiểm tra trước khi dùng cho sản phẩm.
- **Phân tích lỗi trên mẫu traffic thật** (đã che thông tin cá nhân), không chỉ trên tập test tự dựng: câu hỏi thật thường ngắn hơn, sai chính tả, không dấu (Buổi 15).

---

## 🆘 Gặp lỗi? Tra nhanh

| Triệu chứng | Nguyên nhân thường gặp | Cách sửa |
|---|---|---|
| `no CUDA GPUs are available` | Runtime đang là CPU | *Runtime → Change runtime type → T4 GPU*, chạy lại Ô 0b |
| `ModuleNotFoundError: p01kit` | Chưa chạy Ô 0b sau khi khởi động lại | Chạy lại Ô 0b |
| `check_env()` cảnh báo sai phiên bản | Quên khởi động lại sau khi cài | Chạy ô `os.kill(...)` của Ô 0a, rồi Ô 0b |
| `CUDA out of memory` | Còn mô hình cũ trong bộ nhớ, hoặc `batch_size` quá lớn | *Runtime → Restart session*, chạy lại từ Ô 0b; giảm `batch_size` |
| Footprint ~3,4 GB ở Ô 4 | Chưa nén 4-bit | Kiểm tra đã truyền `quantization_config=bnb` |
| `json_valid = 0` ở Ô 13 | Thinking đang bật hoặc `max_new_tokens` quá nhỏ | Giữ mặc định của `evaluate`, dùng `p01kit.MAX_NEW_TOKENS` |
| Colab bị ngắt giữa buổi | Không tương tác lâu hoặc hết phiên | Chạy lại Ô 0b rồi chạy tiếp từ Ô đang dở (kết quả trên Drive vẫn còn) |

---

## 4. Bài tập

**Bắt buộc**
1. Đo `tok/s` của `generate_batch` với `batch_size ∈ {1, 4, 8, 16}` trên 32 prompt; vẽ đồ thị. Ở `batch_size` nào tốc độ không tăng thêm nữa? Vì sao?
   *Gợi ý: mỗi câu trong lô cần chỗ trong bộ nhớ để lưu kết quả trung gian (KV-cache); đến lúc nào đó GPU bận hết công suất.*
2. Tính tay số tham số nếu Qwen3-1.7B **không** dùng chung embedding với `lm_head`. VRAM ở FP16 tăng bao nhiêu?
   *Gợi ý: khi không dùng chung, có thêm một ma trận `lm_head` cùng kích thước `151.936 × 2048`.*
3. Chạy baseline với **thinking bật** trên 20 câu test đầu: `evaluate(model, tok, test[:20], enable_thinking=True, max_new_tokens=1536, batch_size=4)`. So sánh `answer_acc` và thời gian với non-thinking. Ghi vào bảng điểm với `variant="base-thinking-20"`.

**Nâng cao**
4. So sánh footprint khi nạp **8-bit** (`load_in_8bit=True`) và FP16 (không nén). Chênh lệch có khớp quy tắc nhẩm ở mục 1.3 không?
5. Viết hàm `xray_chat(question)` soi top-5 token **đầu tiên của câu trả lời** sau khi áp chat template với `enable_thinking=False`. Token đầu tiên có phải `{` không? Điều đó nói gì về khả năng mô hình trả JSON?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao phải đo baseline trước khi fine-tune?** *(cơ bản)*
   A: Để có mốc so sánh, phát hiện bài toán có thể giải bằng prompt/RAG (không cần fine-tune), và để đặt tiêu chí dừng. Baseline phải đo trên **cùng** tập test, prompt, bộ chấm với các phiên bản sau.
2. **Q: Qwen3-1.7B có bao nhiêu tham số và vì sao có chỗ ghi 1.4B?** *(cơ bản)*
   A: Tổng ≈ 1,72 tỷ; 1,41 tỷ là phần *non-embedding*. Embedding 0,31 tỷ được dùng chung (tie) với `lm_head`.
3. **Q: Vì sao sinh theo lô phải đệm trái còn train lại đệm phải?** *(cơ bản)*
   A: Khi sinh, token mới được nối vào cuối chuỗi nên mọi prompt phải kết thúc thẳng hàng (đệm trái). Khi train, vị trí đệm đã được che bằng `attention_mask` và nhãn `-100`, đệm phải đơn giản và an toàn.
4. **Q: `enable_thinking=False` hoạt động thế nào ở mức token?** *(cơ bản)*
   A: Chat template chèn sẵn `<think>\n\n</think>\n\n` sau `<|im_start|>assistant\n`, mô hình coi như đã nghĩ xong và trả lời ngay.
5. **Q: Vì sao train QLoRA trên T4 phải dùng FP16 chứ không BF16?** *(trung bình)*
   A: T4 (Turing, sm_75) không có đơn vị tính BF16; BF16 cần Ampere (sm_80+). FP16 dải hẹp nên cần loss scaling, đổi lại vẫn tăng tốc được bằng Tensor Core của T4.
6. **Q: GQA là gì và lợi gì khi suy luận?** *(nâng cao)*
   A: Nhiều đầu Query dùng chung một cặp Key/Value (Qwen3-1.7B: 16 Q / 8 KV). KV-cache nhỏ đi 2 lần → phục vụ được ngữ cảnh dài hơn, lô lớn hơn.
7. **Q: Tại sao embedding không bị lượng tử 4-bit trong bitsandbytes?** *(nâng cao)*
   A: bitsandbytes chỉ thay các lớp `nn.Linear` bằng `Linear4bit`; embedding là bảng tra cứu, lượng tử nó gây sai số lớn cho token hiếm mà lợi bộ nhớ không nhiều.

---

## 6. Checklist cuối buổi
- [ ] Vẽ lại được sơ đồ vòng lặp sinh chữ ở mục 1.1 mà không nhìn bài.
- [ ] Giải thích được 2 bản `Qwen3-1.7B` vs `Qwen3-1.7B-Base`.
- [ ] Nhẩm được 3,44 GB (FP16) và giải thích vì sao NF4 là ~1,35 GB chứ không phải 0,86 GB.
- [ ] Viết được `generate_batch` với đệm trái và đo được tốc độ tăng.
- [ ] Bảng điểm có dòng `M00 / base-zero-shot`; đã phân loại 5 lỗi đầu tiên.

## Tài liệu tham khảo
- Qwen3 Technical Report & model card `Qwen/Qwen3-1.7B` (mục *Best Practices* về tham số sampling).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double quantization).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.
