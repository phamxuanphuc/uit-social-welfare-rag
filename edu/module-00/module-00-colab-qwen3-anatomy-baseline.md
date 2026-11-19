# Buổi 00: Làm quen `Qwen3-1.7B` trên Colab T4 và đo baseline

| Mục | Nội dung |
|---|---|
| Thời lượng | 3 giờ (45 phút lý thuyết, 2 giờ 15 phút thực hành). Có thể chia 2 lần ngồi: lần 1 làm Phần A–C, lần 2 làm Phần D–F |
| Cần biết trước | Python cơ bản (biến, vòng lặp, hàm) và biết mở Google Colab. Không cần biết Transformer hay PyTorch |
| Chuẩn bị | Tài khoản Google (Colab T4 miễn phí), xem mẹo tiết kiệm GPU ở [README](../README.md) |
| Sản phẩm | (1) Hàm `xray` soi xác suất token kế tiếp; (2) Hàm `generate_batch` sinh theo lô; (3) Dòng baseline `M00/base-zero-shot` trên 100 câu test |

### Mục tiêu buổi học

1. Hiểu vòng lặp sinh chữ của mô hình ngôn ngữ (tách token, tính xác suất, chọn 1 token, lặp lại).
2. Tự nhẩm dung lượng Qwen3-1.7B (FP16 và NF4) và đối chiếu bằng số đo thật.
3. Đo chênh lệch token và thời gian khi bật/tắt chế độ thinking.
4. Viết hàm sinh theo lô đệm trái để chạy nhiều câu song song.
5. Ghi mốc baseline của mô hình gốc trên 100 câu test cho 15 buổi tiếp theo.

**Tiêu chí hoàn thành:** `check_env()` báo khớp bản ghim và nhận Tesla T4; bảng điểm có dòng `M00/base-zero-shot`; trên Drive có 2 file `p01_runs/outputs/M00__base-zero-shot.json` (Ô 13 tự tạo) và `p01_runs/m00_base_outputs.json` (Ô 14 lưu); bạn giải thích được vì sao mô hình 1,7 tỷ tham số tốn ~1,4 GB VRAM ở 4-bit.

### Bản đồ buổi học và phân bổ thời gian

| Phần | Trọng tâm | Các ô | Thời lượng | Ghi chú đứng lớp |
|---|---|---|---|---|
| Lý thuyết | 5 ý cốt lõi + Từ điển nhanh | Mục 0–1 | 45' | Có thể lồng vào lúc chờ cài gói Ô 0a-2 |
| A. Chuẩn bị | Bật T4, tải repo, cài thư viện, kiểm tra máy | 0a, 0b, 1 | 20' | Nút chờ: Ô 0a-2 cài gói ~2–3' |
| B. Tokenizer | Tiếng Việt tốn token hơn tiếng Anh, token đặc biệt | 2, 3 | 15' | Chạy nhanh (< 10 giây) |
| C. Máy đoán chữ | Nạp 4-bit, soi `xray`, thử temperature, nhẩm VRAM | 4, 5, 6, 7 | 35' | Nút chờ: Ô 4 tải 3,4 GB lần đầu ~1–2' |
| D. Hội thoại | Chat template và đo chi phí thinking | 8, 9 | 20' | Chạy nhanh, để học viên tự thử câu hỏi khác |
| E. Chạy nhanh | Đệm trái và hàm sinh theo lô `generate_batch` | 10, 11 | 20' | So thời gian `batch_size=1` vs `8` |
| F. Baseline | Xem tập test, chấm 100 câu, phân loại 5 câu sai | 12, 13, 14 | 25' | Nút chờ: Ô 13 chạy 100 câu ~2–5' (tranh thủ giải thích `95%CI`) |

---

## 0. Vì sao phải đo baseline trước?

Khi đề xuất fine-tune ở công ty, câu đầu tiên luôn là: "Mô hình gốc đang được bao nhiêu điểm?". Không có mốc ban đầu thì không chứng minh được fine-tune có hiệu quả hay không, cũng không biết khi nào nên dừng.

Xuyên suốt 16 buổi, mọi phiên bản đều chấm trên đúng 100 câu test trong `edu/data/p01_test.jsonl`, cùng prompt và cùng bộ chấm để điểm số so sánh trực tiếp được với nhau.

---

## Từ điển nhanh

Đọc lướt để quen mặt chữ, gặp trong bài thì quay lại tra.

| Thuật ngữ | Nghĩa ngắn gọn |
|---|---|
| Token | Mảnh chữ nhỏ nhất mô hình đọc được (`' bị'` là 1 token; `'Phường'` tách thành `' Ph'` + `'ường'`) |
| Tokenizer | Bộ cắt câu thành token và đổi mỗi mảnh thành một con số (id) |
| Tham số (parameter) | Một con số bên trong mô hình đã chỉnh sẵn lúc huấn luyện (Qwen3-1.7B có ~1,72 tỷ số) |
| Embedding | Bảng tra cứu đổi mỗi token id thành một vector 2048 số để mô hình tính toán |
| Logits | 151.936 điểm số thô mô hình chấm cho từng token trong từ vựng ở vị trí kế tiếp |
| Softmax | Phép đổi logits thành xác suất có tổng bằng 100% |
| Temperature | Núm chỉnh độ liều: thấp thì bám chữ chắc nhất, cao thì đa dạng hơn nhưng dễ bịa |
| Greedy / Sampling | Greedy luôn chọn chữ xác suất cao nhất (tất định). Sampling bốc thăm theo xác suất (`top_k`, `top_p`) |
| FP16 / BF16 | Hai kiểu số thực 16 bit (2 byte). Card T4 chỉ chạy nhanh được FP16 |
| Lượng tử hóa, NF4 | Nén mỗi tham số từ 16 bit xuống 4 bit để nhẹ đi ~4 lần, đổi lại sai số nhỏ |
| VRAM | Bộ nhớ của GPU (Tesla T4 có ~15 GB) |
| KV-cache | Bộ nhớ tạm lưu kết quả các token đã đọc để không phải tính lại ở mỗi bước sinh mới |
| Chat template | Khuôn mẫu bọc câu hỏi theo đúng định dạng hội thoại mà mô hình từng học |
| Thinking mode | Chế độ Qwen3 viết nháp trong `<think>...</think>` trước khi trả lời |
| Batch (sinh theo lô) | Đưa nhiều câu vào GPU xử lý song song thay vì từng câu một |
| Padding (đệm) | Chèn token rỗng để các câu trong cùng lô có độ dài bằng nhau |
| RAG | Tìm sẵn đoạn văn bản liên quan rồi dán vào prompt cho mô hình đọc trước khi trả lời |
| Baseline / zero-shot | Điểm mô hình gốc khi chỉ đưa câu hỏi, không ví dụ mẫu và chưa fine-tune |

---

## 1. Lý thuyết: 5 điều cần hiểu

**Giải thích kiểu Feynman:**
Mô hình ngôn ngữ là máy đoán chữ tiếp theo, giống gợi ý từ trên bàn phím điện thoại nhưng lớn hơn nhiều. "1,7 tỷ tham số" là 1,7 tỷ núm vặn đã chỉnh sẵn lúc huấn luyện. Nén NF4 giống nén ảnh: hạ mỗi núm từ 16 bit xuống 4 bit nên nhẹ đi gần 4 lần, hơi mờ đi một chút. Baseline giống bước lên cân trước ngày đầu tập gym: không có số ban đầu thì không biết tập có tác dụng hay không.

### 1.1. Vòng lặp sinh chữ 4 bước

```text
"Xe ôm bị tai nạn cần nộp hồ sơ tại"
        │  1. Tokenizer cắt câu thành mảnh và đổi sang id
        ▼
[' Xe', ' ô', 'm', ' bị', ' tai', ' nạn', ...]
        │  2. Mô hình (28 lớp xếp chồng) đọc cả chuỗi
        ▼
151.936 điểm số (logits), mỗi token trong từ vựng 1 điểm
        │  3. Softmax đổi điểm số thành xác suất (tổng = 100%)
        ▼
' UBND' 31% · ' phường' 12% · ' cơ' 8% · ...      (số minh họa)
        │  4. Chọn 1 token, nối vào cuối câu
        ▼
Quay lại bước 2, lặp đến khi gặp <|im_end|> hoặc hết max_new_tokens
```

Mỗi vòng chỉ sinh đúng 1 token (200 token trả lời = 200 vòng). Fine-tune chính là chỉnh các tham số để xác suất ở bước 3 nghiêng về câu trả lời ta muốn.

### 1.2. Tiếng Việt tốn token hơn tiếng Anh

Tokenizer học cách cắt từ dữ liệu chủ yếu là tiếng Anh và tiếng Trung, nên chữ tiếng Việt có dấu hay bị băm nhỏ (`ôm` → `' ô'` + `'m'`, `gãy` → `' g'` + `'ã'` + `'y'`). Càng nhiều token thì càng tốn VRAM và chạy càng chậm.

### 1.3. Nhẩm dung lượng mô hình

| Cách lưu | Mỗi tham số | Qwen3-1.7B (~1,72 tỷ tham số) |
|---|---|---|
| FP16 | 2 byte | 1,72 × 2 ≈ 3,44 GB |
| NF4 (4-bit) | 0,5 byte | ~1,35 GB (không phải 0,86 GB) |

Quy tắc nhẩm: lấy số tỷ tham số nhân 2 sẽ ra số GB ở FP16 (mô hình 7B ~14 GB, không vừa T4 15 GB nếu không nén). NF4 ra ~1,35 GB thay vì 0,86 GB vì bảng embedding (0,31 tỷ tham số = 0,62 GB) không bị `bitsandbytes` nén xuống 4-bit mà vẫn giữ 2 byte/tham số.

### 1.4. `Qwen/Qwen3-1.7B` không phải "mô hình base"

- `Qwen/Qwen3-1.7B-Base`: chỉ biết đoán chữ tiếp theo, chưa biết chat.
- `Qwen/Qwen3-1.7B`: đã post-train để biết chat, gọi tool và thinking.

Ta dùng bản đã post-train vì dữ liệu đồ án nhỏ (vài trăm câu), kế thừa sẵn khả năng làm theo chỉ dẫn rẻ hơn dạy lại từ bản Base. Gọi nhầm bản này là "base model" rất hay bị bắt bẻ khi phỏng vấn.

### 1.5. Thinking mode và giới hạn card T4

- **Thinking:** Qwen3 có thể nháp trong `<think>...</think>` trước khi trả lời. Tốt cho bài khó nhưng dài và chậm, nên từ Buổi 00 đến 09 ta tắt thinking để xuất JSON ngắn, nhanh (Buổi 10 quay lại).
- **Card T4 (Turing):** không hỗ trợ phần cứng BF16. Trên Colab miễn phí luôn dùng FP16 (`fp16=True`, `bnb_4bit_compute_dtype=torch.float16`); viết `bf16=True` sẽ báo lỗi ngay.

<details>
<summary>Đọc thêm phần nâng cao (có thể bỏ qua lần đầu): đếm tham số, GQA, NF4 chi tiết, tràn số FP16</summary>

Thông số thật từ `config.json` của Qwen3-1.7B:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_hidden_layers` | 28 | 28 khối Transformer xếp chồng |
| `hidden_size` | 2048 | Chiều vector mỗi token |
| `intermediate_size` | 6144 | Lớp MLP mở rộng lên 6144 rồi co về 2048 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | GQA: 2 đầu Query dùng chung 1 cặp Key/Value → KV-cache giảm 2 lần |
| `head_dim` | 128 | 16 × 128 = 2048 |
| `vocab_size` | 151.936 | Số hàng bảng embedding (tokenizer thật có 151.669 token) |
| `tie_word_embeddings` | `true` | Bảng embedding và `lm_head` dùng chung trọng số → bớt 311 triệu tham số |

Đếm tham số (bỏ qua lớp chuẩn hóa vì rất nhỏ):
1. Chiều K/V = 8 × 128 = 1024.
2. Attention mỗi lớp: `q (2048×2048)` + `k (2048×1024)` + `v (2048×1024)` + `o (2048×2048)` = 12,58 triệu.
3. MLP mỗi lớp: 3 ma trận `2048×6144` (gate, up, down) = 37,75 triệu.
4. 28 lớp × 50,3 triệu = 1,409 tỷ (non-embedding). Cộng embedding `151.936 × 2048` = 0,311 tỷ → tổng 1,720 tỷ.

NF4 chi tiết: 28 lớp Linear tốn 1,409 tỷ × 0,5 byte ≈ 0,70 GB; hằng số double quantization (~0,127 bit/tham số) ≈ 0,02 GB; embedding giữ FP16 = 0,62 GB → tổng ~1,35 GB (~1,26 GiB). `bitsandbytes` chỉ nén các lớp `nn.Linear`, không nén `nn.Embedding`.

FP16 chỉ chứa tối đa ~65.504 nên khi train `Trainer` tự dùng dynamic loss scaling để tránh `NaN`. BF16 cần GPU Ampere (`sm_80`) trở lên (xem thêm Buổi 06).

</details>

---

## 2. Thực hành từng ô trên Colab

## Phần A: Chuẩn bị

> **Bật T4 trước khi chạy Ô 0a:** Vào *Runtime → Change runtime type → T4 GPU* rồi bấm Save. Nếu chạy trên CPU rồi lát nữa mới đổi sang T4, Colab sẽ cấp máy mới tinh và xóa sạch thư viện vừa cài.

### Ô 0a: Tải repo và cài thư viện (chạy 1 lần cho mỗi phiên máy mới)

Vì sao cần tải repo `uit-social-welfare-rag` (đã mở public, không cần token)? Nó chứa 3 thứ cho cả 16 buổi:
- `edu/requirements-colab.txt`: danh sách phiên bản thư viện tương thích.
- `edu/p01kit/`: bộ hàm dùng sẵn (kiểm tra máy, chạy lô, chấm điểm, ghi bảng điểm).
- `edu/data/p01_test.jsonl`: 100 câu test cố định.

Tạo **3 ô Colab riêng** (`0a-1`, `0a-2`, `0a-3`), dán từng dòng và chạy lần lượt:

**Ô 0a-1 (tải repo public):**
```bash
!git clone --depth 1 https://github.com/phamxuanphuc/uit-social-welfare-rag.git /content/uit-social-welfare-rag
```

**Ô 0a-2 (cài thư viện đã ghim, mất ~2–3 phút):**
```bash
!pip install -q -r /content/uit-social-welfare-rag/edu/requirements-colab.txt
```

**Ô 0a-3 (khởi động lại Python để nhận bản thư viện mới):**
```python
import os; os.kill(os.getpid(), 9)  # khởi động lại runtime để nạp thư viện mới (Colab sẽ tự kết nối lại)
```

- Sau Ô 0a-3, thông báo "Your session crashed" là bình thường vì ta chủ động khởi động lại Python. Quên chạy ô này sẽ khiến Python tiếp tục dùng bản `transformers` cũ của Colab.

### Ô 0b: Khởi động phiên (chạy đầu mỗi buổi)

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

- `sys.path.insert` chỉ chỗ tìm `p01kit`; `P01_RUNS` lưu kết quả lên Google Drive để không mất khi Colab ngắt phiên.
- **Kết quả nên thấy:** `p01kit 2.0.0 | model: Qwen/Qwen3-1.7B`, dòng `GPU: {'name': 'Tesla T4', ...}` và báo khớp bản ghim.

### Ô 1 (4 dòng): Kiểm tra GPU

```python
import torch
print(torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Hỗ trợ BF16?", torch.cuda.is_bf16_supported())
```

- **Kết quả nên thấy:** `Tesla T4`, `(7, 5)` (BF16 cần đời `(8, 0)` trở lên), `Hỗ trợ BF16? False`.

---

## Phần B: Mô hình nhìn chữ thế nào

### Ô 2 (6 dòng): Tokenizer tiếng Việt vs tiếng Anh

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(p01kit.MODEL_ID)
for s in ["Tôi chạy xe ôm ở Phường Tân Tạo bị té xe gãy tay",
          "I drive a motorbike taxi in Tan Tao ward and broke my arm"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(len(s.split()), "từ ->", len(ids), "token:", [tok.decode([i]) for i in ids])
```

- Dấu cách ở đầu mảnh (`' bị'`) báo hiệu mở đầu từ mới.
- **Kết quả nên thấy (đã đo):** Câu Việt 13 từ → 19 token (`' ô'`, `'m'`, `' Ph'`, `'ường'`, `' g'`, `'ã'`, `'y'`...); câu Anh 13 từ → 14 token. Prompt RAG của P01 trung bình 836 token (tối đa 885), chậm và tốn VRAM hơn tiếng Anh ~35%.

### Ô 3 (5 dòng): Các token đặc biệt

```python
for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"]:
    print(f"{t:>14} -> id {tok.convert_tokens_to_ids(t)}")
print("eos =", tok.eos_token, "| pad =", tok.pad_token, "| len(tok) =", len(tok))
```

- `<|im_start|>` mở lượt nói; `<|im_end|>` (`eos`) kết thúc lượt trả lời; `<|endoftext|>` (`pad`) dùng để đệm. Thiếu `<|im_end|>` trong dữ liệu train sẽ khiến mô hình nói lan man không dừng.
- **Kết quả nên thấy:** `<|im_start|>`=151644, `<|im_end|>`=151645, `<|endoftext|>`=151643, `<think>`=151667, `</think>`=151668, `<tool_call>`=151657; `len(tok) = 151669`.

---

## Phần C: Máy đoán chữ

### Ô 4 (8 dòng): Nạp mô hình nén 4-bit NF4

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(p01kit.MODEL_ID, quantization_config=bnb,
                                             device_map="auto", dtype=torch.float16)
model.eval()
print(f"Footprint: {model.get_memory_footprint()/1e9:.2f} GB | VRAM đang cấp: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

- Lần đầu chạy sẽ tải ~3,4 GB từ Hugging Face (mất ~1–2 phút) vì trên Hub lưu bản gốc FP16, tải xong `bitsandbytes` mới nén xuống 4-bit trên máy.
- `compute_dtype=float16` giải nén tạm sang FP16 lúc tính; `dtype=float16` giữ embedding/norm ở FP16; `model.eval()` bật chế độ suy luận.
- **Kết quả nên thấy:** Footprint khoảng 1,3–1,5 GB. (Nếu ra ~3,4 GB nghĩa là `quantization_config` chưa có hiệu lực.)

### Ô 5 (10 dòng): Sản phẩm 1 — Máy X-quang token kế tiếp

```python
@torch.inference_mode()
def xray(text: str, k: int = 5, temperature: float = 1.0):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    logits = model(**enc).logits[0, -1].float()          # điểm số tại vị trí CUỐI = dự đoán token kế tiếp
    probs = torch.softmax(logits / temperature, dim=-1)  # đổi điểm số thành xác suất
    vals, idxs = probs.topk(k)                           # lấy k token có xác suất cao nhất
    for p, i in zip(vals, idxs):
        print(f"{tok.decode([i.item()])!r:>16}  {p.item():6.1%}")

xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại")
```

- Cú pháp mới: `@torch.inference_mode()` tắt lưu gradient để nhẹ VRAM; `return_tensors="pt"` + `.to(model.device)` đổi sang tensor PyTorch và đẩy lên GPU; `**enc` trải các khóa trong dict thành tham số hàm; `logits[0, -1]` lấy câu đầu tiên, vị trí token cuối cùng.
- **Kết quả nên thấy:** 5 mảnh chữ (như `' UBND'`, `' Ủy'`, `' cơ'`, `' phường'`...) kèm phần trăm giảm dần.

### Ô 6 (3 dòng): Thử núm vặn temperature

```python
for T in (0.3, 1.0, 1.5):
    print(f"--- temperature={T} ---")
    xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại", temperature=T)
```

- **Kết quả nên thấy:** Thứ tự 5 token giữ nguyên nhưng `T=0.3` dồn gần hết xác suất vào token đầu, còn `T=1.5` dàn trải và hạ tổng top-5 (dễ bịa hơn). Khi chấm P01 ta dùng greedy để kết quả luôn cố định.

### Ô 7 (7 dòng): Nhẩm bộ nhớ và đối chiếu Ô 4

```python
L, H, I, KV, V = 28, 2048, 6144, 1024, 151_936   # số lớp, chiều ẩn, chiều MLP, chiều K/V (8 đầu × 128), số token
attn, mlp = H*H + 2*H*KV + H*H, 3*H*I             # attention (q, k+v, o) và MLP (gate, up, down) mỗi lớp
non_emb, emb = L * (attn + mlp), V * H            # 28 lớp + bảng embedding
print(f"Tham số: {(non_emb + emb)/1e9:.3f} tỷ | FP16: {(non_emb + emb)*2/1e9:.2f} GB")
print(f"NF4 ước tính: {(non_emb*0.5 + non_emb*0.127/8 + emb*2)/1e9:.2f} GB")
print(f"Đo thật (Ô 4): {model.get_memory_footprint()/1e9:.2f} GB")
```

- **Kết quả nên thấy:** `Tham số: 1.720 tỷ | FP16: 3.44 GB`, `NF4 ước tính: 1.35 GB`, số đo thật khoảng 1,3–1,5 GB (lệch nhẹ do các lớp chuẩn hóa nhỏ và bộ đệm).

---

## Phần D: Nói chuyện với mô hình

### Ô 8 (6 dòng): Chat template khi bật/tắt thinking

```python
msgs = [{"role": "system", "content": "Bạn là trợ lý an sinh."},
        {"role": "user", "content": "Xe ôm bị tai nạn cần giấy gì?"}]
for think in (True, False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    print(f"--- enable_thinking={think} ---\n{text!r}\n")
```

- **Kết quả nên thấy:** `think=True` kết thúc bằng `'<|im_start|>assistant\n'`; `think=False` chèn sẵn `'<think>\n\n</think>\n\n'` để mô hình bỏ qua nháp và trả lời luôn.
- Prompt lúc train và lúc dùng phải khớp từng ký tự, nên cả khóa gọi duy nhất `p01kit.prompts.render_prompt()`.

### Ô 9 (10 + 3 dòng): Đo chi phí thinking

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

- `time.perf_counter()` đo thời gian theo giây; `out[0, enc["input_ids"].shape[1]:]` cắt bỏ phần prompt đầu vào để chỉ lấy chữ mô hình mới sinh. Tham số sampling theo chuẩn Qwen3 (không dùng greedy khi bật thinking vì dễ lặp vô hạn).
- **Kết quả nên thấy:** Bật thinking sinh nhiều token hơn hẳn ở cùng `tok/s`, nên tổng thời gian dài hơn tương ứng.

---

## Phần E: Chạy nhanh trên nhiều câu

### Ô 10 (5 dòng): Đệm trái khi sinh theo lô

```python
tok.padding_side = "left"
batch = tok(["Xin chào", "Tôi chạy xe ôm ở Tân Tạo bị té xe"], return_tensors="pt",
            padding=True, add_special_tokens=False)
print(batch["input_ids"])
print(batch["attention_mask"])
```

- Mô hình nối chữ mới vào **mép phải** của cả hai hàng cùng lúc. Câu ngắn (`"Xin chào"`, 3 token) phải đệm 9 token rỗng (`151643`, mask `0`) ở **bên trái** để mép phải thẳng hàng với câu dài (12 token). Đệm phải sẽ kẹt token rỗng giữa câu hỏi và phần trả lời.
- **Kết quả nên thấy:** Hàng 1 bắt đầu bằng 9 số `151643` và 9 số `0` trong `attention_mask`. (Lưu ý: lúc **train** thì ngược lại, dùng đệm phải.)

### Ô 11 (10 + 4 dòng): Sản phẩm 2 — Bộ sinh theo lô

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

- `do_sample=False` bật greedy; đặt `temperature/top_p/top_k = None` để tắt mặc định sampling của Qwen3; `ids[:, enc["input_ids"].shape[1]:]` cắt bỏ prompt của cả lô cùng lúc.
- **Kết quả nên thấy:** `batch_size=8` nhanh hơn `batch_size=1` nhiều lần. Đây cũng chính là `p01kit.infer.generate`.

---

## Phần F: Đo baseline trên tập test P01

### Ô 12 (7 dòng): Xem mẫu tập test P01

```python
from p01kit.data import load_split, dataset_stats
test = load_split("test")
print(dataset_stats(test))
r = test[0]
print("Địa bàn:", r["region"], "\nCâu hỏi:", r["question"])
print("Ngữ cảnh:\n", r["context"][:600], "...")
print("Đáp án vàng:", r["gold"])
```

- **Kết quả nên thấy:** `n=100`, 44% phải từ chối, 10% có văn bản hết hạn, 50% giọng đời thường; `slices`: `seen_doc=24` (có trong train), `unseen_doc=72` (văn bản mới), `ood=4` (ngoài phạm vi). Văn bản trong tập dữ liệu là **giả lập** để học tập.

### Ô 13 (7 dòng): Sản phẩm 3 — Đo baseline zero-shot

```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
res0 = evaluate(model, tok, test, batch_size=8, max_new_tokens=p01kit.MAX_NEW_TOKENS)
print(format_summary(res0["summary"]))
print("Thời gian:", res0["summary"]["seconds"], "giây")
scoreboard.record("M00", "base-zero-shot", res0["summary"], latency_s=res0["summary"]["sec_per_row"], outputs=res0["outputs"])
```

- **Thời gian chờ:** Mất ~2–5 phút trên T4, không in log giữa chừng (đừng bấm dừng). `scoreboard.record` tự tạo `p01_runs/outputs/M00__base-zero-shot.json` và ghi dòng mới vào `leaderboard.csv`.
- **Hôm nay chỉ cần nhìn 2 chỉ số** (6 chỉ số còn lại học ở Buổi 01):
  - `json_valid`: tỉ lệ trả đúng định dạng JSON.
  - `answer_acc`: tỉ lệ quyết định đúng (từ chối đúng lúc hoặc trích đúng văn bản).
- **Cách đọc `95%CI`** (ví dụ minh họa):

```text
json_valid           0.780  95%CI [0.700, 0.860]  (n=100)
answer_acc           0.410  95%CI [0.310, 0.510]  (n=100)
```

  Với 100 câu, điểm thật nằm trong khoảng 31%–51%; hơn kém 2–3 điểm thường chưa đủ kết luận khác biệt thật.
- **Kết quả nên thấy:** Bảng 8 chỉ số kèm `95%CI`, trong đó `json_valid > 0`. Nếu `json_valid = 0`, kiểm tra lại xem thinking có đang bật hay `max_new_tokens` quá nhỏ không.

### Ô 14 (5 dòng): Đọc lỗi thực tế và lưu output

```python
import json
from p01kit.infer import show_errors
show_errors(res0, test, k=5)
with open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json", "w") as f:
    json.dump(res0["outputs"], f, ensure_ascii=False)
```

- Output của `show_errors` có dạng:

```text
--- CT-BHYT::muc_ho_tro::colloquial0 | colloquial | địa bàn=Cần Thơ - Phường Cái Khế | gold_refusal=True gold_doc=None gold_expired=False
Q: Tui là người bán hàng rong, không có tiền mua bảo hiểm cho con...
OUT: {"doc_code": "QĐ-19/2026/UBND-ĐN", ...}
```

- Xếp 5 câu sai vào 5 nhóm dưới đây (cần kiểm tra nhóm (e) thì in thêm `test[i]["context"]`):

| Nhóm lỗi | Dấu hiệu |
|---|---|
| (a) Sai định dạng JSON | Không phải JSON hợp lệ, thiếu khóa hoặc thêm chữ ngoài JSON |
| (b) Không chịu từ chối | `gold_refusal=True` nhưng vẫn trả lời và trích đại văn bản |
| (c) Trích lệch địa bàn | Người hỏi ở Cần Thơ/Đà Nẵng nhưng trích văn bản TP.HCM |
| (d) Bỏ sót cảnh báo hết hạn | `gold_expired=True` nhưng `expired_warning` là `false` |
| (e) Bịa giấy tờ | Liệt kê giấy tờ không có trong ngữ cảnh |

---

## 3. So sánh hiệu quả: bảng điểm sau Buổi 00

```python
scoreboard.show()
```

| module | variant | json_valid | answer_acc | refusal_acc | citation_precision | expired_recall | strict_acc | fairness_gap |
|---|---|---|---|---|---|---|---|---|
| M00 | base-zero-shot | *(số đo của bạn)* | | | | | | |

4 ngưỡng chốt trong [PROJECT-REQUIREMENTS.md](../../PROJECT-REQUIREMENTS.md): `citation_precision ≥ 0.90`, `refusal_acc ≥ 0.90`, `expired_recall = 1.0`, `fairness_gap ≤ 0.10`. Gọi `p01kit.metrics.gate(res0["summary"])` để xem đạt/chưa đạt (sau Buổi 00 hầu hết chưa đạt là bình thường).

---

## Ở công ty thật

- **Baseline là hệ thống đang chạy:** so với quy trình thủ công, FAQ, tìm từ khóa hoặc API mô hình lớn; fine-tune chỉ làm khi thắng về chất lượng, chi phí hoặc độ trễ.
- **Chốt hạ tầng trước khi chọn mô hình:** P01 chạy trên 2 vCPU / 4 GiB (ADR 0001) nên chọn 1,7B từ đầu.
- **Giấy phép:** Apache-2.0 của Qwen3 cần pháp chế duyệt trước khi đưa vào sản phẩm.
- **Traffic thật (đã che PII):** ngắn hơn, sai chính tả, không dấu hơn nhiều so với bộ test tự dựng (Buổi 15).

---

## Gặp lỗi? Tra nhanh

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `no CUDA GPUs are available` | Runtime đang là CPU | *Runtime → Change runtime type → T4 GPU*, chạy lại từ Ô 0a |
| `ModuleNotFoundError: p01kit` | Chưa chạy Ô 0b sau khi restart | Chạy lại Ô 0b |
| `check_env()` báo sai phiên bản | Quên chạy Ô 0a-3 | Chạy Ô 0a-3 (`os.kill`), chờ nối lại rồi chạy Ô 0b |
| `CUDA out of memory` | Còn model cũ trong VRAM hoặc `batch_size` lớn | *Runtime → Restart session*, chạy lại từ Ô 0b hoặc giảm `batch_size` |
| Footprint ~3,4 GB ở Ô 4 | Chưa truyền `quantization_config=bnb` | Kiểm tra lại tham số ở Ô 4 |
| `json_valid = 0` ở Ô 13 | Bật thinking hoặc `max_new_tokens` quá nhỏ | Giữ mặc định của `evaluate`, truyền `p01kit.MAX_NEW_TOKENS` |
| Colab ngắt giữa buổi | Rảnh quá lâu hoặc hết phiên | Chạy lại Ô 0b rồi chạy tiếp (kết quả trên Drive vẫn còn) |

---

## 4. Bài tập

**Bắt buộc**
1. Đo `tok/s` của `generate_batch` với `batch_size` là 1, 4, 8, 16 trên 32 prompt và vẽ đồ thị. Mức nào tốc độ chững lại? Vì sao? *(Gợi ý: KV-cache và giới hạn tính toán của GPU.)*
2. Tính tay số tham số và VRAM FP16 tăng thêm nếu Qwen3-1.7B không tie embedding với `lm_head` (`151.936 × 2048`).
3. Chạy `evaluate(model, tok, test[:20], enable_thinking=True, max_new_tokens=1536, batch_size=4)`, so `answer_acc` và thời gian với non-thinking, ghi bảng điểm với `variant="base-thinking-20"`.

**Nâng cao**
4. Nạp bản 8-bit (`load_in_8bit=True`) và FP16, so footprint với quy tắc nhẩm mục 1.3.
5. Viết `xray_chat(question)` soi top-5 token đầu câu trả lời khi `enable_thinking=False`. Token đầu có phải `{` không?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao phải đo baseline trước khi fine-tune?** (cơ bản)
   A: Để có mốc so sánh, biết khi nào chỉ cần prompt/RAG là đủ và đặt tiêu chí dừng. Phải đo trên cùng tập test, prompt và bộ chấm.
2. **Q: Qwen3-1.7B có bao nhiêu tham số và vì sao có nơi ghi 1.4B?** (cơ bản)
   A: Tổng ~1,72 tỷ; 1,41 tỷ là non-embedding (28 lớp), 0,31 tỷ là bảng embedding dùng chung (tie) với `lm_head`.
3. **Q: Vì sao sinh theo lô đệm trái còn train đệm phải?** (cơ bản)
   A: Lúc sinh, token mới nối vào mép phải nên các câu phải thẳng mép phải (đệm trái). Lúc train, vị trí đệm đã che bằng `attention_mask` và nhãn `-100`, nên đệm phải gọn hơn.
4. **Q: `enable_thinking=False` hoạt động thế nào ở mức token?** (cơ bản)
   A: Chat template chèn sẵn `<think>\n\n</think>\n\n` sau `<|im_start|>assistant\n` để mô hình coi như đã nghĩ xong và trả lời ngay.
5. **Q: Vì sao train QLoRA trên T4 dùng FP16 thay vì BF16?** (trung bình)
   A: T4 (`sm_75`) không có phần cứng cho BF16 (cần Ampere `sm_80+`). FP16 cần loss scaling để tránh tràn nhưng tận dụng được Tensor Core của T4.
6. **Q: GQA là gì và lợi gì khi chạy mô hình?** (nâng cao)
   A: Nhiều đầu Query dùng chung 1 cặp Key/Value (Qwen3-1.7B: 16 Q / 8 KV), giảm một nửa KV-cache nên chạy được ngữ cảnh dài và batch lớn hơn.
7. **Q: Vì sao embedding không bị nén 4-bit trong bitsandbytes?** (nâng cao)
   A: `bitsandbytes` chỉ nén `nn.Linear`. Embedding là bảng tra cứu, nén 4-bit gây sai số nặng cho token hiếm mà tiết kiệm không nhiều.

---

## 6. Checklist cuối buổi

- [ ] Vẽ lại được sơ đồ 4 bước sinh chữ (mục 1.1) không cần nhìn bài.
- [ ] Phân biệt được `Qwen3-1.7B` và `Qwen3-1.7B-Base`.
- [ ] Nhẩm được 3,44 GB (FP16) và giải thích vì sao NF4 ~1,35 GB thay vì 0,86 GB.
- [ ] Viết được `generate_batch` đệm trái và đo tốc độ tăng so với `batch_size=1`.
- [ ] Bảng điểm có `M00 / base-zero-shot` và đã phân loại xong 5 câu sai đầu tiên.

## Tài liệu tham khảo

- Qwen3 Technical Report và model card `Qwen/Qwen3-1.7B` (mục *Best Practices*).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double quantization).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.
