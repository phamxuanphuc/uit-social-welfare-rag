# Buổi 00: Làm quen `Qwen3-1.7B` trên Colab T4 và đo baseline

| Mục | Nội dung |
|---|---|
| Thời lượng | 3 giờ (45 phút lý thuyết, 2 giờ 15 phút thực hành). Có thể tách thành 2 lần ngồi: lần 1 làm Phần A đến C, lần 2 làm Phần D đến F |
| Cần biết trước | Python cơ bản (biến, vòng lặp, hàm) và biết mở Google Colab. Không cần biết Transformer hay PyTorch |
| Chuẩn bị | Tài khoản Google (dùng Colab T4 miễn phí) và repo đồ án clone được (xem [README](README.md)) |
| Sản phẩm cuối buổi | (1) Hàm `xray` soi mô hình đoán chữ tiếp theo; (2) Hàm `generate_batch` hỏi nhiều câu cùng lúc; (3) Dòng điểm baseline `M00/base-zero-shot` trên 100 câu test |
| Cách viết code | Mỗi ô tối đa ~10 dòng, đi theo đúng một nhịp: mục tiêu, code, giải thích, kết quả nên thấy, lỗi hay gặp |

### Mục tiêu buổi học

Sau buổi này, bạn tự làm được 5 việc sau:
1. Giải thích bằng lời của mình cách mô hình ngôn ngữ sinh chữ (tách token, tính xác suất, chọn 1 token rồi lặp lại).
2. Tự nhẩm xem Qwen3-1.7B nặng bao nhiêu GB và kiểm tra lại bằng số đo thật trên GPU.
3. Bật và tắt chế độ thinking để đo xem nó tốn thêm bao nhiêu token và bao nhiêu giây.
4. Chạy mô hình trên nhiều câu cùng lúc (sinh theo lô) và hiểu vì sao phải đệm bên trái.
5. Có điểm baseline của mô hình gốc trên 100 câu test để làm mốc so sánh cho 15 buổi tiếp theo.

**Tiêu chí hoàn thành** (tự kiểm tra trước khi sang Buổi 01):
- `check_env()` báo khớp bản ghim và nhận được GPU Tesla T4.
- Bảng điểm có dòng `M00/base-zero-shot`.
- Trên Google Drive có đủ 2 file: `p01_runs/outputs/M00__base-zero-shot.json` (do `scoreboard.record` ở Ô 13 tự tạo) và `p01_runs/m00_base_outputs.json` (do bạn lưu ở Ô 14).
- Bạn giải thích được cho một người bạn vì sao mô hình 1,7 tỷ tham số chỉ tốn khoảng 1,4 GB VRAM khi nén 4-bit.

### Bản đồ buổi học

| Phần | Câu hỏi chính | Các ô |
|---|---|---|
| A. Chuẩn bị | Đã bật GPU T4 và cài đúng bộ thư viện chưa? | 0a, 0b, 1 |
| B. Mô hình nhìn chữ thế nào | Một câu tiếng Việt bị cắt thành những mảnh gì? | 2, 3 |
| C. Máy đoán chữ | Mô hình nặng bao nhiêu và nó chọn chữ kế tiếp ra sao? | 4, 5, 6, 7 |
| D. Nói chuyện với mô hình | Chat template và chế độ thinking hoạt động kiểu gì? | 8, 9 |
| E. Chạy nhanh | Làm sao hỏi 100 câu mà không phải chờ cả giờ? | 10, 11 |
| F. Baseline | Mô hình gốc đạt bao nhiêu điểm và hay sai kiểu nào? | 12, 13, 14 |

---

## 0. Vì sao phải đo baseline trước?

Đi làm thực tế, khi bạn đề xuất fine-tune một mô hình, câu đầu tiên sếp hỏi thường là: "Mô hình gốc hiện tại được bao nhiêu điểm?". Không có con số ban đầu, bạn không chứng minh được fine-tune có thật sự tốt lên hay không, cũng chẳng biết dừng ở đâu cho hợp lý.

Nguyên tắc giữ xuyên suốt cả 16 buổi: mọi phiên bản đều chấm trên đúng 100 câu test trong `edu/data/p01_test.jsonl`, dùng cùng một khuôn prompt và cùng một bộ chấm. Làm vậy thì điểm của Buổi 00 mới so thẳng được với Buổi 02, 05 hay 14, giống như theo dõi cân nặng bằng cùng một cái cân.

---

## Từ điển nhanh

Đọc lướt qua một lượt để quen mặt chữ. Chỗ nào chưa thấm cứ bỏ qua, gặp trong bài thì quay lại tra.

| Thuật ngữ | Giải thích ngắn gọn |
|---|---|
| Token | Mảnh chữ nhỏ nhất mô hình đọc được. Có từ là 1 token (`' bị'`), có từ bị tách thành nhiều mảnh (`' Ph'` + `'ường'`) |
| Tokenizer | Bộ cắt câu thành token rồi đổi từng mảnh thành một con số (id) |
| Tham số (parameter) | Một con số nằm bên trong mô hình, đã được chỉnh sẵn lúc huấn luyện. Qwen3-1.7B có khoảng 1,72 tỷ con số như vậy |
| Embedding | Bảng tra cứu đổi mỗi token id thành một dãy số (vector 2048 số) để mô hình tính toán được |
| Logits | Điểm số thô mô hình chấm cho toàn bộ 151.936 token trong từ vựng ở vị trí kế tiếp |
| Softmax | Phép biến điểm số thô thành xác suất sao cho tổng tất cả bằng 100% |
| Temperature | Núm chỉnh độ "an toàn" hay "liều": số thấp thì bám sát chữ chắc nhất, số cao thì chọn đa dạng hơn nhưng dễ bịa |
| Greedy / Sampling | Greedy là luôn lấy chữ có xác suất cao nhất (chạy lại ra y hệt). Sampling là bốc thăm theo xác suất (`top_k`, `top_p` khoanh vùng các chữ được phép bốc) |
| FP16 / BF16 | Hai cách lưu số thực bằng 16 bit (2 byte). Card T4 trên Colab miễn phí chỉ chạy nhanh với FP16 |
| Lượng tử hóa, NF4 | Nén mỗi tham số từ 16 bit xuống 4 bit để mô hình nhẹ đi gần 4 lần, chấp nhận sai số nhỏ |
| VRAM | Bộ nhớ riêng của GPU. Tesla T4 có khoảng 15 GB |
| KV-cache | Bộ nhớ tạm lưu kết quả trung gian của các token đã đọc, giúp mô hình khỏi tính lại từ đầu ở mỗi chữ mới sinh |
| Chat template | Khuôn mẫu đóng gói câu hỏi thành đúng định dạng hội thoại mà mô hình từng học |
| Thinking mode | Chế độ Qwen3 viết phần nháp trong `<think>...</think>` trước khi trả lời chính thức |
| Batch (sinh theo lô) | Gom nhiều câu đưa vào GPU xử lý song song thay vì hỏi từng câu một |
| Padding (đệm) | Chèn thêm token rỗng để các câu dài ngắn khác nhau trong cùng một lô có độ dài bằng nhau |
| RAG | Cách tìm sẵn các đoạn văn bản liên quan rồi dán vào prompt cho mô hình đọc trước khi trả lời |
| Baseline / zero-shot | Điểm của mô hình gốc khi chỉ nhận câu hỏi, không đưa ví dụ mẫu và chưa fine-tune |

---

## 1. Lý thuyết: 5 điều cần hiểu trước khi chạy code

**Giải thích kiểu Feynman** (đọc trong 1 phút):
Mô hình ngôn ngữ thực chất là một cái máy đoán chữ tiếp theo, giống gợi ý từ trên bàn phím điện thoại nhưng làm ở quy mô lớn hơn nhiều. Cụm từ "1,7 tỷ tham số" chỉ 1,7 tỷ núm vặn đã được chỉnh sẵn trong quá trình huấn luyện. Nén NF4 giống như nén ảnh: giảm mỗi núm từ 16 bit xuống 4 bit nên nhẹ hơn gần 4 lần, đổi lại hình hơi mờ đi một chút. Còn baseline giống như bước lên cân trước ngày đầu tập gym: không ghi lại cân nặng ban đầu thì sau một tháng bạn khó biết mình có tiến bộ thật không.

### 1.1. Mô hình sinh chữ theo vòng lặp nào?

Toàn bộ những gì bạn học về sau (fine-tune, RAG, agent) đều xoay quanh vòng lặp 4 bước này:

```text
"Xe ôm bị tai nạn cần nộp hồ sơ tại"
        │  1. Tokenizer cắt câu thành từng mảnh và đổi sang id
        ▼
[' Xe', ' ô', 'm', ' bị', ' tai', ' nạn', ...]
        │  2. Mô hình (28 lớp xếp chồng) đọc cả chuỗi
        ▼
151.936 điểm số (logits), mỗi token trong từ vựng nhận 1 điểm
        │  3. Softmax đổi điểm số thành xác suất (tổng = 100%)
        ▼
' UBND' 31% · ' phường' 12% · ' cơ' 8% · ...      (số minh họa)
        │  4. Chọn 1 token, nối vào cuối câu
        ▼
Quay lại bước 2, lặp đến khi gặp <|im_end|> hoặc hết max_new_tokens
```

Mô hình chỉ đoán đúng 1 token ở mỗi lượt. Một câu trả lời dài 200 token tương đương 200 vòng lặp liên tiếp. Khi fine-tune ở các buổi sau, việc ta làm thực chất là chỉnh các núm vặn để xác suất ở bước 3 nghiêng về đúng câu trả lời ta muốn. Ô 5 sẽ cho bạn nhìn thẳng vào bước 3 này.

### 1.2. Vì sao tiếng Việt "đắt" token hơn tiếng Anh?

Tokenizer học cách cắt chữ từ kho dữ liệu huấn luyện, mà phần lớn trong đó là tiếng Anh và tiếng Trung. Những âm tiết tiếng Việt có dấu xuất hiện ít hơn nên hay bị băm nhỏ, chẳng hạn `ôm` bị cắt thành `' ô'` và `'m'`, hay `gãy` tách làm 3 mảnh `' g'`, `'ã'`, `'y'`.

Cắt ra càng nhiều token thì mô hình càng tốn VRAM và trả lời càng chậm. Ở Ô 2 bạn sẽ tự đo con số chênh lệch giữa hai câu tiếng Việt và tiếng Anh có cùng số từ.

### 1.3. Nhẩm nhanh dung lượng mô hình

| Cách lưu | Mỗi tham số tốn | Qwen3-1.7B (khoảng 1,72 tỷ tham số) |
|---|---|---|
| FP16 | 2 byte | 1,72 × 2 ≈ 3,44 GB |
| NF4 (4-bit) | 0,5 byte | Khoảng 1,35 GB (không phải 0,86 GB) |

Mẹo nhẩm nhanh: lấy số tỷ tham số nhân 2 sẽ ra số GB ở FP16. Ví dụ mô hình 7B tốn khoảng 14 GB ở FP16, gần như kịch trần 15 GB của card T4 nếu không nén.

Nhiều người thắc mắc vì sao nén từ 16 bit xuống 4 bit (giảm 4 lần) mà 3,44 GB lại ra ~1,35 GB chứ không phải 0,86 GB. Lý do là bảng embedding (chiếm 0,31 tỷ tham số, tương đương 0,62 GB) không bị thư viện `bitsandbytes` nén xuống 4-bit mà vẫn giữ nguyên 2 byte mỗi tham số. Khi train ở các buổi sau còn cần thêm bộ nhớ cho phần đang học và kết quả trung gian, lúc đó ta sẽ tính kỹ hơn.

### 1.4. `Qwen/Qwen3-1.7B` không phải "mô hình base"

Trên Hugging Face có hai phiên bản rất dễ nhầm:
- `Qwen/Qwen3-1.7B-Base`: chỉ mới học đoán chữ tiếp theo, chưa biết trả lời theo lượt hội thoại.
- `Qwen/Qwen3-1.7B`: đã được huấn luyện thêm (post-train) để biết chat, gọi công cụ (tool) và có chế độ thinking.

Trong đồ án này ta chọn bản đã post-train vì tập dữ liệu của mình nhỏ (chỉ vài trăm câu). Tận dụng sẵn khả năng làm theo chỉ dẫn của nó rẻ và dễ hơn nhiều so với dạy lại từ bản Base. Khi đi phỏng vấn, gọi nhầm bản post-train là "base model" rất hay bị hỏi vặn lại.

### 1.5. Chế độ thinking và giới hạn của card T4

Mặc định Qwen3 có thể viết nháp bên trong thẻ `<think>...</think>` rồi mới chốt câu trả lời. Cách này giúp nó giải bài khó tốt hơn nhưng sinh rất dài và chậm. Từ Buổi 00 đến Buổi 09, ta tắt thinking vì bài toán cần trả ra JSON ngắn và nhanh. Sang Buổi 10 ta mới quay lại khai thác thinking.

Về phần cứng: Tesla T4 trên Colab miễn phí là GPU đời cũ (kiến trúc Turing), không hỗ trợ phần cứng cho BF16. Vì vậy khi chạy trên T4, bạn luôn dùng FP16 (`fp16=True`, `bnb_4bit_compute_dtype=torch.float16`). Nếu viết `bf16=True` thì PyTorch sẽ báo lỗi ngay.

<details>
<summary>Đọc thêm phần nâng cao (có thể bỏ qua ở lần học đầu): đếm tham số bằng tay, GQA, NF4 chi tiết, tràn số FP16</summary>

Thông số thật lấy từ `config.json` của Qwen3-1.7B:

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `num_hidden_layers` | 28 | 28 khối Transformer xếp chồng |
| `hidden_size` | 2048 | Mỗi token được biểu diễn bằng vector 2048 số |
| `intermediate_size` | 6144 | Lớp MLP mở rộng lên 6144 chiều rồi thu về 2048 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 | GQA: 2 đầu Query dùng chung 1 cặp Key/Value, giúp KV-cache nhỏ đi một nửa |
| `head_dim` | 128 | 16 đầu × 128 = 2048 |
| `vocab_size` | 151.936 | Số hàng của bảng embedding (đã đệm cho đẹp); tokenizer thật có 151.669 token |
| `tie_word_embeddings` | `true` | Bảng embedding đầu vào và lớp `lm_head` đầu ra dùng chung trọng số, bớt được 311 triệu tham số |

Đếm nhanh số tham số (bỏ qua các lớp chuẩn hóa vì quá nhỏ):
1. Chiều Key/Value = 8 đầu KV × 128 = 1024 (nhỏ hơn Query 2048 nhờ GQA).
2. Attention mỗi lớp: `q (2048×2048)` + `k (2048×1024)` + `v (2048×1024)` + `o (2048×2048)` = 12,58 triệu.
3. MLP mỗi lớp có 3 ma trận (gate, up, down), mỗi ma trận `2048×6144` = 37,75 triệu.
4. Một lớp xấp xỉ 50,3 triệu; nhân 28 lớp ra 1,409 tỷ (phần non-embedding).
5. Bảng embedding `151.936 × 2048` = 0,311 tỷ. Tổng cộng xấp xỉ 1,720 tỷ tham số.

Tính chi tiết NF4: 28 lớp Linear chiếm 1,409 tỷ × 0,5 byte ≈ 0,70 GB; các hằng số giải nén sau double quantization thêm khoảng 0,127 bit mỗi tham số (theo bài báo QLoRA) ≈ 0,02 GB; bảng embedding giữ FP16 tốn 0,62 GB. Cộng lại ra xấp xỉ 1,35 GB (khoảng 1,26 GiB). Thư viện `bitsandbytes` chỉ thay thế các lớp `nn.Linear`, không đụng tới `nn.Embedding`.

Về lỗi tràn số ở FP16: kiểu FP16 chỉ chứa được số lớn nhất khoảng 65.504, nên lúc huấn luyện `Trainer` phải tự động nhân/chia loss (dynamic loss scaling) để tránh ra `NaN`. Kiểu BF16 có dải số rộng như FP32 nhưng cần GPU đời Ampere (compute capability 8.0) trở lên. Ở Buổi 06 bạn sẽ tự tay gây ra lỗi `NaN` và sửa nó.

</details>

---

## 2. Thực hành từng ô trên Colab

## Phần A: Chuẩn bị

> **Làm việc này trước khi chạy Ô 0a:** Mở Google Colab, vào menu *Runtime → Change runtime type → chọn T4 GPU* rồi bấm Save. Nếu để mặc định là CPU, lát nữa đổi sang T4 thì Colab sẽ cấp máy mới tinh và xóa sạch toàn bộ thư viện bạn vừa cài ở Ô 0a.

### Ô 0a: Tải repo và cài thư viện (chạy 1 lần cho mỗi phiên máy mới)

**Mục tiêu:** Đưa code + dữ liệu đồ án vào Colab và cài đúng phiên bản thư viện đã ghim.

Ô 0a gồm **3 ô Colab riêng biệt** (`0a-1`, `0a-2`, `0a-3`). Bạn tạo 3 ô, dán từng khung code rồi chạy lần lượt, chưa cần hiểu kỹ từng dòng trong ô đầu. Nếu repo đang để private, trước khi chạy hãy bấm biểu tượng hình chìa khóa (*Secrets*) ở thanh bên trái Colab, tạo mục tên `GITHUB_TOKEN` chứa token GitHub của bạn và bật *Notebook access*.

**Ô 0a-1 (tải code từ GitHub):**
```python
import os, subprocess, base64
REPO, URL = "/content/uit-social-welfare-rag", "https://github.com/phamxuanphuc/uit-social-welfare-rag.git"
if not os.path.exists(REPO):
    env = dict(os.environ)
    try:  # repo private: đọc GITHUB_TOKEN từ Colab Secrets; repo public thì bỏ qua
        from google.colab import userdata
        basic = base64.b64encode(f"x-access-token:{userdata.get('GITHUB_TOKEN')}".encode()).decode()
        env |= {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}"}  # token chỉ nằm trong biến môi trường
    except Exception:
        pass
    r = subprocess.run(["git", "clone", "--depth", "1", URL, REPO], env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"git clone lỗi:\n{r.stderr}"
```

**Ô 0a-2 (cài thư viện đã ghim phiên bản, mất khoảng 2 đến 3 phút):**
```bash
!pip install -q -r /content/uit-social-welfare-rag/edu/requirements-colab.txt
```

**Ô 0a-3 (khởi động lại Python để nhận bản thư viện mới):**
```python
import os; os.kill(os.getpid(), 9)  # khởi động lại runtime để nạp thư viện mới (Colab sẽ tự kết nối lại)
```

- **Giải thích:** Sau khi chạy Ô 0a-3, Colab hiện thông báo "Your session crashed" rồi tự nối lại, đó là bình thường vì ta chủ động tắt tiến trình Python cũ. Ô 0a-1 truyền token qua biến môi trường thay vì ghép vào URL để tránh làm lộ token ra màn hình lỗi hoặc lưu vào `.git/config`. Việc ghim phiên bản (`transformers==5.17.0`, `trl==1.15.0`...) giúp code chạy ổn định về sau vì các thư viện LLM đổi tên tham số rất nhanh.
- **Lỗi hay gặp:** Quên chạy Ô 0a-3 khiến Python tiếp tục dùng bản `transformers` cũ có sẵn của Colab, dẫn tới các lỗi khó hiểu ở ô sau.

### Ô 0b: Khởi động phiên (chạy đầu mỗi buổi học)

**Mục tiêu:** Gắn Google Drive để lưu kết quả lâu dài và kiểm tra môi trường.

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

- **Giải thích:** `sys.path.insert` báo cho Python chỗ tìm thư mục `p01kit` của đồ án. Thư mục `P01_RUNS` nằm trên Google Drive nên dù Colab có ngắt kết nối thì bảng điểm và kết quả của bạn vẫn còn.
- **Kết quả nên thấy:** In ra `p01kit 2.0.0 | model: Qwen/Qwen3-1.7B`, dòng `GPU: {'name': 'Tesla T4', ...}` và thông báo môi trường khớp bản ghim.
- **Mẹo tiết kiệm hạn mức:** Đọc thêm mục *Học trên Colab T4 miễn phí* trong [README](README.md) để biết cách ngắt GPU khi không dùng và cách chạy tiếp khi bị rớt phiên.
- **Lỗi hay gặp:** Nếu thấy cảnh báo `transformers: đang là ..., bản ghim là 5.17.0`, bạn quên chạy Ô 0a-3. Nếu thấy cảnh báo `torch` ngoài khoảng hỗ trợ của `unsloth`, báo cho nhóm cập nhật danh sách ghim, đừng tự nâng hay hạ từng gói lẻ.

### Ô 1 (4 dòng): Kiểm tra GPU

**Mục tiêu:** Xác nhận đang có Tesla T4 và thấy tận mắt nó không hỗ trợ BF16.

```python
import torch
print(torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Hỗ trợ BF16?", torch.cuda.is_bf16_supported())
```

- **Giải thích:** `compute capability` là số hiệu đời GPU của NVIDIA. T4 là đời `(7, 5)`, còn BF16 cần đời `(8, 0)` trở lên.
- **Kết quả nên thấy:** `Tesla T4`, `(7, 5)` và `Hỗ trợ BF16? False`.
- **Lỗi hay gặp:** Nếu báo `no CUDA GPUs`, bạn chưa bật T4 trước khi chạy. Hãy đổi sang T4 GPU trong *Runtime → Change runtime type*, sau đó chạy lại từ Ô 0a.

---

## Phần B: Mô hình nhìn chữ thế nào

### Ô 2 (6 dòng): Tokenizer cắt tiếng Việt ra sao

**Mục tiêu:** Xem tận mắt một câu tiếng Việt và câu tiếng Anh cùng nghĩa bị cắt thành bao nhiêu token.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(p01kit.MODEL_ID)
for s in ["Tôi chạy xe ôm ở Phường Tân Tạo bị té xe gãy tay",
          "I drive a motorbike taxi in Tan Tao ward and broke my arm"]:
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(len(s.split()), "từ ->", len(ids), "token:", [tok.decode([i]) for i in ids])
```

- **Giải thích:** `tok(s)` biến chuỗi chữ thành danh sách id số nguyên; `tok.decode([i])` dịch ngược từng id ra mảnh chữ để dễ nhìn. Dấu cách đứng đầu mảnh (như `' bị'`) cho biết mảnh đó mở đầu một từ mới.
- **Kết quả nên thấy (đã đo):** Câu tiếng Việt 13 từ bị cắt thành 19 token (`' ô'`, `'m'`, `' Ph'`, `'ường'`, `' g'`, `'ã'`, `'y'`...), trong khi câu tiếng Anh 13 từ chỉ mất 14 token.
- **Hệ quả ở đồ án P01:** Mỗi prompt RAG của mình dài trung bình 836 token (tối đa 885), tốn bộ nhớ và chậm hơn bản tiếng Anh tương đương khoảng 35%.

### Ô 3 (5 dòng): Các token đặc biệt

**Mục tiêu:** Nhận diện các ký hiệu đánh dấu lượt hội thoại và điểm dừng của Qwen3.

```python
for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>"]:
    print(f"{t:>14} -> id {tok.convert_tokens_to_ids(t)}")
print("eos =", tok.eos_token, "| pad =", tok.pad_token, "| len(tok) =", len(tok))
```

- **Giải thích:** `<|im_start|>` mở một lượt nói, `<|im_end|>` (chính là `eos`, viết tắt của *end of sequence*) báo mô hình dừng trả lời, còn `pad` (`<|endoftext|>`) là token rỗng dùng để đệm ở Ô 10.
- **Kết quả nên thấy:** `<|im_start|>`=151644, `<|im_end|>`=151645, `<|endoftext|>`=151643, `<think>`=151667, `</think>`=151668, `<tool_call>`=151657; `eos = <|im_end|>`, `pad = <|endoftext|>`, `len(tok) = 151669`.
- **Điểm cần nhớ:** Nếu dữ liệu huấn luyện thiếu `<|im_end|>` ở cuối đáp án, mô hình sau khi fine-tune sẽ không biết chỗ dừng và nói lan man tới hết `max_new_tokens`.

---

## Phần C: Máy đoán chữ

### Ô 4 (8 dòng): Nạp mô hình nén 4-bit NF4

**Mục tiêu:** Đưa Qwen3-1.7B lên GPU dưới dạng nén 4-bit và xem dung lượng VRAM thật sự.

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(p01kit.MODEL_ID, quantization_config=bnb,
                                             device_map="auto", dtype=torch.float16)
model.eval()
print(f"Footprint: {model.get_memory_footprint()/1e9:.2f} GB | VRAM đang cấp: {torch.cuda.memory_allocated()/1e9:.2f} GB")
```

- **Lưu ý lúc chạy lần đầu:** Bạn sẽ thấy thanh tiến trình tải khoảng 3,4 GB từ Hugging Face (mất 1 đến 2 phút). Điều này hoàn toàn bình thường: trên mạng chỉ lưu bản gốc FP16, tải về xong thư viện `bitsandbytes` mới nén ngay sang 4-bit trên máy.
- **Giải thích các tham số chính:**
  - `load_in_4bit=True` và `bnb_4bit_quant_type="nf4"`: nén trọng số xuống 4 bit kiểu NormalFloat.
  - `bnb_4bit_use_double_quant=True`: nén thêm các hằng số giải nén để tiết kiệm thêm một ít VRAM.
  - `bnb_4bit_compute_dtype=torch.float16`: khi tính toán, giải nén tạm về FP16 cho hợp với card T4.
  - `dtype=torch.float16`: giữ các phần không nén (như embedding) ở FP16.
  - `model.eval()`: chuyển mô hình sang chế độ chỉ dùng để suy luận, không học.
- **Kết quả nên thấy:** Footprint vào khoảng 1,3 đến 1,5 GB.
- **Lỗi hay gặp:** Nếu in ra ~3,4 GB nghĩa là `quantization_config` chưa có tác dụng. Lưu ý ở `transformers` bản 5 ta dùng `dtype=`, không dùng tên cũ `torch_dtype=`.

### Ô 5 (10 dòng): Sản phẩm 1 — Máy X-quang token kế tiếp

**Mục tiêu:** Nhìn thẳng vào bước 3 của vòng lặp sinh chữ: mô hình đang phân vân giữa những token nào và xác suất từng cái bao nhiêu.

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

- **Giải thích các cú pháp dễ lạ với người mới Python:**
  - `@torch.inference_mode()` là cú pháp gắn nhãn cho hàm bên dưới, báo PyTorch chỉ tính xuôi (không lưu thông tin để học) nên nhanh và nhẹ bộ nhớ hơn.
  - `return_tensors="pt"` đổi kết quả tokenizer sang bảng số PyTorch (tensor), rồi `.to(model.device)` chuyển bảng số đó từ CPU lên GPU.
  - `model(**enc)` dùng dấu `**` để trải các khóa trong dictionary `enc` (`input_ids`, `attention_mask`) thành tham số truyền vào hàm `model(...)`.
  - Kết quả `.logits` là bảng 3 chiều `[số câu, số token, 151.936]`. Viết `[0, -1]` nghĩa là lấy câu đầu tiên (`0`) và vị trí token cuối cùng (`-1`), vì ta muốn đoán chữ đứng ngay sau câu vừa nhập.
- **Kết quả nên thấy:** 5 dòng in ra các mảnh như `' UBND'`, `' Ủy'`, `' cơ'`, `' phường'`... kèm phần trăm giảm dần.
- **Điểm cần nhớ:** Đây chính là phần cốt lõi bên dưới mọi kỹ thuật fine-tune: huấn luyện chỉ nhằm kéo xác suất ở bảng này nghiêng về phía đáp án đúng.

### Ô 6 (3 dòng): Thử vặn núm temperature

**Mục tiêu:** Xem `temperature` làm phân phối xác suất co nhọn lại hay dàn trải ra như thế nào.

```python
for T in (0.3, 1.0, 1.5):
    print(f"--- temperature={T} ---")
    xray("Người lao động chạy xe ôm bị tai nạn cần nộp hồ sơ tại", temperature=T)
```

- **Kết quả nên thấy:** Thứ tự 5 token không đổi nhưng tỉ lệ phần trăm đổi rõ. Ở `T=0.3`, token đứng đầu chiếm gần như trọn vẹn (gần giống greedy). Ở `T=1.5`, xác suất chia đều hơn và tổng top-5 tụt hẳn xuống, nghĩa là mô hình dễ chọn các chữ lạ hơn và cũng dễ bịa hơn.
- **Áp dụng cho P01:** Khi chấm điểm bài toán tra cứu chính sách, ta luôn dùng greedy (chọn token cao nhất) để mỗi lần chạy ra đúng một kết quả cố định.

### Ô 7 (7 dòng): Tự tính bộ nhớ và so với Ô 4

**Mục tiêu:** Kiểm chứng quy tắc nhẩm ở mục 1.3 bằng code rồi đối chiếu với số đo thật của Ô 4.

```python
L, H, I, KV, V = 28, 2048, 6144, 1024, 151_936   # số lớp, chiều ẩn, chiều MLP, chiều K/V (8 đầu × 128), số token
attn, mlp = H*H + 2*H*KV + H*H, 3*H*I             # attention (q, k+v, o) và MLP (gate, up, down) mỗi lớp
non_emb, emb = L * (attn + mlp), V * H            # 28 lớp + bảng embedding
print(f"Tham số: {(non_emb + emb)/1e9:.3f} tỷ | FP16: {(non_emb + emb)*2/1e9:.2f} GB")
print(f"NF4 ước tính: {(non_emb*0.5 + non_emb*0.127/8 + emb*2)/1e9:.2f} GB")
print(f"Đo thật (Ô 4): {model.get_memory_footprint()/1e9:.2f} GB")
```

- **Giải thích dòng NF4:** `non_emb*0.5` là 28 lớp Linear nén 4-bit (0,5 byte/tham số), `non_emb*0.127/8` là bộ nhớ cho các hằng số giải nén, còn `emb*2` là bảng embedding vẫn nằm ở 16-bit (2 byte/tham số).
- **Kết quả nên thấy:** `Tham số: 1.720 tỷ | FP16: 3.44 GB`, `NF4 ước tính: 1.35 GB`, và số đo thật nằm khoảng 1,3 đến 1,5 GB.
- **Thói quen nên tập:** Trước khi nạp một mô hình mới, hãy nhẩm sơ xem nó có vừa GPU không. Ước tính lệch số đo thật 0,1 đến 0,2 GB là bình thường vì còn các lớp chuẩn hóa nhỏ và bộ đệm.

---

## Phần D: Nói chuyện với mô hình

### Ô 8 (6 dòng): Chat template khi bật và tắt thinking

**Mục tiêu:** Xem chuỗi ký tự thật sự đi vào mô hình sau khi áp khuôn hội thoại.

```python
msgs = [{"role": "system", "content": "Bạn là trợ lý an sinh."},
        {"role": "user", "content": "Xe ôm bị tai nạn cần giấy gì?"}]
for think in (True, False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    print(f"--- enable_thinking={think} ---\n{text!r}\n")
```

- **Giải thích:** `apply_chat_template` bọc từng tin nhắn trong cặp `<|im_start|>vai_trò ... <|im_end|>`. Tham số `add_generation_prompt=True` nối thêm `<|im_start|>assistant\n` ở cuối để ra hiệu đến lượt mô hình trả lời.
- **Kết quả nên thấy:** Hai chuỗi gần như y hệt, chỉ khác ở đuôi:
  - Khi `think=True`, đuôi là `'<|im_start|>assistant\n'` (mô hình tự mở `<think>` để nháp).
  - Khi `think=False`, đuôi là `'<|im_start|>assistant\n<think>\n\n</think>\n\n'`. Một khối nháp rỗng được điền sẵn để mô hình hiểu là đã nghĩ xong và trả lời luôn.
- **Lưu ý cho các buổi train:** Prompt lúc huấn luyện và lúc chạy thật phải khớp từng ký tự. Trong khóa này ta luôn gọi qua một chỗ duy nhất là `p01kit.prompts.render_prompt()`.

### Ô 9 (10 + 3 dòng): Đo thử thinking tốn thêm bao nhiêu

**Mục tiêu:** Đo trực tiếp số token và số giây giữa hai chế độ thay vì đoán mò.

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
  - `time.perf_counter()` là đồng hồ bấm giờ chính xác tính bằng giây.
  - `model.generate` tự chạy vòng lặp 4 bước ở mục 1.1 cho tới khi gặp `<|im_end|>` hoặc chạm `max_new_tokens`.
  - Kết quả `out` chứa cả phần prompt đầu vào lẫn phần mới sinh, nên `out[0, enc["input_ids"].shape[1]:]` cắt bỏ số token của prompt để giữ lại đúng phần câu trả lời.
  - Các con số `temperature`, `top_p`, `top_k` lấy theo khuyến nghị chính thức của nhóm Qwen3 (họ cũng khuyên không dùng greedy khi bật thinking vì dễ bị lặp vô hạn).
- **Kết quả nên thấy:** Bật thinking thường sinh số token gấp nhiều lần ở cùng tốc độ `tok/s`, kéo theo thời gian chờ tăng tương ứng.

---

## Phần E: Chạy nhanh trên nhiều câu

### Ô 10 (5 dòng): Vì sao sinh theo lô phải đệm bên trái?

**Mục tiêu:** Hiểu cách xếp các câu dài ngắn khác nhau vào cùng một bảng tensor.

```python
tok.padding_side = "left"
batch = tok(["Xin chào", "Tôi chạy xe ôm ở Tân Tạo bị té xe"], return_tensors="pt",
            padding=True, add_special_tokens=False)
print(batch["input_ids"])
print(batch["attention_mask"])
```

- **Hình dung nhanh:** Xếp hai câu thành hai hàng ngang, mô hình sẽ viết chữ mới nối tiếp vào **mép phải** của cả hai hàng cùng lúc. Câu ngắn hơn (`"Xin chào"`, 3 token) phải đệm thêm token rỗng ở bên **trái** để mép phải của nó thẳng hàng với câu dài (12 token). Nếu đệm bên phải, 9 token rỗng sẽ nằm kẹt giữa `"Xin chào"` và phần trả lời, làm câu trả lời bị hỏng.
- **Giải thích:** `attention_mask` đánh dấu `1` ở các token thật và `0` ở các token đệm để mô hình bỏ qua chúng.
- **Kết quả nên thấy:** Hàng 1 bắt đầu bằng 9 số `151643` (id của token pad) và 9 số `0` tương ứng trong `attention_mask`.
- **Lỗi hay gặp ở các buổi sau:** Khi **train** thì ngược lại, ta đệm bên **phải** (xem `p01kit.sft.PadCollator`). Nhớ kỹ quy tắc: sinh chữ thì đệm trái, train thì đệm phải.

### Ô 11 (10 + 4 dòng): Sản phẩm 2 — Bộ sinh theo lô và đo tốc độ tăng

**Mục tiêu:** Viết hàm hỏi nhiều câu song song và so tốc độ giữa `batch_size=1` và `batch_size=8`.

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
  - Vòng `for` chia danh sách câu hỏi thành từng lô `batch_size` câu.
  - `do_sample=False` bật greedy. Ba tham số `temperature=None, top_p=None, top_k=None` tắt cấu hình sampling mặc định của Qwen3 để tránh cảnh báo.
  - Dấu `:` trong `ids[:, enc["input_ids"].shape[1]:]` nghĩa là cắt bỏ phần prompt cho toàn bộ các hàng trong lô cùng lúc, rồi `batch_decode` dịch cả lô về chữ.
- **Kết quả nên thấy:** `batch_size=8` nhanh hơn `batch_size=1` nhiều lần vì GPU xử lý 8 câu song song ở từng bước. Hàm này cũng chính là `p01kit.infer.generate` dùng lại ở các buổi sau.

---

## Phần F: Đo baseline trên tập test P01

### Ô 12 (7 dòng): Xem thử một câu test P01

**Mục tiêu:** Biết rõ đầu vào và đáp án chuẩn của bài toán trước khi chấm điểm.

```python
from p01kit.data import load_split, dataset_stats
test = load_split("test")
print(dataset_stats(test))
r = test[0]
print("Địa bàn:", r["region"], "\nCâu hỏi:", r["question"])
print("Ngữ cảnh:\n", r["context"][:600], "...")
print("Đáp án vàng:", r["gold"])
```

- **Giải thích:** Mỗi câu test gồm câu hỏi của người dân, 3 đoạn ngữ cảnh chính sách (trong đó có thể có đoạn nhiễu không liên quan hoặc lệch địa bàn) và đáp án chuẩn dạng JSON. Khi ngữ cảnh không chứa văn bản phù hợp, mô hình phải từ chối thay vì bịa.
- **Kết quả nên thấy:** `n=100`, `unanswerable_ratio=0.44` (44% câu phải từ chối), `expired_ratio=0.1`, `colloquial_ratio=0.5` (một nửa viết theo giọng đời thường), và `slices={'seen_doc': 24, 'unseen_doc': 72, 'ood': 4}` (`seen_doc` là văn bản có xuất hiện trong tập train, `unseen_doc` là văn bản mới hoàn toàn, `ood` là câu hỏi ngoài phạm vi). Buổi 01 sẽ đi sâu vào cách chia này.
- **Lưu ý dữ liệu:** Toàn bộ văn bản chính sách trong bộ dữ liệu là giả lập để học tập, không dùng tra cứu thủ tục thật ngoài đời.

### Ô 13 (7 dòng): Sản phẩm 3 — Đo baseline zero-shot và ghi bảng điểm

**Mục tiêu:** Chấm mô hình gốc trên đủ 100 câu test và lưu dòng mốc đầu tiên vào bảng điểm trên Drive.

```python
from p01kit.infer import evaluate
from p01kit.metrics import format_summary
from p01kit import scoreboard
res0 = evaluate(model, tok, test, batch_size=8, max_new_tokens=p01kit.MAX_NEW_TOKENS)
print(format_summary(res0["summary"]))
print("Thời gian:", res0["summary"]["seconds"], "giây")
scoreboard.record("M00", "base-zero-shot", res0["summary"], latency_s=res0["summary"]["sec_per_row"], outputs=res0["outputs"])
```

- **Thời gian chờ:** Ô này chạy 100 câu trên T4 mất khoảng 2 đến 5 phút và không in tiến trình giữa chừng. Thấy Colab im lặng là bình thường, đừng bấm dừng.
- **Giải thích:** `evaluate()` tự tạo prompt chuẩn P01, chạy sinh theo lô bằng greedy, chấm từng câu rồi tổng hợp 8 chỉ số kèm khoảng tin cậy 95%. `scoreboard.record` ghi một dòng vào `leaderboard.csv` trên Drive và tự tạo file `p01_runs/outputs/M00__base-zero-shot.json` lưu output thô từng câu.
- **Hôm nay chỉ cần nhìn 2 chỉ số trước** (6 chỉ số còn lại học ở Buổi 01):
  - `json_valid`: tỉ lệ câu trả lời đúng định dạng JSON. Số này thấp nghĩa là mô hình chưa quen khuôn JSON của P01.
  - `answer_acc`: tỉ lệ quyết định đúng (từ chối đúng lúc hoặc trích đúng mã văn bản). Đây là con số dễ hình dung nhất cho chất lượng tổng thể.
- **Đọc khoảng tin cậy `95%CI`:** Chẳng hạn dòng in ra có dạng dưới đây (số minh họa, số thật sẽ phụ thuộc lượt chạy của bạn):

```text
json_valid           0.780  95%CI [0.700, 0.860]  (n=100)
answer_acc           0.410  95%CI [0.310, 0.510]  (n=100)
```

  Khoảng `[0.310, 0.510]` có nghĩa là vì ta chỉ đo trên 100 câu, điểm thật của mô hình nằm đâu đó trong khoảng 31% đến 51%. Nếu hai mô hình chỉ hơn nhau 2 đến 3 điểm phần trăm thì chưa chắc đã khác nhau thật (Buổi 01 sẽ học cách kiểm định).
- **Kết quả nên thấy:** Bảng 8 chỉ số kèm `95%CI` và `n`, với `json_valid > 0`. Điểm từ chối và cảnh báo hết hạn ở mô hình gốc thường thấp, đó là lý do ta cần fine-tune ở các buổi sau.
- **Lỗi hay gặp:** Nếu `json_valid = 0`, kiểm tra lại xem bạn có đổi `enable_thinking=True` hoặc hạ `max_new_tokens` quá nhỏ không.

### Ô 14 (5 dòng): Đọc lỗi thực tế (error analysis) và lưu output

**Mục tiêu:** Đọc trực tiếp 5 câu mô hình làm sai để biết điểm yếu của nó, đồng thời lưu output ra Drive cho Buổi 01 dùng lại miễn phí.

```python
import json
from p01kit.infer import show_errors
show_errors(res0, test, k=5)
with open(f"{os.environ['P01_RUNS']}/m00_base_outputs.json", "w") as f:
    json.dump(res0["outputs"], f, ensure_ascii=False)
```

- **Giải thích:** `show_errors` in 5 câu sai đầu tiên theo dạng:

```text
--- CT-BHYT::muc_ho_tro::colloquial0 | colloquial | địa bàn=Cần Thơ - Phường Cái Khế | gold_refusal=True gold_doc=None gold_expired=False
Q: Tui là người bán hàng rong, không có tiền mua bảo hiểm cho con...
OUT: {"doc_code": "QĐ-19/2026/UBND-ĐN", ...}
```

  Nhìn dòng tiêu đề bạn đối chiếu được ngay địa bàn của người hỏi, có phải từ chối không (`gold_refusal`), văn bản đúng (`gold_doc`) và văn bản có hết hạn không (`gold_expired`). Nếu muốn soi kỹ nhóm lỗi (e) cho một câu cụ thể, bạn có thể in thêm `test[i]["context"]` của câu đó.
- **Việc cần làm:** Phân loại 5 câu sai vừa in vào 5 nhóm dưới đây rồi ghi lại vào vở:

| Nhóm lỗi | Dấu hiệu nhận biết |
|---|---|
| (a) Sai định dạng JSON | Không phải JSON hợp lệ, thiếu khóa hoặc viết thêm chữ ngoài JSON |
| (b) Không chịu từ chối | `gold_refusal=True` nhưng mô hình vẫn trả lời và trích đại văn bản |
| (c) Trích nhầm văn bản lệch địa bàn | Người hỏi ở Cần Thơ/Đà Nẵng nhưng mô hình trích văn bản TP.HCM |
| (d) Bỏ sót cảnh báo hết hạn | `gold_expired=True` nhưng `expired_warning` trong output là `false` |
| (e) Bịa giấy tờ | Liệt kê giấy tờ không hề có trong ngữ cảnh |

---

## 3. So sánh hiệu quả: bảng điểm sau Buổi 00

Mở bảng điểm bất cứ lúc nào bằng một dòng:

```python
scoreboard.show()
```

| module | variant | json_valid | answer_acc | refusal_acc | citation_precision | expired_recall | strict_acc | fairness_gap |
|---|---|---|---|---|---|---|---|---|
| M00 | base-zero-shot | *(số đo của bạn)* | | | | | | |

Trong [PROJECT-REQUIREMENTS.md](../PROJECT-REQUIREMENTS.md), dự án chốt sẵn 4 ngưỡng bắt buộc: `citation_precision ≥ 0.90`, `refusal_acc ≥ 0.90`, `expired_recall = 1.0`, `fairness_gap ≤ 0.10`. Bạn có thể gọi `p01kit.metrics.gate(res0["summary"])` để xem chỉ số nào đạt, chỉ số nào chưa. Sau Buổi 00 hầu hết ngưỡng chưa đạt là chuyện bình thường.

---

## Ở công ty thật

- **Baseline thật là hệ thống đang chạy:** Không chỉ so với mô hình gốc, ở doanh nghiệp ta còn phải so với cách làm hiện tại (nhân viên tra thủ công, trang FAQ, tìm theo từ khóa hoặc gọi API mô hình lớn). Fine-tune chỉ đáng đầu tư khi thắng các phương án đó về chất lượng, chi phí hoặc độ trễ.
- **Chốt giới hạn phần cứng trước khi chọn mô hình:** Dự án P01 đặt mục tiêu chạy trên máy CPU 2 vCPU / 4 GiB (ADR 0001), nên cỡ 1,7B được chọn ngay từ đầu. Ở công ty, nhóm kỹ sư cũng tính trước chi phí suy luận theo số request mỗi ngày trước khi thử nghiệm.
- **Kiểm tra giấy phép mô hình:** Qwen3 dùng giấy phép Apache-2.0. Khi làm sản phẩm thật, bộ phận pháp chế sẽ rà soát giấy phép này trước khi cho phép triển khai.
- **Phân tích lỗi trên câu hỏi thật:** Câu hỏi người dùng thật (đã che thông tin cá nhân) thường ngắn, viết tắt, sai chính tả và không dấu, khó hơn nhiều so với bộ test tự dựng (sẽ bàn kỹ ở Buổi 15).

---

## Gặp lỗi? Tra nhanh

| Triệu chứng | Nguyên nhân hay gặp | Cách xử lý |
|---|---|---|
| `no CUDA GPUs are available` | Runtime đang chạy CPU | Vào *Runtime → Change runtime type → T4 GPU*, rồi chạy lại từ Ô 0a |
| `ModuleNotFoundError: p01kit` | Chưa chạy Ô 0b sau khi khởi động lại | Chạy lại Ô 0b |
| `check_env()` báo sai phiên bản | Quên chạy Ô 0a-3 sau khi cài thư viện | Chạy Ô 0a-3 (`os.kill(...)`), chờ nối lại rồi chạy Ô 0b |
| `CUDA out of memory` | Còn mô hình cũ trong VRAM hoặc `batch_size` quá lớn | *Runtime → Restart session*, chạy lại từ Ô 0b hoặc giảm `batch_size` |
| Footprint ~3,4 GB ở Ô 4 | Chưa bật nén 4-bit | Kiểm tra xem đã truyền `quantization_config=bnb` vào `from_pretrained` chưa |
| `json_valid = 0` ở Ô 13 | Vô tình bật thinking hoặc đặt `max_new_tokens` quá nhỏ | Giữ mặc định của `evaluate`, truyền `p01kit.MAX_NEW_TOKENS` |
| Colab ngắt phiên giữa buổi | Để tab rảnh quá lâu hoặc hết phiên miễn phí | Chạy lại Ô 0b rồi chạy tiếp ô đang làm dở (file trên Drive vẫn còn) |

---

## 4. Bài tập

**Bắt buộc**
1. Đo tốc độ `tok/s` của hàm `generate_batch` với `batch_size` lần lượt là 1, 4, 8, 16 trên 32 câu prompt rồi vẽ đồ thị. Đến mức `batch_size` nào thì tốc độ bắt đầu chững lại? Vì sao? (Gợi ý: mỗi câu trong lô đều chiếm một phần bộ nhớ cho KV-cache, và tới một ngưỡng nhất định thì GPU đã chạy hết công suất.)
2. Tự tính số tham số nếu Qwen3-1.7B không dùng chung bảng embedding với `lm_head`. Dung lượng VRAM ở FP16 sẽ tăng thêm bao nhiêu GB? (Gợi ý: khi tách riêng, mô hình có thêm một ma trận kích thước `151.936 × 2048`.)
3. Chạy thử baseline có bật thinking trên 20 câu test đầu: `evaluate(model, tok, test[:20], enable_thinking=True, max_new_tokens=1536, batch_size=4)`. So sánh `answer_acc` và thời gian chạy với bản tắt thinking, rồi ghi vào bảng điểm với `variant="base-thinking-20"`.

**Nâng cao**
4. Thử nạp mô hình ở chế độ 8-bit (`load_in_8bit=True`) và FP16 (không nén), so sánh footprint với bản 4-bit ở Ô 4. Chênh lệch có khớp quy tắc nhẩm ở mục 1.3 không?
5. Viết hàm `xray_chat(question)` soi top-5 token đầu tiên của câu trả lời sau khi áp chat template với `enable_thinking=False`. Token đứng đầu có phải ký tự `{` không? Kết quả đó cho biết điều gì về khả năng trả JSON của mô hình?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao phải đo baseline trước khi fine-tune?** (cơ bản)
   A: Để có mốc so sánh, sớm phát hiện trường hợp chỉ cần chỉnh prompt hoặc RAG là đủ (không cần fine-tune), và đặt tiêu chí dừng rõ ràng. Baseline bắt buộc phải đo trên cùng tập test, cùng prompt và cùng bộ chấm với các bản sau.
2. **Q: Qwen3-1.7B có bao nhiêu tham số và vì sao có nơi ghi 1.4B?** (cơ bản)
   A: Tổng cộng khoảng 1,72 tỷ tham số, trong đó 1,41 tỷ là phần non-embedding (28 lớp Transformer). Phần 0,31 tỷ còn lại là bảng embedding, dùng chung trọng số (tie) với lớp `lm_head`.
3. **Q: Vì sao sinh theo lô phải đệm trái còn train lại đệm phải?** (cơ bản)
   A: Lúc sinh chữ, mô hình nối token mới vào cuối chuỗi nên các câu phải thẳng hàng ở mép phải (tức là đệm bên trái). Lúc train, các vị trí đệm đã bị che bằng `attention_mask` và nhãn `-100`, nên đệm phải gọn và an toàn hơn.
4. **Q: `enable_thinking=False` hoạt động thế nào ở mức token?** (cơ bản)
   A: Chat template chèn sẵn `<think>\n\n</think>\n\n` ngay sau `<|im_start|>assistant\n`. Mô hình thấy khối nghĩ đã đóng nên bỏ qua bước nháp và trả lời luôn.
5. **Q: Vì sao train QLoRA trên card T4 phải dùng FP16 thay vì BF16?** (trung bình)
   A: Card T4 (kiến trúc Turing, `sm_75`) không có mạch phần cứng cho BF16; BF16 cần GPU đời Ampere (`sm_80`) trở lên. FP16 có dải số hẹp nên cần cơ chế loss scaling, đổi lại vẫn tận dụng được Tensor Core của T4.
6. **Q: GQA là gì và giúp ích gì khi chạy mô hình?** (nâng cao)
   A: GQA (Grouped-Query Attention) cho phép nhiều đầu Query dùng chung một cặp Key/Value (ở Qwen3-1.7B là 16 đầu Q dùng chung 8 đầu KV). Nhờ vậy bộ nhớ KV-cache giảm đi một nửa, chạy được ngữ cảnh dài hơn và kích thước lô lớn hơn.
7. **Q: Vì sao bảng embedding không bị nén 4-bit trong bitsandbytes?** (nâng cao)
   A: Thư viện `bitsandbytes` chỉ thay các lớp `nn.Linear` bằng `Linear4bit`. Bảng embedding là bảng tra cứu, nén xuống 4-bit dễ gây sai số nặng cho các token hiếm trong khi dung lượng tiết kiệm được không đáng kể.

---

## 6. Checklist cuối buổi

- [ ] Tự vẽ lại được sơ đồ 4 bước vòng lặp sinh chữ ở mục 1.1 mà không cần mở bài.
- [ ] Giải thích được sự khác nhau giữa `Qwen3-1.7B` và `Qwen3-1.7B-Base`.
- [ ] Nhẩm được 3,44 GB (FP16) và giải thích được vì sao NF4 ra khoảng 1,35 GB thay vì 0,86 GB.
- [ ] Viết được hàm `generate_batch` đệm trái và đo được mức tăng tốc so với `batch_size=1`.
- [ ] Bảng điểm trên Drive có dòng `M00 / base-zero-shot` và đã tự phân loại xong 5 câu sai đầu tiên.

## Tài liệu tham khảo

- Qwen3 Technical Report và model card `Qwen/Qwen3-1.7B` (mục *Best Practices* về tham số sampling).
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (NF4, double quantization).
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models*.
