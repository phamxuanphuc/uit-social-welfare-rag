# Buổi 13 — Triển khai: Merge FP16, GGUF + `imatrix` thật, Giải mã ràng buộc JSON & Đo độ trễ CPU

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (45' lý thuyết · 3h15 lab, trong đó ~15–20' biên dịch llama.cpp) |
| **Đầu vào** | Adapter tốt nhất theo bảng điểm (mặc định `m09/grpo`) |
| **🎯 Sản phẩm** | (1) Mô hình gộp **`merged_fp16`**; (2) **5 file GGUF** (F16, Q8_0, Q4_K_M, Q4_K_M+imatrix, Q3_K_M ± imatrix) build bằng llama.cpp thật; (3) **Bảng đánh đổi** kích thước – chất lượng – tốc độ; (4) Thí nghiệm **constrained decoding** (JSON Schema) vs sinh tự do; (5) Số đo **TTFT / token/s trên CPU** để dự báo cho Azure 2 vCPU |
| **Chỉ số đo** | MB · 8 chỉ số P01 · `json_valid` · TTFT (s) · token/s (prefill & sinh) |
| **Chế độ code** | 🧩 Khối gộp (Python + shell) |

---

## 0. Bức tranh lớn

Một mô hình chỉ có giá trị khi **chạy được ở nơi người dùng cần**. ADR 0001 của đồ án chốt: **CPU-only** (Azure Container Apps 2 vCPU / 4 GiB, không PyTorch trong container) với `llama-cpp-python` + `Qwen3-1.7B-Q4_K_M.gguf`. Buổi này đi trọn con đường từ adapter LoRA → file GGUF → đo trên CPU, và trả lời bằng số: **lượng tử hóa làm mất bao nhiêu chất lượng, imatrix cứu lại được bao nhiêu, độ trễ có chấp nhận được không**.

---

## 1. Lý thuyết

### 1.1. Merge đúng cách
- **Sai:** gọi `merge_and_unload()` trên mô hình đang nạp 4-bit rồi lưu — cộng ΔW (FP16) vào trọng số **đã bị làm tròn 4-bit** rồi lượng tử lần nữa → sai số kép.
- **Đúng:** lưu adapter → nạp base **FP16 sạch** → `merge_and_unload()` → lưu → lượng tử hóa **một lần** sang GGUF.

### 1.2. Các định dạng lượng tử
| Định dạng | Chạy tốt trên | Ý tưởng |
|---|---|---|
| **GGUF** (llama.cpp) | CPU, CPU+GPU, Mac | K-quants: chia khối, mỗi khối có scale/min; nhiều mức bit (Q8_0, Q6_K, Q4_K_M, Q3_K_M, IQ…) |
| **AWQ** | GPU (vLLM) | Dò kênh quan trọng dựa trên **activation** và **nhân scale** để bảo vệ chúng trước khi lượng tử |
| **GPTQ** | GPU (vLLM, ExLlama) | Lượng tử từng lớp, bù sai số bằng thông tin bậc hai (Hessian xấp xỉ) |
| **bitsandbytes NF4** | GPU (train QLoRA) | Lượng tử lúc nạp; tiện để train, không phải định dạng phục vụ tối ưu |

### 1.3. Importance matrix (imatrix)
- Chạy mô hình F16 trên **văn bản hiệu chuẩn**, ghi lại độ lớn activation của từng cột trọng số → biết cột nào **quan trọng**.
- Khi lượng tử, `llama-quantize --imatrix` tối ưu sai số **có trọng số** theo tầm quan trọng đó.
- Hiệu quả **lớn dần khi số bit giảm** (Q3, IQ) và nhỏ ở Q8. Các định dạng IQ (ví dụ IQ2) **bắt buộc** có imatrix.
- Văn bản hiệu chuẩn nên **giống phân phối thật** (prompt + đáp án P01 tiếng Việt) và **chỉ lấy từ tập train** — dùng test là rò rỉ.

### 1.4. Giải mã ràng buộc (constrained decoding)
- llama.cpp chuyển **JSON Schema → ngữ pháp GBNF**; ở mỗi bước sinh, các token làm hỏng cú pháp bị **chặn** (logit = −∞).
- Đảm bảo 100% JSON đúng schema, kể cả **thứ tự khóa** khi schema liệt kê theo thứ tự (`verification_reasoning` đầu tiên — ADR 0002).
- Không sửa được **nội dung sai** (văn bản nhầm vẫn là JSON hợp lệ) → vẫn cần fine-tune + guardrail.

### 1.5. Hai con số độ trễ
- **TTFT** (time to first token) ≈ thời gian **prefill** (xử lý ~840 token prompt) — phụ thuộc độ dài prompt; trên CPU thường là phần lớn độ trễ cảm nhận.
- **TPOT** / token/s sinh — quyết định tốc độ "gõ chữ" khi stream ~230 token JSON.
- Tổng ≈ TTFT + số_token_sinh × TPOT.

---

## 2. Thực hành

### Khối 1 — Khởi động & chọn adapter cuối
```python
import os, sys, json, time, subprocess
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
import torch, p01kit
from p01kit.data import load_split
from p01kit.modeling import adapter_path, free, load_tokenizer
from p01kit import scoreboard
train, val, test = (load_split(s) for s in ("train", "val", "test"))
FINAL = adapter_path("m09", "grpo")          # đổi theo kết luận bảng điểm của bạn
WORK = "/content/deploy"; os.makedirs(WORK, exist_ok=True)
tok = load_tokenizer()
scoreboard.show()
```

### Khối 2 — Merge LoRA vào base FP16 (trên CPU, ~3,4 GB RAM)
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
base16 = AutoModelForCausalLM.from_pretrained(p01kit.MODEL_ID, dtype=torch.float16, device_map="cpu")
merged = PeftModel.from_pretrained(base16, str(FINAL)).merge_and_unload()
merged.save_pretrained(f"{WORK}/merged_fp16", safe_serialization=True); tok.save_pretrained(f"{WORK}/merged_fp16")
print(sorted(os.listdir(f"{WORK}/merged_fp16")))
del base16, merged; free()
```
- 📌 **Kết quả nên thấy:** `config.json`, `model.safetensors` (~3,4 GB), các file tokenizer. Không còn `adapter_config.json` → đây là mô hình độc lập.

### Khối 3 — Biên dịch llama.cpp (ghim tag, chỉ build đúng kiến trúc T4) — hoặc khôi phục từ Drive
```python
from p01kit.colab import restore_llamacpp_build, save_llamacpp_build
if not os.path.exists("/content/llama.cpp"):        # mã nguồn cần cho convert_hf_to_gguf.py (clone nông, vài giây)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "b12084", "https://github.com/ggml-org/llama.cpp", "/content/llama.cpp"], check=True)
CACHED = restore_llamacpp_build()
print("✅ Dùng lại bản build trên Drive — bỏ qua 2 ô biên dịch" if CACHED else "Chưa có bản build hợp lệ → ô dưới sẽ biên dịch")
```
```bash
!test -x /content/llama.cpp/build/bin/llama-server || cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
!test -x /content/llama.cpp/build/bin/llama-server || cmake --build /content/llama.cpp/build -j $(nproc) --target llama-quantize llama-imatrix llama-server llama-bench
```
```python
if not CACHED:
    print("Đã lưu bản build lên Drive:", save_llamacpp_build())   # Buổi 15 và lần chạy lại Buổi 13 sẽ dùng lại
print(sorted(os.listdir("/content/llama.cpp/build/bin")))
```
- **Giải thích:** `--branch b12084` ghim phiên bản (tái lập được); `CMAKE_CUDA_ARCHITECTURES=75` chỉ biên dịch cho T4 → nhanh hơn nhiều so với build mọi kiến trúc; chỉ build 4 công cụ cần dùng.
- 💡 **Colab miễn phí:** runtime miễn phí chỉ có 2 vCPU nên biên dịch CUDA mất nhiều thời gian (ghi lại số phút của bạn). Bản build (`build/bin`, gồm chương trình và thư viện `.so`) được nén lên `p01_runs/cache/` ngay sau lần biên dịch đầu; mọi phiên sau chỉ giải nén và chạy thử `llama-server --version`. Nếu Colab đổi image và bản lưu không chạy được, nó bị xóa và ô biên dịch tự chạy lại.
- ⚠️ **Không** cài `requirements` của llama.cpp vào môi trường này — file đó có thể hạ cấp `torch`/`transformers` đã ghim. Script chuyển đổi tự dùng thư viện `gguf-py` đi kèm repo.

### Khối 4 — Chuyển sang GGUF F16 & tạo văn bản hiệu chuẩn (chỉ từ TRAIN)
```python
from p01kit.llamacpp import calibration_text
LC = "/content/llama.cpp"
subprocess.run(["python", f"{LC}/convert_hf_to_gguf.py", f"{WORK}/merged_fp16", "--outfile", f"{WORK}/p01-f16.gguf", "--outtype", "f16"], check=True)
open(f"{WORK}/calib.txt", "w").write(calibration_text(train, tok))
print(f"F16: {os.path.getsize(f'{WORK}/p01-f16.gguf')/1e6:.0f} MB | calib: {len(open(f'{WORK}/calib.txt').read())} ký tự")
```

### Khối 5 — Tạo imatrix thật & lượng tử hóa nhiều mức
```bash
!/content/llama.cpp/build/bin/llama-imatrix -m /content/deploy/p01-f16.gguf -f /content/deploy/calib.txt -o /content/deploy/imatrix.gguf -ngl 99 -c 1024 --chunks 100
```
```python
BIN = f"{LC}/build/bin"
variants = {"Q8_0": ("Q8_0", False), "Q4_K_M": ("Q4_K_M", False), "Q4_K_M-imat": ("Q4_K_M", True),
            "Q3_K_M": ("Q3_K_M", False), "Q3_K_M-imat": ("Q3_K_M", True)}
for name, (qtype, imat) in variants.items():
    cmd = [f"{BIN}/llama-quantize"] + (["--imatrix", f"{WORK}/imatrix.gguf"] if imat else []) + \
          [f"{WORK}/p01-f16.gguf", f"{WORK}/p01-{name}.gguf", qtype]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"{name:<12} {os.path.getsize(f'{WORK}/p01-{name}.gguf')/1e6:>6.0f} MB")
```
- **Giải thích:** `-c 1024` = độ dài mỗi đoạn hiệu chuẩn (đủ chứa 1 mẫu P01); `--chunks 100` giới hạn thời gian. Bản llama.cpp cũ lưu imatrix dạng `.dat`; bản mới lưu GGUF — `llama-quantize` đọc được cả hai.
- 📌 **Kết quả nên thấy:** kích thước giảm dần F16 (~3.400 MB) → Q8_0 → Q4_K_M (~1.100 MB) → Q3_K_M; bản có/không imatrix **cùng kích thước** (imatrix không đổi số bit, chỉ đổi cách làm tròn).

### Khối 6 — Phục vụ bằng `llama-server` & chấm điểm trên 100 câu test
```python
import requests
from p01kit.llamacpp import complete, ANSWER_JSON_SCHEMA
from p01kit.prompts import render_prompt
from p01kit.metrics import score_one, summarize, format_summary

def start_server(gguf, ngl=99, threads=None, port=8080, slots=1):
    cmd = [f"{BIN}/llama-server", "-m", gguf, "-c", str(2048 * slots), "-ngl", str(ngl), "--port", str(port), "-np", str(slots)]
    if threads: cmd += ["-t", str(threads)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill(); raise RuntimeError("llama-server không khởi động được")

from concurrent.futures import ThreadPoolExecutor
from p01kit.runs import once
prompts = [render_prompt(tok, r) for r in test]           # cùng prompt với lúc train (enable_thinking=False)
def eval_gguf(name, schema=True, slots=4):
    def run():
        proc = start_server(f"{WORK}/p01-{name}.gguf", slots=slots)
        try:
            with ThreadPoolExecutor(slots) as ex:              # 4 request song song = 4 slot của llama-server
                outs = list(ex.map(lambda p: complete(p, json_schema=ANSWER_JSON_SCHEMA if schema else None)["text"], prompts))
        finally:
            proc.terminate(); proc.wait()
        s = summarize([score_one(o, r) for o, r in zip(outs, test)])
        scoreboard.record("M13", f"gguf-{name}{'' if schema else '-noschema'}", s, outputs=outs)
        return s
    return once("m13", f"eval_{name}{'' if schema else '_noschema'}", run)   # bị ngắt → biến thể đã chấm được bỏ qua

gguf_res = {"F16": eval_gguf("f16")}
for name in variants:
    gguf_res[name] = eval_gguf(name)
gguf_res["Q4_K_M-imat (không schema)"] = eval_gguf("Q4_K_M-imat", schema=False)
for k, s in gguf_res.items():
    print(f"{k:<28} json={s['json_valid']:.3f} schema={s['schema_ok']:.3f} answer={s['answer_acc']:.3f} strict={s['strict_acc']:.3f} expired={s['expired_recall']}")
```
- **Giải thích:** đo chất lượng trên **GPU** (`-ngl 99`) cho nhanh — kết quả số học của GGUF trên GPU và CPU gần như tương đương; độ trễ CPU đo riêng ở Khối 7.
- 💡 **Colab miễn phí:** chấm 8 biến thể là phần tốn GPU nhất buổi. `slots=4` cho `llama-server` xử lý 4 câu cùng lúc (`-c` là tổng ngữ cảnh, chia đều cho các slot) thay vì từng câu một; `once` lưu điểm từng biến thể lên Drive. Lưu ý: file GGUF nằm ở ổ đĩa tạm `/content/deploy` (bản gộp FP16 + 6 file GGUF, ~12 GB, không vừa Drive miễn phí) — nếu runtime bị ngắt, phải chạy lại Khối 2–5 để tạo lại file, nhưng các biến thể đã chấm không phải chấm lại.
- ⚠️ Với nhiều slot, llama.cpp ghép các câu thành batch nên kết quả greedy có thể lệch rất nhẹ so với `slots=1` (thứ tự cộng số thực khác nhau). So sánh các biến thể với nhau vẫn công bằng vì cùng cấu hình; độ trễ một người dùng ở Khối 7 vẫn đo với 1 slot.
- 📌 **Cách đọc:** F16 là trần chất lượng của bản gộp; Q8_0 gần như không mất; so `Q4_K_M` với `Q4_K_M-imat` và `Q3_K_M` với `Q3_K_M-imat` để thấy imatrix có tác dụng **lớn hơn ở mức bit thấp** hay không. Dòng "không schema" cho biết constrained decoding đóng góp bao nhiêu vào `json_valid`/`schema_ok`.

### Khối 7 — Độ trễ trên CPU: TTFT, token/s, và dự báo cho Azure 2 vCPU
```python
import statistics as st
proc = start_server(f"{WORK}/p01-Q4_K_M-imat.gguf", ngl=0, threads=2)      # mô phỏng 2 vCPU
try:
    runs = [complete(p, json_schema=ANSWER_JSON_SCHEMA) for p in prompts[:10]]
finally:
    proc.terminate(); proc.wait()
ttft = [r["ttft_s"] for r in runs]; tps = [r["tok_per_s"] for r in runs if r["tok_per_s"]]; tot = [r["total_s"] for r in runs]
print(f"CPU 2 luồng — TTFT trung vị {st.median(ttft):.1f}s | sinh {st.median(tps):.1f} tok/s | tổng trung vị {st.median(tot):.1f}s/câu")
```
```bash
!/content/llama.cpp/build/bin/llama-bench -m /content/deploy/p01-Q4_K_M-imat.gguf -ngl 0 -t 2 -p 840 -n 230
```
- **Giải thích:** `llama-bench -p 840 -n 230` đo tốc độ **prefill 840 token** (đúng độ dài prompt P01) và **sinh 230 token** (đúng độ dài JSON trung bình) — số liệu chuẩn hóa để so sánh với phần cứng khác.
- ⚠️ CPU Colab khác CPU Azure → coi đây là **ước lượng bậc độ lớn**; quyết định cuối phải đo trên chính môi trường deploy (ghi vào Release Dossier).

### Khối 8 — Lưu sản phẩm lên Drive
```python
import shutil
dst = adapter_path("m13", "gguf"); os.makedirs(dst, exist_ok=True)
for f in ("p01-Q4_K_M-imat.gguf", "imatrix.gguf"):
    shutil.copy(f"{WORK}/{f}", dst / f)
json.dump({k: {m: s.get(m) for m in ("json_valid", "schema_ok", "answer_acc", "strict_acc", "expired_recall")} for k, s in gguf_res.items()},
          open(dst / "quant_report.json", "w"), indent=2, ensure_ascii=False)
print(sorted(os.listdir(dst)))
```

### 🔁 Lối tắt bằng Unsloth (sau khi đã hiểu quy trình thủ công)
```python
# from unsloth import FastLanguageModel   # (runtime mới, import trước transformers)
# model, tok = FastLanguageModel.from_pretrained(str(FINAL), max_seq_length=p01kit.MAX_SEQ_LEN, load_in_4bit=True)
# model.save_pretrained_gguf(f"{WORK}/unsloth_gguf", tok, quantization_method="q4_k_m", imatrix_file=f"{WORK}/imatrix.gguf")
```
- Unsloth tự giải nén về 16-bit trước khi gọi llama.cpp (tránh lỗi merge 4-bit) và từ bản 2026.11 nhận `imatrix_file=`. Dùng lối tắt khi đã hiểu và kiểm chứng được từng bước bên dưới.

---

## 3. So sánh hiệu quả

### 📊 Trước → Sau: từng câu, cùng bộ chấm
```python
from p01kit.compare import compare_runs
rep = compare_runs(("M09", "grpo"), ("M13", "gguf-Q4_K_M-imat"), test)
```
- **Đọc báo cáo theo thứ tự:** (1) bảng chỉ số — Δ có ✅/🔻 và khoảng tin cậy của bản *sau*; (2) McNemar — chỉ các câu **đổi kết quả** mang thông tin, p < 0,05 mới gọi là cải thiện thật; (3) lát cắt — cải thiện có đều ở giọng đời thường và văn bản chưa thấy không; (4) **ví dụ cụ thể** — đọc tận mắt câu được sửa và câu bị làm hỏng.
- 🎯 **Buổi này:** Cùng mô hình, chỉ khác **lượng tử hóa + JSON Schema**: câu bị làm hỏng là cái giá của Q4. Đổi bản *sau* thành `("M13", "gguf-Q3_K_M")` để thấy cái giá tăng khi xuống 3 bit.


| Biến thể | MB | json_valid | schema_ok | answer_acc | strict_acc | expired_recall |
|---|---|---|---|---|---|---|
| Adapter trên 4-bit (bảng điểm M09) | ~70 + base | | | | | |
| GGUF F16 | | | | | | |
| Q8_0 | | | | | | |
| Q4_K_M | | | | | | |
| **Q4_K_M + imatrix** | | | | | | |
| Q3_K_M | | | | | | |
| Q3_K_M + imatrix | | | | | | |
| Q4_K_M + imatrix, **không schema** | | | | | | |

| CPU 2 luồng (Q4_K_M+imatrix) | TTFT | token/s sinh | Tổng/câu | pp840 (llama-bench) | tg230 |
|---|---|---|---|---|---|
| Số đo | | | | | |

> 📌 **Quyết định deploy:** chọn mức lượng tử nhỏ nhất mà mọi ngưỡng chốt trước vẫn đạt (kết hợp guardrail ở Buổi 14). Nếu tổng độ trễ/câu trên CPU vượt mức chấp nhận, cân nhắc: rút ngắn `verification_reasoning`, giảm số văn bản ngữ cảnh, hoặc stream kết quả.

---

## 4. Phục vụ nhiều adapter & huấn luyện đa GPU (kiến thức bắt buộc khi phỏng vấn)

### 4.1. vLLM Multi-LoRA (tham khảo — chạy trên runtime GPU riêng)
```bash
# vllm serve Qwen/Qwen3-1.7B --dtype half --enable-lora --max-lora-rank 16 \
#   --lora-modules p01=/content/drive/MyDrive/p01_runs/m09/grpo agent=/content/drive/MyDrive/p01_runs/m11/agent
```
- Một bản base trong VRAM, nhiều adapter nạp động theo request (`"model": "p01"` hoặc `"agent"` trong API kiểu OpenAI). **PagedAttention** quản lý KV-cache theo trang như bộ nhớ ảo → batch lớn, ít phân mảnh. **Continuous batching** ghép request mới vào batch đang chạy.

### 4.2. Huấn luyện đa GPU
| Kỹ thuật | Chia gì | Khi nào dùng |
|---|---|---|
| **DDP** | Dữ liệu; mỗi GPU giữ **toàn bộ** mô hình | Mô hình vừa 1 GPU, muốn nhanh hơn |
| **FSDP / ZeRO-3** | Chia cả **tham số, gradient, trạng thái optimizer** | Mô hình không vừa 1 GPU |
| ZeRO-1 / ZeRO-2 | Chỉ chia optimizer / + gradient | Trung gian, ít giao tiếp hơn ZeRO-3 |
| Tensor / Pipeline parallel | Chia **trong lớp** / **theo lớp** | Mô hình rất lớn, phục vụ suy luận lớn |

---

## 🏢 Ở công ty thật

- **Phục vụ bằng container có cấu hình ghim** (tag image, hash mô hình), triển khai qua CI/CD có revision và rollback, kiểm thử tải trước khi mở traffic (Buổi 15).
- **Chọn engine theo phần cứng và chi phí mỗi 1.000 request ở SLO yêu cầu**: trên GPU thường là vLLM/TGI/SGLang (continuous batching); trên CPU hoặc thiết bị biên là llama.cpp/ONNX.
- **Đánh giá lại trên chính artifact deploy** (file GGUF), không suy ra từ adapter, và ghi hash vào hồ sơ phát hành.

---

## 5. Bài tập

**Bắt buộc**
1. Thêm biến thể **IQ3_XXS** (bắt buộc imatrix). Thử lượng tử **không** truyền imatrix — thông báo lỗi là gì? Đo chất lượng khi có imatrix.
2. **Imatrix sai miền:** tạo imatrix từ văn bản tiếng Anh ngẫu nhiên (ví dụ đoạn Wikipedia) thay cho calib P01. So Q3_K_M-imat (P01) và Q3_K_M-imat (tiếng Anh).
3. Đo TTFT khi rút ngữ cảnh còn 2 văn bản (sửa `context_ids`) — TTFT giảm tuyến tính theo số token prompt không?

**Nâng cao**
4. Chạy `llama-cpp-python` (CPU wheel) đúng như ADR 0001: nạp GGUF, dùng `LlamaGrammar.from_json_schema(json.dumps(ANSWER_JSON_SCHEMA))`, so với `llama-server`.
5. Viết `Dockerfile` `python:3.11-slim` tối giản chỉ gồm FastAPI + llama-cpp-python + GGUF; đo kích thước image và RAM lúc chạy (mục tiêu < 2,5 GB).

---

## 6. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Vì sao không merge LoRA trên mô hình 4-bit?**
   A: Trọng số 4-bit đã bị làm tròn; cộng ΔW rồi lượng tử lại gây sai số kép. Đúng: merge trên base FP16/BF16 sạch, lượng tử một lần.
2. **Q: imatrix là gì, vì sao giúp?**
   A: Ma trận tầm quan trọng thu từ activation trên văn bản hiệu chuẩn; lượng tử tối ưu sai số có trọng số theo cột quan trọng. Lợi rõ ở mức bit thấp; dữ liệu hiệu chuẩn nên cùng miền và không dùng test.
3. **Q: GGUF vs AWQ vs GPTQ?**
   A: GGUF cho CPU/hỗn hợp qua llama.cpp; AWQ bảo vệ kênh quan trọng theo activation bằng scale, chạy nhanh trên GPU/vLLM; GPTQ lượng tử từng lớp có bù sai số bậc hai.
4. **Q: Constrained decoding giải quyết và không giải quyết được gì?**
   A: Đảm bảo output đúng cú pháp/schema (kể cả thứ tự khóa); không đảm bảo nội dung đúng — vẫn cần mô hình tốt và hậu kiểm.
5. **Q: TTFT và TPOT phụ thuộc vào gì?**
   A: TTFT chủ yếu theo độ dài prompt (prefill, compute-bound); TPOT theo băng thông bộ nhớ khi đọc trọng số mỗi bước sinh (memory-bound) — lượng tử nhỏ hơn giúp TPOT rõ rệt.
6. **Q: vLLM phục vụ nhiều LoRA thế nào?**
   A: Giữ một base trong VRAM, nạp adapter nhỏ theo yêu cầu, gộp request khác adapter trong cùng batch (kernel multi-LoRA); PagedAttention và continuous batching tối ưu throughput.
7. **Q: DDP khác FSDP?**
   A: DDP nhân bản toàn bộ mô hình trên mỗi GPU, chỉ chia dữ liệu; FSDP/ZeRO-3 chia nhỏ tham số, gradient, optimizer state qua các GPU để train mô hình không vừa một card.

---

## 7. Checklist cuối buổi
- [ ] Merge trên FP16, build llama.cpp đúng tag, có imatrix thật từ dữ liệu train.
- [ ] Bảng 8 dòng chất lượng–kích thước + bảng độ trễ CPU.
- [ ] `m13/gguf/` (GGUF chọn deploy + imatrix + `quant_report.json`) trên Drive.

## Tài liệu tham khảo
- llama.cpp — `tools/quantize`, `tools/imatrix`, `tools/server` README; GBNF grammars.
- Lin et al., *AWQ* (2023); Frantar et al., *GPTQ* (2022).
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023).
- Rajbhandari et al., *ZeRO* (2020); PyTorch FSDP documentation.
