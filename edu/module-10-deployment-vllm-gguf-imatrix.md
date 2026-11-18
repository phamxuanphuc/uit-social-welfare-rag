# Buổi 10 (Module 10): Triển khai Production — Merge 4-bit, `imatrix GGUF`, `vLLM`, Multi-LoRA & Distributed

- **Bối cảnh thực tế (Job Interview Focus):** Làm xong model trên Colab chỉ là một nửa chặng đường. Khi đi làm, bạn phải trả lời được: làm sao gộp LoRA mà không tụt chất lượng 4-bit, tạo **`imatrix`** thật bằng `llama.cpp`, phục vụ nhiều khách hàng cùng lúc bằng **`vLLM` Multi-LoRA**, đo **TTFT / Throughput**, và chọn giữa `GGUF`, `AWQ`, `GPTQ` hay `DDP/FSDP`.
- **🎯 Sản phẩm cuối buổi:** **Quy trình Xuất `Qwen3-1.7B-Q4_K_M.gguf` kèm `imatrix` Thật** (không phải file txt tạo suông) + Bảng đo độ trễ **TTFT & Tokens/giây** trên CPU/GPU.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Cạm bẫy chết người khi gộp Adapter (`Merge LoRA`) trên bản 4-bit:**
   - Nhiều người nạp model ở `4-bit` rồi gọi ngay `model.merge_and_unload()` và lưu ra đĩa $\rightarrow$ bạn vừa cộng ma trận `16-bit` vào trọng số đã bị làm tròn `4-bit`, gây sai số lượng tử hóa kép!
   - **Cách chuẩn:** Lưu riêng **Adapter LoRA nhẹ (~30 MB)**, sau đó nạp lại model gốc ở **`16-bit` (`float16`/`bfloat16`)**, gọi `merge_and_unload()` trên nền 16-bit sạch, rồi mới lượng tử hóa 1 lần duy nhất xuống **`Q4_K_M.gguf`** (hoặc dùng `Unsloth` `save_pretrained_gguf`, vì bên dưới Unsloth tự động giải nén về 16-bit rồi mới gọi `llama.cpp` nén GGUF).
2. **So sánh 3 chuẩn Lượng tử hóa phổ biến khi phỏng vấn:**
   - **`GGUF` (`llama.cpp`):** Tối ưu chạy **CPU** hoặc kết hợp CPU+GPU (rất hợp hạ tầng Azure Container Apps không có GPU chi phí $0 của P01).
   - **`AWQ` (*Activation-aware Weight Quantization*):** Bảo vệ top 1% trọng số quan trọng dựa trên kích hoạt, chạy siêu nhanh trên **GPU bằng `vLLM`**.
   - **`GPTQ`:** Lượng tử hóa 4-bit theo từng lớp trên GPU bằng ma trận Hessian, lâu hơn AWQ khi nén nhưng hỗ trợ rộng trên `vLLM`/`ExLlamaV2`.
3. **Phục vụ Multi-LoRA trên `vLLM` & Đào tạo đa GPU (`DDP`, `FSDP`, `DeepSpeed`):**
   - **`vLLM` + `PagedAttention` + Multi-LoRA:** 1 GPU giữ chung đúng 1 bản gốc `Qwen3-1.7B` (3.4 GB) và nạp động hàng chục tờ giấy LoRA khác nhau (`An sinh`, `Y tế`, `Pháp luật`) cho từng request mà không tốn thêm GPU!
   - **Đa GPU (`DDP` vs `FSDP` vs `DeepSpeed ZeRO`):** `DDP` sao chép toàn bộ model sang mọi GPU (chỉ chia dữ liệu); `FSDP` / `DeepSpeed ZeRO-3` **cắt nhỏ** cả Model Weights, Gradients và Optimizer States chia đều qua các GPU để train model 70B+.

---

## 2. Thực hành Xuất `imatrix Q4_K_M.gguf` Thật & Đo Benchmark Trên Colab

### Bước 1: Merge LoRA Đúng Chuẩn (Lưu Adapter $\rightarrow$ Merge ở 16-bit)
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 1. Chỉ lưu Adapter LoRA (~30 MB)
model_fast.save_pretrained("adapter_p01")
tokenizer.save_pretrained("adapter_p01")

# 2. (Chuẩn công nghiệp khi tự merge bằng PEFT): Nạp bản FP16 sạch rồi mới merge_and_unload()
# base_fp16 = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cpu")
# merged_model = PeftModel.from_pretrained(base_fp16, "adapter_p01").merge_and_unload()
# merged_model.save_pretrained("merged_fp16_p01")
print("Đã lưu Adapter sạch; sẵn sàng merge 16-bit và nén GGUF!")
```

### Bước 2: Tạo `imatrix` Thật bằng Công cụ `llama.cpp` (`llama-imatrix` + `llama-quantize`)
*(Nếu chạy bằng `Unsloth` hoặc biên dịch `llama.cpp` trên Colab, quy trình tạo Importance Matrix thật gồm đúng 3 lệnh shell sau)*:
```bash
# 1. Xuất bản 16-bit F16 GGUF từ thư mục merged_fp16_p01 (dùng convert_hf_to_gguf.py của llama.cpp)
# python llama.cpp/convert_hf_to_gguf.py merged_fp16_p01 --outfile qwen3-1.7b-p01-f16.gguf --outtype f16

# 2. Chạy llama-imatrix trên file câu tiếng Việt P01 để sinh file trọng số quan trọng (imatrix.dat)
# ./llama.cpp/build/bin/llama-imatrix -m qwen3-1.7b-p01-f16.gguf -f imatrix_calibration_vi.txt -o imatrix.dat --chunks 50

# 3. Nén từ F16 xuống Q4_K_M CÓ TRUYỀN --imatrix imatrix.dat (bảo toàn dấu tiếng Việt!)
# ./llama.cpp/build/bin/llama-quantize --imatrix imatrix.dat qwen3-1.7b-p01-f16.gguf qwen3-1.7b-p01-Q4_K_M.gguf Q4_K_M
```
- **Lưu ý thực tế:** Dùng `model_fast.save_pretrained_gguf("...", tokenizer, quantization_method="q4_k_m")` của `Unsloth` trên Colab chỉ chạy lượng tử hóa `Q4_K_M` tiêu chuẩn; nếu muốn có **Importance Matrix (`imatrix`) thật**, bắt buộc chạy bước `./llama-imatrix` rồi `./llama-quantize --imatrix imatrix.dat` của `llama.cpp` như trên!

### Bước 3: Đo Độ trễ Thực tế (`TTFT` & `Tokens/sec`)
```python
import time

def benchmark_latency(model_obj, prompt_txt: str):
    inp = tokenizer([prompt_txt], return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model_obj.generate(**inp, max_new_tokens=100, do_sample=False)
    elapsed = time.perf_counter() - t0
    gen_tokens = out.shape[-1] - inp["input_ids"].shape[-1]
    print(f"⏱️ Sinh {gen_tokens} tokens trong {elapsed:.2f}s -> Tốc độ: {gen_tokens / elapsed:.1f} tokens/s")

benchmark_latency(model_fast, format_dpo_prompt(json.dumps([POLICIES[0]], ensure_ascii=False), "Chạy xe ôm bị tai nạn cần giấy gì?"))
```

### Bước 4: Lệnh Phục vụ Multi-LoRA trên `vLLM` (Tham khảo khi công ty có GPU)
```bash
# 1 GPU chạy đồng thời Qwen3-1.7B gốc + nhiều Adapter chuyên biệt:
# vllm serve Qwen/Qwen3-1.7B --enable-lora --lora-modules welfare=phamxuanphuc/p01-lora legal=phamxuanphuc/legal-lora
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Tại sao không nên gọi `merge_and_unload()` trực tiếp trên model đang nạp ở 4-bit `bitsandbytes`?*
   - **Đáp:** Vì cộng trọng số LoRA (16-bit) vào trọng số đã nén 4-bit gây mất độ chính xác nặng (*quantization error*). Luôn phải lưu Adapter riêng, nạp lại bản gốc `FP16/BF16` rồi mới merge, sau đó mới nén 1 lần về GGUF/AWQ.
2. **Hỏi:** *`TTFT` và `TPOT` là gì khi đo hiệu năng phục vụ LLM?*
   - **Đáp:** **TTFT** (*Time To First Token*) là độ trễ từ lúc gửi request đến khi hiện chữ đầu tiên (phụ thuộc độ dài prompt/context); **TPOT** (*Time Per Output Token*) là thời gian trung bình sinh mỗi chữ tiếp theo (quyết định cảm giác gõ nhanh hay chậm khi stream).
3. **Hỏi:** *Phân biệt `DDP`, `FSDP` và `DeepSpeed ZeRO` khi train trên cụm nhiều GPU?*
   - **Đáp:** `DDP` sao chép nguyên mô hình sang mọi GPU (chỉ chia batch dữ liệu). `FSDP` và `DeepSpeed ZeRO` chia mảnh (shard) Optimizer states (ZeRO-1), Gradients (ZeRO-2) và cả Model Parameters (ZeRO-3) qua các GPU để train model rất lớn không vừa 1 card.
