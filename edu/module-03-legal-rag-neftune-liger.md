# Buổi 3 (Module 3): Pháp luật & Thủ tục Hành chính Việt Nam — `NEFTune`, `Unsloth + Liger Kernel` & Công bằng Ngôn ngữ

- **Dự án tham chiếu:** `duyet/vietnamese-legal-instruct` (127k mẫu `vbpl.vn`), `fisherman611/vietnamese-legal-chatbot` (Hybrid Search Dense + BM25 + Reranker), `tangbamiinh/traffic-law-assistant` (phân tách thực thể pháp lý), các mô hình của `AITeamVN`, `5CD-AI`, `bkai-foundation-models`, `ura-hcmut`.
- **Mục tiêu đầu ra:** Hiểu và thực hành ánh xạ ngôn ngữ đời thường sang thuật ngữ pháp lý (`ColloquialMapping`), bật **`NEFTune` (`neftune_noise_alpha=5`)** chống học vẹt, kích hoạt **`Unsloth + Liger Kernel`** giảm 60% VRAM cho văn bản dài, và đo độ chênh lệch Công bằng (**Fairness Gap $\le 10\%$**).

---

## 1. Giải thích Bản chất theo Phương pháp Feynman

1. **Vấn đề "Rào cản ngôn ngữ hành chính" (Trục Công bằng — Fairness trong Rubric P01):**
   - **Hình ảnh Feynman:** Văn bản pháp luật trên `vbpl.vn` viết bằng ngôn ngữ trang trọng (*"Lao động không có giao kết hợp đồng lao động bị suy giảm khả năng lao động"*), nhưng người dân ngoài đời lại hỏi bằng từ truyền miệng (*"Tôi chạy xe ôm công nghệ bị té xe nghỉ làm tuần nay"*).
   - Nếu chỉ tìm kiếm và huấn luyện trên câu chữ luật cứng nhắc, người biết chữ nghĩa hành chính sẽ được AI trả lời đúng `95%`, còn người lao động nghèo nói chuyện đời thường chỉ nhận được câu trả lời đúng `50%` $\rightarrow$ **Hệ thống AI gây bất công xã hội!**
   - **Giải pháp kép (học từ `fisherman611` & `tangbamiinh`):**
     1. Ở tầng tìm kiếm: Kết hợp **Bảng từ đồng nghĩa đời thường (`ColloquialMapping`)** + **Hybrid Search (Dense ngữ nghĩa + BM25 từ khóa)**.
     2. Ở tầng Fine-tune LLM: Dùng **`NEFTune`** để mô hình học *ý nghĩa khái quát* thay vì học thuộc lòng từ khóa.
2. **`NEFTune` (`Noisy Embedding Fine-Tuning`, `neftune_noise_alpha=5`) hoạt động ra sao?**
   - **Hình ảnh Feynman:** Nếu bạn học thuộc lòng một bài văn trong phòng kín yên tĩnh tuyệt đối, ra phòng thi chỉ cần giám thị đổi 2 chữ trong đề là bạn đứng hình (*Overfitting*). `NEFTune` giống như việc bật tiếng ồn trắng nhẹ ($\alpha = 5$) vào tai người học trong lúc ôn bài: nó cộng một chút nhiễu ngẫu nhiên vào tọa độ từ vựng (`Embedding`), buộc bộ não `Qwen3-1.7B` không thể nhớ vẹt mặt chữ mà phải nắm bắt **ý nghĩa vùng lân cận**.
3. **Tại sao đến Buổi 3 lại bật `Unsloth` + `Liger Kernel`?**
   - Văn bản pháp luật và thủ tục hành chính thường dài (1,000 – 2,048 tokens). Ở tầng cuối cùng của Qwen3, phép tính `Softmax + Cross-Entropy` phải nhân với bảng từ điển **151,936 từ**, tạo ra một bảng số tạm thời chiếm tới **4–6 GB VRAM**.
   - **`Liger Kernel`** tính gộp phép toán đó thành từng lát nhỏ ngay trong bộ nhớ siêu tốc của GPU (*SRAM*) mà không tạo bảng số khổng lồ trên VRAM $\rightarrow$ **giảm ngay ~60% VRAM** và tăng tốc gấp đôi!

---

## 2. Thực hành 5 Bước trên Google Colab

### Bước 1: Cài đặt `Unsloth` & `Liger Kernel` và Nạp mô hình tăng tốc
```bash
!pip install -q unsloth liger-kernel
```
```python
from unsloth import FastLanguageModel

model_fast, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
```

---

### Bước 2: Thiết lập `ColloquialMapping` & Cặp câu hỏi Song song (Chuẩn vs. Đời thường)
```python
COLLOQUIAL_MAP = {
    "xe ôm": "lao động phi chính thức vận tải xe hai bánh",
    "chạy grab": "lao động phi chính thức lái xe công nghệ",
    "bán vé số": "lao động tự do bán vé số lẻ",
    "phụ hồ": "lao động tự do ngành xây dựng",
    "tiền trọ": "hỗ trợ tiền thuê nhà trọ",
    "sổ hộ khẩu": "giấy xác nhận thông tin cư trú CT07",
}

def expand_colloquial(query: str) -> str:
    added = [formal for slang, formal in COLLOQUIAL_MAP.items() if slang in query.lower()]
    return f"{query} ({', '.join(added)})" if added else query

paired_eval = [
    {
        "formal": "Hồ sơ đề nghị hưởng trợ cấp tai nạn lao động đối với lao động phi chính thức tại Phường Tân Tạo gồm những giấy tờ gì?",
        "colloquial": "Tôi chạy xe ôm ở Phường Tân Tạo mới bị té xe nghỉ làm thì đem giấy gì lên phường xin hỗ trợ vậy?",
        "expected_doc": "NQ-01/2026/HĐND-HCM",
    },
    {
        "formal": "Thủ tục đăng ký hỗ trợ 100% mức đóng Bảo hiểm y tế cho con của người lao động tự do thu nhập thấp tại Phường Tân Tạo?",
        "colloquial": "Tôi đi phụ hồ nghèo quá muốn xin thẻ bảo hiểm đi khám bệnh cho đứa nhỏ ở Phường Tân Tạo cần giấy tờ chi?",
        "expected_doc": "QĐ-15/2026/UBND-HCM",
    },
]
print("Ví dụ mở rộng câu đời thường:\n", expand_colloquial(paired_eval[0]["colloquial"]))
```

---

### Bước 3: Chuẩn bị dữ liệu Pháp luật Hành chính (kết cấu Thực thể kiểu `tangbamiinh` & `vbpl.vn`)
```python
import json
from datasets import Dataset

legal_context = (
    "[Văn bản 1] NQ-01/2026/HĐND-HCM (Điều 4, Khoản 2 - Hiệu lực: 2026-01-15, Phiên bản: v2.1, Địa bàn: TP.HCM - Phường Tân Tạo): "
    "Đối tượng: Lao động phi chính thức (xe ôm, bán vé số, phụ hồ). Hồ sơ gồm: (1) Đơn Mẫu 01-ASXH, (2) CCCD hoặc CT07, "
    "(3) Xác nhận Tổ dân phố, (4) Giấy ra viện. Nơi nộp: Bộ phận Một cửa UBND Phường Tân Tạo.\n"
    "[Văn bản 2] QĐ-15/2026/UBND-HCM (Điều 3, Khoản 1 - Hiệu lực: 2026-03-01, Phiên bản: v1.0, Địa bàn: TP.HCM - Phường Tân Tạo): "
    "Đối tượng: Con của lao động tự do thu nhập thấp. Hồ sơ gồm: (1) Tờ khai Mẫu TK1-TS, (2) Giấy khai sinh của con, "
    "(3) Xác nhận thu nhập thấp, (4) Bản sao CCCD. Nơi nộp: Điểm tiếp nhận ASXH Phường Tân Tạo."
)

train_pairs = [
    (
        "Tôi chạy grab ở Phường Tân Tạo bị tai nạn giao thông cần nộp giấy tờ gì và ở đâu?",
        {
            "verification_reasoning": "Đối tượng chạy grab là lao động phi chính thức tại Phường Tân Tạo, khớp NQ-01/2026/HĐND-HCM (Điều 4, Khoản 2).",
            "doc_code": "NQ-01/2026/HĐND-HCM",
            "article": "Điều 4, Khoản 2",
            "required_documents": ["Đơn Mẫu 01-ASXH", "CCCD hoặc CT07", "Xác nhận Tổ dân phố", "Giấy ra viện"],
            "submission_place": "Bộ phận Một cửa UBND Phường Tân Tạo"
        }
    ),
    (
        "Bán vé số ở Phường Tân Tạo muốn làm thẻ BHYT cho con đi học cần giấy gì?",
        {
            "verification_reasoning": "Hỗ trợ thẻ BHYT cho con lao động tự do tại Phường Tân Tạo, khớp QĐ-15/2026/UBND-HCM (Điều 3, Khoản 1).",
            "doc_code": "QĐ-15/2026/UBND-HCM",
            "article": "Điều 3, Khoản 1",
            "required_documents": ["Tờ khai Mẫu TK1-TS", "Giấy khai sinh của con", "Xác nhận thu nhập thấp", "Bản sao CCCD"],
            "submission_place": "Điểm tiếp nhận ASXH Phường Tân Tạo"
        }
    ),
] * 8

ds_module3 = Dataset.from_list([
    {
        "messages": [
            {"role": "system", "content": "Trả lời thủ tục hành chính bằng JSON chuẩn theo đúng văn bản pháp luật trong ngữ cảnh."},
            {"role": "user", "content": format_rag_input(legal_context, expand_colloquial(q))},
            {"role": "assistant", "content": json.dumps(ans, ensure_ascii=False)},
        ]
    }
    for q, ans in train_pairs
])
```

---

### Bước 4: Huấn luyện với `DoRA + rsLoRA + NEFTune (alpha=5)`
```python
from trl import SFTConfig, SFTTrainer

model_fast = FastLanguageModel.get_peft_model(
    model_fast,
    r=16,
    lora_alpha=32,
    use_dora=True,
    use_rslora=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

trainer_m3 = SFTTrainer(
    model=model_fast,
    processing_class=tokenizer,
    train_dataset=ds_module3,
    args=SFTConfig(
        output_dir="outputs_module3",
        max_steps=30,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        neftune_noise_alpha=5,
        logging_steps=5,
        fp16=True,
        report_to="none",
    ),
)
trainer_m3.train()
```

---

### Bước 5: Đo Độ Lệch Công Bằng (`Fairness Gap`: Câu chuẩn vs. Câu đời thường)
```python
FastLanguageModel.for_inference(model_fast)

for item in paired_eval:
    for style in ("formal", "colloquial"):
        q_expanded = expand_colloquial(item[style])
        res = ask_rag(model_fast, legal_context, q_expanded, use_lora=True)
        matched = item["expected_doc"] in res
        print(f"[{style.upper():10s}] Khớp {item['expected_doc']} = {matched} | Output: {res[:110]}...")
```

---

## 3. Câu hỏi Tự kiểm tra cuối Buổi 3 (Checkpoint)
1. Tại sao phải kết hợp cả `expand_colloquial()` (từ điển) lẫn `neftune_noise_alpha=5` (nhiễu embedding)? *(Trả lời: Từ điển bắt các từ lóng phổ biến đã biết, còn NEFTune giúp mô hình tổng quát hóa với các cách diễn đạt đời thường mới chưa có trong từ điển)*.
