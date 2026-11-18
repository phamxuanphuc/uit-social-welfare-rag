# Buổi 7 (Module 7): Kỹ thuật Dữ liệu — Sinh Dữ liệu Tổng hợp (Synthetic Data), Lọc & Khử Trùng lặp

- **Bối cảnh thực tế (Job Interview Focus):** Ở công ty thật, **80% thời gian kỹ sư Fine-tune nằm ở khâu chuẩn bị và làm sạch dữ liệu**, không phải ở việc gõ lệnh `.train()`. Bài này dạy bạn cách tự tạo bộ dữ liệu **vài trăm đến hàng nghìn mẫu chất lượng cao** cho bất kỳ lĩnh vực nào mà không làm bẩn tập Test.
- **🎯 Sản phẩm cuối buổi:** **Pipeline Tạo & Lọc Dữ liệu Sạch (`Data-Factory`)** — sinh đa dạng cách hỏi (chuẩn vs. đời thường), lọc bỏ mẫu lỗi JSON/bịa trích dẫn, khử trùng lặp (Deduplication) bằng TF-IDF Cosine, và kiểm tra không rò rỉ dữ liệu sang tập Test.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Synthetic Data & Teacher-Student Distillation là gì?**
   - **Hình ảnh Feynman:** Bạn muốn đào tạo một nhân viên thực tập (`Qwen3-1.7B`) nhưng chỉ có 5 văn bản luật khô khan. Bạn nhờ một chuyên gia lớn (`Qwen3-32B` hoặc Gemini/GPT — gọi là **Teacher**) đọc từng điều khoản và đóng vai người dân hỏi theo 10 cách khác nhau (giọng miền Nam, miền Bắc, người lớn tuổi, sinh viên) kèm đáp án JSON chuẩn, sau đó dùng chính bộ dữ liệu đó dạy lại `Qwen3-1.7B` (**Student**).
2. **Tại sao bắt buộc phải Khử trùng lặp (Deduplication) & Lọc chất lượng (Quality Filtering)?**
   - Nếu sinh 500 câu hỏi mà có 300 câu giống nhau tới 90% (*"Tôi chạy xe ôm..."* đổi mỗi 1 chữ), mô hình sẽ **học vẹt cực nặng** và sụp đổ đa dạng ngôn ngữ.
   - Bắt buộc qua 3 cửa ải: (1) **Schema & Grounding Filter** (loại bỏ mọi mẫu LLM Teacher tự bịa ra ngoài văn bản gốc), (2) **Exact & Near Deduplication** (loại các câu có độ tương đồng Cosine > `0.85`), (3) **Leakage Check** (kiểm tra Cosine giữa mọi câu `Train` và mọi câu `Test` phải < `0.85`).

---

## 2. Thực hành Pipeline `Data-Factory` trên Google Colab

### Bước 1: Sinh Đa Dạng Cách Hỏi từ Văn bản Gốc (Teacher Prompt Template)
```python
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

synthetic_candidates = [
    {"scenario_id": "SCN_01", "style": "colloquial", "doc_code": "NQ-01/2026/HĐND-HCM", "q": "Chạy xe ôm ở Phường Tân Tạo bị té xe nghỉ làm thì đem giấy gì lên phường?", "excerpt": "Hỗ trợ tai nạn lao động cho lao động phi chính thức"},
    {"scenario_id": "SCN_01", "style": "colloquial", "doc_code": "NQ-01/2026/HĐND-HCM", "q": "Chạy xe ôm ở Phường Tân Tạo bị té xe nghỉ làm thì cần giấy gì?", "excerpt": "Hỗ trợ tai nạn lao động cho lao động phi chính thức"},  # Gần trùng lặp!
    {"scenario_id": "SCN_01", "style": "formal", "doc_code": "NQ-01/2026/HĐND-HCM", "q": "Thành phần hồ sơ đề nghị trợ cấp tai nạn lao động cho lao động phi chính thức tại Phường Tân Tạo?", "excerpt": "Hỗ trợ tai nạn lao động cho lao động phi chính thức"},
    {"scenario_id": "SCN_02", "style": "colloquial", "doc_code": "QĐ-15/2026/UBND-HCM", "q": "Làm thẻ BHYT cho con phụ hồ ở Phường Tân Tạo nộp ở đâu?", "excerpt": "Hỗ trợ 100% thẻ BHYT và miễn giảm 50% học phí"},
    {"scenario_id": "SCN_02", "style": "colloquial", "doc_code": "QĐ-15/2026/UBND-HCM", "q": "Bịa đặt: Có được tặng 50 triệu tiền mặt mua nhà không?", "excerpt": "Tặng ngay 50 triệu đồng mua biệt thự"},  # Lỗi bịa đặt không nằm trong nguồn!
]
```

### Bước 2: Bộ Lọc Grounding (Loại bỏ mọi mẫu Teacher LLM tự bịa đặt)
```python
policy_by_code = {p["doc_code"]: p for p in POLICIES}

def filter_grounded_samples(rows: list) -> list:
    kept = []
    for r in rows:
        src = policy_by_code.get(r["doc_code"])
        if src and r["excerpt"].lower()[:20] in src["content"].lower():
            kept.append(r)
        else:
            print(f"🗑️ Loại mẫu bịa đặt/sai nguồn: {r['q']}")
    return kept

grounded_rows = filter_grounded_samples(synthetic_candidates)
print(f"Sau Grounding Filter: giữ {len(grounded_rows)}/{len(synthetic_candidates)} mẫu.")
```

### Bước 3: Bộ Khử Trùng Lặp (Near-Deduplication với ngưỡng Cosine `0.85`)
```python
def deduplicate_queries(rows: list, threshold: float = 0.85) -> list:
    if len(rows) <= 1:
        return rows
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([r["q"] for r in rows])
    sim_matrix = cosine_similarity(vec)
    kept_indices = []
    for i in range(len(rows)):
        is_dup = any(sim_matrix[i, j] >= threshold for j in kept_indices)
        if is_dup:
            print(f"♻️ Loại mẫu trùng lặp (Cosine >= {threshold}): {rows[i]['q']}")
        else:
            kept_indices.append(i)
    return [rows[i] for i in kept_indices]

clean_rows = deduplicate_queries(grounded_rows, threshold=0.80)
print(f"Sau Dedup: giữ {len(clean_rows)} mẫu sạch duy nhất.")
```

### Bước 4: Kiểm tra Rò rỉ Dữ liệu (`Train` vs. `Test` Cross-Leakage Audit)
```python
def check_train_test_leakage(train_qs: list, test_qs: list, threshold: float = 0.80):
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(train_qs + test_qs)
    sims = cosine_similarity(vec.transform(test_qs), vec.transform(train_qs))
    leaks = [(test_qs[i], train_qs[j], sims[i, j]) for i in range(len(test_qs)) for j in range(len(train_qs)) if sims[i, j] >= threshold]
    print(f"🔍 Kiểm tra rò rỉ Train/Test: phát hiện {len(leaks)} cặp nghi ngờ lộ đề.")
    return leaks

check_train_test_leakage([r["q"] for r in clean_rows], ["Chạy xe ôm ở Tân Tạo cần giấy gì khi tai nạn?"])
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Nếu dùng GPT-4/Claude sinh 5,000 mẫu Synthetic Data để fine-tune Qwen3-1.7B, rủi ro lớn nhất là gì và xử lý thế nào?*
   - **Đáp:** Rủi ro gồm: (1) **Model collapse / lặp mẫu** (các câu giống hệt cấu trúc nhau) $\rightarrow$ xử lý bằng **Deduplication** (MinHash / Cosine) và đa dạng hóa persona/style trong prompt sinh; (2) **Hallucination lan truyền** từ Teacher $\rightarrow$ xử lý bằng **Programmatic Grounding Check** & Schema validator trước khi đưa vào train; (3) **Lộ đề sang Test** $\rightarrow$ chia theo `scenario_id` và chạy **Cross-Leakage Audit**.
2. **Hỏi:** *Tại sao lọc dữ liệu ít mẫu nhưng sạch (`1,000` mẫu chuẩn, giống triết lý LIMA) thường cho kết quả tốt hơn `50,000` mẫu nhiễu?*
   - **Đáp:** Vì chất lượng căn chỉnh hành vi (alignment) phụ thuộc vào tính nhất quán của format và độ chính xác logic. Dữ liệu nhiễu chứa nhãn mâu thuẫn làm gradient triệt tiêu lẫn nhau và dạy mô hình thói quen đoán bừa.
