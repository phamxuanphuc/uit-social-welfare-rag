# Buổi 9 (Module 9): Fine-tune Embedding & Reranker Tiếng Việt (Tối ưu Tầng Truy xuất RAG P01)

- **Bối cảnh thực tế (Job Interview Focus):** Trong một hệ thống RAG, **70% lỗi sai đến từ tầng Truy xuất (Retriever tìm trượt văn bản gốc)** chứ không phải do LLM! Các nhà tuyển dụng vị trí RAG/AI Engineer luôn hỏi: *"Làm sao cải thiện độ chính xác khi người dùng gõ từ lóng miền địa phương mà Embedding gốc không hiểu?"*
- **🎯 Sản phẩm cuối buổi:** **Bộ Embedding Tiếng Việt Tinh chỉnh (`BAAI/bge-m3` / `sentence-transformers`)** sử dụng **`MultipleNegativesRankingLoss` + Hard Negatives** giúp kéo cặp *(Câu hỏi đời thường $\leftrightarrow$ Văn bản luật)* lại gần nhau hơn, đo `Recall@1` Trước vs. Sau trên tập Test giữ riêng.

---

## 1. Giải thích theo Phương pháp Feynman

1. **Embedding vs. Cross-Encoder Reranker khác nhau thế nào?**
   - **Embedding (Bi-Encoder — như `BAAI/bge-m3`):** Biến *Câu hỏi* thành 1 mũi tên tọa độ và biến *Văn bản* thành 1 mũi tên tọa độ **độc lập nhau**, rồi đo góc giữa 2 mũi tên (Cosine). **Siêu nhanh** để quét 1 triệu văn bản, nhưng kém tinh tế với từ đồng nghĩa hiếm.
   - **Reranker (Cross-Encoder — như `FlashRank` / `bge-reranker`):** Đưa cả `[Câu hỏi + Văn bản]` vào đọc cùng lúc để từng chữ soi chiếu chéo lẫn nhau (`Self-Attention`). **Rất chính xác** nhưng chậm, nên chỉ dùng để chấm lại Top-10 văn bản sau khi Embedding đã lọc xong.
2. **`Hard Negative Mining` & `MultipleNegativesRankingLoss` (MNRL):**
   - **Hình ảnh Feynman:** Muốn dạy phân biệt hàng thật vs. hàng giả, nếu đưa cho học sinh xem 1 chiếc iPhone và... 1 quả táo (**Easy Negative**), học sinh học chẳng tiến bộ gì. Phải đặt 1 văn bản đúng của *Phường Tân Tạo* bên cạnh 1 văn bản gần giống hệt nhưng thuộc *Đà Nẵng* hoặc *đã hết hạn* (**Hard Negative**)!
   - `MNRL` tự động kéo `(Câu hỏi, Văn bản đúng)` xích lại gần `1.0` và đẩy `(Câu hỏi, Văn bản nhiễu)` ra xa `0.0`.

---

## 2. Thực hành Fine-tune Embedding Tiếng Việt trên Google Colab

### Bước 1: Cài đặt `sentence-transformers` & Nạp Mô hình Embedding Đa ngôn ngữ
```bash
!pip install -q "sentence-transformers>=3.0.0" "datasets==3.5.0"
```
```python
import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader

embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5" if False else "intfloat/multilingual-e5-small")
print("Đã nạp Embedding Model! Số chiều vector:", embed_model.get_sentence_embedding_dimension())
```

### Bước 2: Đo `Recall@1` 🔴 TRƯỚC Fine-tune (Câu hỏi Đời thường vs. Kho Văn bản P01)
```python
corpus_docs = [p["content"] for p in POLICIES]
eval_pairs = [
    ("Tôi chạy xe ôm bị té xe nghỉ làm tuần nay cần hỗ trợ gì?", 0),
    ("Phụ hồ nghèo muốn xin thẻ bảo hiểm y tế cho con đi học?", 1),
    ("Làm sao xin trợ cấp mất việc học nghề tại Liên Chiểu?", 2),
    ("Hỗ trợ tiền phòng trọ cho công nhân giai đoạn 2024?", 3),
]

def eval_recall_at_1(model_emb) -> float:
    doc_vecs = model_emb.encode(["passage: " + d for d in corpus_docs], normalize_embeddings=True)
    hits = 0
    for q, true_idx in eval_pairs:
        q_vec = model_emb.encode("query: " + q, normalize_embeddings=True)
        pred_idx = int(np.argmax(cos_sim(q_vec, doc_vecs)[0]))
        hits += int(pred_idx == true_idx)
    return hits / len(eval_pairs) * 100

recall_before = eval_recall_at_1(embed_model)
print(f"🔴 Recall@1 TRƯỚC Fine-tune Embedding: {recall_before:.1f}%")
```

### Bước 3: Đóng gói Bộ Ba `(Anchor, Positive, Hard Negative)` & Huấn luyện `MNRL`
```python
train_triplets = [
    InputExample(texts=[
        "query: Chạy grab bị tai nạn giao thông ở Tân Tạo xin tiền hỗ trợ",
        "passage: " + POLICIES[0]["content"],
        "passage: " + POLICIES[2]["content"],  # Hard Negative: Cùng nói lao động tự do nhưng ở Đà Nẵng!
    ]),
    InputExample(texts=[
        "query: Bán vé số nghèo xin thẻ BHYT và giảm học phí cho con",
        "passage: " + POLICIES[1]["content"],
        "passage: " + POLICIES[0]["content"],
    ]),
    InputExample(texts=[
        "query: Thợ hồ mất việc ở Đà Nẵng muốn đăng ký học nghề miễn phí",
        "passage: " + POLICIES[2]["content"],
        "passage: " + POLICIES[0]["content"],
    ]),
    InputExample(texts=[
        "query: Công nhân ở trọ muốn hỏi tiền thuê nhà trọ năm 2024",
        "passage: " + POLICIES[3]["content"],
        "passage: " + POLICIES[1]["content"],
    ]),
] * 4

train_loader = DataLoader(train_triplets, shuffle=True, batch_size=2)
train_loss = losses.MultipleNegativesRankingLoss(embed_model)
embed_model.fit(train_objectives=[(train_loader, train_loss)], epochs=3, warmup_steps=2, show_progress_bar=False)
```

### Bước 4: Đo `Recall@1` 🟢 SAU Fine-tune Embedding
```python
recall_after = eval_recall_at_1(embed_model)
print(f"🏆 Recall@1 Embedding Tiếng Việt: Trước = {recall_before:.1f}% ---> Sau = {recall_after:.1f}%")
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Tại sao hệ thống RAG lớn luôn dùng 2 tầng: Bi-Encoder (Embedding) rồi mới đến Cross-Encoder (Reranker)?*
   - **Đáp:** Bi-Encoder tính trước vector cho toàn bộ tài liệu và lưu vào Vector DB (Qdrant), lúc truy vấn chỉ cần đo Cosine trong vài mili-giây (`O(1)` với HNSW) nhưng thiếu tương tác từng token giữa Query và Doc. Cross-Encoder đưa cả cặp vào Transformer cho độ chính xác cao hơn hẳn nhưng tốn `O(N)` lượt chạy, nên chỉ chạy lại trên Top-10 ứng viên.
2. **Hỏi:** *Hard Negative trong huấn luyện Embedding là gì và tại sao nó quan trọng?*
   - **Đáp:** Hard Negative là văn bản *sai* nhưng chứa nhiều từ khóa giống câu hỏi hoặc cùng chủ đề (ví dụ cùng là chính sách lao động tự do nhưng sai địa bàn). Huấn luyện với Hard Negative ép vector phân biệt được ranh giới ngữ nghĩa tinh tế thay vì chỉ khớp từ khóa bề mặt.
