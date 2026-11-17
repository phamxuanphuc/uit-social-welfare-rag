# 0002. Dual-Mode RAG with Verification-First JSON Schema & Hard Safety Guardrails

- **Status:** accepted

Để vừa khai thác Rationale Distillation từ Qwen3-1.7B, vừa đảm bảo Constrained JSON Decoding của `llama-cpp-python` không bị lệch định dạng, chúng tôi đặt `verification_reasoning` làm key đầu tiên ngay trong JSON Schema (`RAGAnswerSchema`) thay vì thẻ văn bản tự do bên ngoài.

Bên cạnh đó, hệ thống áp dụng kiến trúc Dual-Mode: khi có cấu hình Qdrant Cloud và file `.gguf` sẽ chạy pipeline Qdrant + FlashRank + llama.cpp; khi chạy CI (`pytest`) hoặc môi trường chưa tải model 1.1 GB sẽ tự động dùng Local Hybrid Engine trên `data/mock_policies.json`. Mọi phản hồi đều đi qua lớp hậu kiểm Python cưỡng bức bật `expired_warning=True` nếu tài liệu gốc có `status == "expired"`, đồng thời lưu biến động trạng thái văn bản vào SQLite/Qdrant thay vì ghi đè file seed.
