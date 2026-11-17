# 0001. CPU-Only GGUF & ONNX Inference Runtime

- **Status:** accepted

Đề tài yêu cầu triển khai trên hạ tầng chi phí $0 (Azure Container Apps 2 CPU / 4 GiB RAM, Scale-to-Zero). Nếu cài đặt PyTorch hoặc TensorFlow vào container thực thi, kích thước image và bộ nhớ RAM khi khởi động sẽ vượt ngưỡng 2.5–4 GiB.

Chúng tôi quyết định tách riêng môi trường huấn luyện trên Google Colab T4 (Unsloth/TRL/PEFT/ORPO/GRPO) và môi trường phục vụ trên CPU: sử dụng `llama-cpp-python` nạp `Qwen3-1.7B-Q4_K_M.gguf`, kết hợp `fastembed` (ONNX) cho Dense/Sparse BM25 và `FlashRank` (ONNX) cho reranking, tuyệt đối không cài đặt PyTorch hay TensorFlow trong `requirements.txt`.
