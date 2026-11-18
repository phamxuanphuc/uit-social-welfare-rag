"""p01kit — bộ công cụ dùng chung cho lộ trình fine-tune Qwen3-1.7B (đồ án P01).

Nguyên tắc thiết kế:
- Mọi thứ *không phải trọng tâm bài học* (nạp dữ liệu, chấm điểm, bảng điểm, guardrail)
  nằm ở đây, được kiểm thử bằng `pytest` trên CPU.
- Mọi thứ *là trọng tâm bài học* (masking, LoRA, DPO, GRPO...) học viên tự viết trong
  từng buổi, rồi so với bản trong p01kit để tự kiểm tra.
- Các module cần GPU (`modeling`, `infer`) chỉ import torch/transformers bên trong hàm,
  nên `import p01kit` luôn nhẹ và chạy được ở mọi nơi.
"""

__version__ = "2.0.0"

MODEL_ID = "Qwen/Qwen3-1.7B"
UNSLOTH_MODEL_ID = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
# Đo bằng tokenizer thật trên tập train: prompt ~840 token (tối đa 885), đáp án ~230 (tối đa 308).
MAX_SEQ_LEN = 1280
MAX_NEW_TOKENS = 384
