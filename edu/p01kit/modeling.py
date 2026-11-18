"""Nạp mô hình/adapter chuẩn hóa cho mọi buổi (cần GPU; torch/transformers import trong hàm).

Kỷ luật checkpoint (chống "nhiễm adapter" giữa các buổi):
- Mỗi buổi lưu adapter vào `runs/<module>/<tên>`.
- Buổi sau LUÔN nạp base sạch + adapter cụ thể, không dùng lại biến mô hình của buổi trước.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

from . import MODEL_ID


def runs_dir() -> Path:
    root = Path(os.environ.get("P01_RUNS", "/content/drive/MyDrive/p01_runs"))
    if not root.parent.exists():
        root = Path("p01_runs")
    root.mkdir(parents=True, exist_ok=True)
    return root


def adapter_path(module: str, name: str = "adapter") -> Path:
    return runs_dir() / module / name


def bnb_4bit_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16)


def compute_dtype():
    """T4 (Turing, sm_75) không có BF16 phần cứng -> float16. GPU Ampere+ -> bfloat16."""
    import torch

    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_id: str = MODEL_ID):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def load_model(model_id: str = MODEL_ID, adapter: str | Path | None = None, four_bit: bool = True, trainable: bool = False):
    """Base sạch (4-bit NF4 mặc định) + adapter LoRA tùy chọn."""
    import torch
    from transformers import AutoModelForCausalLM

    kw = {"device_map": "auto", "dtype": compute_dtype()}
    if four_bit:
        kw["quantization_config"] = bnb_4bit_config()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if trainable and four_bit:
        from peft import prepare_model_for_kbit_training

        # norm -> FP32, bật input grads + gradient checkpointing cho nền 4-bit đóng băng
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    if trainable:
        # Trọng số học được PHẢI là FP32 khi train fp16=True (GradScaler không unscale được grad FP16).
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    else:
        model.eval()
    torch.manual_seed(0)
    return model


def free(*names: str, scope: dict | None = None) -> None:
    """Xóa biến khỏi `scope` (thường là globals()) và giải phóng VRAM."""
    import torch

    if scope is not None:
        for n in names:
            scope.pop(n, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram_gb() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {"allocated": round(torch.cuda.memory_allocated() / 2**30, 2),
            "peak": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "total": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}


def count_trainable(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "pct": round(100 * trainable / max(1, total), 3)}


def set_seed(seed: int = 42) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
