"""Chạy lại an toàn khi Colab miễn phí ngắt giữa buổi.

- `once` / `@resumable`: kết quả của mỗi biến thể (dict JSON) được ghi lên Drive ngay khi xong; chạy lại ô
  thì biến thể đã xong được bỏ qua, chỉ chạy phần còn thiếu.
- `resume_point`: checkpoint mới nhất của một lượt train dài để `trainer.train(resume_from_checkpoint=...)`.
"""

from __future__ import annotations

import functools
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from .modeling import runs_dir


def result_path(module: str, key: str) -> Path:
    if not re.fullmatch(r"[\w.\-]+", key):
        raise ValueError(f"Khóa kết quả chỉ gồm chữ, số, '_', '-', '.': {key!r}")
    return runs_dir() / module / "results" / f"{key}.json"


def once(module: str, key: str, fn: Callable[[], Any], force: bool = False) -> Any:
    """Trả kết quả đã lưu nếu có; nếu chưa thì chạy `fn()`, ghi JSON (ghi tạm rồi đổi tên) và trả kết quả.

    Giá trị trả về luôn là bản đã qua JSON (tuple → list) để lần chạy đầu và lần chạy lại giống hệt nhau.
    `force=True` khi bạn CỐ Ý chạy lại (ví dụ đã sửa code) — đừng dùng để "đo lại test cho đẹp".
    """
    path = result_path(module, key)
    if path.exists() and not force:
        print(f"[p01kit] {module}/{key}: đã có kết quả trên Drive → bỏ qua (force=True để chạy lại)")
        return json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(fn(), ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return json.loads(text)


def resumable(module: str):
    """Trang trí hàm chạy một biến thể: tham số đầu tiên là tên biến thể, dùng làm khóa kết quả.

        @resumable("m03")
        def run(name, **lora_kw): ...
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(name: str, *args, force: bool = False, **kwargs):
            return once(module, name, lambda: fn(name, *args, **kwargs), force=force)
        return wrapper
    return deco


def resume_point(output_dir: str | Path) -> str | None:
    """Đường dẫn `checkpoint-N` có N lớn nhất trong `output_dir`, hoặc None nếu chưa có (lần chạy đầu)."""
    d = Path(output_dir)
    if not d.is_dir():
        return None
    ckpts = [(int(m.group(1)), p) for p in d.iterdir()
             if p.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and (p / "trainer_state.json").exists()]
    return str(max(ckpts)[1]) if ckpts else None
