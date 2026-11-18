"""Tiện ích cho Colab T4 MIỄN PHÍ: kiểm tra môi trường, lưu/khôi phục bản build llama.cpp trên Drive.

Bản miễn phí xóa sạch runtime sau mỗi phiên → mọi thứ cài/biên dịch phải làm lại. Hai việc tốn thời gian
nhất là cài thư viện (có thể kéo theo bản `torch` khác) và biên dịch llama.cpp có CUDA trên 2 vCPU.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Mapping

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements-colab.txt"
# Ràng buộc của unsloth==2026.11.3 (đọc từ metadata PyPI): torch>=2.4.0,<2.15.0
TORCH_MIN, TORCH_MAX = (2, 4), (2, 15)
LLAMA_CPP_TAG, CUDA_ARCH = "b12084", "75"
LLAMA_CPP_BUILD = Path("/content/llama.cpp/build")


def canonical(name: str) -> str:
    """Tên gói theo PEP 503: `unsloth_zoo` ≡ `unsloth-zoo` ≡ `Unsloth.Zoo`."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: str | Path = REQUIREMENTS) -> dict[str, str]:
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[canonical(name)] = ver.strip()
    return pins


def version_tuple(v: str) -> tuple[int, ...]:
    """'2.9.1+cu126' → (2, 9, 1)."""
    return tuple(int(x) for x in re.findall(r"\d+", v.split("+", 1)[0])[:3])


def torch_ok(v: str | None) -> bool:
    return v is not None and TORCH_MIN <= version_tuple(v)[:2] < TORCH_MAX


def compare_pins(installed: Mapping[str, str | None], pins: Mapping[str, str]) -> list[str]:
    inst = {canonical(k): v for k, v in installed.items()}
    problems = []
    for name, want in pins.items():
        have = inst.get(name)
        if have is None:
            problems.append(f"{name}: chưa cài (cần {want})")
        elif have != want:
            problems.append(f"{name}: đang là {have}, bản ghim là {want}")
    return problems


def installed_versions(names) -> dict[str, str | None]:
    out = {}
    for n in names:
        try:
            out[canonical(n)] = md.version(n)
        except md.PackageNotFoundError:
            out[canonical(n)] = None
    return out


def gpu_info() -> dict | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "capability": f"{p.major}.{p.minor}", "vram_gb": round(p.total_memory / 2**30, 1),
            "cuda": torch.version.cuda}


def check_env(installed: Mapping[str, str | None] | None = None, gpu: dict | None = None,
              pins: Mapping[str, str] | None = None, verbose: bool = True) -> dict:
    """Chạy đầu mỗi buổi: bản cài có đúng bản ghim không, `torch` có nằm trong khoảng unsloth hỗ trợ không, có GPU không.

    Sai bản ghim thường do quên *Restart session* sau khi cài — lỗi phát sinh sẽ rất khó hiểu ở các ô sau.
    """
    pins = dict(pins or read_pins())
    if installed is None:
        installed = installed_versions(list(pins) + ["torch", "xformers"])
        gpu = gpu_info()
    problems = compare_pins(installed, pins)
    torch_v = {canonical(k): v for k, v in installed.items()}.get("torch")
    if not torch_ok(torch_v):
        problems.append(f"torch {torch_v}: ngoài khoảng unsloth hỗ trợ "
                        f"[{'.'.join(map(str, TORCH_MIN))}, {'.'.join(map(str, TORCH_MAX))})")
    if gpu is None:
        problems.append("không thấy GPU: Runtime → Change runtime type → T4 GPU (bỏ qua nếu ô này chỉ cần CPU)")
    if verbose:
        print("GPU:", gpu or "—")
        print("torch", torch_v, "| xformers", {canonical(k): v for k, v in installed.items()}.get("xformers"))
        print("✅ Môi trường khớp bản ghim" if not problems else "⚠️ " + "\n⚠️ ".join(problems))
    return {"ok": not problems, "problems": problems, "installed": dict(installed), "gpu": gpu}


# ---------------------------------------------------------------------------
# Bản build llama.cpp: biên dịch 1 lần ở Buổi 13, các phiên sau khôi phục từ Drive
# ---------------------------------------------------------------------------

def llamacpp_cache_file(runs: str | Path | None = None, tag: str = LLAMA_CPP_TAG, arch: str = CUDA_ARCH) -> Path:
    if runs is None:
        from .modeling import runs_dir

        runs = runs_dir()
    return Path(runs) / "cache" / f"llama.cpp-{tag}-sm{arch}-bin.tgz"


def save_llamacpp_build(build_dir: str | Path = LLAMA_CPP_BUILD, cache: str | Path | None = None) -> Path:
    """Nén thư mục `build/bin` (chương trình + thư viện .so) lên Drive, chừng vài chục MB."""
    cache = Path(cache) if cache else llamacpp_cache_file()
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        tar.add(Path(build_dir) / "bin", arcname="bin")
    tmp.replace(cache)
    return cache


def restore_llamacpp_build(build_dir: str | Path = LLAMA_CPP_BUILD, cache: str | Path | None = None) -> bool:
    """Giải nén bản build từ Drive về đúng đường dẫn cũ rồi chạy thử `llama-server --version`.

    Trả về False (→ phải biên dịch lại) nếu chưa có bản lưu hoặc bản lưu không chạy được, ví dụ khi Colab đổi
    image và thư viện CUDA không còn khớp. Bản hỏng bị xóa để ô biên dịch (`test -x ... || cmake ...`) chạy lại.
    """
    cache = Path(cache) if cache else llamacpp_cache_file()
    if not cache.exists():
        return False
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(cache, "r:gz") as tar:
        tar.extractall(build_dir, filter="data")
    server = build_dir / "bin" / "llama-server"
    server.chmod(0o755)
    try:
        ok = subprocess.run([str(server), "--version"], capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        shutil.rmtree(build_dir / "bin", ignore_errors=True)
    return ok
