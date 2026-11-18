import os
import stat

from p01kit.colab import (canonical, check_env, compare_pins, read_pins, restore_llamacpp_build,
                          save_llamacpp_build, torch_ok, version_tuple)

PINS = {"transformers": "5.17.0", "unsloth-zoo": "2026.11.3"}


def test_requirements_pin_unsloth_zoo_with_unsloth():
    pins = read_pins()
    assert pins["unsloth-zoo"] == pins["unsloth"], "unsloth_zoo phải ghim cùng đợt phát hành với unsloth"
    assert {"transformers", "trl", "peft", "datasets", "bitsandbytes", "accelerate"} <= set(pins)


def test_version_parsing_and_torch_range():
    assert version_tuple("2.9.1+cu126") == (2, 9, 1)
    assert canonical("Unsloth_Zoo") == canonical("unsloth-zoo") == "unsloth-zoo"
    assert torch_ok("2.4.0") and torch_ok("2.14.9+cu128")
    assert not torch_ok("2.15.0") and not torch_ok("2.3.1") and not torch_ok(None)


def test_compare_pins_reports_missing_and_mismatch():
    probs = compare_pins({"transformers": "5.16.0"}, PINS)
    assert any("5.16.0" in p for p in probs) and any("unsloth-zoo: chưa cài" in p for p in probs)
    assert compare_pins({"transformers": "5.17.0", "unsloth_zoo": "2026.11.3"}, PINS) == []


def test_check_env_ok_and_problems():
    gpu = {"name": "Tesla T4", "capability": "7.5", "vram_gb": 14.7, "cuda": "12.6"}
    good = {"transformers": "5.17.0", "unsloth_zoo": "2026.11.3", "torch": "2.9.0+cu126", "xformers": None}
    assert check_env(good, gpu, PINS, verbose=False)["ok"]
    bad = check_env({**good, "torch": "2.15.1"}, None, PINS, verbose=False)
    assert not bad["ok"] and any("torch 2.15.1" in p for p in bad["problems"])
    assert any("GPU" in p for p in bad["problems"])


def _fake_build(root, script):
    b = root / "build" / "bin"
    b.mkdir(parents=True)
    exe = b / "llama-server"
    exe.write_text(script)
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    (b / "libggml.so").write_bytes(b"\x7fELF-fake")
    return root / "build"


def test_llamacpp_build_roundtrip(tmp_path):
    build = _fake_build(tmp_path / "a", "#!/bin/sh\necho 'version: b12084'\n")
    cache = save_llamacpp_build(build, tmp_path / "drive" / "cache" / "llama.tgz")
    assert cache.exists() and not cache.with_suffix(".tmp").exists()
    target = tmp_path / "b" / "build"
    assert restore_llamacpp_build(target, cache)
    assert (target / "bin" / "libggml.so").exists() and os.access(target / "bin" / "llama-server", os.X_OK)


def test_restore_fails_without_cache_or_with_broken_binary(tmp_path):
    assert not restore_llamacpp_build(tmp_path / "x", tmp_path / "khong-co.tgz")
    build = _fake_build(tmp_path / "a", "#!/bin/sh\nexit 127\n")          # ví dụ: thiếu libcudart sau khi Colab đổi image
    cache = save_llamacpp_build(build, tmp_path / "c.tgz")
    assert not restore_llamacpp_build(tmp_path / "b", cache)
    assert not (tmp_path / "b" / "bin").exists()                         # xóa bản hỏng → ô biên dịch sẽ build lại


def test_export_results_keeps_small_result_files_only(tmp_path):
    import zipfile

    from p01kit.colab import export_results

    runs = tmp_path / "p01_runs"
    for rel, data in {"leaderboard.csv": "module,variant\n", "outputs/M00__base-zero-shot.json": "[]",
                      "m03/results/V2_dora_r16.json": "{}", "m06/sft_best/selection.json": "{}",
                      "m06/sft_best/adapter_model.safetensors": "x" * 1000, "m13/gguf/p01-Q4_K_M-imat.gguf": "x",
                      "m09/ckpt/checkpoint-16/optimizer.pt": "x"}.items():
        (runs / rel).parent.mkdir(parents=True, exist_ok=True)
        (runs / rel).write_text(data)
    names = set(zipfile.ZipFile(export_results(runs, tmp_path / "out.zip")).namelist())
    assert {"leaderboard.csv", "outputs/M00__base-zero-shot.json", "m03/results/V2_dora_r16.json",
            "m06/sft_best/selection.json", "ENV.json"} == names


def test_export_results_without_any_run(tmp_path):
    import pytest

    from p01kit.colab import export_results

    with pytest.raises(FileNotFoundError):
        export_results(tmp_path, tmp_path / "x.zip")
