"""Kiểm thử chất lượng các file bài học: code chạy được về mặt cú pháp, không dùng API đã bị bỏ,
ghim phiên bản, import Unsloth đúng thứ tự, đủ các mục sư phạm bắt buộc."""

import ast
import re
from pathlib import Path

import pytest

EDU = Path(__file__).resolve().parent.parent
MODULES = sorted(EDU.glob("module-*.md"))
BLOCK_RE = re.compile(r"```(\w+)\n(.*?)```", re.S)
import textwrap  # noqa: E402  (khối code nằm trong danh sách Markdown bị thụt lề)

REQUIRED_SECTIONS = ["Kết quả nên thấy", "Bài tập", "Câu hỏi phỏng vấn", "Checklist cuối buổi", "So sánh hiệu quả",
                     "Ở công ty thật", "Mục tiêu buổi học", "Tiêu chí hoàn thành", "Giải thích kiểu Feynman"]
# API đã bị bỏ/đổi tên ở bộ thư viện đã ghim (transformers 5.17, trl 1.15) — không được xuất hiện trong code.
REMOVED_APIS = ["warmup_ratio", "torch_dtype=", "max_prompt_length", "group_by_length", "rpo_alpha",
                "evaluation_strategy", "tokenizer=tok", "dataset_text_field", "from trl import ORPO"]


def blocks(path: Path, lang: str):
    return [textwrap.dedent(code) for lg, code in BLOCK_RE.findall(path.read_text(encoding="utf-8")) if lg == lang]


def test_has_16_modules():
    assert [p.name[:9] for p in MODULES] == [f"module-{i:02d}" for i in range(16)]


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_python_blocks_parse(path):
    for i, code in enumerate(blocks(path, "python")):
        try:
            ast.parse(code)
        except SyntaxError as e:  # pragma: no cover - thông báo lỗi rõ ràng
            pytest.fail(f"{path.name} khối python #{i}: {e}\n{code}")


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_removed_apis(path):
    code = "\n".join(blocks(path, "python"))
    for api in REMOVED_APIS:
        assert api not in code, f"{path.name} dùng API đã bị bỏ: {api}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_pip_installs_are_pinned(path):
    for code in blocks(path, "bash"):
        for line in code.splitlines():
            if "pip install" in line and "-r " not in line:
                pkgs = [t.strip('"\'') for t in line.split("install", 1)[1].split() if not t.startswith("-")]
                for pkg in pkgs:
                    assert "==" in pkg or pkg.startswith(("git+", "/", ".")), f"{path.name}: chưa ghim phiên bản '{pkg}'"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_unsloth_imported_first(path):
    code = "\n".join(line for b in blocks(path, "python") for line in b.splitlines() if not line.lstrip().startswith("#"))
    if "from unsloth import" not in code:
        return
    first_unsloth = code.index("from unsloth import")
    for other in ("from trl", "from transformers", "import transformers", "from peft"):
        if other in code:
            assert first_unsloth < code.index(other), f"{path.name}: '{other}' đứng trước unsloth"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_required_sections(path):
    text = path.read_text(encoding="utf-8")
    for sec in REQUIRED_SECTIONS:
        assert sec in text, f"{path.name} thiếu mục '{sec}'"
    assert text.count("**Q:") >= 5, f"{path.name} cần ít nhất 5 câu hỏi phỏng vấn"


@pytest.mark.parametrize("path", MODULES + [EDU / "README.md"], ids=lambda p: p.name)
def test_no_credentials_in_urls(path):
    """Token ghép vào URL (`https://{token}@github.com/...`) bị in ra khi lệnh lỗi và lưu trong .git/config."""
    for lang in ("python", "bash"):
        for code in blocks(path, lang):
            # `https://{auth}github.com` (biến chứa "token@") hoặc `https://{user}:{token}@host`
            assert not re.search(r"https?://(\{|[^\s/\"']*\{[^}]*\}[^\s/\"']*@)", code), \
                f"{path.name}: không ghép token/biến vào URL — truyền qua biến môi trường của git"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_p01kit_imports_exist(path):
    """Mọi `from p01kit.x import a, b` trong bài học phải trỏ tới tên có thật."""
    import importlib
    import sys

    sys.path.insert(0, str(EDU))
    for code in blocks(path, "python"):
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("p01kit"):
                mod = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(mod, alias.name) or importlib.util.find_spec(f"{node.module}.{alias.name}"), \
                        f"{path.name}: {node.module} không có '{alias.name}'"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_undefined_names(path):
    """Ghép các ô Python theo thứ tự như một notebook: mọi tên phải được import/định nghĩa trước khi dùng."""
    checker = pytest.importorskip("pyflakes.checker")
    from pyflakes.messages import UndefinedName

    tree = ast.parse("\n".join(blocks(path, "python")))
    # `del model; free()` cấp ngoài cùng chỉ để giải phóng VRAM; pyflakes kiểm thân hàm SAU mọi lệnh del,
    # nên giữ lại sẽ báo nhầm cho hàm đã định nghĩa (và dùng xong) trước khi del.
    tree.body = [n for n in tree.body if not isinstance(n, ast.Delete)]
    undefined = sorted({m.message_args[0] for m in checker.Checker(tree).messages if isinstance(m, UndefinedName)})
    assert not undefined, f"{path.name}: dùng tên chưa import/định nghĩa: {undefined}"
