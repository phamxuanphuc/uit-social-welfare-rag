import html as html_lib
from html.parser import HTMLParser
from pathlib import Path
import re

EDU = Path(__file__).resolve().parent.parent
MD_M00 = EDU / "module-00-colab-qwen3-anatomy-baseline.md"
HTML_M00 = EDU / "module-00-colab-qwen3-anatomy-baseline.html"
DIAGRAM_M00 = EDU / "module-00-diagram.html"

VOID_TAGS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}


class TagBalanceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"Mismatched closing </{tag}>; stack={self.stack[-4:]}")
            return
        self.stack.pop()


def test_module00_html_exists_and_balanced():
    assert HTML_M00.exists(), f"Missing {HTML_M00}"
    text = HTML_M00.read_text(encoding="utf-8")
    parser = TagBalanceParser()
    parser.feed(text)
    assert not parser.errors, parser.errors
    assert not parser.stack, f"Unclosed tags: {parser.stack}"


def test_module00_html_is_self_contained():
    text = HTML_M00.read_text(encoding="utf-8")
    assert not re.search(r"<link[^>]+href=[\"']https?://", text, re.I)
    assert not re.search(r"<script[^>]+src=[\"']https?://", text, re.I)
    assert "@import" not in text
    assert "fonts.googleapis.com" not in text


def test_module00_html_has_all_cells_and_code_blocks():
    md_text = MD_M00.read_text(encoding="utf-8")
    html_text = HTML_M00.read_text(encoding="utf-8")

    expected_cells = ["0a", "0b"] + [str(i) for i in range(1, 15)]
    for cell_id in expected_cells:
        assert f'data-cell="{cell_id}"' in html_text, f"Missing cell {cell_id}"

    md_blocks = re.findall(r"```(?:python|bash)\n(.*?)```", md_text, re.S)
    html_code_blocks = [
        html_lib.unescape(block).strip()
        for block in re.findall(r"<pre><code>(.*?)</code></pre>", html_text, re.S)
    ]
    assert len(html_code_blocks) == len(md_blocks)
    for idx, md_code in enumerate(md_blocks):
        assert md_code.strip() in html_code_blocks, f"Code block #{idx} missing or altered in HTML"


def test_module00_html_has_theme_and_interactive_widgets():
    text = HTML_M00.read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in text
    assert 'data-theme="dark"' in text
    assert 'data-theme="light"' in text
    assert "prefers-reduced-motion" in text
    for widget_id in ("lab-xray", "lab-tokens", "lab-vram", "lab-think", "pad-grid"):
        assert f'id="{widget_id}"' in text
    assert len(re.findall(r'class="qa-item"', text)) == 7


def test_module00_diagram_exists_balanced_and_self_contained():
    assert DIAGRAM_M00.exists(), f"Missing {DIAGRAM_M00}"
    text = DIAGRAM_M00.read_text(encoding="utf-8")
    parser = TagBalanceParser()
    parser.feed(text)
    assert not parser.errors, parser.errors
    assert not parser.stack, f"Unclosed tags: {parser.stack}"
    assert not re.search(r"<link[^>]+href=[\"']https?://", text, re.I)
    assert not re.search(r"<script[^>]+src=[\"']https?://", text, re.I)
    assert "@import" not in text
    assert "prefers-color-scheme: dark" in text
    assert "prefers-reduced-motion" in text


def test_module00_diagram_has_all_architecture_nodes_and_views():
    text = DIAGRAM_M00.read_text(encoding="utf-8")
    assert 'id="arch-svg"' in text
    for step in range(6):
        assert f'data-step="{step}"' in text
    for view_id in ("param-bar", "vram-nf4-bar", "xray-mini", "pad-box"):
        assert f'id="{view_id}"' in text
    assert "tie_word_embeddings = true" in text
    assert "M00 / base-zero-shot" in text
