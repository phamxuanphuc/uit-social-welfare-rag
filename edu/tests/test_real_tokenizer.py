"""Kiểm thử với tokenizer Qwen3-1.7B thật (chỉ tải ~11 MB tokenizer, không tải trọng số).
Tự bỏ qua khi không có mạng và chưa có cache."""

import json

import pytest

from p01kit import MODEL_ID
from p01kit.cot import to_mixed_example
from p01kit.prompts import to_prompt_completion
from p01kit.sft import IGNORE_INDEX, encode_assistant_only, encode_prompt_completion
from p01kit.tools import TOOLS


@pytest.fixture(scope="module")
def qwen_tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:  # pragma: no cover - phụ thuộc mạng
        pytest.skip(f"Không tải được tokenizer: {e}")


def test_eos_and_prompt_boundary(qwen_tok, splits):
    assert qwen_tok.eos_token == "<|im_end|>"
    for r in splits["train"][:40] + splits["test"][:20]:
        ex = to_prompt_completion(qwen_tok, r)
        enc = encode_prompt_completion(qwen_tok, ex["prompt"], ex["completion"])
        learned = qwen_tok.decode([t for t in enc["labels"] if t != IGNORE_INDEX])
        assert learned == ex["completion"]


def test_sequence_lengths_fit_budget(qwen_tok, splits):
    from p01kit import MAX_NEW_TOKENS, MAX_SEQ_LEN

    lens, comp = [], []
    for r in splits["train"] + splits["test"]:
        ex = to_prompt_completion(qwen_tok, r)
        lens.append(len(qwen_tok(ex["prompt"] + ex["completion"], add_special_tokens=False)["input_ids"]))
        comp.append(len(qwen_tok(ex["completion"], add_special_tokens=False)["input_ids"]))
    assert max(lens) <= MAX_SEQ_LEN, max(lens)
    assert max(comp) <= MAX_NEW_TOKENS, max(comp)


def test_thinking_examples_boundary(qwen_tok, splits, corpus):
    r = splits["train"][3]
    for think in (True, False):
        ex = to_mixed_example(qwen_tok, r, think=think, corpus=corpus)
        enc = encode_prompt_completion(qwen_tok, ex["prompt"], ex["completion"])
        assert qwen_tok.decode([t for t in enc["labels"] if t != IGNORE_INDEX]) == ex["completion"]


def test_multiturn_tool_masking(qwen_tok):
    conv = [{"role": "system", "content": "Bạn là trợ lý P01."},
            {"role": "user", "content": "Tui chạy xe ôm ở Tân Tạo bị té xe"},
            {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": "lookup_policy", "arguments": {"region": "TP.HCM - Phường Tân Tạo", "topic": "tai_nan"}}}]},
            {"role": "tool", "content": json.dumps({"doc_code": "NQ-01/2026/HĐND-HCM"}, ensure_ascii=False)},
            {"role": "assistant", "content": "Theo NQ-01/2026/HĐND-HCM..."},
            {"role": "user", "content": "Nộp ở đâu?"},
            {"role": "assistant", "content": "Bộ phận Một cửa UBND Phường Tân Tạo."}]
    text = qwen_tok.apply_chat_template(conv, tools=TOOLS, tokenize=False, enable_thinking=False)
    enc = encode_assistant_only(qwen_tok, text)
    learned = qwen_tok.decode([t for t in enc["labels"] if t != IGNORE_INDEX])
    assert "<tool_call>" in learned and "Bộ phận Một cửa" in learned
    assert "<tool_response>" not in learned and "Tui chạy xe ôm" not in learned and "<tools>" not in learned
