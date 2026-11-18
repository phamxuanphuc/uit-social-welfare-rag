import sys
from pathlib import Path

import pytest

EDU = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EDU))


@pytest.fixture(scope="session")
def corpus():
    from p01kit.data import load_corpus

    return load_corpus()


@pytest.fixture(scope="session")
def splits(corpus):
    from p01kit.data import build_dataset

    return build_dataset(corpus)


class CharTokenizer:
    """Tokenizer giả lập: mỗi ký tự là 1 token, có offset_mapping — đủ để kiểm thử masking trên CPU."""

    eos_token = "<|im_end|>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True, tools=None):
        s = ""
        for m in messages:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        return s


@pytest.fixture
def char_tok():
    return CharTokenizer()
