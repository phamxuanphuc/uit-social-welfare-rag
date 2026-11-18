import json
import re

import pytest

from p01kit.cot import reasoning_trace, to_mixed_example
from p01kit.dedup import leakage_audit, near_dedup
from p01kit.grounding import check_grounding, hallucination_rate
from p01kit.prompts import SYSTEM_PROMPT, build_messages, pick_fewshot, render_prompt, to_prompt_completion
from p01kit.rewards import reward_citation, reward_expired, reward_format
from p01kit.sft import IGNORE_INDEX, PadCollator, assistant_spans, encode_assistant_only, encode_prompt_completion


def test_prompt_completion_masking(char_tok, splits):
    r = splits["train"][0]
    ex = to_prompt_completion(char_tok, r)
    enc = encode_prompt_completion(char_tok, ex["prompt"], ex["completion"])
    n_prompt = len(ex["prompt"])
    assert all(x == IGNORE_INDEX for x in enc["labels"][:n_prompt])
    assert "".join(chr(x) for x in enc["labels"][n_prompt:]) == ex["completion"]
    assert ex["completion"].endswith("<|im_end|>")
    assert ex["prompt"].endswith("<think>\n\n</think>\n\n")


def test_truncation_guard(char_tok):
    with pytest.raises(ValueError):
        encode_prompt_completion(char_tok, "x" * 50, "y", max_length=10)
    enc = encode_prompt_completion(char_tok, "x" * 5, "y" * 20, max_length=10)
    assert len(enc["input_ids"]) == 10


def test_assistant_only_multiturn(char_tok):
    text = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ1<|im_end|>\n<|im_start|>assistant\nA1<|im_end|>\n"
            "<|im_start|>user\n<tool_response>R</tool_response><|im_end|>\n<|im_start|>assistant\nA2<|im_end|>\n")
    spans = assistant_spans(text)
    assert [text[a:b] for a, b in spans] == ["A1<|im_end|>", "A2<|im_end|>"]
    enc = encode_assistant_only(char_tok, text)
    learned = "".join(chr(t) for t in enc["labels"] if t != IGNORE_INDEX)
    assert learned == "A1<|im_end|>A2<|im_end|>"


def test_pad_collator():
    batch = PadCollator(pad_token_id=0, pad_to_multiple_of=8)([
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1], "labels": [-100, 6, 7]},
        {"input_ids": [5], "attention_mask": [1], "labels": [5]},
    ])
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"][1].tolist() == [5] + [-100] * 7
    assert batch["attention_mask"][1].sum().item() == 1


def test_fewshot_and_prompt(char_tok, splits):
    shots = pick_fewshot(splits["train"], k=3)
    assert shots[1]["is_unanswerable"] and len(shots) == 3
    msgs = build_messages(splits["test"][0], SYSTEM_PROMPT, shots)
    assert len(msgs) == 1 + 2 * 3 + 1
    assert render_prompt(char_tok, splits["test"][0]).count("<|im_start|>") == 3


def test_rewards(splits):
    r = next(x for x in splits["train"] if x["is_expired"])
    good = json.dumps(r["gold"], ensure_ascii=False)
    miss = json.dumps(dict(r["gold"], expired_warning=False), ensure_ascii=False)
    gold = [json.dumps(r["gold"], ensure_ascii=False)] * 3
    comps = [good, miss, "không phải json"]
    assert reward_format(comps) == [1.0, 1.0, -1.0]
    assert reward_citation(comps, gold=gold) == [1.0, 1.0, -1.0]
    assert reward_expired(comps, gold=gold) == [1.0, -2.0, -2.0]
    conv = [[{"role": "assistant", "content": good}]]
    assert reward_format(conv) == [1.0]


def test_grounding(splits):
    r = next(x for x in splits["test"] if not x["is_unanswerable"])
    assert check_grounding(json.dumps(r["gold"], ensure_ascii=False), r["context"])["grounded"]
    fake = dict(r["gold"], required_documents=r["gold"]["required_documents"] + ["Sổ hộ khẩu bản gốc"])
    res = check_grounding(fake, r["context"])
    assert not res["grounded"] and res["hallucinated"] == ["Sổ hộ khẩu bản gốc"]
    assert hallucination_rate([json.dumps(r["gold"], ensure_ascii=False)], [r]) == 0.0


def test_dedup_and_leakage():
    texts = ["Chạy xe ôm bị té xe cần giấy gì?", "Chạy xe ôm bị té xe thì cần giấy gì?", "Xin thẻ BHYT cho con ở đâu?"]
    kept, dropped = near_dedup(texts, threshold=0.8)
    assert kept == [0, 2] and dropped[0][:2] == (1, 0)
    assert leakage_audit(texts[:1], ["Chạy xe ôm bị té xe cần giấy gì ạ?"], 0.8)
    assert not leakage_audit(texts[:1], ["Vay vốn mở quán ở đâu?"], 0.8)


def test_cot_trace_and_mixed_examples(char_tok, splits, corpus):
    r = next(x for x in splits["train"] if x["is_expired"])
    tr = reasoning_trace(r, corpus)
    assert len(re.findall(r"^Văn bản \d", tr, flags=re.M)) == 3 and "expired_warning" in tr
    think = to_mixed_example(char_tok, r, think=True, corpus=corpus)
    plain = to_mixed_example(char_tok, r, think=False, corpus=corpus)
    assert think["completion"].startswith("<think>\n") and think["prompt"].endswith("<|im_start|>assistant\n")
    assert plain["prompt"].endswith("</think>\n\n") and plain["completion"].startswith("{")
