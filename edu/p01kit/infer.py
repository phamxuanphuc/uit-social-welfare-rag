"""Sinh câu trả lời theo lô (batched generation) và chạy đánh giá trên tập test.

Tối ưu chính so với bản cũ (hỏi từng câu một): gom lô + left-padding -> nhanh hơn nhiều lần
trên T4, vì GPU xử lý song song nhiều chuỗi trong cùng một bước sinh token.
"""

from __future__ import annotations

import time

from .metrics import score_one, summarize
from .prompts import SYSTEM_PROMPT, render_prompt


def generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 384, batch_size: int = 8,
             do_sample: bool = False, temperature: float = 0.7, top_p: float = 0.8, top_k: int = 20,
             seed: int | None = None) -> list[str]:
    """Sinh theo lô. Greedy mặc định (đánh giá tất định); bật do_sample để đo độ ổn định.

    Tham số sampling mặc định theo khuyến nghị của Qwen3 cho chế độ non-thinking (T=0.7, p=0.8, k=20).
    """
    import torch

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # BẮT BUỘC khi sinh theo lô với decoder-only
    outs: list[str] = []
    if seed is not None:
        torch.manual_seed(seed)
    gen_kw = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.pad_token_id, "do_sample": do_sample}
    if do_sample:
        gen_kw.update(temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        gen_kw.update(temperature=None, top_p=None, top_k=None)
    try:
        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.inference_mode():
                ids = model.generate(**enc, **gen_kw)
            outs += tokenizer.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        tokenizer.padding_side = old_side
    return outs


def evaluate(model, tokenizer, rows: list[dict], system_prompt: str = SYSTEM_PROMPT, fewshot: list[dict] | None = None,
             enable_thinking: bool = False, **gen_kw) -> dict:
    """Chạy mô hình trên `rows`, chấm điểm, trả về {outputs, scores, summary, seconds}."""
    prompts = [render_prompt(tokenizer, r, system_prompt, fewshot, enable_thinking=enable_thinking) for r in rows]
    t0 = time.perf_counter()
    outputs = generate(model, tokenizer, prompts, **gen_kw)
    secs = time.perf_counter() - t0
    scores = [score_one(o, r) for o, r in zip(outputs, rows)]
    summary = summarize(scores)
    summary["seconds"] = round(secs, 1)
    summary["sec_per_row"] = round(secs / max(1, len(rows)), 3)
    return {"outputs": outputs, "scores": scores, "summary": summary, "seconds": secs}


def show_errors(result: dict, rows: list[dict], k: int = 5) -> None:
    """In k ca sai để phân tích lỗi (error analysis) — kỹ năng phỏng vấn rất hay bị hỏi."""
    shown = 0
    for out, s, r in zip(result["outputs"], result["scores"], rows):
        if s["answer_ok"] and s.get("strict_ok", True):
            continue
        print(f"--- {r['id']} | {r['style']} | gold_refusal={r['gold']['is_refusal']} gold_doc={r['gold']['doc_code']}")
        print("Q:", r["question"])
        print("OUT:", out[:400].replace("\n", " "))
        shown += 1
        if shown >= k:
            break
