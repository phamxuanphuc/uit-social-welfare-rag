"""So sánh TRƯỚC / SAU fine-tune trên cùng câu hỏi, cùng bộ chấm — không chỉ một con số, mà từng câu.

Ba câu hỏi một báo cáo trước/sau phải trả lời:
1. Chỉ số nào đổi, đổi bao nhiêu, có vượt nhiễu không (Δ, khoảng tin cậy, McNemar)?
2. Đổi ở nhóm nào (giọng chuẩn/đời thường, văn bản đã thấy/chưa thấy)?
3. Đổi ở **câu nào** — câu nào được sửa đúng, câu nào bị làm hỏng, và hỏng kiểu gì?
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .metrics import extract_json, score_one, summarize
from .ops import error_category
from .stats import mcnemar_exact

KEY_METRICS = ("json_valid", "refusal_acc", "citation_precision", "answer_acc", "expired_recall", "docs_f1",
               "strict_acc", "fairness_gap", "acc_unseen_doc")
LOWER_IS_BETTER = {"fairness_gap", "false_expired_rate"}


def _brief(output: str) -> str:
    obj, err = extract_json(output)
    if obj is None:
        return f"❌ {err}: {output.strip()[:80]!r}"
    if obj.get("is_refusal"):
        return f"từ chối (vẫn ghi mã {obj['doc_code']})" if obj.get("doc_code") else "từ chối"
    warn = " ⚠️hết hạn" if obj.get("expired_warning") else ""
    return f"{obj.get('doc_code')}{warn}"


def _gold_brief(row: dict) -> str:
    g = row["gold"]
    if g["is_refusal"]:
        return "từ chối"
    return f"{g['doc_code']}{' ⚠️hết hạn' if g['expired_warning'] else ''}"


def before_after(before: Sequence[str], after: Sequence[str], rows: Sequence[dict], before_name: str = "trước",
                 after_name: str = "sau", k_examples: int = 3) -> dict:
    """Chấm lại cả hai bộ output bằng `score_one` rồi so sánh từng chỉ số, từng lát cắt, từng câu."""
    if not (len(before) == len(after) == len(rows)):
        raise ValueError("Hai bộ output phải cùng số câu với `rows` (cùng tập test, cùng thứ tự)")
    sb = [score_one(o, r) for o, r in zip(before, rows)]
    sa = [score_one(o, r) for o, r in zip(after, rows)]
    mb, ma = summarize(sb), summarize(sa)
    table = []
    for k in KEY_METRICS:
        b, a = mb.get(k), ma.get(k)
        delta = None if b is None or a is None else round(a - b, 4)
        better = None if delta is None or delta == 0 else (delta < 0) == (k in LOWER_IS_BETTER)
        table.append({"metric": k, "before": b, "after": a, "delta": delta, "better": better,
                      "before_ci": mb.get(f"{k}_ci"), "after_ci": ma.get(f"{k}_ci")})
    tests = {k: mcnemar_exact([bool(x[k]) for x in sb], [bool(x[k]) for x in sa]) for k in ("answer_ok", "strict_ok")}
    slices = []
    for key in ("style", "slice"):
        for val in sorted({r[key] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[key] == val]
            acc = lambda s: round(sum(bool(s[i]["answer_ok"]) for i in idx) / len(idx), 4)
            slices.append({"slice": f"{key}={val}", "n": len(idx), "before": acc(sb), "after": acc(sa),
                           "delta": round(acc(sa) - acc(sb), 4)})
    fixed = [i for i in range(len(rows)) if not sb[i]["strict_ok"] and sa[i]["strict_ok"]]
    broken = [i for i in range(len(rows)) if sb[i]["strict_ok"] and not sa[i]["strict_ok"]]

    def example(i):
        return {"id": rows[i]["id"], "style": rows[i]["style"], "question": rows[i]["question"],
                "gold": _gold_brief(rows[i]), "before": _brief(before[i]), "after": _brief(after[i]),
                "error_before": error_category(sb[i]), "error_after": error_category(sa[i])}

    return {"before_name": before_name, "after_name": after_name, "n": len(rows), "table": table,
            "mcnemar": tests, "slices": slices,
            "transitions": {"sửa_đúng": len(fixed), "làm_hỏng": len(broken),
                            "vẫn_đúng": sum(sb[i]["strict_ok"] and sa[i]["strict_ok"] for i in range(len(rows))),
                            "vẫn_sai": sum(not sb[i]["strict_ok"] and not sa[i]["strict_ok"] for i in range(len(rows)))},
            "errors": {"before": dict(Counter(error_category(s) for s in sb)),
                       "after": dict(Counter(error_category(s) for s in sa))},
            "fixed_examples": [example(i) for i in fixed[:k_examples]],
            "broken_examples": [example(i) for i in broken[:k_examples]]}


def _fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def to_markdown(rep: dict) -> str:
    b, a = rep["before_name"], rep["after_name"]
    lines = [f"### {b} → {a} ({rep['n']} câu, cùng bộ chấm)", "", f"| Chỉ số | {b} | {a} | Δ | 95% CI ({a}) |",
             "|---|---|---|---|---|"]
    for r in rep["table"]:
        mark = "" if r["better"] is None else (" ✅" if r["better"] else " 🔻")
        ci = r["after_ci"]
        ci_s = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        d = "—" if r["delta"] is None else f"{r['delta']:+.3f}{mark}"
        lines.append(f"| {r['metric']} | {_fmt(r['before'])} | {_fmt(r['after'])} | {d} | {ci_s} |")
    t = rep["mcnemar"]
    lines += ["", "**McNemar** (chỉ các câu bất đồng mang thông tin): " + " · ".join(
        f"`{k}`: {v['c']} câu sửa đúng / {v['b']} câu làm hỏng, p = {v['p_value']:.3f}" for k, v in t.items()), ""]
    tr = rep["transitions"]
    lines += [f"**Từng câu (strict):** ✅ sửa đúng {tr['sửa_đúng']} · 🔻 làm hỏng {tr['làm_hỏng']} · "
              f"giữ đúng {tr['vẫn_đúng']} · vẫn sai {tr['vẫn_sai']}", "", "| Lát cắt | n | trước | sau | Δ |", "|---|---|---|---|---|"]
    lines += [f"| {s['slice']} | {s['n']} | {s['before']:.3f} | {s['after']:.3f} | {s['delta']:+.3f} |" for s in rep["slices"]]
    eb, ea = rep["errors"]["before"], rep["errors"]["after"]
    lines += ["", "| Loại lỗi | trước | sau |", "|---|---|---|"]
    lines += [f"| {k} | {eb.get(k, 0)} | {ea.get(k, 0)} |" for k in sorted(set(eb) | set(ea))]
    for title, key in (("✅ Ví dụ được sửa đúng", "fixed_examples"), ("🔻 Ví dụ bị làm hỏng", "broken_examples")):
        if rep[key]:
            lines += ["", f"**{title}**", "", "| Câu hỏi | Đáp án đúng | trước | sau |", "|---|---|---|---|"]
            lines += [f"| {e['question'][:110]} | {e['gold']} | {e['before']} | {e['after']} |" for e in rep[key]]
    return "\n".join(lines)


def show(rep: dict) -> str:
    """Hiển thị đẹp trong Colab/Jupyter (Markdown), in chữ nếu chạy ngoài notebook."""
    md = to_markdown(rep)
    try:
        from IPython.display import Markdown, display

        display(Markdown(md))
    except ImportError:
        print(md)
    return md


def compare_runs(before: tuple[str, str], after: tuple[str, str], rows: Sequence[dict], k_examples: int = 3,
                 display: bool = True) -> dict:
    """So 2 lượt đã ghi bảng điểm kèm output, ví dụ `compare_runs(("M00", "base-zero-shot"), ("M06", "sft_best"), test)`."""
    from .scoreboard import load_outputs

    rep = before_after(load_outputs(*before), load_outputs(*after), rows, "/".join(before), "/".join(after), k_examples)
    rep["markdown"] = show(rep) if display else to_markdown(rep)
    return rep


def journey(runs: Sequence[tuple[str, str]], rows: Sequence[dict], metrics: Sequence[str] = KEY_METRICS,
            display: bool = True) -> str:
    """Bảng "hành trình" nhiều mốc trên cùng tập test; mỗi mốc kèm p-value McNemar so với mốc trước nó."""
    from .scoreboard import load_outputs

    head = "| Mốc | " + " | ".join(metrics) + " | p (so với mốc trước) |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    prev = None
    for run in runs:
        scores = [score_one(o, r) for o, r in zip(load_outputs(*run), rows, strict=True)]
        s = summarize(scores, with_ci=False)
        p = "—" if prev is None else f"{mcnemar_exact([x['strict_ok'] for x in prev], [x['strict_ok'] for x in scores])['p_value']:.3f}"
        lines.append(f"| {'/'.join(run)} | " + " | ".join(_fmt(s.get(m)) for m in metrics) + f" | {p} |")
        prev = scores
    md = "\n".join(lines)
    if display:
        try:
            from IPython.display import Markdown, display as _d

            _d(Markdown(md))
        except ImportError:
            print(md)
    return md
