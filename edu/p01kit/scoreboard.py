"""Bảng điểm xuyên suốt (leaderboard) — mỗi buổi ghi thêm dòng, đo trên CÙNG tập test.

Lưu vào Google Drive để không mất khi Colab ngắt phiên.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

COLUMNS = ["time", "module", "variant", "n", "json_valid", "refusal_acc", "citation_precision", "answer_acc",
           "expired_recall", "false_expired_rate", "docs_f1", "strict_acc", "fairness_gap", "acc_unseen_doc",
           "latency_s", "notes"]


def default_path() -> Path:
    root = Path(os.environ.get("P01_RUNS", "/content/drive/MyDrive/p01_runs"))
    if not root.parent.exists():
        root = Path("p01_runs")
    root.mkdir(parents=True, exist_ok=True)
    return root / "leaderboard.csv"


# Output thô của một số lượt chạy cũ được lưu bằng tên riêng trước khi `record(outputs=...)` có mặt.
LEGACY_OUTPUTS = {("M00", "base-zero-shot"): "m00_base_outputs.json", ("M02", "sft-lora-r16"): "m02_outputs.json",
                  ("M06", "sft_best"): "m06_sft_best_outputs.json"}


def outputs_path(module: str, variant: str, root: str | Path | None = None) -> Path:
    root = Path(root) if root else default_path().parent
    safe = re.sub(r"[^\w.+-]", "_", variant)          # Python 3.11: không đặt backslash trong biểu thức f-string
    return root / "outputs" / f"{module}__{safe}.json"


def save_outputs(module: str, variant: str, outputs: list[str], root: str | Path | None = None) -> Path:
    path = outputs_path(module, variant, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(list(outputs), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_outputs(module: str, variant: str, root: str | Path | None = None) -> list[str]:
    """Output thô từng câu của một lượt đã ghi bảng điểm — để chấm lại hoặc so sánh trước/sau."""
    path = outputs_path(module, variant, root)
    if not path.exists() and (module, variant) in LEGACY_OUTPUTS:
        path = path.parent.parent / LEGACY_OUTPUTS[(module, variant)]
    if not path.exists():
        raise FileNotFoundError(f"Chưa có output của {module}/{variant} ({path}). Chạy lại ô đánh giá của buổi đó "
                                f"(scoreboard.record(..., outputs=res['outputs'])) trước khi so sánh.")
    return json.loads(path.read_text(encoding="utf-8"))


def record(module: str, variant: str, summary: dict, path: str | Path | None = None, latency_s: float | None = None,
           notes: str = "", outputs: list[str] | None = None) -> dict:
    """Ghi một dòng bảng điểm. Truyền `outputs` (output thô từng câu test) để so sánh trước/sau ở các buổi sau."""
    path = Path(path) if path else default_path()
    if outputs is not None:
        save_outputs(module, variant, outputs, path.parent)
    row = {c: "" for c in COLUMNS}
    row.update({"time": time.strftime("%Y-%m-%d %H:%M"), "module": module, "variant": variant, "notes": notes,
                "latency_s": "" if latency_s is None else round(latency_s, 3)})
    for c in COLUMNS:
        if c in summary and summary[c] is not None:
            row[c] = summary[c]
    if "acc_unseen_doc" in summary and summary["acc_unseen_doc"] is not None:
        row["acc_unseen_doc"] = summary["acc_unseen_doc"]
    # Chạy lại một ô (ví dụ sau khi Colab ngắt) THAY dòng cũ của cùng (module, variant) thay vì ghi trùng.
    rows = [r for r in load(path) if (r.get("module"), r.get("variant")) != (module, variant)] + [row]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return row


def load(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else default_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def show(path: str | Path | None = None, cols=("module", "variant", "n", "json_valid", "refusal_acc",
                                                 "citation_precision", "expired_recall", "strict_acc",
                                                 "fairness_gap", "acc_unseen_doc")) -> str:
    rows = load(path)

    def fmt(v):
        try:
            return f"{float(v):.3f}" if "." in str(v) else str(v)
        except ValueError:
            return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
    table = "\n".join(lines)
    print(table)
    return table
