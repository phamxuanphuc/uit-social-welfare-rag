"""Bảng điểm xuyên suốt (leaderboard) — mỗi buổi ghi thêm dòng, đo trên CÙNG tập test.

Lưu vào Google Drive để không mất khi Colab ngắt phiên.
"""

from __future__ import annotations

import csv
import os
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


def record(module: str, variant: str, summary: dict, path: str | Path | None = None, latency_s: float | None = None,
           notes: str = "") -> dict:
    path = Path(path) if path else default_path()
    row = {c: "" for c in COLUMNS}
    row.update({"time": time.strftime("%Y-%m-%d %H:%M"), "module": module, "variant": variant, "notes": notes,
                "latency_s": "" if latency_s is None else round(latency_s, 3)})
    for c in COLUMNS:
        if c in summary and summary[c] is not None:
            row[c] = summary[c]
    if "acc_unseen_doc" in summary and summary["acc_unseen_doc"] is not None:
        row["acc_unseen_doc"] = summary["acc_unseen_doc"]
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
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
