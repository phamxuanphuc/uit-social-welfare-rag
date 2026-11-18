"""Tập văn bản lõi đã kiểm hiệu lực — nguồn đáp án chuẩn (gold) cho RAFT và đánh giá P01.

`data/real/core_set.csv` do người soạn và kiểm tay. Mỗi dòng là một văn bản trong kho
`data/real/manifest.csv`, kèm trạng thái hiệu lực **đã kiểm** và bằng chứng:

- `check = primary`: điều khoản hiệu lực của một văn bản mới hơn (có trong kho) bãi bỏ / thay thế văn bản này.
- `check = secondary`: nguồn ngoài kho (cổng chính phủ, văn bản hợp nhất) — ghi URL ở `evidence`.
- `check = portal_only`: chưa kiểm, tạm dùng trạng thái trên cổng (phải trùng với cổng).

Script này kiểm tính hợp lệ của file, liệt kê chỗ lệch giữa cổng và kết quả kiểm, rồi xuất các đoạn
(Điều) của tập lõi với trạng thái đã kiểm ghi đè trạng thái trên cổng.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

STATUSES = ("active", "partial", "expired")
CHECKS = ("primary", "secondary", "portal_only")
FIELDS = ("id", "so_ky_hieu", "group", "status", "check", "effective_until", "superseded_by", "evidence",
          "checked_on", "note")


def load_csv(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def portal_status(m: dict) -> str:
    """Trạng thái trên cổng theo cùng thang `STATUSES` (manifest: validity + partially_expired)."""
    if m["validity"] == "expired":
        return "expired"
    if m["validity"] == "active":
        return "partial" if m.get("partially_expired") == "True" else "active"
    return "unknown"


def _iso(s: str) -> date | None:
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def validate(core: list[dict], manifest: dict[str, dict], today: date) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for n, r in enumerate(core, start=2):                     # dòng 1 là tiêu đề
        where = f"dòng {n} ({r.get('so_ky_hieu', '?')})"
        missing = [k for k in FIELDS if k not in r]
        if missing:
            errors.append(f"{where}: thiếu cột {missing}")
            continue
        if r["id"] in seen:
            errors.append(f"{where}: trùng id {r['id']}")
        seen.add(r["id"])
        m = manifest.get(r["id"])
        if m is None:
            errors.append(f"{where}: id {r['id']} không có trong manifest")
            continue
        if m["so_ky_hieu"].strip() != r["so_ky_hieu"].strip() and r["so_ky_hieu"] not in m["so_ky_hieu"]:
            errors.append(f"{where}: số ký hiệu khác manifest ({m['so_ky_hieu']})")
        if r["status"] not in STATUSES:
            errors.append(f"{where}: status '{r['status']}' không thuộc {STATUSES}")
        if r["check"] not in CHECKS:
            errors.append(f"{where}: check '{r['check']}' không thuộc {CHECKS}")
            continue
        if r["check"] == "portal_only":
            if r["status"] != portal_status(m):
                errors.append(f"{where}: portal_only nhưng status '{r['status']}' khác cổng '{portal_status(m)}'")
            if r["checked_on"]:
                errors.append(f"{where}: portal_only không được có checked_on")
        else:
            if not r["evidence"].strip():
                errors.append(f"{where}: check={r['check']} cần evidence")
            d = _iso(r["checked_on"])
            if d is None or d > today:
                errors.append(f"{where}: checked_on '{r['checked_on']}' sai định dạng hoặc ở tương lai")
            if r["check"] == "secondary" and not r["evidence"].startswith("https://"):
                errors.append(f"{where}: secondary cần evidence là URL https")
        if r["effective_until"]:
            if _iso(r["effective_until"]) is None:
                errors.append(f"{where}: effective_until '{r['effective_until']}' không phải YYYY-MM-DD")
            elif r["status"] == "active":
                errors.append(f"{where}: status active nhưng có effective_until")
        if r["status"] == "expired" and r["check"] != "portal_only" and not r["effective_until"]:
            errors.append(f"{where}: đã kiểm là expired thì cần effective_until")
    return errors


def disagreements(core: list[dict], manifest: dict[str, dict]) -> list[dict]:
    """Văn bản mà kết quả kiểm khác trạng thái trên cổng."""
    out = []
    for r in core:
        m = manifest.get(r["id"])
        if m and r["check"] != "portal_only" and portal_status(m) != r["status"]:
            out.append({"so_ky_hieu": r["so_ky_hieu"], "portal": portal_status(m), "checked": r["status"],
                        "check": r["check"], "superseded_by": r["superseded_by"]})
    return out


def export_articles(core: list[dict], articles_path: str | Path, out_path: str | Path) -> int:
    """Lọc các đoạn của tập lõi; `validity` lấy từ kết quả kiểm, giữ trạng thái cổng ở `validity_portal`."""
    by_id = {r["id"]: r for r in core}
    n = 0
    with open(articles_path, encoding="utf-8") as fi, open(out_path, "w", encoding="utf-8") as fo:
        for line in fi:
            a = json.loads(line)
            r = by_id.get(a["doc_id"])
            if r is None:
                continue
            a.update({"validity_portal": a["validity"],
                      "validity": "expired" if r["status"] == "expired" else "active",
                      "partially_expired": r["status"] == "partial",
                      "validity_check": r["check"], "effective_until": r["effective_until"] or None,
                      "superseded_by": r["superseded_by"] or None, "group": r["group"]})
            fo.write(json.dumps(a, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--core", default="data/real/core_set.csv")
    ap.add_argument("--manifest", default="data/real/manifest.csv")
    ap.add_argument("--articles", default="data/real/welfare_articles.jsonl")
    ap.add_argument("--out", default="data/real/core_articles.jsonl")
    ap.add_argument("--today", default=date.today().isoformat())
    args = ap.parse_args(argv)

    core = load_csv(args.core)
    manifest = {m["id"]: m for m in load_csv(args.manifest)}
    errors = validate(core, manifest, date.fromisoformat(args.today))
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    by_check = {c: sum(r["check"] == c for r in core) for c in CHECKS}
    print(f"{len(core)} văn bản lõi — " + ", ".join(f"{c}: {k}" for c, k in by_check.items()))
    for d in disagreements(core, manifest):
        print(f"  LỆCH CỔNG  {d['so_ky_hieu']}: cổng={d['portal']} → đã kiểm={d['checked']} ({d['check']})"
              + (f", thay bởi {d['superseded_by']}" if d["superseded_by"] else ""))
    if Path(args.articles).exists():
        n = export_articles(core, args.articles, args.out)
        print(f"{n} đoạn → {args.out}")
    else:
        print(f"bỏ qua xuất đoạn: chưa có {args.articles} (chạy data_pipeline.vbpl_welfare trước)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
