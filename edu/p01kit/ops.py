"""Vận hành sau phát hành (Buổi 15): cổng champion/challenger, độ trễ & năng lực, drift, KillSwitch, đồng thuận nhãn.

Mọi hàm ở đây là Python thuần, chạy trên CPU và được kiểm thử bằng `pytest` — đúng như ở công ty,
nơi các quyết định "có phát hành không / có dừng hệ thống không" phải là code có test, không phải cảm tính.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

from .metrics import gate, summarize
from .stats import mcnemar_exact


# ---------------------------------------------------------------------------
# Độ trễ & năng lực phục vụ
# ---------------------------------------------------------------------------

def percentile(xs: Sequence[float], q: float) -> float:
    """Phân vị q ∈ [0, 100], nội suy tuyến tính (giống numpy.percentile mặc định)."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def latency_report(samples_s: Sequence[float]) -> dict:
    """p50/p95/p99 — SLO ở công ty đặt trên phân vị cao, không đặt trên trung bình."""
    return {"n": len(samples_s), "mean": round(sum(samples_s) / len(samples_s), 3) if samples_s else float("nan"),
            **{f"p{q}": round(percentile(samples_s, q), 3) for q in (50, 95, 99)},
            "max": round(max(samples_s), 3) if samples_s else float("nan")}


def capacity_plan(peak_rpm: float, service_time_s: float, slots_per_replica: int = 1,
                  target_utilization: float = 0.7) -> dict:
    """Số replica cần cho lưu lượng đỉnh (định luật Little: tải = tốc độ đến × thời gian phục vụ).

    `slots_per_replica` = số request một replica xử lý song song (`llama-server -np`).
    Giữ mức sử dụng ≤ 70% để hàng đợi không bùng nổ khi lưu lượng dao động.
    """
    if peak_rpm < 0 or service_time_s <= 0 or slots_per_replica < 1 or not 0 < target_utilization <= 1:
        raise ValueError("Tham số năng lực không hợp lệ")
    offered_load = peak_rpm / 60 * service_time_s            # số request đang xử lý đồng thời trung bình
    replicas = max(1, math.ceil(offered_load / (slots_per_replica * target_utilization)))
    return {"offered_load": round(offered_load, 3), "replicas": replicas,
            "utilization": round(offered_load / (replicas * slots_per_replica), 3)}


# ---------------------------------------------------------------------------
# Cổng phát hành champion vs challenger
# ---------------------------------------------------------------------------

GATE_METRICS = ("refusal_acc", "citation_precision", "answer_acc", "expired_recall", "strict_acc")
SLICE_KEYS = ("style", "slice")


def _by_id(scores: Iterable[dict]) -> dict[str, dict]:
    return {s["id"]: s for s in scores}


def compare_release(champion: Sequence[dict], challenger: Sequence[dict], max_slice_drop: float = 0.05,
                    alpha: float = 0.05, min_slice_n: int = 10) -> dict:
    """So mô hình mới (challenger) với mô hình đang chạy (champion) trên CÙNG tập câu hỏi.

    Đầu vào là danh sách kết quả `metrics.score_one`. Quyết định:
    - `reject`: challenger trượt ngưỡng chốt trước, hoặc kém hơn champion có ý nghĩa thống kê (McNemar).
    - `hold`:   không kém có ý nghĩa nhưng có lát cắt tụt quá `max_slice_drop` → cần người xem xét.
    - `promote`: qua mọi ngưỡng và không lát cắt nào tụt quá ngưỡng.
    Ngưỡng tuyệt đối thôi là chưa đủ: mô hình mới có thể qua ngưỡng mà vẫn tệ hơn bản đang chạy.
    """
    a, b = _by_id(champion), _by_id(challenger)
    if set(a) != set(b):
        raise ValueError("Champion và challenger phải được chấm trên cùng tập câu hỏi")
    ids = sorted(a)
    sa, sb = summarize([a[i] for i in ids], with_ci=False), summarize([b[i] for i in ids], with_ci=False)
    deltas = {k: None if sa.get(k) is None or sb.get(k) is None else round(sb[k] - sa[k], 4) for k in GATE_METRICS}
    tests = {k: mcnemar_exact([bool(a[i][k]) for i in ids], [bool(b[i][k]) for i in ids])
             for k in ("answer_ok", "strict_ok")}

    slices = []
    for key in SLICE_KEYS:
        for val in sorted({a[i][key] for i in ids}):
            sel = [i for i in ids if a[i][key] == val]
            if len(sel) < min_slice_n:
                continue
            acc_a = sum(bool(a[i]["answer_ok"]) for i in sel) / len(sel)
            acc_b = sum(bool(b[i]["answer_ok"]) for i in sel) / len(sel)
            slices.append({"slice": f"{key}={val}", "n": len(sel), "champion": round(acc_a, 4),
                           "challenger": round(acc_b, 4), "delta": round(acc_b - acc_a, 4)})

    reasons = []
    failed_gates = [k for k, (_, ok) in gate(sb).items() if not ok]
    if failed_gates:
        reasons.append("trượt ngưỡng chốt trước: " + ", ".join(failed_gates))
    worse = [k for k, t in tests.items() if t["p_value"] < alpha and t["b"] > t["c"]]
    if worse:
        reasons.append("kém hơn champion có ý nghĩa thống kê: " + ", ".join(worse))
    dropped = [s["slice"] for s in slices if s["delta"] < -max_slice_drop]
    if reasons:
        decision = "reject"
    elif dropped:
        decision = "hold"
        reasons.append(f"lát cắt tụt > {max_slice_drop:.0%}: " + ", ".join(dropped))
    else:
        decision = "promote"
    return {"decision": decision, "reasons": reasons, "champion": {k: sa.get(k) for k in GATE_METRICS},
            "challenger": {k: sb.get(k) for k in GATE_METRICS}, "delta": deltas, "mcnemar": tests, "slices": slices}


def shadow_disagreement(champion_answers: Sequence[dict], challenger_answers: Sequence[dict],
                        keys: Sequence[str] = ("is_refusal", "doc_code", "expired_warning")) -> dict:
    """Chạy bóng (shadow): challenger nhận cùng traffic thật nhưng KHÔNG trả cho người dùng.

    Không có nhãn vàng cho traffic thật → đo tỉ lệ bất đồng theo từng trường để biết nên xem tay câu nào.
    """
    if len(champion_answers) != len(challenger_answers):
        raise ValueError("Hai luồng phải cùng số request")
    n = len(champion_answers)
    per_key = {k: sum((x or {}).get(k) != (y or {}).get(k) for x, y in zip(champion_answers, challenger_answers))
               for k in keys}
    idx = [i for i, (x, y) in enumerate(zip(champion_answers, challenger_answers))
           if any((x or {}).get(k) != (y or {}).get(k) for k in keys)]
    return {"n": n, "disagree_rate": round(len(idx) / n, 4) if n else float("nan"),
            "by_key": {k: round(v / n, 4) if n else float("nan") for k, v in per_key.items()}, "review_idx": idx}


# ---------------------------------------------------------------------------
# Drift: phân phối đầu vào/đầu ra trong cửa sổ giám sát so với lúc phát hành
# ---------------------------------------------------------------------------

PSI_WARN, PSI_ALERT = 0.10, 0.25


def psi(expected: Mapping[str, float], actual: Mapping[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index cho biến phân loại: Σ (a − e) · ln(a / e) trên tỉ lệ từng nhóm.

    Quy ước phổ biến trong ngành: < 0,10 ổn định · 0,10–0,25 cần theo dõi · > 0,25 thay đổi đáng kể.
    Nhóm chỉ xuất hiện ở một phía được làm trơn bằng `eps` để không chia cho 0.
    """
    te, ta = sum(expected.values()), sum(actual.values())
    if te <= 0 or ta <= 0:
        raise ValueError("Phân phối rỗng")
    total = 0.0
    for k in set(expected) | set(actual):
        e = max(expected.get(k, 0) / te, eps)
        a = max(actual.get(k, 0) / ta, eps)
        total += (a - e) * math.log(a / e)
    return round(total, 4)


def drift_report(baseline: Sequence[dict], window: Sequence[dict], features: Sequence[str]) -> dict:
    """PSI cho từng đặc trưng của bản ghi giám sát (ví dụ `region`, `no_diacritics`, `override`)."""
    out = {}
    for f in features:
        e, a = Counter(str(r.get(f)) for r in baseline), Counter(str(r.get(f)) for r in window)
        v = psi(e, a)
        out[f] = {"psi": v, "level": "alert" if v > PSI_ALERT else "warn" if v > PSI_WARN else "ok",
                  "baseline": dict(e.most_common(5)), "window": dict(a.most_common(5))}
    return out


# ---------------------------------------------------------------------------
# KillSwitch (PROJECT-REQUIREMENTS §3) & đồng thuận nhãn của cán bộ
# ---------------------------------------------------------------------------

def should_kill(audit_logs: Sequence[dict], confirmed_errors: Sequence[dict], expired_codes: set[str],
                max_expired_leak: float = 0.0, max_error_rate: float = 0.15, min_requests: int = 50) -> dict:
    """Hai điều kiện dừng khẩn cấp trong một cửa sổ giám sát.

    1. Tỉ lệ câu trả lời trích văn bản HẾT HẠN mà không bật `expired_warning` > `max_expired_leak` (mặc định 0%).
    2. Tỉ lệ báo lỗi = số trace bị báo sai *đã được cán bộ xác nhận* / số request trong cửa sổ > `max_error_rate`.
    Điều kiện 1 kích hoạt ngay cả khi ít dữ liệu (một ca lọt cũng là quá nhiều); điều kiện 2 chỉ xét khi
    đủ `min_requests` request để không dừng hệ thống vì nhiễu của vài báo cáo đầu tiên.
    """
    answered = [r for r in audit_logs if r.get("doc_code") and not r.get("is_refusal")]
    leaks = [r["trace_id"] for r in answered if r["doc_code"] in expired_codes and not r.get("expired_warning")]
    leak_rate = len(leaks) / len(answered) if answered else 0.0
    seen_ids = {r["trace_id"] for r in audit_logs}
    errors = {e["trace_id"] for e in confirmed_errors if e["trace_id"] in seen_ids}
    error_rate = len(errors) / len(audit_logs) if audit_logs else 0.0
    reasons = []
    if leak_rate > max_expired_leak:
        reasons.append(f"lọt văn bản hết hạn không cảnh báo: {len(leaks)} ca ({leak_rate:.1%})")
    if len(audit_logs) >= min_requests and error_rate > max_error_rate:
        reasons.append(f"tỉ lệ báo lỗi đã xác nhận {error_rate:.1%} > {max_error_rate:.0%}")
    return {"kill": bool(reasons), "reasons": reasons, "expired_leak_rate": round(leak_rate, 4),
            "error_rate": round(error_rate, 4), "n_requests": len(audit_logs), "leaked_trace_ids": leaks}


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Độ đồng thuận giữa 2 người gán nhãn, đã trừ phần trùng do ngẫu nhiên.

    Ngành dữ liệu thường coi κ ≥ 0,8 là tốt, 0,6–0,8 là chấp nhận được; thấp hơn → sửa hướng dẫn gán nhãn
    trước khi dùng nhãn để huấn luyện lại.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Hai bộ nhãn phải cùng độ dài và không rỗng")
    n = len(labels_a)
    po = sum(x == y for x, y in zip(labels_a, labels_b)) / n
    ca, cb = Counter(labels_a), Counter(labels_b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


ERROR_LABELS = ("dung", "le_ra_tu_choi", "tu_choi_nham", "khong_canh_bao_het_han", "sai_van_ban", "sai_giay_to_noi_nop")


def error_category(score: dict) -> str:
    """Quy một kết quả `score_one` về đúng MỘT nhãn trong hướng dẫn gán nhãn (ưu tiên lỗi nguy hiểm trước)."""
    if score["strict_ok"]:
        return "dung"
    gold_refusal = score["docs_f1"] is None
    if gold_refusal:
        return "le_ra_tu_choi"
    if not score["cited"]:
        return "tu_choi_nham"
    if not score["citation_ok"]:
        return "sai_van_ban"
    if not score["expired_ok"]:
        return "khong_canh_bao_het_han"
    return "sai_giay_to_noi_nop"
