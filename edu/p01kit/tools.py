"""Function calling cho Qwen3: định nghĩa tool, phân tích `<tool_call>`, vòng lặp agent.

Qwen3 dùng định dạng Hermes: mô hình in
    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>
và nhận kết quả trong lượt `tool` (template render thành `<tool_response>...</tool_response>`).
"""

from __future__ import annotations

import json
import re
from typing import Callable

from .data import Corpus, applicable, load_corpus, resolve
from .guardrails import _fold

_CORPUS: Corpus | None = None
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _corpus() -> Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = load_corpus()
    return _CORPUS


def lookup_policy(region: str, topic: str) -> dict:
    """Tra cứu văn bản an sinh xã hội còn áp dụng cho một địa bàn và một chủ đề.

    Args:
        region: Tên địa bàn đầy đủ, ví dụ "TP.HCM - Phường Tân Tạo".
        topic: Mã chủ đề, một trong: tai_nan, bhyt_con, hoc_nghe, mat_viec, tien_tro, thien_tai, hoc_phi, bhxh_tn, vay_von, cao_tuoi, mai_tang.
    """
    c = _corpus()
    code = c.region_code(region)
    if code is None:
        return {"error": f"Địa bàn không hợp lệ: {region}"}
    if topic not in c.topics:
        return {"error": f"Chủ đề không hợp lệ: {topic}"}
    p = resolve(topic, code, c.policies)
    if p is None:
        return {"found": False, "message": "Không có văn bản phù hợp; đề nghị liên hệ cán bộ."}
    return {"found": True, "doc_code": p.doc_code, "status": p.status, "region": c.region_name(p.region),
            "article": p.article, "benefit": p.benefit, "required_documents": list(p.required_documents),
            "submission_place": p.submission_place}


def check_document_status(doc_code: str) -> dict:
    """Kiểm tra trạng thái hiệu lực hiện hành của một văn bản theo mã.

    Args:
        doc_code: Mã văn bản, ví dụ "NQ-01/2026/HĐND-HCM".
    """
    p = _corpus().by_code(doc_code)
    if p is None:
        return {"error": f"Không tìm thấy văn bản {doc_code}"}
    return {"doc_code": p.doc_code, "status": p.status, "effective_date": p.effective_date, "expiry_date": p.expiry_date}


TOOLS = [lookup_policy, check_document_status]
_REGISTRY: dict[str, Callable] = {f.__name__: f for f in TOOLS}


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            calls.append({"name": None, "arguments": None, "error": "invalid_json"})
            continue
        calls.append({"name": obj.get("name"), "arguments": obj.get("arguments") or {}})
    return calls


def execute_tool_call(call: dict) -> str:
    fn = _REGISTRY.get(call.get("name"))
    if fn is None:
        return json.dumps({"error": f"Tool không tồn tại: {call.get('name')}"}, ensure_ascii=False)
    try:
        return json.dumps(fn(**call["arguments"]), ensure_ascii=False)
    except TypeError as e:  # thiếu/sai tham số
        return json.dumps({"error": f"Sai tham số: {e}"}, ensure_ascii=False)


def run_agent(generate_fn: Callable[[list[dict]], str], messages: list[dict], max_steps: int = 3) -> dict:
    """Vòng lặp agent: sinh -> nếu có tool_call thì chạy tool, nối kết quả -> sinh tiếp.

    `generate_fn(messages) -> str` là hàm sinh của mô hình (đã áp chat template với tools=TOOLS).
    """
    msgs = list(messages)
    trace = []
    for _ in range(max_steps):
        out = generate_fn(msgs)
        calls = [c for c in parse_tool_calls(out) if c.get("name")]
        if not calls:
            msgs.append({"role": "assistant", "content": out.strip()})
            return {"final": out.strip(), "messages": msgs, "trace": trace}
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
        for c in calls:
            result = execute_tool_call(c)
            trace.append({"call": c, "result": result})
            msgs.append({"role": "tool", "content": result})
    return {"final": None, "messages": msgs, "trace": trace, "error": "max_steps"}


def guess_topic(question: str) -> str | None:
    """Bộ nhận diện chủ đề bằng từ khóa — baseline 'không cần LLM' để so sánh với agent."""
    q = _fold(question)
    rules = [("tai_nan", ["tai nan", "te xe", "xe tong", "bi thuong", "rot gian"]),
             ("bhyt_con", ["bao hiem y te", "the bao hiem", "bhyt"]),
             ("hoc_nghe", ["hoc nghe", "dao tao nghe"]),
             ("mat_viec", ["mat viec", "nghi viec", "cho nghi"]),
             ("tien_tro", ["nha tro", "phong tro", "o tro", "tien tro", "thue tro", "thue nha", "thue phong", "tien phong"]),
             ("thien_tai", ["ngap", "lu", "trieu cuong", "thien tai"]),
             ("hoc_phi", ["hoc phi", "dong hoc"]),
             ("bhxh_tn", ["bao hiem xa hoi", "luong huu", "bhxh"]),
             ("vay_von", ["vay"]),
             ("cao_tuoi", ["gia", "cao tuoi", "tuoi"]),
             ("mai_tang", ["mat", "qua doi", "hau su", "mai tang"])]
    for topic, kws in rules:
        if any(re.search(rf"\b{re.escape(k)}\b", q) for k in kws):
            return topic
    return None


__all__ = ["TOOLS", "lookup_policy", "check_document_status", "parse_tool_calls", "execute_tool_call",
           "run_agent", "guess_topic", "applicable"]
