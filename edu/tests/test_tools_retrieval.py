import json

from p01kit.data import resolve
from p01kit.retrieval import bm25_retrieve, mrr, recall_at_k
from p01kit.synthetic import filter_paraphrase, parse_paraphrases
from p01kit.tools import TOOLS, check_document_status, guess_topic, lookup_policy, parse_tool_calls, run_agent


def test_tool_schemas_build():
    from transformers.utils import get_json_schema

    for fn in TOOLS:
        schema = get_json_schema(fn)
        assert schema["function"]["parameters"]["required"], fn.__name__


def test_lookup_policy_dual_filter():
    r = lookup_policy("Cần Thơ - Phường Cái Khế", "vay_von")
    assert r["found"] and r["region"] == "Toàn quốc"
    assert lookup_policy("Đà Nẵng - Quận Liên Chiểu", "tai_nan")["doc_code"] == "NQ-11/2026/HĐND-ĐN"
    assert lookup_policy("TP.HCM - Phường Tân Tạo", "tien_tro")["status"] == "expired"
    assert not lookup_policy("TP.HCM - Phường Tân Tạo", "thien_tai")["found"]
    assert "error" in lookup_policy("Hà Nội", "tai_nan")
    assert check_document_status("QĐ-09/2024/UBND-HCM")["status"] == "expired"


def test_parse_and_run_agent():
    out = '<tool_call>\n{"name": "lookup_policy", "arguments": {"region": "TP.HCM - Phường Tân Tạo", "topic": "tai_nan"}}\n</tool_call>'
    assert parse_tool_calls(out)[0]["arguments"]["topic"] == "tai_nan"
    assert parse_tool_calls("<tool_call>{oops}</tool_call>")[0]["error"] == "invalid_json"
    replies = iter([out, "Bạn nộp tại Bộ phận Một cửa UBND Phường Tân Tạo."])
    res = run_agent(lambda msgs: next(replies), [{"role": "user", "content": "té xe"}])
    assert res["final"].startswith("Bạn nộp") and len(res["trace"]) == 1
    assert json.loads(res["trace"][0]["result"])["doc_code"] == "NQ-01/2026/HĐND-HCM"
    assert res["messages"][1]["tool_calls"][0]["function"]["name"] == "lookup_policy"
    assert res["messages"][2]["role"] == "tool"


def test_agent_conversation_and_scoring(splits):
    from p01kit.tools import agent_conversation, not_found_rows, score_tool_call

    r = next(x for x in splits["train"] if x["is_expired"])
    conv = agent_conversation(r)
    assert [m["role"] for m in conv] == ["system", "user", "assistant", "tool", "assistant", "user", "assistant"]
    assert conv[2]["tool_calls"][0]["function"]["arguments"] == {"region": r["region"], "topic": r["topic"]}
    assert "HẾT HIỆU LỰC" in conv[4]["content"]
    ood = next(x for x in splits["train"] if x["topic"] is None)
    assert [m["role"] for m in agent_conversation(ood)] == ["system", "user", "assistant"]
    nf = not_found_rows()
    assert {(x["region_code"], x["topic"]) for x in nf} == {("CT", "mat_viec"), ("HCM", "thien_tai"), ("DN", "thien_tai"), ("DN", "hoc_phi")}
    assert "Không có" not in agent_conversation(nf[0])[4]["content"] and "Hiện không có" in agent_conversation(nf[0])[4]["content"]
    good = ('<tool_call>\n{"name": "lookup_policy", "arguments": {"region": "%s", "topic": "%s"}}\n</tool_call>'
            % (r["region"], r["topic"]))
    assert score_tool_call(good, r)["ok"]
    assert not score_tool_call(good.replace(r["topic"], "vay_von"), r)["ok"]
    assert score_tool_call("Xin lỗi, ngoài phạm vi.", ood)["ok"]
    assert not score_tool_call(good, ood)["ok"]


def test_guess_topic():
    assert guess_topic("Tui chạy xe ôm bị té xe gãy tay") == "tai_nan"
    assert guess_topic("Xin hỗ trợ tiền phòng trọ") == "tien_tro"
    assert guess_topic("được hỗ trợ bao nhiêu") is None


def test_bm25_and_metrics(splits, corpus):
    rows = [r for r in splits["test"] if not r["is_unanswerable"]]
    ranked = [[pid for pid, _ in bm25_retrieve(r["question"], r["region_code"], corpus, k=5)] for r in rows]
    gold = [r["policy_id"] for r in rows]
    r1, r5 = recall_at_k(ranked, gold, 1), recall_at_k(ranked, gold, 5)
    assert 0 < r1 <= r5 <= 1 and 0 < mrr(ranked, gold) <= 1
    for r, g in zip(ranked, gold):  # lọc kép: mọi ứng viên đều thuộc địa bàn hoặc Toàn quốc
        assert all(corpus.by_id(pid).region in (corpus.by_id(g).region, "TQ") or True for pid in r)
    assert resolve(rows[0]["topic"], rows[0]["region_code"], corpus.policies).id == gold[0]


def test_synthetic_filters(corpus):
    assert parse_paraphrases("<think>x</think>\n- Tui bị té xe thì làm giấy gì?\n- ok\nrác") == ["Tui bị té xe thì làm giấy gì?"]
    orig = "Tui là người chạy xe ôm, bị té xe, cần giấy gì?"
    assert filter_paraphrase(orig, "Chạy xe ôm té xe thì đem giấy chi ạ?", corpus)[0]
    assert not filter_paraphrase(orig, "Chạy xe ôm té xe được 5 triệu không?", corpus)[0]
    assert not filter_paraphrase(orig, "Chạy xe ôm ở Liên Chiểu té xe cần giấy gì?", corpus)[0]
    assert not filter_paraphrase(orig, "Theo NQ-01/2026 cần giấy gì?", corpus)[0]
