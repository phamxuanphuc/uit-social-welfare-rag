# Buổi 11 — Function Calling & Agent nhiều lượt trên Qwen3-1.7B

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 3,5 giờ (45' lý thuyết · 2h45 lab) |
| **Đầu vào** | `p01kit/tools.py` (tool `lookup_policy`, `check_document_status`, vòng lặp agent); kỹ năng masking nhiều lượt |
| **🎯 Sản phẩm** | (1) Bộ chấm **tool-call accuracy**; (2) Adapter **`m11/agent`** gọi tool + hội thoại 2 lượt; (3) **Agent chạy thật end-to-end**: sinh `<tool_call>` → Python thực thi → trả lời → trả lời câu hỏi nối tiếp; (4) So sánh với baseline **không cần LLM** (từ khóa) |
| **Chỉ số đo** | % gọi tool đúng (tên, địa bàn, chủ đề) · % từ chối đúng câu ngoài phạm vi · % trả lời cuối đúng mã văn bản & cảnh báo hết hạn |
| **Chế độ code** | 🧩 Khối gộp |

---

## 0. Bức tranh lớn

Hầu hết vị trí AI Engineer hiện nay đều làm **agent**: LLM không trả lời từ trí nhớ mà **gọi công cụ** (tra database, gọi API) rồi tổng hợp. Với P01, thay vì nhồi 3 văn bản vào prompt (RAFT), agent tự gọi `lookup_policy(region, topic)` — tool này áp dụng **luật nghiệp vụ chính xác** (lọc kép địa bàn, ưu tiên bản còn hiệu lực). LLM chỉ còn 2 việc: **hiểu câu hỏi đời thường thành mã chủ đề** và **diễn đạt kết quả**.

Một câu hỏi kiểm tra tư duy kỹ sư: *có cần LLM cho việc hiểu chủ đề không?* Bộ nhận diện bằng từ khóa `guess_topic` đã đạt **88,5%** trên 96 câu test (đã đo). Agent phải **vượt** con số đó mới đáng chi phí.

---

## 1. Lý thuyết

### 1.1. Định dạng tool của Qwen3 (kiểu Hermes)
- Truyền danh sách hàm Python vào `apply_chat_template(..., tools=TOOLS)`. Transformers đọc **type hint + docstring kiểu Google** (`Args:`) để sinh JSON Schema; thiếu mô tả tham số → lỗi `DocstringParsingException` (lỗi đã gặp ở giáo trình cũ).
- Schema được chèn vào system prompt trong `<tools>…</tools>`; mô hình gọi hàm bằng:
  ```
  <tool_call>
  {"name": "lookup_policy", "arguments": {"region": "...", "topic": "..."}}
  </tool_call>
  ```
- Kết quả tool truyền lại bằng message `role="tool"`; template Qwen3 render nó thành lượt **user** chứa `<tool_response>…</tool_response>`.
- Huấn luyện nên dùng trường chuẩn `tool_calls` của message assistant (không tự gõ chuỗi `<tool_call>` vào `content`) để template render đúng định dạng mô hình đã quen.

### 1.2. Che nhãn cho hội thoại nhiều lượt có tool
Chỉ học **nội dung các lượt assistant** (gồm cả `<tool_call>` và `<|im_end|>`); che system (kể cả khối `<tools>` dài), user, và `<tool_response>`. Mô hình không cần học đóng vai người dùng hay đóng vai database. `p01kit.sft.encode_assistant_only` làm điều này bằng `offset_mapping` — đã có test với tokenizer Qwen3 thật (`tests/test_real_tokenizer.py`).

### 1.3. Ba loại lỗi của agent
| Lỗi | Ví dụ | Phòng tránh |
|---|---|---|
| Không gọi tool khi cần | Trả lời luôn từ trí nhớ → bịa | Dữ liệu có tool call cho mọi câu trong phạm vi |
| Gọi sai tham số | `topic="tai nạn"` thay vì `"tai_nan"`, sai tên địa bàn | Mô tả enum trong docstring; validate tham số trước khi chạy (`execute_tool_call` bắt `TypeError`) |
| Gọi tool khi không nên | Câu hỏi visa cũng tra chính sách | Mẫu từ chối không gọi tool |

---

## 2. Thực hành

### Khối 1 — Khởi động, xem schema tool mà mô hình nhìn thấy
```python
import os, sys, json, time
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
from unsloth import FastLanguageModel
import torch, p01kit
from datasets import Dataset
from transformers.utils import get_json_schema
from p01kit.data import load_split
from p01kit.tools import (TOOLS, AGENT_SYSTEM, FOLLOW_UP, agent_conversation, not_found_rows, score_tool_call,
                          guess_topic, run_agent, lookup_policy)
from p01kit.infer import generate
from p01kit.modeling import adapter_path, free
train, val, test = (load_split(s) for s in ("train", "val", "test"))
print(json.dumps(get_json_schema(TOOLS[0]), ensure_ascii=False, indent=1)[:700])
```
- 📌 **Kết quả nên thấy:** schema có `"required": ["region", "topic"]` và mô tả tiếng Việt lấy từ docstring.

### Khối 2 — Baseline không cần LLM: nhận diện chủ đề bằng từ khóa
```python
topic_rows = [r for r in test if r["topic"]]
kw_acc = sum(guess_topic(r["question"]) == r["topic"] for r in topic_rows) / len(topic_rows)
for style in ("formal", "colloquial"):
    rs = [r for r in topic_rows if r["style"] == style]
    print(style, round(sum(guess_topic(r["question"]) == r["topic"] for r in rs) / len(rs), 3))
print("Từ khóa — chủ đề đúng:", round(kw_acc, 3))
```
- 📌 **Kết quả nên thấy (tất định):** tổng **0,885** trên 96 câu. So sánh 2 giọng để thấy từ khóa yếu ở đâu.

### Khối 3 — Đo base Qwen3 gọi tool zero-shot
```python
def first_turn_prompts(tok, rows):
    return [tok.apply_chat_template(
        [{"role": "system", "content": AGENT_SYSTEM},
         {"role": "user", "content": f"Địa bàn đã chọn: {r['region']}\nCâu hỏi: {r['question']}"}],
        tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in rows]

def tool_eval(model, tok, rows, label):
    outs = generate(model, tok, first_turn_prompts(tok, rows), max_new_tokens=96, batch_size=8)
    sc = [score_tool_call(o, r) for o, r in zip(outs, rows)]
    inscope = [s for s in sc if s["should_call"]]
    res = {"tool_call_acc": round(sum(s["ok"] for s in inscope) / len(inscope), 3),
           "called_rate": round(sum(s["called"] for s in inscope) / len(inscope), 3),
           "topic_acc": round(sum(s.get("topic_ok", False) for s in inscope) / len(inscope), 3),
           "region_acc": round(sum(s.get("region_ok", False) for s in inscope) / len(inscope), 3),
           "ood_ok": [s["ok"] for s in sc if not s["should_call"]]}
    print(label, res)
    return outs, sc, res

base, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=2048, load_in_4bit=True)
FastLanguageModel.for_inference(base)
outs_b, sc_b, res_b = tool_eval(base, tok, test, "base zero-shot")
print(outs_b[0])
del base; free()
```
- 📌 **Điều cần quan sát:** Qwen3 đã được post-train cho tool calling nên `called_rate` thường cao ngay từ đầu; điểm yếu hay nằm ở **`topic_acc`** (mã chủ đề là quy ước riêng của dự án) — đây là chỗ fine-tune phát huy tác dụng.

### Khối 4 — Dựng dữ liệu hội thoại agent & che nhãn nhiều lượt
```python
from p01kit.sft import encode_assistant_only
nf = not_found_rows()
nf_train, nf_test = [r for r in nf if r["id"].endswith("-0")], [r for r in nf if r["id"].endswith("-1")]
convs = [agent_conversation(r) for r in train + nf_train]
texts = [tok.apply_chat_template(c, tools=TOOLS, tokenize=False, enable_thinking=False) for c in convs]
encoded = [encode_assistant_only(tok, t) for t in texts]
ex = encoded[0]
print("Ví dụ — tổng token:", len(ex["labels"]), "| được học:", sum(l != -100 for l in ex["labels"]))
print(tok.decode([t for t in ex["labels"] if t != -100]))
ds_agent = Dataset.from_list(encoded)
```
- 📌 **Kết quả nên thấy:** 276 hội thoại (268 + 8 ca "không có văn bản"), dài tối đa **893** token; phần được học gồm `<tool_call>…</tool_call><|im_end|>`, câu trả lời và câu trả lời nối tiếp — **không** có `<tools>`, câu hỏi người dùng hay `<tool_response>`.
- 🧠 `nf_train` / `nf_test` dùng **khuôn câu khác nhau** cho cùng tổ hợp (địa bàn, chủ đề) — đủ để kiểm tra agent xử lý kết quả rỗng, dù chưa phải tập giữ riêng hoàn hảo (ghi chú điều này trong báo cáo).

### Khối 5 — Train adapter agent (Unsloth + Trainer với dữ liệu đã che nhãn)
```python
from transformers import Trainer, TrainingArguments
from p01kit.sft import PadCollator
model, tok = FastLanguageModel.from_pretrained(p01kit.MODEL_ID, max_seq_length=1024, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0, bias="none", random_state=42,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_gradient_checkpointing="unsloth")
args = TrainingArguments(output_dir=str(adapter_path("m11", "ckpt")), seed=42, per_device_train_batch_size=4,
                         gradient_accumulation_steps=4, num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine",
                         warmup_steps=0.05, optim="paged_adamw_8bit", fp16=True, logging_steps=5, save_strategy="no",
                         report_to=os.environ.get("REPORT_TO", "none"), remove_unused_columns=False)
Trainer(model=model, args=args, train_dataset=ds_agent, data_collator=PadCollator(tok.pad_token_id)).train()
model.save_pretrained(adapter_path("m11", "agent")); tok.save_pretrained(adapter_path("m11", "agent"))
FastLanguageModel.for_inference(model)
outs_a, sc_a, res_a = tool_eval(model, tok, test, "agent fine-tuned")
_, _, res_nf = tool_eval(model, tok, nf_test, "agent — ca không có văn bản")
```
- **Giải thích:** dùng `Trainer` thường (không phải `SFTTrainer`) vì dữ liệu **đã được che nhãn sẵn** theo từng lượt — bạn kiểm soát hoàn toàn token nào được học.

### Khối 6 — Chạy agent THẬT end-to-end (2 lượt)
```python
def make_gen(model, tok):
    def gen(messages):
        p = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return generate(model, tok, [p], max_new_tokens=256, batch_size=1)[0]
    return gen

gen = make_gen(model, tok)
demo = next(r for r in test if r["is_expired"] and r["style"] == "colloquial")
msgs = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": f"Địa bàn đã chọn: {demo['region']}\nCâu hỏi: {demo['question']}"}]
turn1 = run_agent(gen, msgs)
print("🛠️ Tool:", turn1["trace"]); print("💬 Lượt 1:", turn1["final"])
turn2 = run_agent(gen, turn1["messages"] + [{"role": "user", "content": FOLLOW_UP}])
print("💬 Lượt 2:", turn2["final"])
```
- 📌 **Kết quả nên thấy:** trace có đúng 1 lời gọi `lookup_policy` với địa bàn và chủ đề đúng; câu trả lời lượt 1 có cảnh báo "HẾT HIỆU LỰC"; lượt 2 trả lời nơi nộp **dựa trên ngữ cảnh lượt trước**, không gọi tool lại.

### Khối 7 — 📊 Đánh giá end-to-end trên toàn bộ test
```python
def e2e_ok(r, final):
    if r["topic"] is None:
        return final is not None and "ngoài phạm vi" in final
    exp = lookup_policy(r["region"], r["topic"])
    if not exp.get("found"):
        return final is not None and "không có" in final.lower()
    warn_ok = ("HẾT HIỆU LỰC" in (final or "")) == (exp["status"] == "expired")
    return final is not None and exp["doc_code"] in final and warn_ok

t0 = time.perf_counter()
e2e = [e2e_ok(r, run_agent(gen, [{"role": "system", "content": AGENT_SYSTEM},
                                {"role": "user", "content": f"Địa bàn đã chọn: {r['region']}\nCâu hỏi: {r['question']}"}])["final"])
       for r in test]
print(f"End-to-end đúng: {sum(e2e)}/{len(e2e)} | {(time.perf_counter() - t0) / len(test):.2f} s/câu")
```
- **Giải thích:** "đúng" = trích đúng mã văn bản mà luật nghiệp vụ trả về **và** cảnh báo hết hạn khớp trạng thái. Chạy tuần tự (batch 1) vì mỗi câu có 2 lần sinh phụ thuộc nhau.

---

## 3. So sánh hiệu quả

| Phương án | Chủ đề đúng | Gọi tool đúng hoàn toàn | Từ chối đúng câu ngoài phạm vi | End-to-end đúng | s/câu |
|---|---|---|---|---|---|
| Từ khóa `guess_topic` (không LLM) | 0,885 | — | — | — | ~0 |
| Base Qwen3 zero-shot | | | | — | |
| **Agent fine-tuned** | | | | | |

> 📌 **So với RAFT (Buổi 05–09):** agent đẩy phần "đọc–lọc văn bản" sang code (tool), LLM chỉ hiểu ý định. Ưu: chính xác tuyệt đối về luật khi chọn đúng chủ đề, prompt ngắn hơn. Nhược: phụ thuộc chủ đề phải nằm trong danh mục; không xử lý được câu hỏi về **chi tiết nội dung** văn bản. Thảo luận: production nên kết hợp thế nào?

---

## 🏢 Ở công ty thật

- **Tool ở công ty là API thật** (CSDL hồ sơ, hệ thống một cửa) có xác thực, giới hạn tần suất, timeout. Agent phải xử lý lỗi tool, thử lại có giới hạn, và **không** tự thực hiện hành động không đảo ngược được khi chưa có người xác nhận.
- **Quyền tối thiểu**: tool tra cứu chỉ được đọc; mọi lời gọi tool được ghi nhật ký kiểm toán kèm `trace_id`.
- **Đánh giá agent theo nhiệm vụ hoàn thành đầu-cuối** như Khối 7, cộng với kiểm thử an toàn cho đối số của tool (prompt injection đi qua tham số).

---

## 4. Bài tập

**Bắt buộc**
1. **Validate tham số:** viết lớp Pydantic `LookupArgs(region: Literal[...], topic: Literal[...])`, dùng trước `execute_tool_call`; khi sai, trả lỗi về cho mô hình ở lượt `tool` và cho nó **tự sửa** (thêm 10 hội thoại "sửa lỗi" vào dữ liệu train).
2. Thêm hội thoại dùng tool thứ 2 `check_document_status` (người dân hỏi "Quyết định QĐ-09/2024 còn hiệu lực không?"). Đo tool selection accuracy giữa 2 tool.
3. Đo độ chính xác agent riêng cho câu đời thường vs câu chuẩn. Agent có thu hẹp Fairness Gap so với từ khóa?

**Nâng cao**
4. **Gọi nhiều tool song song:** câu hỏi kép ("té xe và muốn làm bảo hiểm cho con") cần 2 lời gọi `lookup_policy`. Tạo 20 mẫu, train và đánh giá.
5. Đánh giá trên một phần `phamhai/Vietnamese-Function-Calling-Test` (Apache-2.0) để đo năng lực tool calling tổng quát **trước/sau** fine-tune — fine-tune hẹp có làm giảm năng lực chung không?

---

## 5. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Qwen3 nhận định nghĩa tool và trả lời gọi tool thế nào?**
   A: Schema JSON của hàm được chèn vào system prompt trong `<tools>`; mô hình sinh `<tool_call>{"name":..., "arguments":...}</tool_call>`; kết quả trả về qua message role `tool`, được render thành `<tool_response>`.
2. **Q: Tính loss thế nào khi fine-tune hội thoại nhiều lượt có tool?**
   A: Chỉ tính trên token của các lượt assistant (kể cả lời gọi tool và token kết thúc lượt); che system, user và tool response bằng -100.
3. **Q: Làm sao giảm lỗi gọi sai tham số?**
   A: Enum/mô tả rõ trong schema, dữ liệu train có tham số chuẩn hóa, validate bằng Pydantic trước khi chạy, trả lỗi để mô hình tự sửa, và đo tool-call accuracy theo từng trường.
4. **Q: Khi nào KHÔNG nên dùng LLM agent?**
   A: Khi luật đơn giản giải quyết đủ tốt (ví dụ từ khóa đạt 88,5%) và chi phí/độ trễ quan trọng; LLM chỉ đáng khi vượt baseline đơn giản có ý nghĩa ở các ca khó (diễn đạt đời thường, câu hỏi kép).
5. **Q: Vì sao dùng trường `tool_calls` thay vì tự viết chuỗi `<tool_call>` vào content khi tạo dữ liệu?**
   A: Để chat template render đúng định dạng mà mô hình đã được post-train, tránh lệch khoảng trắng/xuống dòng giữa train và suy luận.
6. **Q: Agent khác RAG ở điểm nào?**
   A: RAG đưa văn bản truy xuất vào ngữ cảnh để LLM đọc; agent để LLM quyết định gọi công cụ nào với tham số nào, logic nghiệp vụ nằm trong công cụ. Hai cách bổ sung nhau.

---

## 6. Checklist cuối buổi
- [ ] Bảng 3 phương án có tool-call accuracy và end-to-end.
- [ ] Demo agent 2 lượt chạy thật, có trace.
- [ ] `m11/agent` trên Drive.

## Tài liệu tham khảo
- Qwen documentation — *Function Calling* (định dạng Hermes).
- Hugging Face — *Chat Templates: Tool use / function calling*.
- Patil et al., *Gorilla* & Berkeley Function Calling Leaderboard (BFCL).
- Liu et al., *APIGen / xLAM function-calling-60k* (2024).
