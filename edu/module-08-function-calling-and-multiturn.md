# Buổi 8 (Module 8): Function Calling, Tool Use & Hội thoại Nhiều lượt (`Multi-turn`) trên `Qwen3-1.7B`

- **Bối cảnh thực tế (Job Interview Focus):** Hầu hết các job AI Engineer hiện nay không chỉ bắt LLM trả lời 1 lượt, mà làm **Agent gọi công cụ (`<tool_call>`)** — ví dụ: gọi hàm tra cứu database theo địa bàn, gọi API kiểm tra trạng thái văn bản, hoặc giữ ngữ cảnh qua **nhiều lượt hỏi đáp (Multi-turn)** khi người dân hỏi bổ sung (*"Vậy nộp ở đâu?"*).
- **🎯 Sản phẩm cuối buổi:** **Trợ lý Agent `Qwen3-1.7B` Gọi Tool & Hội thoại 2 Lượt** — tự sinh chuẩn thẻ `<tool_call>{"name": "lookup_policy", "arguments": ...}</tool_call>` của Qwen3, nhận kết quả `<tool_response>` và trả lời tự nhiên ở lượt tiếp theo.

---

## 1. Giải thích theo Phương pháp Feynman

1. **`Function Calling` là gì?**
   - **Hình ảnh Feynman:** Mô hình `Qwen3-1.7B` không có kết nối internet hay quyền mở thẳng file SQLite/Qdrant của công ty. Thay vào đó, ta đưa cho nó một **bảng thực đơn công cụ (`tools`)** gồm tên hàm và các tham số cần điền. Khi cần dữ liệu mới, mô hình in ra một phiếu gọi món đặc biệt **`<tool_call>{...}</tool_call>`**; code Python của ta chạy hàm thật lấy kết quả rồi nhét lại vào thẻ **`<tool_response>...</tool_response>`** để AI đọc và trả lời người dân!
2. **Multi-turn Loss Masking (Khác gì Single-turn?):**
   - Trong một cuộc hội thoại 3 lượt (`User 1 -> Assistant 1 -> User 2 -> Assistant 2`), **chỉ các lượt của `Assistant` mới được giữ nhãn để tính Loss**; toàn bộ lượt của `User` và kết quả trả về từ `<tool_response>` đều phải gắn `-100` (bị che đi) vì mô hình không cần học cách đóng vai người dùng hay đóng vai Database!

---

## 2. Thực hành Từng Bước trên Google Colab (Chuẩn Template `<tools>` Gốc của Qwen3)

### Bước 1: Định nghĩa Danh sách Tool Python & Xem `chat_template` Tự Bọc `<tools>` của Qwen3
```python
import json, torch

def lookup_policy(region: str, keyword: str) -> dict:
    """Tra cứu văn bản an sinh theo địa bàn và từ khóa."""
    for p in POLICIES:
        if p["region"] == region and keyword.lower() in (p["content"] + " " + p["doc_code"]).lower():
            return p
    return {"error": "Không tìm thấy văn bản phù hợp"}

tools_schema = [lookup_policy]
sample_msgs = [{"role": "user", "content": "Tôi chạy xe ôm ở Phường Tân Tạo bị tai nạn thì tra quy định nào?"}]

rendered_tool_prompt = tokenizer.apply_chat_template(
    sample_msgs, tools=tools_schema, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
print("Chuỗi <tools> gốc của Qwen3:\n", rendered_tool_prompt[:500])
```

### Bước 2: Đo 🔴 TRƯỚC Fine-tune (Thử gọi Tool trên câu hỏi Multi-turn)
```python
def run_agent_turn(model_obj, messages: list) -> str:
    txt = tokenizer.apply_chat_template(messages, tools=tools_schema, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inp = tokenizer([txt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model_obj.generate(**inp, max_new_tokens=160, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=False)

print("🔴 TRƯỚC FINE-TUNE (Thử sinh <tool_call>):\n", run_agent_turn(model_fast, sample_msgs))
```

### Bước 3: Đóng gói Dữ liệu Multi-turn Tool-Use (`<tool_call>` $\rightarrow$ `<tool_response>` $\rightarrow$ Câu trả lời)
```python
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

multi_turn_conversations = [
    [
        {"role": "system", "content": "Bạn là trợ lý P01. Luôn gọi lookup_policy trước khi trả lời quy định địa phương."},
        {"role": "user", "content": "Chạy xe ôm ở TP.HCM - Phường Tân Tạo bị tai nạn cần văn bản nào?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "lookup_policy", "arguments": {"region": "TP.HCM - Phường Tân Tạo", "keyword": "xe ôm"}}\n</tool_call>'},
        {"role": "tool", "content": json.dumps(POLICIES[0], ensure_ascii=False)},
        {"role": "assistant", "content": "Theo NQ-01/2026/HĐND-HCM, bạn cần chuẩn bị Đơn Mẫu 01-ASXH, CCCD/CT07, xác nhận Tổ dân phố và giấy ra viện."},
        {"role": "user", "content": "Vậy tôi nộp các giấy đó ở đâu?"},
        {"role": "assistant", "content": "Bạn nộp trực tiếp tại Bộ phận Một cửa UBND Phường Tân Tạo."},
    ],
    [
        {"role": "system", "content": "Bạn là trợ lý P01. Luôn gọi lookup_policy trước khi trả lời quy định địa phương."},
        {"role": "user", "content": "Phụ hồ tại Đà Nẵng - Quận Liên Chiểu mất việc muốn học nghề tra ở đâu?"},
        {"role": "assistant", "content": '<tool_call>\n{"name": "lookup_policy", "arguments": {"region": "Đà Nẵng - Quận Liên Chiểu", "keyword": "học nghề"}}\n</tool_call>'},
        {"role": "tool", "content": json.dumps(POLICIES[2], ensure_ascii=False)},
        {"role": "assistant", "content": "Theo NQ-08/2026/HĐND-ĐN, bạn nộp Phiếu Mẫu ĐK-02 và CCCD tại Trung tâm Dịch vụ Việc làm Quận Liên Chiểu."},
        {"role": "user", "content": "Hồ sơ có cần xác nhận tạm trú không?"},
        {"role": "assistant", "content": "Có, bạn cần bổ sung giấy xác nhận tạm trú từ 06 tháng trở lên tại Quận Liên Chiểu."},
    ],
]

ds_tools = Dataset.from_list([
    {"text": tokenizer.apply_chat_template(conv, tools=tools_schema, tokenize=False, add_generation_prompt=False, enable_thinking=False)}
    for conv in multi_turn_conversations
])
```

### Bước 4: Huấn luyện Multi-turn Tool Calling với `train_on_responses_only`
```python
FastLanguageModel.for_training(model_fast)
trainer_m8 = SFTTrainer(
    model=model_fast, processing_class=tokenizer, train_dataset=ds_tools, dataset_text_field="text",
    args=SFTConfig(output_dir="out_m8_tools", num_train_epochs=8, per_device_train_batch_size=1, learning_rate=2e-4, fp16=True, report_to="none"),
)
trainer_m8 = train_on_responses_only(trainer_m8, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
trainer_m8.train()
```
- **Lưu ý kỹ thuật:** `train_on_responses_only` tự động che `-100` cho tất cả các lượt `user` và `tool` trong cả đoạn hội thoại nhiều lượt, chỉ tính Loss trên các lượt `assistant`!

### Bước 5: Thực thi Vòng lặp Agent Thật (Model gọi `<tool_call>` $\rightarrow$ Python chạy `lookup_policy()` $\rightarrow$ Trả lời!)
```python
FastLanguageModel.for_inference(model_fast)
live_msgs = [
    {"role": "system", "content": "Bạn là trợ lý P01. Luôn gọi lookup_policy trước khi trả lời quy định địa phương."},
    {"role": "user", "content": "Tôi bán vé số ở TP.HCM - Phường Tân Tạo bị tai nạn thì xem văn bản nào?"},
]
turn1 = run_agent_turn(model_fast, live_msgs)
print("🟢 Lượt 1 (AI tự sinh Tool Call):\n", turn1)
```

---

## 3. Câu hỏi Phỏng vấn Thực chiến (Kèm Đáp án)
1. **Hỏi:** *Khi fine-tune hội thoại nhiều lượt (Multi-turn + Tool Use), việc tính Loss phải xử lý thế nào?*
   - **Đáp:** Phải áp dụng **multi-turn assistant-only loss masking**: chỉ tính Loss trên các token do `assistant` sinh ra (gồm cả `<tool_call>`), còn `system`, `user` và `<tool_response>` đều phải gắn `label = -100`.
2. **Hỏi:** *Làm sao tránh AI bịa tên hàm hoặc sai kiểu dữ liệu tham số khi gọi Tool?*
   - **Đáp:** Kết hợp truyền đúng JSON Schema định nghĩa hàm vào `tools=`, huấn luyện trên tập mẫu đa dạng cả ca gọi tool lẫn ca trả lời trực tiếp, và xác thực tham số bằng **Pydantic** trước khi thực thi hàm thật.
