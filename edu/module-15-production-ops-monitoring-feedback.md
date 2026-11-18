# Buổi 15 — Vận hành sau phát hành: API, Champion/Challenger, Tải, Drift, Phản hồi & KillSwitch

| Mục | Nội dung |
|---|---|
| **Thời lượng** | 4 giờ (1h lý thuyết · 3h lab) |
| **Đầu vào** | `m13/gguf/p01-Q4_K_M-imat.gguf` (Buổi 13) · `m06_sft_best_outputs.json` (Buổi 06) · số đo "tổng/câu trên CPU 2 luồng" ở Khối 7 Buổi 13 |
| **🎯 Sản phẩm** | (1) **API FastAPI chạy thật** bọc guardrail 4 tầng, `trace_id`, nhật ký kiểm toán, báo lỗi, KillSwitch; (2) **Quyết định cổng phát hành** champion vs challenger (`promote` / `hold` / `reject`) có lý do; (3) **Bảng tải** p50/p95 theo số người dùng đồng thời + **kế hoạch số replica**; (4) **Báo cáo drift** 2 tuần traffic mô phỏng; (5) **Vòng phản hồi**: người dân báo lỗi → 2 cán bộ duyệt (đo κ) → KillSwitch/huấn luyện lại; (6) **`ops_report.json`** + runbook sự cố |
| **Chỉ số đo** | Quyết định cổng & p-value McNemar · p50/p95/p99 (s) · request/s · PSI từng đặc trưng · độ chính xác hệ thống theo tuần · tỉ lệ báo lỗi · Cohen's κ |
| **Chế độ code** | 🧩 Khối gộp — phần lớn logic nằm trong `p01kit/ops.py`, `p01kit/service.py`, `p01kit/traffic.py` (đã có test CPU) |

> ⚠️ Traffic ở buổi này là **mô phỏng tất định** từ tập val/test (`p01kit/traffic.py`) vì lớp học chưa có người dùng thật. Văn bản chính sách vẫn là **giả lập**.

---

## 0. Bức tranh lớn

Buổi 14 kết thúc bằng một hồ sơ phát hành. Ở công ty, **ngày phát hành là ngày bắt đầu phần việc khó nhất**: mô hình gặp câu hỏi không giống tập test, người dùng gõ không dấu, chính sách thay đổi, máy chủ quá tải lúc cao điểm, và lỗi chỉ được biết **vài ngày sau** khi người dân bị trả hồ sơ. Các bài báo kinh điển về hệ thống ML ở quy mô lớn (Sculley và cộng sự, 2015; Breck và cộng sự, 2017) đều nhấn mạnh: phần mô hình chỉ là một ô nhỏ — phần lớn công sức nằm ở phục vụ, giám sát, dữ liệu phản hồi và quy trình phát hành.

Buổi này đi trọn vòng đời đó trên chính hệ thống P01:

```
 offline eval ──► cổng champion/challenger ──► shadow ──► canary ──► 100% traffic
      ▲                                                                   │
      │            huấn luyện lại (CT) ◄── nhãn đã duyệt ◄── báo lỗi ◄─────┤
      │                                                                   ▼
 tập đánh giá mới ◄─────────────────────────── giám sát: SLO · drift · KillSwitch
```

---

## 1. Lý thuyết

### 1.1. Champion / challenger — vì sao "qua ngưỡng" chưa đủ để phát hành
- **Champion** = mô hình đang phục vụ; **challenger** = ứng viên mới. Ngưỡng chốt trước (Buổi 14) là **sàn tuyệt đối**; nhưng một mô hình qua sàn vẫn có thể **kém hơn** bản đang chạy ở một nhóm người dùng.
- Cổng `p01kit.ops.compare_release` ra 3 quyết định:

| Quyết định | Điều kiện |
|---|---|
| `reject` | Trượt ngưỡng chốt trước, **hoặc** kém champion có ý nghĩa thống kê (McNemar, p < 0,05) |
| `hold` | Không kém có ý nghĩa, nhưng **một lát cắt** (giọng đời thường, văn bản mới…) tụt > 5 điểm → người phụ trách phải xem |
| `promote` | Qua mọi ngưỡng, không lát cắt nào tụt quá ngưỡng |

- Phải chấm **hệ thống sẽ deploy** (mô hình + guardrail, đúng artifact GGUF) — `ops.system_scores` — không chấm mô hình trần.
- **Shadow**: challenger nhận bản sao traffic thật nhưng **không** trả kết quả cho người dùng → đo tỉ lệ bất đồng (`ops.shadow_disagreement`) và xem tay các câu bất đồng. **Canary**: trả kết quả challenger cho 5–10% người dùng, theo dõi chỉ số, rồi mới mở rộng. **Rollback**: quay về revision cũ trong vài phút (Azure Container Apps giữ nhiều revision).

### 1.2. SLO, phân vị và năng lực phục vụ
- **SLO** (Service Level Objective) đặt trên **phân vị cao**, ví dụ "p95 < 20 s, tỉ lệ lỗi 5xx < 1%". Trung bình che mất những người chờ lâu nhất.
- **Định luật Little:** số request đang xử lý = tốc độ đến × thời gian phục vụ. Với 30 request/phút và 12 s/request → trung bình 6 request đồng thời. Một replica `llama-server -np 2` xử lý 2 request song song → cần ⌈6 / (2 × 0,7)⌉ = 5 replica nếu giữ mức sử dụng 70% (`ops.capacity_plan`).
- Vì sao **không chạy 100% công suất**: khi mức sử dụng tiến tới 1, hàng đợi (và p95) tăng **phi tuyến**. Khối 4 cho bạn thấy tận mắt.
- Trên CPU, nhiều slot song song **chia nhau** cùng số lõi → mỗi request chậm đi. Phải đo, không suy ra từ số đo 1 request.

### 1.3. Ba loại drift trong P01
| Loại | Định nghĩa | Ví dụ P01 | Phát hiện bằng |
|---|---|---|---|
| **Data drift** | Phân phối đầu vào P(x) đổi | Người dân gõ **không dấu**, dồn về một phường, dán SĐT vào câu hỏi | PSI trên đặc trưng request (không cần nhãn) |
| **Concept drift** | Quan hệ P(y\|x) đổi — cùng câu hỏi, đáp án đúng đã khác | Văn bản **hết hiệu lực**, có văn bản mới thay thế | Nhật ký phê duyệt văn bản (ApprovalAudit) + nhãn trễ |
| **Output drift** | Hành vi mô hình đổi | Tỉ lệ từ chối, tỉ lệ guardrail phải ghi đè tăng vọt | PSI trên `is_refusal`, `override`, `confidence` |

- **PSI** (Population Stability Index) = Σ (a − e)·ln(a/e) trên tỉ lệ từng nhóm. Quy ước ngành: < 0,10 ổn định · 0,10–0,25 theo dõi · > 0,25 thay đổi đáng kể. PSI chỉ báo **có thay đổi**, không báo **chất lượng giảm** — phải nối với nhãn.

### 1.4. Nhãn trễ (DelayedLabel) và hướng dẫn gán nhãn
- Nhãn thật đến **muộn, thưa và lệch**: người dân chỉ báo khi bị trả hồ sơ; có người báo nhầm câu đúng. Vì vậy mọi báo cáo phải được **cán bộ duyệt** theo một **hướng dẫn gán nhãn** chung trước khi tính tỉ lệ lỗi hay dùng để huấn luyện lại.
- Hướng dẫn gán nhãn P01 (`ops.ERROR_LABELS`, ưu tiên lỗi nguy hiểm trước — gán **một** nhãn):

| Nhãn | Khi nào |
|---|---|
| `le_ra_tu_choi` | Câu ngoài phạm vi / không có văn bản phù hợp nhưng hệ thống vẫn trả lời |
| `tu_choi_nham` | Có văn bản đúng trong kho nhưng hệ thống từ chối |
| `sai_van_ban` | Trích văn bản khác văn bản người dân cần |
| `khong_canh_bao_het_han` | Trích đúng văn bản đã hết hạn nhưng không cảnh báo |
| `sai_giay_to_noi_nop` | Đúng văn bản nhưng sai/thiếu giấy tờ hoặc nơi nộp |
| `dung` | Báo nhầm — câu trả lời đúng |

- **Cohen's κ** đo đồng thuận giữa 2 cán bộ đã trừ phần trùng ngẫu nhiên. κ ≥ 0,8 tốt; 0,6–0,8 chấp nhận được; thấp hơn → **sửa hướng dẫn**, chưa dùng nhãn. Ca bất đồng → người thứ ba phân xử (adjudication).

### 1.5. KillSwitch, runbook và huấn luyện lại liên tục (CT)
- KillSwitch (PROJECT-REQUIREMENTS §3, `ops.should_kill`): (1) lọt văn bản hết hạn không cảnh báo **> 0%** — một ca là đủ; (2) tỉ lệ báo lỗi **đã được cán bộ xác nhận** > 15% khi đã đủ ≥ 50 request. Khi bật: API **không gọi mô hình**, trả hướng dẫn liên hệ Bộ phận Một cửa.
- **Runbook** = tài liệu "khi chuông báo kêu thì làm gì": ai trực, kiểm tra gì trước, rollback thế nào, báo cho ai, khi nào được tắt KillSwitch.
- **CT không phải "cứ có dữ liệu mới là train lại"**: phải có điều kiện kích hoạt (drift + chất lượng giảm), chỉ dùng nhãn đã duyệt, và **không bao giờ** để câu hỏi của tập test lọt vào train. Traffic thật nên được tách thành một **tập đánh giá mới** (`test_v2`) đóng băng, giữ tập test cũ để so sánh xuyên phiên bản.

---

## 2. Thực hành

### Khối 1 — Khởi động, llama.cpp, `llama-server` 4 slot trên GPU
```bash
!pip install -q fastapi==0.162.0 uvicorn==0.54.0 httpx==0.28.1
```
```python
import os, sys, json, time, shutil, subprocess, threading
sys.path.insert(0, "/content/uit-social-welfare-rag/edu")
from google.colab import drive; drive.mount("/content/drive")
os.environ["P01_RUNS"] = "/content/drive/MyDrive/p01_runs"
import requests, p01kit
from concurrent.futures import ThreadPoolExecutor
from p01kit import scoreboard
from p01kit.data import load_corpus, load_split, resolve
from p01kit.llamacpp import complete, ANSWER_JSON_SCHEMA
from p01kit.modeling import adapter_path, load_tokenizer
from p01kit.prompts import render_prompt
corpus = load_corpus()
train, val, test = (load_split(s) for s in ("train", "val", "test"))
tok = load_tokenizer()
GGUF = "/content/deploy/p01-Q4_K_M-imat.gguf"
os.makedirs("/content/deploy", exist_ok=True)
if not os.path.exists(GGUF):
    shutil.copy(adapter_path("m13", "gguf") / "p01-Q4_K_M-imat.gguf", GGUF)
from p01kit.colab import restore_llamacpp_build, save_llamacpp_build
CACHED = restore_llamacpp_build()        # bản build đã lưu ở Buổi 13 → không phải biên dịch lại
print("✅ llama.cpp khôi phục từ Drive" if CACHED else "Chưa có bản build trên Drive → ô dưới sẽ biên dịch llama-server")
```
```bash
!test -x /content/llama.cpp/build/bin/llama-server || (git clone --depth 1 --branch b12084 https://github.com/ggml-org/llama.cpp /content/llama.cpp && cmake -S /content/llama.cpp -B /content/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build /content/llama.cpp/build -j $(nproc) --target llama-server)
```
```python
if not CACHED:
    save_llamacpp_build()

def start_server(gguf, ngl=99, slots=4, threads=None, port=8080):
    cmd = ["/content/llama.cpp/build/bin/llama-server", "-m", gguf, "-c", str(2048 * slots), "-np", str(slots),
           "-ngl", str(ngl), "--port", str(port)] + (["-t", str(threads)] if threads else [])
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill(); raise RuntimeError("llama-server không khởi động được")

LLM = start_server(GGUF)
def llm_fn(row):
    return complete(render_prompt(tok, row), json_schema=ANSWER_JSON_SCHEMA)["text"]
```
- **Giải thích:** `-c` của `llama-server` là **tổng** ngữ cảnh chia đều cho các slot → 4 slot × 2048 token (prompt ~840 + đáp án ≤ 384 là vừa). `llm_fn(row)` là "ổ cắm" duy nhất giữa mô hình và phần còn lại của hệ thống — đổi mô hình chỉ cần đổi hàm này.
- ⚠️ Đo chất lượng trên GPU cho nhanh; **độ trễ phải đo trên CPU** (Khối 4) vì môi trường deploy là CPU (ADR 0001).

### Khối 2 — Cổng phát hành: champion (`sft_best`) vs challenger (GGUF đang định deploy)
```python
from p01kit.metrics import summarize
from p01kit.ops import compare_release, system_scores
champ_out = json.load(open(f"{os.environ['P01_RUNS']}/m06_sft_best_outputs.json"))
with ThreadPoolExecutor(4) as ex:                                   # 4 luồng = 4 slot của llama-server
    chall_out = list(ex.map(llm_fn, test))
champion, challenger = system_scores(champ_out, test, corpus), system_scores(chall_out, test, corpus)
rel = compare_release(champion, challenger)
print("Quyết định:", rel["decision"], "|", "; ".join(rel["reasons"]) or "không có lý do chặn")
for k in rel["delta"]:
    print(f"{k:<20} champion={rel['champion'][k]}  challenger={rel['challenger'][k]}  Δ={rel['delta'][k]}")
print("McNemar:", {k: round(v["p_value"], 3) for k, v in rel["mcnemar"].items()})
for s in rel["slices"]:
    print(f"  {s['slice']:<22} n={s['n']:<3} {s['champion']:.3f} → {s['challenger']:.3f} (Δ {s['delta']:+.3f})")
scoreboard.record("M15", "gguf-system-challenger", summarize(challenger))
gate_dir = adapter_path("m15", "gate"); os.makedirs(gate_dir, exist_ok=True)
json.dump({"champion": champion, "challenger": challenger}, open(gate_dir / "scores.json", "w"), ensure_ascii=False)
```
- **Giải thích:** cả hai bên đều được chấm **sau guardrail** trên cùng 100 câu. Champion là `sft_best` vì đó là "bản đang chạy" giả định của lớp; nếu nhóm bạn đã deploy bản khác, dùng output của bản đó.
- 📌 **Kết quả nên thấy:** một quyết định kèm lý do cụ thể. `hold` là kết quả **bình thường** với tập test 100 câu — lát cắt nhỏ dao động mạnh; khi đó người phụ trách (Evaluation & QA Lead) phải xem tay các câu tụt trước khi ký.
- ⚠️ Không "sửa" `max_slice_drop` sau khi thấy kết quả để ra `promote` — giống như đổi ngưỡng chốt trước.

### Khối 3 — Dựng API và kiểm tra khói (smoke test)
```python
import uvicorn
from p01kit.service import create_app
ADMIN_KEY = os.environ.get("P01_ADMIN_KEY") or os.urandom(16).hex()   # khóa demo, không ghi vào notebook
OFFICER = {"X-API-Key": ADMIN_KEY, "X-Role": "Officer"}
app = create_app(llm_fn, corpus, model_version="p01-qwen3-1.7b-Q4_K_M-imat", admin_key=ADMIN_KEY)
api = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning"))
threading.Thread(target=api.run, daemon=True).start(); time.sleep(2)
API = "http://127.0.0.1:8000"
print(requests.get(f"{API}/health").json())
r = requests.post(f"{API}/v1/answer", json={"region": "TP.HCM - Phường Tân Tạo",
                  "question": "SĐT tui 090.123.4567, tui chạy xe ôm té gãy tay thì được hỗ trợ gì, cần giấy gì?"}).json()
print(json.dumps(r, ensure_ascii=False, indent=1))
print("Nhật ký:", json.dumps(app.state.audit[-1], ensure_ascii=False))
```
- 📌 **Kết quả nên thấy:** response có `trace_id`, `confidence`, `residual_risk`, dòng *GIẢ LẬP*; trong nhật ký, `question_masked` chứa `[ĐÃ_CHE_ĐỊNH_DANH]` thay cho số điện thoại và **không** có câu hỏi thô.
- **Giải thích:** đây là đúng luồng của hệ thống thật trong PROJECT-REQUIREMENTS: chặn thiếu địa bàn/lệch địa bàn **trước khi gọi LLM**, truy xuất BM25 lọc kép `region IN [selected, "Toàn quốc"]`, khóa metadata sau khi sinh, ghi nhật ký đã che PII.

### Khối 4 — Kiểm thử tải trên CPU & kế hoạch replica
```python
from p01kit.ops import capacity_plan, latency_report
from p01kit.traffic import simulate_requests
LLM.terminate(); LLM.wait()
LLM = start_server(GGUF, ngl=0, slots=2, threads=2)                 # mô phỏng 1 replica Azure 2 vCPU, 2 slot
def call(r):
    t0 = time.perf_counter()
    requests.post(f"{API}/v1/answer", json={"region": r["region"], "question": r["question"]}, timeout=600).raise_for_status()
    return time.perf_counter() - t0
load = []
for conc in (1, 2, 4):
    reqs = simulate_requests(val, 8, seed=conc)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(conc) as ex:
        lat = list(ex.map(call, reqs))
    load.append({"concurrency": conc, **latency_report(lat), "rps": round(len(reqs) / (time.perf_counter() - t0), 3)})
    print(load[-1])
CPU_SERVICE_S = load[0]["p50"]                                       # thời gian phục vụ 1 request khi không chờ
print(capacity_plan(peak_rpm=30, service_time_s=CPU_SERVICE_S, slots_per_replica=2))
```
- **Giải thích:** đồng thời 1 → 2 → 4 trên **một** replica 2 slot: từ 2 lên 4, request phải **xếp hàng** → p95 tăng mạnh trong khi request/s gần như không tăng. Đó là lý do mở rộng bằng **thêm replica**, không phải dồn thêm request vào một máy.
- 📌 **Kết quả nên thấy:** p50 ở đồng thời 1 xấp xỉ "tổng/câu" đo ở Khối 7 Buổi 13 (cộng thêm vài mili giây của API + BM25). `capacity_plan` cho số replica cần cho giả định 30 request/phút lúc cao điểm — **ghi rõ đây là giả định**, ở công ty con số này đến từ log hoặc dự báo nghiệp vụ.
- ⚠️ Buổi này đo tải trên Colab (CPU khác Azure). Bài tập 2 yêu cầu đo lại trong container đúng cấu hình deploy.

### Khối 5 — Hai tuần traffic mô phỏng & báo cáo drift
```python
from p01kit.ops import drift_report, system_label
from p01kit.traffic import request_features
LLM.terminate(); LLM.wait(); LLM = start_server(GGUF)                # quay lại GPU cho nhanh
by_id = {r["id"]: r for r in val + test}
week1 = simulate_requests(val + test, 160, seed=101)
week2 = simulate_requests(val + test, 160, seed=202, no_diacritics_share=0.5, ood_share=0.2, pii_share=0.15,
                          region_weights={"HCM": 0.7, "DN": 0.15, "CT": 0.15})
def run_week(reqs):
    def one(q):
        res = requests.post(f"{API}/v1/answer", json={"region": q["region"], "question": q["question"]}, timeout=600).json()
        src = by_id[q["source_id"]]
        need = resolve(src["topic"], src["region_code"], corpus.policies)
        return {**request_features(q["region"], q["question"]), "trace_id": res["trace_id"],
                "is_refusal": res["answer"]["is_refusal"], "confidence": res["confidence"],
                "truth": system_label(res["answer"], need)}
    with ThreadPoolExecutor(4) as ex:
        return list(ex.map(one, reqs))
mon1, mon2 = run_week(week1), run_week(week2)
FEATS = ["region", "no_diacritics", "has_pii", "length", "is_refusal", "confidence"]
for f, v in drift_report(mon1, mon2, FEATS).items():
    print(f"{f:<14} PSI={v['psi']:<7} {v['level']:<6} tuần1={v['baseline']}  tuần2={v['window']}")
def acc(m, cond=lambda x: True):
    sel = [x for x in m if cond(x)]
    return round(sum(x["truth"] == "dung" for x in sel) / max(1, len(sel)), 3)
print("Độ chính xác hệ thống — tuần 1:", acc(mon1), "| tuần 2:", acc(mon2),
      "| tuần 2, câu KHÔNG dấu:", acc(mon2, lambda x: x["no_diacritics"]), "| câu có dấu:", acc(mon2, lambda x: not x["no_diacritics"]))
```
- **Giải thích:** tuần 2 "bẻ" phân phối theo 4 hướng hay gặp ngoài đời. PSI cảnh báo **trước** khi có nhãn; nhãn `truth` ở đây chỉ có được vì traffic là mô phỏng — ngoài đời bạn chỉ biết chất lượng qua nhãn trễ (Khối 6).
- 📌 **Kết quả nên thấy:** `no_diacritics` và `has_pii` ở mức **alert**; `region` ít nhất **warn**; `is_refusal` tăng vì tuần 2 có nhiều câu ngoài phạm vi hơn (đúng ra phải từ chối — drift đầu ra **không** đồng nghĩa với lỗi).
- 🧠 **Chẩn đoán theo tầng** khi câu không dấu có độ chính xác thấp hơn: BM25 của `p01kit` đã chuẩn hóa bỏ dấu (`guardrails._fold`) nên tầng truy xuất gần như không bị ảnh hưởng; phần giảm đến từ **mô hình** đọc câu không dấu kém hơn — mô hình chưa từng thấy câu không dấu khi train. Sửa bằng dữ liệu (bài tập 4), và chứng minh chẩn đoán bằng số đo (bài tập 1) trước khi sửa.
- ⚠️ Câu `has_pii` vẫn được trả lời bình thường vì guardrail che số trước khi gửi cho mô hình — kiểm tra lại trong `app.state.audit`.

### Khối 6 — Vòng phản hồi: báo lỗi → 2 cán bộ duyệt → tỉ lệ lỗi → KillSwitch
```python
from collections import Counter
from p01kit.ops import cohen_kappa, should_kill
from p01kit.traffic import simulate_feedback, simulate_reviews
mon = mon1 + mon2
reports = simulate_feedback([m["trace_id"] for m in mon], [m["truth"] for m in mon], seed=7)
print(f"{len(reports)} báo cáo / {len(mon)} lượt; câu sai thật: {sum(m['truth'] != 'dung' for m in mon)}")
officer_a, officer_b = simulate_reviews(reports, seed=1, accuracy=0.92), simulate_reviews(reports, seed=2, accuracy=0.80)
print("Cohen's κ giữa 2 cán bộ:", cohen_kappa(officer_a, officer_b))
final = [a if a == b else r["true_label"] for r, a, b in zip(reports, officer_a, officer_b)]   # bất đồng → người thứ 3 phân xử
print("Nhãn sau phân xử:", Counter(final).most_common())
for r, lab in zip(reports, final):
    requests.post(f"{API}/v1/admin/review", json={"trace_id": r["trace_id"], "label": lab}, headers=OFFICER)
expired_codes = {p.doc_code for p in corpus.policies if not p.is_active}
for name, m in (("tuần 1", mon1), ("tuần 2", mon2)):
    ids = {x["trace_id"] for x in m}
    logs = [a for a in app.state.audit if a["trace_id"] in ids]
    errs = [{"trace_id": r["trace_id"]} for r, lab in zip(reports, final) if lab != "dung" and r["trace_id"] in ids]
    print(name, should_kill(logs, errs, expired_codes))
print("API:", requests.get(f"{API}/health").json()["kill_switch"])
```
- **Giải thích:** chỉ khoảng một nửa số câu sai được báo (người dân không phải lúc nào cũng quay lại), và có vài báo cáo nhầm → tỉ lệ báo lỗi **thấp hơn** tỉ lệ sai thật. Đó là lý do ngưỡng 15% là ngưỡng **trên nhãn đã duyệt** và vẫn cần đánh giá offline định kỳ.
- 📌 **Kết quả nên thấy:** hai cán bộ trùng nhãn ở khoảng 3/4 số báo cáo, nhưng κ chỉ khoảng **0,5–0,75** (đo trên 40 lần mô phỏng với độ chính xác 92% và 80%; vài chục báo cáo nên κ dao động mạnh). κ thấp hơn tỉ lệ trùng vì đã **trừ phần trùng do ngẫu nhiên** — khi một nhãn chiếm đa số, hai người đoán bừa cũng trùng nhiều. Với κ ở mức này, ca bất đồng **phải** qua người thứ ba. Đối chiếu nhãn chiếm đa số với chẩn đoán ở Khối 5. Tuần 2 có thể vượt 15% hoặc không, tùy mô hình của bạn: **ghi lại và giải thích**.
- ⚠️ Khi tự viết mô phỏng, **không dùng chung seed** cho bước sinh nhãn và bước sinh báo cáo: hai chuỗi ngẫu nhiên trùng nhau sẽ tạo tương quan giả (ví dụ không câu sai nào được báo).

### Khối 7 — Chaos test: deploy "quên" guardrail → KillSwitch tự bật
```python
old = [p for p in corpus.policies if not p.is_active]
bad_app = create_app(llm_fn, corpus, model_version="chaos-no-guardrail", admin_key=ADMIN_KEY, enforce_guardrails=False)
from fastapi.testclient import TestClient
chaos = TestClient(bad_app)
for p in old:
    region = corpus.region_name(p.region if p.region != "TQ" else "HCM")
    out = chaos.post("/v1/answer", json={"region": region, "question": f"Cho hỏi {p.benefit_name} cần giấy tờ gì?"}).json()
    print(p.doc_code, "→", out["answer"]["doc_code"], "| cảnh báo hết hạn:", out["answer"]["expired_warning"])
print("KillSwitch:", chaos.get("/health").json()["kill_switch"])
print("Sau khi bật, request mới:", chaos.post("/v1/answer", json={"region": "TP.HCM - Phường Tân Tạo", "question": "Hỗ trợ tiền trọ?"}).json()["answer"]["submission_place"])
```
- **Giải thích:** mô phỏng một lần deploy lỗi cấu hình (tắt tầng 3). Nếu mô hình từng quên cờ hết hạn (Buổi 09 đo `expired_recall`), một ca lọt là đủ để KillSwitch tự bật và mọi request sau đó được chuyển về Bộ phận Một cửa **mà không gọi mô hình**. Nếu mô hình của bạn luôn bật cờ đúng, KillSwitch không bật — đó cũng là một kết quả cần ghi.
- 🧠 Chaos test là thói quen của đội vận hành: **cố tình** làm hỏng một thành phần trong môi trường thử để chứng minh cơ chế an toàn hoạt động, thay vì tin rằng nó hoạt động.

### Khối 8 — Báo cáo vận hành & quyết định huấn luyện lại
```python
drift = drift_report(mon1, mon2, FEATS)
ops_report = {
    "date": time.strftime("%Y-%m-%d"), "model_version": "p01-qwen3-1.7b-Q4_K_M-imat",
    "release_gate": {k: rel[k] for k in ("decision", "reasons", "delta", "slices")},
    "load": load, "capacity_assumption": {"peak_rpm": 30, "service_time_s": CPU_SERVICE_S},
    "drift": {f: v["psi"] for f, v in drift.items()},
    "system_accuracy": {"week1": acc(mon1), "week2": acc(mon2)},
    "feedback": {"reports": len(reports), "kappa": cohen_kappa(officer_a, officer_b), "labels": Counter(final)},
    "ct_trigger": drift["no_diacritics"]["level"] == "alert" and acc(mon2) < acc(mon1) - 0.05,
    "new_eval_set": sorted({by_id[q["source_id"]]["id"] for q, m in zip(week2, mon2) if m["truth"] != "dung"}),
}
out = adapter_path("m15", "ops"); os.makedirs(out, exist_ok=True)
json.dump(ops_report, open(out / "ops_report.json", "w"), ensure_ascii=False, indent=2, default=list)
print(json.dumps({k: ops_report[k] for k in ("release_gate", "drift", "system_accuracy", "ct_trigger")}, ensure_ascii=False)[:1500])
api.should_exit = True; LLM.terminate()
```
- **Giải thích `new_eval_set`:** các câu tuần 2 bị sai đều đến từ **val/test** → **tuyệt đối không** đưa vào train (rò rỉ test). Ở công ty, traffic thật được tách thành tập đánh giá mới đóng băng; dữ liệu train mới phải sinh từ **kịch bản mới** (ví dụ thêm biến thể không dấu cho các câu train — bài tập 4).
- 📌 `ct_trigger = True` khi có drift đầu vào **và** chất lượng giảm > 5 điểm. Drift mà chất lượng không giảm → chỉ theo dõi, không train lại.

---

## 3. So sánh hiệu quả

| Cổng phát hành (sau guardrail, 100 câu test) | answer_acc | strict_acc | refusal_acc | citation_precision | Quyết định |
|---|---|---|---|---|---|
| Champion `sft_best` (Buổi 06) | | | | | — |
| Challenger GGUF Q4_K_M+imatrix | | | | | |

| Tải CPU (1 replica, 2 slot, 2 luồng) | p50 (s) | p95 (s) | request/s |
|---|---|---|---|
| Đồng thời 1 | | | |
| Đồng thời 2 | | | |
| Đồng thời 4 | | | |

| Giám sát | Tuần 1 | Tuần 2 | PSI | Mức |
|---|---|---|---|---|
| Tỉ lệ câu không dấu | | | | |
| Tỉ lệ câu có PII | | | | |
| Tỉ lệ từ chối | | | | |
| Độ chính xác hệ thống | | | — | — |
| Tỉ lệ báo lỗi đã duyệt | | | — | — |

> 📌 **Đọc bảng:** tách bạch **lỗi mô hình** (cổng phát hành, chất lượng trên ngữ cảnh dựng sẵn) với **lỗi hệ thống** (truy xuất, phân phối đầu vào, năng lực phục vụ). Phần lớn sự cố ngoài đời thuộc nhóm thứ hai.

---

## 4. 🏢 Ở công ty thật

- **Cổng phát hành chạy trong CI**, không chạy tay: mỗi pull request đổi mô hình/prompt/guardrail kích hoạt job đánh giá trên tập cố định và chặn merge nếu `compare_release` ra `reject`. Repo này có sẵn ví dụ CI chạy bộ test (`.github/workflows/edu-tests.yml`).
- **Nhật ký và chỉ số đi về hệ thống giám sát tập trung** (Azure Monitor, Grafana/Prometheus, Datadog…) thay vì list trong RAM; cảnh báo gửi về kênh trực (on-call) kèm đường dẫn runbook.
- **Traffic thật có nhãn rất ít** — đội thường dành ngân sách cho một nhóm gán nhãn định kỳ (ví dụ mỗi tuần duyệt ngẫu nhiên 100 lượt), đo κ và cập nhật hướng dẫn gán nhãn theo phiên bản.
- **Quyền riêng tư là yêu cầu pháp lý**, không chỉ là "thói quen tốt": dữ liệu cá nhân của người dân phải có căn cứ xử lý, thời hạn lưu (TTL 30 ngày trong PROJECT-REQUIREMENTS), quyền xóa theo `trace_id`, và người phụ trách. Khi làm thật, hãy hỏi bộ phận pháp chế về quy định bảo vệ dữ liệu cá nhân hiện hành.
- **Rollback được tập dượt** trước ngày cần dùng; thời gian rollback là một chỉ số được theo dõi.

---

## 5. Bài tập

**Bắt buộc**
1. **Chẩn đoán tầng gây lỗi cho câu không dấu:** (a) đo Recall@3 của `bm25_retrieve` trên 100 câu test có dấu và sau `strip_diacritics`; (b) đo mô hình trên **ngữ cảnh dựng sẵn** với câu hỏi đã bỏ dấu. Lỗi nằm ở tầng nào? Thử tắt `_fold` trong `retrieval.tokenize` để thấy điều gì xảy ra nếu tầng truy xuất không chuẩn hóa.
2. **Đóng gói & đo đúng môi trường deploy:** viết `Dockerfile` `python:3.11-slim` gồm FastAPI + `llama-cpp-python` (CPU) + GGUF, chạy với `--cpus 2 --memory 4g`, lặp lại Khối 4. So với số đo Colab.
3. **Cổng phát hành trong CI:** viết script `release_gate.py` đọc 2 file điểm JSON (lưu từ Khối 2), gọi `compare_release` và trả mã thoát ≠ 0 khi `reject`; thêm một job vào workflow GitHub Actions.
4. **Dữ liệu train mới không rò rỉ:** sinh biến thể không dấu **chỉ từ tập train** (`strip_diacritics`), train lại theo cấu hình Buổi 06 và đo trên `new_eval_set` + tập test cũ. Kiểm toán rò rỉ bằng `p01kit.dedup` trước khi train.
5. **Tự gán nhãn:** cùng một bạn trong nhóm, mỗi người gán nhãn độc lập 30 báo cáo (in `question_masked` + câu trả lời) theo bảng 1.4. Tính κ. Chỗ nào hai người hiểu hướng dẫn khác nhau? Sửa hướng dẫn và gán lại.

**Nâng cao**
6. **Canary:** viết một `llm_fn` định tuyến 10% request sang challenger (theo băm `trace_id` để cùng người dùng luôn vào cùng một bản), so chỉ số hai nhánh sau 200 request.
7. Xuất chỉ số `/metrics` dạng Prometheus (đếm request theo `override`, histogram độ trễ) và vẽ dashboard p95 + tỉ lệ từ chối theo thời gian.

---

## 6. Câu hỏi phỏng vấn (kèm đáp án)

1. **Q: Mô hình mới vượt mọi ngưỡng chốt trước. Bạn có deploy luôn không?**
   A: Chưa. So với champion trên cùng tập câu hỏi (McNemar) và theo từng lát cắt; ngưỡng tuyệt đối không bắt được trường hợp mô hình mới tệ hơn ở một nhóm người dùng. Sau đó chạy shadow → canary rồi mới mở 100%, luôn sẵn sàng rollback.
2. **Q: Vì sao SLO đặt trên p95 chứ không phải độ trễ trung bình?**
   A: Trung bình che những người chờ lâu nhất — chính là người bỏ đi hoặc phàn nàn. Phân phối độ trễ LLM lệch phải (câu dài, hàng đợi), nên p95/p99 phản ánh trải nghiệm tệ nhất mà ta cam kết chịu được.
3. **Q: Hệ thống báo PSI cao ở đặc trưng "câu không dấu". Bạn làm gì?**
   A: PSI chỉ báo phân phối đổi. Kiểm tra chất lượng trên lát cắt đó bằng nhãn trễ hoặc gán nhãn mẫu; nếu giảm, đo riêng từng tầng (truy xuất: Recall@k trên câu không dấu; mô hình: độ chính xác trên ngữ cảnh dựng sẵn) và sửa ở đúng tầng — không phải lúc nào cũng là train lại mô hình.
4. **Q: Dùng phản hồi của người dùng để train lại có rủi ro gì?**
   A: Nhãn thưa và lệch (chỉ người bị lỗi mới báo), có báo nhầm, có thể bị lợi dụng; câu hỏi có thể trùng tập test gây rò rỉ; vòng lặp phản hồi làm mô hình chỉ giỏi trên những gì nó đã thấy. Cần duyệt nhãn theo hướng dẫn, đo κ, tách tập đánh giá mới và kiểm toán rò rỉ trước khi train.
5. **Q: Cohen's κ = 0,45 giữa hai người gán nhãn nghĩa là gì?**
   A: Đồng thuận chỉ ở mức trung bình sau khi trừ phần trùng ngẫu nhiên — hướng dẫn gán nhãn đang mơ hồ. Chưa dùng nhãn để train; xem các ca bất đồng, sửa định nghĩa/ví dụ trong hướng dẫn, gán lại, đến khi κ ≥ 0,6–0,8.
6. **Q: Tính số máy cần cho 30 request/phút, mỗi request 12 s, mỗi máy xử lý 2 request song song.**
   A: Định luật Little: tải = 0,5 request/s × 12 s = 6 request đồng thời. Giữ mức sử dụng 70%: ⌈6 / (2 × 0,7)⌉ = 5 máy. Sau đó kiểm chứng bằng load test vì trên CPU, chạy song song làm mỗi request chậm đi.
7. **Q: KillSwitch của dự án hoạt động thế nào và ai được tắt nó?**
   A: Tự bật khi có một ca trích văn bản hết hạn mà không cảnh báo, hoặc tỉ lệ lỗi đã duyệt > 15% trên ≥ 50 request; khi bật, API không gọi mô hình và chuyển người dân về Bộ phận Một cửa. Chỉ Policy Approver & Safety Officer (header `X-Role: Officer` + khóa) được tắt, sau khi rollback/sửa lỗi; tắt sẽ mở cửa sổ giám sát mới.

---

## 7. Checklist cuối buổi
- [ ] Quyết định cổng phát hành (`promote`/`hold`/`reject`) có lý do và p-value; không đổi ngưỡng sau khi xem kết quả.
- [ ] API chạy thật: smoke test có `trace_id`, nhật ký không chứa PII thô, endpoint quản trị từ chối khi thiếu khóa/vai trò.
- [ ] Bảng tải p50/p95 theo đồng thời + số replica cho giả định cao điểm (ghi rõ giả định).
- [ ] Báo cáo drift 2 tuần, chỉ ra đặc trưng nào gây giảm chất lượng và ở tầng nào.
- [ ] Vòng phản hồi: κ giữa 2 cán bộ, nhãn sau phân xử, kết quả `should_kill` từng tuần; chaos test đã chạy.
- [ ] `m15/ops/ops_report.json` trên Drive; một trang runbook (ai trực, kiểm tra gì, rollback ra sao, khi nào tắt KillSwitch).

## Tài liệu tham khảo
- Sculley et al., *Hidden Technical Debt in Machine Learning Systems*, NeurIPS 2015.
- Breck et al., *The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction*, IEEE Big Data 2017.
- Beyer et al. (eds.), *Site Reliability Engineering*, O'Reilly 2016 — chương *Service Level Objectives*.
- Little, J. D. C., *A Proof for the Queuing Formula: L = λW*, Operations Research 1961.
- Cohen, J., *A Coefficient of Agreement for Nominal Scales*, Educational and Psychological Measurement 1960.
