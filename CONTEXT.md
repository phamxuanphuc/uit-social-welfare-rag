# Social Welfare RAG (SDG 1.3 - P01)

Hệ thống trợ lý tra cứu quyền lợi an sinh và hồ sơ thủ tục cho người lao động phi chính thức dựa trên văn bản công khai (giả lập), tuân thủ khung Responsible AI, NIST AI RMF và MLOps.

## Language

**PolicyDocument**:
Văn bản quy phạm hoặc hướng dẫn chương trình an sinh xã hội có gắn mã văn bản, điều khoản, địa bàn, ngày hiệu lực, phiên bản và trạng thái.
_Avoid_: Bài viết, tài liệu chung

**Jurisdiction**:
Địa bàn hành chính giả lập (`region`) nơi áp dụng **PolicyDocument**, hoặc cấp `"Toàn quốc"` cho quy định dùng chung.
_Avoid_: Khu vực, chỗ ở

**InformalWorker**:
Người lao động phi chính thức (xe ôm, bán vé số, phụ hồ, bán hàng rong) sử dụng ngôn ngữ đời thường để tra cứu quyền lợi.
_Avoid_: Khách hàng, user

**ColloquialMapping**:
Bảng ánh xạ cụm từ đời thường của **InformalWorker** sang thuật ngữ pháp lý chuẩn nhằm giảm thiên lệch truy xuất.
_Avoid_: Từ điển

**ValidityStatus**:
Trạng thái hiệu lực của **PolicyDocument**, chỉ nhận một trong hai giá trị `"active"` hoặc `"expired"`.
_Avoid_: Trạng thái chung

**VerificationReasoning**:
Khối kiểm chứng bắt buộc nằm ở trường đầu tiên của **ConstrainedAnswer** để đối chiếu địa bàn, hiệu lực và độ đủ của ngữ cảnh trước khi sinh kết quả.
_Avoid_: Giải thích tự do

**ConstrainedAnswer**:
Phản hồi JSON tuân thủ schema cố định gồm danh sách giấy tờ, nơi nộp, hậu quả nộp thiếu, trích dẫn, cảnh báo hết hiệu lực và đường chuyển cán bộ.
_Avoid_: Câu trả lời văn bản thuần

**CrossRegionConflict**:
Tình huống người dùng chọn một **Jurisdiction** trên giao diện nhưng nội dung câu hỏi lại nêu đích danh một địa bàn khác; hệ thống phải cảnh báo lệch địa bàn ngay.
_Avoid_: Lỗi tìm kiếm

**ConfidenceTier**:
Nhãn mức độ chắc chắn thân thiện với **InformalWorker** (`Cao` / `Trung bình` / `Thấp - Cần hỏi cán bộ`) đi kèm dòng giới hạn sử dụng.
_Avoid_: Điểm cosine thô

**DelayedLabel**:
Phản hồi báo lỗi thực tế gắn với `trace_id` do người dùng hoặc cán bộ gửi sau khi sử dụng để phục vụ tái huấn luyện liên tục (CT).
_Avoid_: Đánh giá sao

**ApprovalAudit**:
Nhật ký bất biến ghi lại việc **Policy Approver & Safety Officer** phê duyệt chuyển đổi **ValidityStatus** của **PolicyDocument**.
_Avoid_: Log hệ thống

**KillSwitch**:
Cơ chế dừng khẩn cấp chuyển hệ thống về danh bạ Cán bộ Một cửa hoặc Rollback revision khi tỷ lệ lọt văn bản `expired` không cảnh báo `> 0%` hoặc tỷ lệ báo lỗi **DelayedLabel** `> 15%`.
_Avoid_: Tắt server

**ResidualRisk**:
Rủi ro còn lại bắt buộc công bố cho **InformalWorker** rằng kết quả AI chỉ hỗ trợ chuẩn bị hồ sơ ban đầu, không thay thế quyết định xét duyệt hành chính.
_Avoid_: Miễn trừ chung

**ReleaseDossier**:
Hồ sơ một lần phát hành ghi nhận `dataset_version`, `model_version`, `decision_threshold` và kết quả kiểm thử 4 mốc.
_Avoid_: Changelog

## Governance Roles (`GOVERN`)

- **Data Steward**: Quản lý kho **PolicyDocument**, cấu trúc metadata, **ColloquialMapping** và kiểm soát chia tập không rò rỉ dữ liệu.
- **Evaluation & QA Lead**: Chạy kiểm thử **ReleaseDossier** qua 4 mốc, đo độ ổn định 3 lần chạy và theo dõi cảnh báo Data/Concept Drift.
- **Policy Approver & Safety Officer**: Duyệt đổi **ValidityStatus** tạo **ApprovalAudit**, xác minh ca `expired` và nắm quyền kích hoạt **KillSwitch**.

## Relationships

- Một **Jurisdiction** sở hữu độc lập các **PolicyDocument** của riêng địa bàn đó, đồng thời kết hợp lọc kép với nhóm `"Toàn quốc"` (`region IN [selected_region, "Toàn quốc"]`).
- Một **PolicyDocument** có đúng một **ValidityStatus** hiện hành và tạo ra một **ApprovalAudit** mỗi khi cán bộ cập nhật trạng thái.
- Một lượt hỏi của **InformalWorker** sinh ra một **ConstrainedAnswer** bắt đầu bằng **VerificationReasoning**, kèm **ConfidenceTier**, **ResidualRisk** và tối đa một **DelayedLabel** gắn qua `trace_id`.

## Example dialogue

> **Dev:** "Nếu **PolicyDocument** tại **Jurisdiction** Phường Tân Tạo hết hiệu lực, ta có xóa khỏi kho tìm kiếm không?"
> **Domain expert:** "Không — giữ nguyên với **ValidityStatus** là `expired` để hệ thống truy xuất được, bật `expired_warning` cảnh báo đỏ và hướng dẫn chuyển **Policy Approver & Safety Officer** xác minh."

## Flagged ambiguities

- "Lọc cứng `region`" xung đột với quy định chung về hậu quả thiếu giấy tờ — đã chốt: lọc kép `region IN [selected_region, "Toàn quốc"]` và kiểm tra **CrossRegionConflict**.
- "Khối `<verification>` ngoài JSON" xung đột với Constrained JSON Decoding — đã chốt: đưa `verification_reasoning` làm trường đầu tiên trong JSON Schema.
- "Cập nhật trạng thái văn bản" tránh ghi đè file seed `data/mock_policies.json` — đã chốt: lưu biến động vào DB (**ApprovalAudit**) và đồng bộ RAM/Qdrant.
- "Chia tập RAFT" tránh rò rỉ dữ liệu biến thể câu hỏi giữa Train và Test — đã chốt: chia `80/10/10` theo nhóm kịch bản gốc (`Scenario-Group Split`).
- "Che PII có dấu cách/chấm" tránh che nhầm số tiền hay tổng đài — đã chốt: Regex bắt CCCD (9–12 số) và SĐT di động kể cả khi viết cách bằng khoảng trắng/chấm/gạch ngang, không đụng số tiền và hotline công khai; không dùng LLM bên thứ 3 và hỗ trợ xóa theo `trace_id` / TTL 30 ngày.
