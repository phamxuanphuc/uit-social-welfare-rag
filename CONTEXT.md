# Social Welfare RAG (SDG 1.3 - P01)

Hệ thống trợ lý tra cứu quyền lợi an sinh và hồ sơ thủ tục cho người lao động phi chính thức dựa trên văn bản công khai (giả lập), tuân thủ khung Responsible AI và MLOps.

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

**DelayedLabel**:
Phản hồi báo lỗi thực tế gắn với `trace_id` do người dùng hoặc cán bộ gửi sau khi sử dụng để phục vụ tái huấn luyện liên tục.
_Avoid_: Đánh giá sao

**ApprovalAudit**:
Nhật ký bất biến ghi lại việc cán bộ chuyên trách phê duyệt chuyển đổi **ValidityStatus** của **PolicyDocument**.
_Avoid_: Log hệ thống

## Relationships

- Một **Jurisdiction** sở hữu độc lập các **PolicyDocument** của riêng địa bàn đó, đồng thời kết hợp lọc kép với nhóm `"Toàn quốc"` (`region IN [selected_region, "Toàn quốc"]`).
- Một **PolicyDocument** có đúng một **ValidityStatus** hiện hành và tạo ra một **ApprovalAudit** mỗi khi cán bộ cập nhật trạng thái.
- Một lượt hỏi của **InformalWorker** sinh ra một **ConstrainedAnswer** bắt đầu bằng **VerificationReasoning**, kèm tối đa một **DelayedLabel** gắn qua `trace_id`.

## Example dialogue

> **Dev:** "Nếu **PolicyDocument** tại **Jurisdiction** Phường Tân Tạo hết hiệu lực, ta có xóa khỏi kho tìm kiếm không?"
> **Domain expert:** "Không — giữ nguyên với **ValidityStatus** là `expired` để hệ thống truy xuất được, bật `expired_warning` cảnh báo đỏ và hướng dẫn chuyển cán bộ xác minh."

## Flagged ambiguities

- "Lọc cứng `region`" xung đột với quy định chung về hậu quả thiếu giấy tờ — đã chốt: lọc kép `region IN [selected_region, "Toàn quốc"]`.
- "Khối `<verification>` ngoài JSON" xung đột với Constrained JSON Decoding — đã chốt: đưa `verification_reasoning` làm trường đầu tiên trong JSON Schema.
- "Cập nhật trạng thái văn bản" tránh ghi đè file seed `data/mock_policies.json` — đã chốt: lưu biến động vào DB (**ApprovalAudit**) và đồng bộ RAM/Qdrant.
