# Quy tắc làm việc cho Agent (`AGENTS.md`)

## 1. Quy trình Git & Quy tắc viết Commit Message
- **Commit sau mỗi thay đổi:** Sau mỗi lần thêm, sửa hoặc xóa nội dung trong dự án, bắt buộc tạo một `git commit` riêng cho phạm vi thay đổi đó. Không gộp nhiều thay đổi không liên quan vào cùng một commit.
- **Định dạng Commit Message (Conventional Commits — tiếng Anh):**
  - Cấu trúc: `<type>(<scope>): <short summary>` (hoặc `<type>: <short summary>` nếu phạm vi chung).
  - Các `<type>` chuẩn:
    - `feat`: Thêm tính năng mới
    - `fix`: Sửa lỗi
    - `test`: Thêm hoặc cập nhật bài test
    - `docs`: Cập nhật tài liệu
    - `refactor`: Tái cấu trúc mã nguồn (không đổi hành vi)
    - `chore`: Cấu hình, dọn dẹp, bảo trì
  - **Yêu cầu nội dung:** Dùng động từ nguyên mẫu (imperative mood, ví dụ: `add`, `fix`, `update`), ngắn gọn (≤ 72 ký tự ở dòng tiêu đề), mô tả đúng *cái gì* và *tại sao* thay đổi.

## 2. Bắt buộc viết Test & Xác thực trước khi bàn giao
- **Test đi kèm thay đổi:** Sau mỗi thay đổi về mã nguồn hoặc logic xử lý, bắt buộc phải viết mới hoặc cập nhật các bài test tương ứng (unit test / integration test / edge-case test).
- **Cổng xác thực (Verification Gate):** Trước khi bàn giao kết quả cho người dùng, phải chạy toàn bộ bộ test và các lệnh kiểm tra liên quan, đảm bảo **100% test thông qua (pass)** và không phát sinh lỗi hồi quy.

## 3. An toàn dữ liệu & Mã nguồn
- Tách biệt dữ liệu giả lập (mock data / fixtures) khỏi mã nguồn xử lý.
- Tuyệt đối không commit API key, thông tin bí mật hoặc dữ liệu định danh cá nhân (PII) thật vào repository.
