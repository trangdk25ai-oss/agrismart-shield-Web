# Kịch bản Demo AgriSmart Shield (không cần thực địa, không cần Internet)

## 1. Chuẩn bị (trước buổi trình bày 1 ngày)

| Việc | Kiểm tra |
|---|---|
| Laptop đã cài Phần 1, có `web/models/model_int8.onnx` và `demo/samples/` | `python web/run.py` → trang hiện "Mô hình: model_int8.onnx" |
| Điện thoại Android/iPhone có trình duyệt (Chrome/Safari) | mở được địa chỉ `http://<IP laptop>:8000` |
| **Mạng riêng**: bật điểm phát sóng trên điện thoại, laptop kết nối vào | không phụ thuộc Wi-Fi hội trường; tắt dữ liệu di động để chứng minh chạy không cần Internet |
| Tường lửa Windows cho phép Python trên mạng Private | điện thoại mở được trang |
| 3–5 lá thật (mua ở chợ / vườn nhà) có triệu chứng, 1 lá khỏe | chụp thử trước, chọn lá cho kết quả rõ |
| Tờ in ảnh mẫu (`demo/samples/to_in_A4.html`) | dự phòng khi không có lá thật |
| Báo cáo `reports/bao_cao_kiem_thu.md`, video quay màn hình dự phòng | phòng khi máy trục trặc |
| Cắm sạc laptop; tắt cập nhật tự động, chế độ ngủ | |

## 2. Kịch bản 5 phút

| Thời gian | Nội dung | Thao tác |
|---|---|---|
| 0:00–0:40 | Vấn đề: nông dân ước lượng mức bệnh bằng mắt, phun theo cảm tính | slide |
| 0:40–1:20 | Kiến trúc: điện thoại chỉ là trình duyệt; mô hình INT8 < 10 MB, Severity, Knowledge Engine chạy trên laptop; không cần Internet | slide sơ đồ triển khai; chỉ biểu tượng điểm phát sóng |
| 1:20–2:30 | **Ảnh mẫu có đáp án**: thẻ "Ảnh mẫu" → chọn 1 ảnh nhẹ, 1 ảnh nặng | trang hiện bảng "Đáp án – Hệ thống – ✓/✗" |
| 2:30–3:30 | **Chụp trực tiếp** lá thật (hoặc ảnh in) bằng điện thoại | lớp phủ, % tổn thương, mức độ; chạm lá khác nếu ảnh có nhiều lá |
| 3:30–4:10 | Phác đồ: đổi bình 16 L ↔ 2 L (liều đổi theo), "Tải lịch phun (.ics)" | mở tệp bằng ứng dụng Lịch (thử trước trên đúng điện thoại Demo: khả năng mở .ics tùy ứng dụng lịch) |
| 4:10–4:30 | Ảnh mờ: chụp rung tay → hệ thống yêu cầu chụp lại | |
| 4:30–5:00 | Số liệu kiểm thử: bảng chỉ tiêu nghiệm thu, độ bền 10 điều kiện, giới hạn | slide từ `bao_cao_kiem_thu.md` |

## 3. Dự phòng

| Sự cố | Xử lý |
|---|---|
| Điện thoại không vào được trang | dùng trình duyệt ngay trên laptop (`http://127.0.0.1:8000`), chọn ảnh từ máy |
| Lá thật cho kết quả lạ | chuyển sang ảnh mẫu có đáp án; giải thích giới hạn dữ liệu (chưa có ảnh Đà Nẵng) |
| Laptop hỏng | phát video quay sẵn |

## 4. Câu hỏi dễ gặp

- **Không thực địa thì sao biết chạy được ngoài ruộng?** – Chưa khẳng định được. Đã đo trên tập kiểm tra độc lập,
  10 điều kiện chụp mô phỏng và ảnh chụp lại bằng điện thoại (số liệu trong báo cáo); kiểm thử ngoài ruộng là
  việc của giai đoạn sau, chạy lại cùng các script.
- **Vì sao là trang web mà không phải ứng dụng?** – Không cần cài đặt, chạy trên mọi điện thoại có trình duyệt,
  một bản mã duy nhất cho Severity và Knowledge Engine (không phải viết lại bằng ngôn ngữ thứ hai); vẫn không cần
  Internet khi dùng mạng cục bộ. Đổi lại: cần một máy tính chạy máy chủ trong cùng mạng.
- **Liều lượng có dùng được ngay không?** – Không. Tri thức IPM đang là DỰ THẢO, liều là dữ liệu mẫu, trang web
  luôn cảnh báo; phải được cán bộ BVTV thẩm định theo Danh mục thuốc BVTV hiện hành.
