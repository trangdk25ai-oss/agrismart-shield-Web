# Thiết kế hệ thống AgriSmart Shield (bản 3 – trang web)

Bản đầy đủ có sơ đồ nằm trong tệp HTML đi kèm (mục "Phân tích thiết kế hệ thống"). Tệp này tóm tắt các quyết định
và hợp đồng dữ liệu để người sửa mã tra cứu nhanh.

## 1. Bối cảnh và tác nhân

| Tác nhân | Dùng hệ thống để |
|---|---|
| Người trồng (nông dân, hộ gia đình) | chụp lá, xem % tổn thương, mức độ, phác đồ, lịch phun |
| Cán bộ BVTV / chuyên gia | thẩm định và nhập phác đồ (`knowledge_base/ipm_rules.json`), đánh giá mù ảnh mẫu |
| Nhóm phát triển | chuẩn bị dữ liệu, huấn luyện, xuất mô hình, kiểm thử, trình diễn |
| Ban giám khảo | xem Demo, đọc báo cáo kiểm thử |

## 2. Yêu cầu

Chức năng: (F1) nhận ảnh từ camera/thư viện; (F2) kiểm tra ảnh mờ; (F3) phân đoạn lá và vết bệnh;
(F4) tính Severity Index theo lá; (F5) phân mức và tra phác đồ IPM theo (cây, bệnh, %); (F6) tính lượng chế phẩm theo
loại bình; (F7) lịch phun và tệp .ics; (F8) ảnh mẫu có đáp án cho Demo; (F9) lịch sử trên thiết bị.

Phi chức năng: (N1) không cần Internet khi dùng; (N2) mô hình < 10 MB, suy luận < 100 ms trên laptop chạy Demo;
(N3) không lưu ảnh người dùng; (N4) chạy trên Windows/Ubuntu/macOS, điện thoại chỉ cần trình duyệt;
(N5) một bản cài đặt duy nhất cho thuật toán; (N6) mọi số liệu kiểm thử tái lập được bằng script.

## 3. Kiến trúc

```
Trình duyệt điện thoại ──HTTP (Wi-Fi / điểm phát sóng, không Internet)──► Laptop: web/run.py (uvicorn)
  web/static (HTML/CSS/JS)                                                  └ web/app/main.py   (API)
                                                                              └ web/app/service.py (nạp 1 lần)
                                                                                  ├ agrismart.seg_runtime  (ONNX INT8)
                                                                                  ├ agrismart.infer.analyze (luồng)
                                                                                  ├ agrismart.severity
                                                                                  └ agrismart.knowledge_engine ◄ knowledge_base/ipm_rules.json
```

Chuỗi xử lý một yêu cầu `/api/diagnose`: đọc ảnh (giới hạn dung lượng, xoay EXIF, thu về 1280 px) → Blur Check →
(mờ và chưa `force` thì dừng, trả `BLURRY`) → mô hình → Severity theo lá, chọn lá chứa tâm ảnh → Knowledge Engine →
lớp phủ PNG trong suốt + ảnh JPEG → JSON.

## 4. Quyết định thiết kế

| Vấn đề | Chọn | Phương án khác | Lý do |
|---|---|---|---|
| Nơi chạy mô hình | Máy chủ Python trên laptop | Chạy trong trình duyệt (ONNX Runtime Web / LiteRT.js) | dùng lại 100% mã Python đã kiểm thử; không phải viết lại giải mã mask, Severity, Knowledge Engine bằng JavaScript |
| Khung máy chủ | FastAPI + uvicorn | Flask; Gradio/Streamlit | API JSON rõ ràng, kiểm tra kiểu dữ liệu, TestClient để kiểm thử; giao diện tự thiết kế theo 3 màn hình của quy trình |
| Định dạng mô hình | ONNX INT8 (ultralytics `format=onnx, quantize=8`) | LiteRT INT8; PyTorch .pt | onnxruntime chạy trên Windows/Linux/macOS, không cần PyTorch; LiteRT chỉ Linux x86/macOS |
| Giải mã đầu ra | tự viết (numpy), bám ultralytics 8.4.176 | gọi ultralytics khi chạy | máy chủ nhẹ (không PyTorch); đã đối chiếu với ultralytics trên mô hình thật |
| Nhắc lịch phun | tệp .ics | thông báo đẩy (Web Push) | Web Push cần HTTPS và Internet; .ics nhập vào ứng dụng Lịch có sẵn |
| Lưu lịch sử | localStorage của trình duyệt, không lưu ảnh | cơ sở dữ liệu máy chủ | không thu thập dữ liệu người dùng; đủ cho Demo |
| Không có mô hình | chế độ thử bằng luật màu, ghi rõ trên giao diện | báo lỗi, dừng | chạy thử giao diện ngay sau Phần 1 |

## 5. Hợp đồng dữ liệu

- Nhãn: YOLOv8-seg polygon chuẩn hóa; lớp 0 = `leaf`; lớp bệnh `<cây>__<bệnh>` (`ai/configs/class_map.yaml`).
- Mô hình: `model_int8.onnx` (metadata `names`, `imgsz`) + `model_meta.json` (s07).
- Tri thức: `knowledge_base/ipm_rules.json` theo `ipm_rules.schema.json`; ngưỡng mặc định < 5 / [5; 15] / > 15 %.
- Ảnh mẫu: `demo/samples/manifest.json` – `id, file, crop_code, gt_class, gt_severity_pct, gt_level_code, source_id, license`.
- Kết quả `/api/diagnose`: `status` (OK, HEALTHY, NO_LEAF, NO_LEAF_WITH_LESIONS, LEAF_TOO_SMALL, BLURRY), `blur`, `detections`
  (hộp chuẩn hóa 0–1), `leaves` (hộp, %, lớp trội), `recommendation` (đầu ra Knowledge Engine), `legend`,
  `image_jpeg_b64`, `overlay_png_b64`, `timings_ms`, `ground_truth` (khi dùng ảnh mẫu).

## 6. Rủi ro kỹ thuật

| Rủi ro | Giảm thiểu |
|---|---|
| Điện thoại không vào được máy chủ (tường lửa, Wi-Fi chặn máy nhìn nhau) | dùng điểm phát sóng điện thoại; mở cổng; dự phòng chạy trên trình duyệt laptop |
| Trình duyệt chặn camera trực tiếp trên HTTP | dùng ô chọn tệp có `capture` (mở ứng dụng camera), không dùng getUserMedia |
| Lệch kết quả giữa bộ chạy ONNX và ultralytics | kiểm thử tensor giả + đối chiếu trên mô hình thật (mục Xác minh) |
| Lượng tử hóa INT8 làm giảm mAP | s08 compare, tiêu chí ≤ 2 điểm %; nếu không đạt dùng ONNX FP32 (`--format onnx-fp32`) |
| INT8 (QDQ) không nhanh hơn FP32 trên CPU của laptop | đo cả hai bằng s08 latency; chọn theo số đo; giảm `imgsz` nếu chưa đạt < 100 ms |
| Nhiều người dùng cùng lúc | mô hình chạy tuần tự (khóa); đủ cho Demo, triển khai thật cần nhiều tiến trình |
