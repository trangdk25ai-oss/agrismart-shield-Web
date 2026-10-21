# Trang web AgriSmart Shield (Bước 5)

Máy chủ Python (FastAPI) chạy trên laptop; điện thoại mở trang bằng trình duyệt qua cùng mạng Wi-Fi hoặc
điểm phát sóng. **Không cần Internet** và không cần cài ứng dụng trên điện thoại. Toàn bộ xử lý (Blur Check,
mô hình, Severity Index, Knowledge Engine) chạy trên laptop; ảnh chỉ xử lý trong bộ nhớ, không lưu lại.

## Chạy

```bash
# đã cài ở Phần 1: python -m pip install -r web/requirements.txt
python web/run.py                                   # tự tìm mô hình trong web/models/
python web/run.py --model web/models/model_int8.onnx --port 8000
```

Màn hình in ra địa chỉ cho điện thoại, ví dụ `http://192.168.1.5:8000`. Chưa có mô hình thì trang chạy ở
**chế độ thử**: tách lá và vùng đổi màu bằng luật màu sắc, người dùng tự chọn bệnh – chỉ để chạy thử giao diện,
không phải chẩn đoán (trang hiển thị cảnh báo).

| Tham số / biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `--model` / `AGS_MODEL` | tự tìm `web/models/model_int8.onnx`, `*.onnx`, `*.tflite`, `*.pt` | tệp mô hình |
| `--conf` / `AGS_CONF` | 0,25 | ngưỡng tin cậy |
| `--blur-threshold` / `AGS_BLUR_THRESHOLD` | 100 | ngưỡng Blur Check (hiệu chỉnh bằng `s09_robustness.py`) |
| `--samples` / `AGS_SAMPLES_DIR` | `demo/samples` | bộ ảnh mẫu Demo (`s10_demo_pack.py`) |
| `AGS_MAX_UPLOAD_MB` | 12 | dung lượng ảnh tối đa |
| `AGS_MAX_SIDE` | 1280 | ảnh được thu về cạnh dài này trước khi xử lý |

## Điện thoại không mở được trang

1. Điện thoại và laptop phải chung một mạng (cùng Wi-Fi, hoặc laptop bắt điểm phát sóng của điện thoại).
2. Windows: lần đầu chạy, hộp thoại Tường lửa hỏi cho phép Python → chọn cho phép trên mạng **riêng (Private)**.
   Nếu mạng đang đặt là **công cộng (Public)**, đổi sang Private trong Cài đặt → Mạng & Internet → Wi-Fi → (tên mạng).
3. Ubuntu có bật `ufw`: `sudo ufw allow 8000/tcp`.
4. Mạng Wi-Fi công cộng/khách sạn thường chặn các máy nhìn thấy nhau (client isolation): dùng điểm phát sóng của điện thoại.

## Chạy trong WSL (Ubuntu trên Windows)

Trên chính máy tính: mở `http://127.0.0.1:8000` bằng trình duyệt Windows (WSL tự chuyển cổng về `localhost`).
Với điện thoại: WSL 2 mặc định dùng mạng NAT với địa chỉ riêng (172.x.x.x) nên điện thoại **không** vào được
địa chỉ đó. Chọn một trong hai cách (theo tài liệu Microsoft “Accessing network applications with WSL”):

1. **Mạng mirrored** (Windows 11 22H2 trở lên): tạo/sửa tệp `%UserProfile%\.wslconfig`:
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   rồi chạy `wsl --shutdown` trong PowerShell, mở lại Ubuntu. Cho phép kết nối vào cổng 8000 qua tường lửa
   Hyper-V (PowerShell quyền quản trị):
   ```powershell
   New-NetFirewallHyperVRule -Name "AgriSmart8000" -DisplayName "AgriSmart Shield 8000" -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 8000
   ```
   Điện thoại mở `http://<IP Wi-Fi của Windows>:8000` (xem bằng `ipconfig`).
2. **Chuyển cổng** (chế độ NAT, PowerShell quyền quản trị; địa chỉ WSL đổi sau mỗi lần khởi động lại):
   ```powershell
   $ip = (wsl hostname -I).Trim().Split()[0]
   netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$ip
   New-NetFirewallRule -DisplayName "AgriSmart Shield 8000" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
   ```
   Điện thoại mở `http://<IP Wi-Fi của Windows>:8000`. Gỡ chuyển cổng khi xong:
   `netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0`.

Đơn giản nhất cho buổi Demo: chạy máy chủ bằng Python cài trực tiếp trên Windows (`python web\run.py`), không qua WSL.

## API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/health` | tình trạng máy chủ |
| GET | `/api/meta` | cây trồng, bệnh, loại bình, chế độ, thông tin mô hình, địa chỉ cho điện thoại |
| POST | `/api/diagnose` | multipart: `image` hoặc `sample_id`; `crop_code`, `tank_code`, `start_date`, `force`, `lesion_class` |
| POST | `/api/recommend` | JSON: tính lại phác đồ khi đổi bình / chọn lá khác (không chạy lại mô hình) |
| GET | `/api/calendar.ics` | tệp lịch phun (.ics) để thêm vào ứng dụng Lịch |
| GET | `/api/samples`, `/api/samples/{id}/thumb` | ảnh mẫu Demo kèm đáp án |

Trang tài liệu API tự sinh: `/docs` (cần Internet để tải giao diện Swagger).

## Cấu trúc

```
web/
  run.py            khởi động uvicorn, in địa chỉ mạng cục bộ
  app/main.py       định tuyến API (FastAPI)
  app/service.py    nạp mô hình + tri thức một lần; gọi agrismart.infer.analyze
  app/imaging.py    đọc ảnh tải lên (giới hạn dung lượng, xoay theo EXIF), mã hóa ảnh trả về
  app/calendar_ics.py  sinh tệp .ics (RFC 5545)
  app/samples.py    bộ ảnh mẫu Demo
  static/           giao diện (HTML/CSS/JS thuần, không thư viện ngoài)
  models/           đặt model_int8.onnx + model_meta.json (không đưa lên kho mã)
  tests/            kiểm thử API (pytest)
```
