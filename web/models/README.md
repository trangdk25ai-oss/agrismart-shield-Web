# Thư mục mô hình của máy chủ web

Chép vào đây 2 tệp do `ai/scripts/s07_export.py` sinh ra (trên Colab: thư mục `MyDrive/AgriSmart/model_for_web`):

- `model_int8.onnx` – mô hình YOLOv8n-seg lượng tử hóa INT8 (chạy bằng onnxruntime trên Windows/Linux/macOS)
- `model_meta.json` – danh sách lớp, kích thước đầu vào, kích thước tệp

Máy chủ ưu tiên `model_int8.onnx`, sau đó `*.onnx`, `*.tflite` (cần `ai-edge-litert`, chỉ Linux/macOS), `*.pt`
(cần ultralytics + PyTorch). Không có tệp nào thì chạy ở chế độ thử. Không đưa tệp mô hình lên kho mã công khai
(đã có trong `.gitignore`); chia sẻ qua Google Drive / GitHub Release.
