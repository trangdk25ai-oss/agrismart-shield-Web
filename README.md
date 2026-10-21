# AgriSmart Shield

Trang web chẩn đoán bệnh lá cây chạy **không cần Internet**: phân đoạn phiến lá và vết bệnh bằng YOLOv8n-seg
(INT8, < 10 MB), tính **Severity Index** (% diện tích tổn thương) và tra **phác đồ IPM** (chế phẩm, công thức pha
bình 16 L / 2 L, lịch phun). Máy chủ Python chạy trên laptop; điện thoại mở trang bằng trình duyệt qua Wi-Fi hoặc
điểm phát sóng. Dự thi Danang AI4Life 2026.

Bản 3 so với bản trước: Bước 5 đổi từ ứng dụng Flutter sang **trang web**; Bước 6 đổi từ kiểm thử thực địa sang
**kiểm thử thay thế + kịch bản Demo**; thêm tài liệu thiết kế hệ thống (`docs/thiet_ke_he_thong.md`).

## Cấu trúc

```
ai/                       Bước 1–4 và công cụ kiểm thử Bước 6 (Python)
  agrismart/              lõi dùng chung: severity, knowledge_engine, seg_runtime, infer, color_rule,
                          robustness, yolo_io, blur, metrics, sources, image_hash, annotation_import…
  scripts/s00..s11        pipeline đánh số theo thứ tự chạy
  configs/                class_map.yaml, dataset.yaml, plantseg115_classes.yaml, plantdoc_map.yaml
  tests/                  pytest (không cần GPU/torch)
web/                      Bước 5: máy chủ FastAPI (app/), giao diện (static/), mô hình (models/), kiểm thử (tests/)
demo/                     Bước 6: phiếu đánh giá chuyên gia mẫu; bộ ảnh mẫu demo/samples (sinh bởi s10, không commit)
knowledge_base/           ipm_rules.json + JSON Schema + validate_kb.py + kb_add_rules.py
data_sources/catalog.yaml danh mục 19 nguồn dữ liệu (link, DOI, giấy phép, cách nối vào bộ mã)
docs/                     thiết kế hệ thống, gán nhãn, nguồn dữ liệu, kế hoạch kiểm thử, kịch bản Demo
notebooks/                notebook Google Colab cho Phần 2 (huấn luyện, xuất INT8, kiểm thử độ bền, ảnh mẫu)
scripts/                  setup_ubuntu.sh, phan1_windows.ps1 (cài + chạy thử Phần 1)
.vscode/                  tác vụ, cấu hình chạy/gỡ lỗi, tiện ích đề xuất cho VS Code
Makefile                  lệnh tắt cho Ubuntu (make help)
pytest.ini                pytest chạy cả ai/tests và web/tests
```

## Chạy trên máy – 3 phần

| Phần | Làm gì | Cần gì | Lệnh |
|---|---|---|---|
| 1 | Cài đặt, kiểm thử, chuẩn bị dữ liệu, gán nhãn; chạy thử trang web | Máy tính thường, Python 3.10–3.13 | Ubuntu: `bash scripts/setup_ubuntu.sh` · Windows: `powershell -ExecutionPolicy Bypass -File scripts\phan1_windows.ps1` |
| 2 | Huấn luyện, đánh giá, xuất ONNX INT8, kiểm thử độ bền, tạo ảnh mẫu | Google Colab (GPU T4) | `notebooks/AgriSmart_Colab_HuanLuyen.ipynb` |
| 3 | Chạy trang web, mở trên điện thoại | Laptop đã làm Phần 1 + điện thoại có trình duyệt, chung mạng | chép mô hình vào `web/models/`, `python web/run.py` |

VS Code: mở thư mục dự án, cài tiện ích đề xuất, chạy **Terminal → Run Task → AgriSmart: …**.

## Các bước pipeline (sau khi kích hoạt `.venv`)

```bash
# Bước 1 – dữ liệu
python ai/scripts/s00_build_class_map.py --plantseg-diseases "tomato early blight" "tomato late blight" "tomato septoria leaf spot" --extra cucumber__downy_mildew cucumber__anthracnose citrus__anthracnose
python ai/scripts/s01_convert_masks_to_yolo.py --images data_raw/plantseg/images --lesion-masks data_raw/plantseg/annotations --mask-mode value --class-map ai/configs/class_map.yaml --bootstrap-leaf --out datasets/agrismart_all --prefix plantseg_
python ai/scripts/s00b_pick_for_labeling.py --source-id cucumber_recognition --src data_raw/cucumber_recognition --map "Downy Mildew=cucumber__downy_mildew" --per-class 150 --dedup-against datasets/agrismart_all/images --out data_raw/can_gan_nhan
python ai/scripts/s00c_auto_label.py --dir data_raw/can_gan_nhan --class-map ai/configs/class_map.yaml   # nhãn nháp; rà soát trên Labelme, tích cờ da_ra_soat
python ai/scripts/s01b_import_annotations.py --format labelme --labelme-dir data_raw/can_gan_nhan --class-map ai/configs/class_map.yaml --out datasets/agrismart_all --prefix lbl_
python ai/scripts/s02_split_dataset.py --src datasets/agrismart_all --out datasets/agrismart --class-map ai/configs/class_map.yaml
python ai/scripts/s03_augment_train.py --data datasets/agrismart --copies 2
python ai/scripts/s04_check_dataset.py --data-yaml datasets/agrismart/dataset.yaml
# Bước 2–3 – trên Colab (notebook), hoặc máy có GPU sau khi cài ai/requirements.txt
python ai/scripts/s05_train.py --data datasets/agrismart/dataset.yaml --epochs 100 --batch 16
python ai/scripts/s06_evaluate.py --weights runs/segment/agrismart_v8n/weights/best.pt --data datasets/agrismart/dataset.yaml --out reports/eval_fp32.json
python ai/scripts/s07_export.py --weights runs/segment/agrismart_v8n/weights/best.pt --data datasets/agrismart/dataset.yaml --copy-to web/models
python ai/scripts/s06_evaluate.py --weights web/models/model_int8.onnx --data datasets/agrismart/dataset.yaml --out reports/eval_int8.json
python ai/scripts/s08_benchmark.py compare --fp32 reports/eval_fp32.json --int8 reports/eval_int8.json --out reports/compare.json
# Bước 5 – trang web
python web/run.py
# Bước 6 – kiểm thử thay thực địa (docs/ke_hoach_kiem_thu.md)
python ai/scripts/s08_benchmark.py latency --model web/models/model_int8.onnx --images datasets/agrismart/images/test --out reports/latency.json
python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx --data datasets/agrismart/dataset.yaml --out reports/robustness
python ai/scripts/s10_demo_pack.py --data datasets/agrismart/dataset.yaml --n 24 --source-map plantseg_=plantseg
python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx --recapture-dir demo/chup_lai --out reports/recapture
python ai/scripts/s11_test_report.py --reports reports --out reports/bao_cao_kiem_thu.md
```

## Nguồn dữ liệu

`python ai/scripts/datasets_catalog.py list` liệt kê 19 nguồn (PlantSeg, PlantDoc và 17 nguồn bổ sung) kèm
mức phù hợp, giấy phép và cách nối vào bộ mã. Chi tiết: `docs/nguon_du_lieu.md`. Sinh trích dẫn cho hồ sơ:
`python ai/scripts/datasets_catalog.py cite --ids plantseg plantdoc --manifest data_raw/can_gan_nhan/manifest.csv`.

## Thêm loại bệnh mới

Chọn bệnh từ PlantSeg và/hoặc khai báo bệnh tự gán nhãn bằng `ai/scripts/s00_build_class_map.py`. Gán nhãn bằng
Labelme/CVAT/Roboflow, nhập bằng `ai/scripts/s01b_import_annotations.py`, tạo quy tắc IPM chờ nhập bằng
`knowledge_base/kb_add_rules.py`. Trang web tự nhận lớp mới từ mô hình, không cần sửa mã. Chi tiết: `docs/huong_dan_gan_nhan.md`.

## Lưu ý pháp lý – chuyên môn

`knowledge_base/ipm_rules.json` đang ở trạng thái **DRAFT**: liều lượng là dữ liệu mẫu (`rate_verified: false`) và
trang web luôn hiển thị cảnh báo. Trước khi phát hành, cán bộ chuyên môn BVTV phải thẩm định, đối chiếu Danh mục
thuốc BVTV được phép sử dụng tại Việt Nam hiện hành và nhãn sản phẩm, rồi chuyển `review_status` sang `APPROVED`.

Mỗi nguồn dữ liệu có giấy phép riêng (PlantSeg: CC BY-NC-ND 4.0 – không thương mại, không phân phối bản chỉnh sửa).
Không đưa bộ dữ liệu đã chuyển đổi, ảnh mẫu `demo/samples` hay mô hình lên kho mã công khai; chỉ công khai script.

## Giấy phép

Ultralytics YOLO phát hành theo AGPL-3.0, có giấy phép Enterprise cho trường hợp không công khai mã nguồn. AGPL-3.0
có điều khoản riêng cho phần mềm dùng qua mạng (Điều 13); phạm vi áp dụng đối với mô hình đã xuất chạy trên máy chủ
web chưa đủ cơ sở kết luận. Khi dự thi nên công khai mã theo AGPL; hỏi ý kiến pháp lý trước khi thương mại hóa.
