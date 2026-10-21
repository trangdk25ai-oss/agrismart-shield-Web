# Sử dụng các nguồn dữ liệu

Danh mục đầy đủ (19 nguồn) nằm trong `data_sources/catalog.yaml`. Mọi lệnh dưới đây là **một dòng**, chạy
được trên cả Ubuntu (bash) và Windows (PowerShell) sau khi đã kích hoạt môi trường ảo `.venv`.

```bash
python ai/scripts/datasets_catalog.py list                          # toàn bộ nguồn
python ai/scripts/datasets_catalog.py list --crop cucumber           # theo cây trồng
python ai/scripts/datasets_catalog.py list --gap cucumber__downy_mildew   # nguồn có ảnh cho lớp còn thiếu
python ai/scripts/datasets_catalog.py show riceseg_5932
```

## 1. Mức phù hợp và cách nối vào bộ mã

| Mức | Ý nghĩa | Cách dùng |
|---|---|---|
| A | Có mask vết bệnh | `s01` (`--mask-mode value` hoặc `binary`) |
| B | Có mask/polygon lá hoặc nhãn mức độ | bổ trợ lớp `leaf`, kiểm tra chéo Severity |
| C | Chỉ phân loại / hộp bao | `s00b` chọn ảnh → gán polygon → `s01b`; hoặc tập kiểm tra `s06b` |
| D | Ngoài phạm vi hiện tại | để dành khi mở rộng |
| Q | Chưa đọc được trang chi tiết | tự kiểm tra trước khi dùng |

## 2. Công thức theo từng nguồn

### PlantSeg (A) – nguồn huấn luyện chính
```bash
python ai/scripts/s00_build_class_map.py --plantseg-diseases "tomato early blight" "tomato late blight" "tomato septoria leaf spot" --extra cucumber__downy_mildew cucumber__anthracnose citrus__anthracnose --out ai/configs/class_map.yaml --labels-dir ai/configs/labels
python ai/scripts/s01_convert_masks_to_yolo.py --images data_raw/plantseg/images --lesion-masks data_raw/plantseg/annotations --mask-mode value --class-map ai/configs/class_map.yaml --leaf-masks data_raw/leaf_masks --out datasets/agrismart_all --prefix plantseg_
```
Thiếu mask lá thì thay `--leaf-masks …` bằng `--bootstrap-leaf` và rà soát `leaf_review.csv`.

### Nguồn chỉ có nhãn phân loại (C) – lấp 3 bệnh còn thiếu
Ví dụ Cucumber Disease Recognition (sương mai, thán thư dưa chuột). Tên thư mục lớp trong `--map` phải lấy
đúng theo bộ dữ liệu đã tải.
```bash
python ai/scripts/s00b_pick_for_labeling.py --source-id cucumber_recognition --src "data_raw/cucumber_recognition" --map "Downy Mildew=cucumber__downy_mildew" --map "Anthracnose=cucumber__anthracnose" --per-class 150 --dedup-against datasets/agrismart_all/images --out data_raw/can_gan_nhan
python ai/scripts/s00b_pick_for_labeling.py --source-id lemon_large_bd --src "data_raw/lemon_large_bd" --map "Anthracnose=citrus__anthracnose" --per-class 150 --dedup-against datasets/agrismart_all/images --out data_raw/can_gan_nhan
```
- `s00b` bỏ ảnh tăng cường có sẵn, bỏ ảnh trùng/gần trùng (dHash), bỏ ảnh nhỏ hơn 256 px.
- Chạy lại chỉ bổ sung phần còn thiếu cho đủ `--per-class`.
- Nguồn ảnh được ghi vào `manifest.csv`.

Gán polygon theo `docs/huong_dan_gan_nhan.md` (Labelme: `labelme data_raw/can_gan_nhan --labels ai/configs/labels/labelme_labels.txt`), rồi:
```bash
python ai/scripts/s01b_import_annotations.py --format labelme --labelme-dir data_raw/can_gan_nhan --class-map ai/configs/class_map.yaml --out datasets/agrismart_all --prefix lbl_
```

### PlantDoc (C) – tập kiểm tra độc lập
```bash
python ai/scripts/datasets_catalog.py dedup --a data_raw/plantdoc/TEST --b datasets/agrismart/images/train --out reports/trung_plantdoc_train.csv
python ai/scripts/s06b_eval_bbox.py --weights runs/segment/agrismart_v8n/weights/best.pt --csv data_raw/plantdoc/test_labels.csv --images data_raw/plantdoc/TEST --map ai/configs/plantdoc_map.yaml --exclude-list reports/trung_plantdoc_train.csv --out reports/eval_plantdoc.json
```
Kết quả: độ chính xác mức lá, độ nhạy theo bệnh, tỷ lệ báo nhầm trên lá khỏe.

### RiceSeg-5932 + Rice Leaf Disease Image Samples (A) – khi mở rộng sang lúa
1. Đặt ảnh và mask theo cùng cấu trúc thư mục, trùng tên file.
2. Kiểm tra mask trước:
   ```bash
   python ai/scripts/tools/inspect_masks.py data_raw/riceseg/masks --n 50
   ```
3. Khai báo lớp bằng `s00 --plantseg-crops rice` hoặc `--extra rice__…`, và khai báo `folder_to_target` trong `class_map.yaml`.
4. Chuyển nhãn:
   ```bash
   python ai/scripts/s01_convert_masks_to_yolo.py --images data_raw/rice_sethy --lesion-masks data_raw/riceseg/masks --mask-mode binary --class-map ai/configs/class_map.yaml --bootstrap-leaf --out datasets/agrismart_all --prefix riceseg_
   ```
Mask JPG được ngưỡng hóa ở 127. Nếu chỉ lấy mọi pixel > 0, diện tích vết bệnh bị phóng đại (thử trên một mask tròn chuẩn: lệch +28%).

### BRACOT (B) – mask lá cà phê định dạng VIA
```bash
python ai/scripts/s01b_import_annotations.py --format via --via-json data_raw/bracot/annotations.json --images data_raw/bracot/images --via-label-attr <tên_thuộc_tính> --class-map ai/configs/class_map.yaml --out datasets/agrismart_all --prefix bracot_
```
- Mở file JSON để biết tên thuộc tính chứa nhãn.
- Nếu tên nhãn trong file khác tên lớp, khai báo ánh xạ trong `label_aliases`.

### RoCoLe, BRACOL (B) – kiểm tra chéo Severity
Hai bộ có thang mức bệnh theo % diện tích lá. Dùng để đối chiếu cách phân mức của thuật toán Severity, **không** trộn vào tập huấn luyện cà chua, dưa chuột, cây có múi.

### Roboflow Universe, bộ phân đoạn dưa chuột (Q)
Xuất định dạng **COCO Segmentation**, rồi nhập bằng `s01b --format coco`. Phải rà lại polygon lá trước khi dùng.

## 3. Danh sách kiểm tra trước khi đưa nguồn vào huấn luyện
- [ ] Đọc mục giấy phép trên trang gốc. Giấy phép NC/ND (như PlantSeg) không cho thương mại hóa hoặc phân phối bản chỉnh sửa.
- [ ] Chỉ dùng ảnh gốc, bỏ ảnh tăng cường có sẵn trong bộ dữ liệu.
- [ ] Loại ảnh trùng giữa các nguồn (`s00b --dedup-against`, `datasets_catalog.py dedup`).
- [ ] Cán bộ BVTV xác nhận tên bệnh của bộ nước ngoài tương ứng triệu chứng tại địa phương.
- [ ] Dùng `--prefix` khác nhau cho mỗi nguồn để truy nguồn từng ảnh.
- [ ] Sinh trích dẫn cho hồ sơ: `python ai/scripts/datasets_catalog.py cite --ids plantseg plantdoc --manifest data_raw/can_gan_nhan/manifest.csv`.
