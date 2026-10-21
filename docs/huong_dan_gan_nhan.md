# Hướng dẫn bổ sung loại bệnh mới và gán nhãn

Áp dụng khi muốn mô hình nhận dạng thêm bệnh: bệnh có sẵn trong PlantSeg (115 bệnh) hoặc bệnh tự chụp,
tự gán nhãn tại địa phương (ví dụ sương mai dưa chuột, thán thư dưa chuột, thán thư cây có múi).

Lệnh trong tài liệu viết cho Windows PowerShell (`.\.venv\Scripts\python`). Trên Ubuntu, chạy
`source .venv/bin/activate` rồi thay `.\.venv\Scripts\python` bằng `python`, dấu `\` trong đường dẫn bằng `/`,
và dấu xuống dòng `` ` `` bằng `\`. Nguồn ảnh bổ sung để gán nhãn: xem `docs/nguon_du_lieu.md` và `s00b_pick_for_labeling.py`.

## 1. Quy trình tổng thể

| Bước | Việc | Công cụ / lệnh |
|---|---|---|
| 1 | Chọn danh sách bệnh, sinh `class_map.yaml` và danh sách nhãn | `ai/scripts/s00_build_class_map.py` |
| 2 | Chụp ảnh và gán nhãn bệnh tự thu thập | Labelme (offline) hoặc CVAT / Roboflow |
| 3 | Chuyển nhãn PlantSeg và nhãn tự gán vào CÙNG một thư mục | `s01` (PlantSeg), `s01b` (nhãn tự gán) |
| 4 | Tạo quy tắc IPM "chờ nhập" cho bệnh mới, chuyên gia BVTV điền phác đồ | `knowledge_base/kb_add_rules.py` |
| 5 | Chia tập, tăng cường, kiểm tra, huấn luyện lại, cập nhật trang web | `s02` → `s08` |

**Lưu ý:**
- Mỗi lần thêm hoặc bớt bệnh là phải **huấn luyện lại** mô hình, vì số lớp và chỉ số lớp thay đổi.
- Luôn chạy lại `s01`/`s01b` từ nhãn gốc. Không sửa tay các file `.txt` đã sinh trước đó.

## 2. Bước 1 – Chọn bệnh

```powershell
# Xem PlantSeg có bệnh nào của một loại cây (mã cây: tomato, cucumber, citrus, bellpepper, rice…)
.\.venv\Scripts\python ai\scripts\s00_build_class_map.py --list cucumber citrus

# Ví dụ: 3 bệnh cà chua + phấn trắng dưa chuột (PlantSeg) + 3 bệnh tự gán nhãn
.\.venv\Scripts\python ai\scripts\s00_build_class_map.py `
    --plantseg-diseases "tomato early blight" "tomato late blight" "tomato septoria leaf spot" "cucumber powdery mildew" `
    --extra cucumber__downy_mildew cucumber__anthracnose citrus__anthracnose `
    --out ai\configs\class_map.yaml --labels-dir ai\configs\labels
```

**Quy ước tên lớp tự gán nhãn: `<cây>__<bệnh>`**
- Gồm chữ thường không dấu, số và gạch dưới; giữa tên cây và tên bệnh là hai dấu gạch dưới.
- Mã cây không chứa dấu gạch dưới.
- Ví dụ: `cucumber__downy_mildew`, `citrus__anthracnose`, `rice__blast`.

Lệnh tạo ra 2 file danh sách nhãn dùng cho công cụ gán nhãn:
- `ai/configs/labels/cvat_labels.json`: dán vào ô **Raw** khi tạo nhãn trong CVAT.
- `ai/configs/labels/labelme_labels.txt`: dùng cho tham số `--labels` của Labelme.

## 3. Bước 2 – Gán nhãn

### Chọn công cụ
- **Labelme** (khuyến nghị cho máy văn phòng, chạy offline):
  ```powershell
  .\.venv\Scripts\python -m pip install labelme
  .\.venv\Scripts\labelme data_raw\tu_gan_nhan --labels ai\configs\labels\labelme_labels.txt
  ```
  - Dùng công cụ **Create Polygons**.
  - Mỗi ảnh sinh một file `.json` cạnh ảnh.
  - Labelme 6 trở lên mặc định tự lưu và không nhúng ảnh vào JSON. Bản cũ (5.x) cần thêm `--nodata --autosave`; Labelme 7 đã bỏ hai tham số này (gõ vào sẽ báo lỗi).
  - Có thể dùng công cụ AI-Polygon/AI-Mask của Labelme để khoanh vùng nhanh; `s01b` đọc được cả polygon, circle và mask.
- **CVAT** (web, có công cụ AI/SAM để khoanh vùng nhanh):
  - Tạo project, vào mục Labels, chọn **Raw** và dán nội dung `cvat_labels.json`.
  - Sau khi gán xong, xuất định dạng **COCO 1.0**; file nhãn là `annotations/instances_default.json`.
- **Roboflow**: dùng Smart Polygon, xuất định dạng **COCO Segmentation**.

### Nhãn nháp tự động để rà soát (s00c) – nhanh hơn vẽ từ đầu
Ảnh chọn từ bộ dữ liệu phân loại (s00b) chỉ biết tên bệnh theo thư mục. `s00c_auto_label.py` tự tách phiến lá
(ảnh nền trắng: vùng không trắng; ảnh ngoài vườn: ExG + Otsu) và vùng đổi màu trong lá, rồi ghi tệp nhãn Labelme
cạnh ảnh với cờ `da_ra_soat: false`:
```bash
python ai/scripts/s00c_auto_label.py --dir data_raw/can_gan_nhan --class-map ai/configs/class_map.yaml
labelme data_raw/can_gan_nhan --labels ai/configs/labels/labelme_labels.txt --flags da_ra_soat
```
- Không ghi đè tệp `.json` đã có (nhãn đã vẽ tay); ảnh không thấy vùng đổi màu thì không có tệp nháp – gán tay.
- `auto_label_report.csv` ghi từng ảnh: kết quả, phương pháp, tỷ lệ vùng đổi màu ước tính, ghi chú nghi ngờ.
- Trên Labelme: sửa điểm, xóa vùng sai (lá vàng, cháy nắng, bóng đổ, vết côn trùng cũng bị máy tính là vết bệnh),
  vẽ thêm vết bị sót, rồi **tích cờ `da_ra_soat`** ở khung Flags và bấm Next (tự lưu).
- Ô tích trong Image List chỉ cho biết ảnh đã có tệp nhãn (kể cả nháp) – dựa vào cờ `da_ra_soat` để biết đã rà soát.
- `s01b` bỏ qua tệp nháp chưa tích cờ (`skip_nhan_nhap_chua_ra_soat`). `--include-unreviewed` nhập cả nhãn chưa rà
  soát – không khuyến nghị; nếu dùng phải ghi rõ trong báo cáo vì nhãn khi đó là nhãn do luật màu sinh ra.

### Quy tắc gán nhãn (bắt buộc, để thuật toán Severity Index tính đúng)
1. **Phiến lá (`leaf`)**:
   - Vẽ một polygon bao **trọn** phiến lá, kể cả vùng vết bệnh và mép lá bị cháy.
   - Không khoét lỗ ở chỗ vết bệnh.
   - Không gộp nhiều lá vào một polygon.
2. **Vết bệnh (`<cây>__<bệnh>`)**:
   - Mỗi vết hoặc mỗi mảng vết dính liền là một polygon riêng, vẽ theo viền vùng mô bệnh (bao cả quầng vàng nếu đó là đặc trưng của bệnh).
   - Vết bệnh phải nằm trong polygon lá.
   - Vết nhỏ hơn khoảng 4×4 pixel sẽ bị bỏ khi chuyển đổi.
3. **Lá nào trong ảnh phải gán:**
   - Gán mọi lá thấy rõ từ khoảng một nửa diện tích trở lên.
   - Lá mờ hoặc bị che khuất nhiều thì không gán; tránh chọn ảnh có quá nhiều lá như vậy.
4. **Khi không chắc là bệnh gì:** không đưa ảnh vào bộ dữ liệu, hoặc hỏi cán bộ BVTV. **Không đoán nhãn.**
5. **Lá khỏe:**
   - Chỉ gán `leaf`.
   - Đây là mẫu âm giúp mô hình bớt báo nhầm; đề xuất chiếm khoảng 10–20% số ảnh.
6. **Tên nhãn:**
   - Phải đúng danh sách nhãn đã sinh ở Bước 1.
   - Nếu buộc phải dùng tên khác, khai báo ở mục `label_aliases` trong `class_map.yaml`.

### Chụp ảnh tại địa phương
- Mỗi ảnh tập trung một lá chính, khoảng cách 15–25 cm.
- Chụp trong nhiều điều kiện ánh sáng (nắng, râm, sau mưa), ở nhiều vườn, nhiều thời điểm.
- Rải đều 3 mức bệnh (khởi phát, nhẹ, nặng). Đề xuất tối thiểu 150–200 ảnh cho mỗi bệnh; con số này là kinh nghiệm, không phải tiêu chuẩn.
- Đặt tên file có mã vườn và mã cây, ví dụ `DN01_cay05_la02.jpg`. Khi chia tập, dùng `--group-regex "^.*?(DN\d+_cay\d+)"` để ảnh của cùng một cây nằm cùng một tập.

### Kiểm soát chất lượng
- Người thứ hai rà lại ngẫu nhiên 10–20% số ảnh.
- Thống nhất cách vẽ viền vết bệnh trước khi gán hàng loạt.
- Xem `import_report.json`, đặc biệt các dòng `unknown_label:…` (nhãn không có trong danh sách).
- Chạy `s04` để kiểm tra vết bệnh có nằm trong lá hay không.

## 4. Bước 3 – Nhập nhãn vào bộ dữ liệu

```powershell
# Dữ liệu PlantSeg (mask theo mã lớp)
.\.venv\Scripts\python ai\scripts\s01_convert_masks_to_yolo.py --images data_raw\plantseg\images `
    --lesion-masks data_raw\plantseg\annotations --mask-mode value --class-map ai\configs\class_map.yaml `
    --leaf-masks data_raw\leaf_masks --out datasets\agrismart_all

# Nhãn tự gán bằng Labelme
.\.venv\Scripts\python ai\scripts\s01b_import_annotations.py --format labelme --labelme-dir data_raw\tu_gan_nhan `
    --class-map ai\configs\class_map.yaml --out datasets\agrismart_all --prefix dn_

# Hoặc nhãn xuất từ CVAT/Roboflow (COCO)
.\.venv\Scripts\python ai\scripts\s01b_import_annotations.py --format coco `
    --coco-json data_raw\cvat\annotations\instances_default.json --images data_raw\cvat\images `
    --class-map ai\configs\class_map.yaml --out datasets\agrismart_all --prefix cvat_
```

## 5. Bước 4 – Phác đồ IPM cho bệnh mới

```powershell
.\.venv\Scripts\python knowledge_base\kb_add_rules.py --class-map ai\configs\class_map.yaml
```

- Lệnh thêm quy tắc **CHỜ NHẬP** (`rule_status: "PENDING"`) cho các bệnh chưa có trong `ipm_rules.json`. File cũ được lưu thành `.bak`.
- Với bệnh có quy tắc chờ nhập, trang web vẫn khoanh vùng, tính % tổn thương và mức độ, nhưng **không đưa phác đồ hay liều lượng**. Thay vào đó, trang web báo "phác đồ chưa được cán bộ chuyên môn nhập".
- Cán bộ BVTV điền các mục sau rồi đổi `rule_status` thành `ACTIVE`:
  - tên bệnh tiếng Việt, tác nhân;
  - `actions` cho từng mức;
  - `active_ingredient`, `dosage_guide` (chỉ dùng sản phẩm có trong Danh mục thuốc BVTV được phép sử dụng);
  - `spray_plan`.
- Sau đó chạy `knowledge_base\validate_kb.py` để kiểm tra. Hệ thống không cho đặt `ACTIVE` khi còn thiếu hành động cho một mức nào đó.

## 6. Bước 5 – Huấn luyện lại và cập nhật trang web

1. Chạy `s02` → `s04` trên máy, rồi `s05` → `s08` trên Colab.
2. Chép `model_int8.onnx` và `model_meta.json` (thư mục `MyDrive/AgriSmart/model_for_web` trên Drive) vào `web/models/`.
   Tri thức IPM được máy chủ đọc thẳng từ `knowledge_base/ipm_rules.json`, không cần chép.
3. Khởi động lại máy chủ (`python web/run.py`). Trang web tự cập nhật danh sách cây trồng, bệnh và màu lớp phủ theo
   danh sách lớp trong mô hình, **không cần sửa mã**. Cây trồng không có bệnh nào trong mô hình sẽ tự ẩn.
