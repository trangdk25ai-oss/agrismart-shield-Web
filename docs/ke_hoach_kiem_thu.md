# Bước 6 – Kế hoạch kiểm thử thay thế thực địa

Quy trình gốc đặt Bước 6 là kiểm thử thực địa. Nhóm **không đi thực địa được**, nên thay bằng 5 hình thức
làm được trong phòng, mỗi hình thức có script chạy tự động và số liệu đưa thẳng vào hồ sơ.

| # | Hình thức | Trả lời câu hỏi | Công cụ | Cần gì |
|---|---|---|---|---|
| 1 | Đánh giá trên tập kiểm tra độc lập | Mô hình đúng đến đâu trên ảnh chưa từng thấy? | `s06_evaluate.py`, `s06b_eval_bbox.py` (PlantDoc) | Colab |
| 2 | Kiểm thử độ bền – 10 điều kiện chụp mô phỏng | Ảnh mờ, tối, chói, chụp xa, nghiêng… làm kết quả kém đi bao nhiêu? | `s09_robustness.py --data` | Colab hoặc laptop |
| 3 | Chụp lại ảnh in / màn hình bằng điện thoại | Qua camera thật (ánh sáng, nét, góc) kết quả còn giữ được không? | `s10_demo_pack.py` (tờ in A4) → `s09_robustness.py --recapture-dir` | Máy in màu hoặc màn hình, 1 điện thoại |
| 4 | Đo tốc độ trên máy chạy Demo | Đáp ứng < 100 ms suy luận? Người dùng chờ bao lâu? | `s08_benchmark.py latency`, số đo trên trang web | Laptop chạy Demo |
| 5 | (Tùy chọn) Chuyên gia BVTV đánh giá mù ảnh mẫu | Con số của hệ thống có gần với đánh giá của người làm nghề? | `demo/danh_gia_chuyen_gia_mau.csv` → `s11_test_report.py` | 1 cán bộ BVTV, 30–60 phút, làm từ xa được |

Cuối cùng `s11_test_report.py` gộp tất cả thành `reports/bao_cao_kiem_thu.md`.

## 1. Chỉ tiêu nghiệm thu

| Chỉ tiêu | Ngưỡng | Căn cứ |
|---|---|---|
| Kích thước mô hình INT8 | < 10 MB | Quy trình gốc, Bước 3.1 |
| Mask mAP50-95 giảm sau INT8 | ≤ 2 điểm % (tuyệt đối) | Quy trình gốc, Bước 3.2 (ghi rõ cách hiểu "2%") |
| Thời gian suy luận | < 100 ms | Quy trình gốc đo trên điện thoại; nay đo trên laptop chạy Demo – ghi rõ cấu hình máy |
| Sai số Severity (MAE) trên tập kiểm tra | ≤ 3 điểm % | Đề xuất |
| Đúng mức xử lý trên tập kiểm tra | ≥ 85% | Đề xuất |
| Nhận đúng bệnh (ảnh gốc tập kiểm tra) | ≥ 85% | Đề xuất |
| Độ bền: đúng mức thấp nhất (trừ mờ nặng, rung tay) | ≥ 75% | Đề xuất |
| Blur Check: phát hiện ảnh mờ nặng / báo nhầm ảnh nét | ≥ 90% / ≤ 10% | Đề xuất |
| Chụp lại ảnh in: MAE / đúng mức | ≤ 5 điểm % / ≥ 70% | Đề xuất |
| So với chuyên gia (nếu làm) | MAE ≤ 5 điểm % | Đề xuất |

Các ngưỡng "Đề xuất" do nhóm đặt để có tiêu chí đo được; **phải thống nhất trong nhóm trước khi đưa vào hồ sơ**.

## 2. Cách làm từng hình thức

### 2.1 Tập kiểm tra độc lập
Tập test (10%) được `s02` tách theo nhóm cây, không dùng khi huấn luyện. Chạy ô 8–10 của notebook Colab.
PlantDoc (ô 12) là ảnh từ nguồn khác, loại ảnh trùng trước khi đánh giá – gần với "ảnh lạ" nhất mà không cần ra ruộng.

### 2.2 Độ bền
```bash
python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx \
    --data datasets/agrismart/dataset.yaml --split test --out reports/robustness
```
Các phép biến đổi (lệch nét, rung tay, thiếu/thừa sáng, tương phản thấp, chụp xa, nén JPEG, nghiêng 20°,
nhiễu) không làm đổi tỷ lệ diện tích vết bệnh/lá, nên đáp án tính từ nhãn gốc vẫn đúng cho ảnh đã biến đổi.
Script cũng in **ngưỡng Blur Check gợi ý** từ phân bố điểm nét của ảnh gốc và ảnh làm mờ: cập nhật
`DEFAULT_BLUR_THRESHOLD` (ai/agrismart/blur.py) hoặc chạy máy chủ với `--blur-threshold`.

### 2.3 Chụp lại ảnh in / màn hình
1. `python ai/scripts/s10_demo_pack.py --data datasets/agrismart/dataset.yaml --n 24 --source-map plantseg_=plantseg`
2. Mở `demo/samples/to_in_A4.html`, in màu khổ A4 (hoặc hiển thị từng ảnh toàn màn hình trên máy tính).
3. Mỗi ảnh chụp 3 lần bằng điện thoại: ánh sáng phòng, gần cửa sổ, nghiêng ~20°. Đặt tên `s07_1.jpg`, `s07_2.jpg`…
4. Chép vào `demo/chup_lai/` rồi chạy:
```bash
python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx --recapture-dir demo/chup_lai --out reports/recapture
```
Giới hạn cần ghi trong báo cáo: ảnh in có màu và độ phản xạ khác lá thật; kết quả cho biết ảnh hưởng của camera
và cách chụp, **không** thay được ảnh lá thật ngoài ruộng.

### 2.4 Tốc độ
```bash
python ai/scripts/s08_benchmark.py latency --model web/models/model_int8.onnx \
    --images datasets/agrismart/images/test --runs 30 --out reports/latency.json
```
Trên trang web, mục "Chi tiết kỹ thuật" hiển thị thời gian từng chặng và cả vòng (gồm truyền ảnh qua Wi-Fi).

Đo **cả bản INT8 và bản FP32** (`s07_export.py --format onnx-fp32`) trên đúng laptop chạy Demo: với onnxruntime trên
CPU, bản INT8 dạng QDQ không chắc nhanh hơn FP32 (khi dựng tài liệu, trên máy ảo 2 vCPU, bản INT8 640 px chậm hơn
bản FP32). Bản INT8 đáp ứng tiêu chí < 10 MB; chọn bản chạy Demo theo số đo, ghi rõ lựa chọn trong hồ sơ. Nếu cả hai
chưa đạt < 100 ms: giảm kích thước đầu vào khi xuất (`--imgsz 480` hoặc `416`) rồi đánh giá lại độ chính xác.

### 2.5 Chuyên gia đánh giá mù (tùy chọn)
1. Gửi thư mục `demo/samples` (chỉ ảnh, **không** gửi manifest.json vì có đáp án) cho cán bộ BVTV.
2. Chuyên gia điền `demo/danh_gia_chuyen_gia.csv` (chép từ `danh_gia_chuyen_gia_mau.csv`): % tổn thương ước lượng
   và mức (Khỏe / Khởi phát / Nhiễm nhẹ / Nhiễm nặng) cho từng ảnh.
3. Chạy hệ thống trên chính bộ ảnh mẫu rồi lập báo cáo:
```bash
python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx --recapture-dir demo/samples \
    --condition-name anh_mau --out reports/samples
python ai/scripts/s11_test_report.py --reports reports --out reports/bao_cao_kiem_thu.md
```
Báo cáo so sánh 3 phía: hệ thống – chuyên gia – đáp án từ nhãn. Sai lệch lớn giữa chuyên gia và nhãn là thông tin
quý: cho thấy nhãn cần rà lại hoặc cách ước lượng bằng mắt khác cách đếm pixel.

## 3. Những gì KHÔNG thay thế được (nêu trong hồ sơ)

- Ảnh lá thật tại Đà Nẵng: giống cây, giai đoạn bệnh, nền ruộng, lá ướt, nhiều lá chồng nhau.
- Mức hại cấp đồng ruộng: Severity của một lá chưa phải tỷ lệ bệnh/chỉ số bệnh theo phương pháp điều tra
  (QCVN 01-38:2010/BNNPTNT – kiểm tra tình trạng hiệu lực trước khi viện dẫn).
- Hành vi người dùng thật (nông dân) với giao diện.

Đề xuất cho giai đoạn sau: khi có điều kiện, chụp tối thiểu 30 lá/bệnh tại ruộng, đưa vào tập kiểm tra riêng và
chạy lại đúng các script trên – không cần sửa mã.
