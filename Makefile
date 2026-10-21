# AgriSmart Shield – lệnh tắt cho Ubuntu / macOS (GNU make). Windows: dùng scripts/phan1_windows.ps1
# và các lệnh "python ..." tương ứng trong README.
#   make help        xem danh sách lệnh
#   make setup       tạo .venv + cài thư viện Phần 1 và máy chủ web (không cần GPU, không cần PyTorch);
#                    các lệnh khác tự chạy bước này nếu chưa cài
# Dùng sẵn một môi trường khác (conda, venv khác): không cần make, chạy thẳng các lệnh "python ..." trong README.
#   make check       kiểm thử + kiểm tra tri thức + chạy thử pipeline dữ liệu tổng hợp
#   make web         chạy trang web (Phần 3)
PYTHON ?= python3
VENV   ?= .venv
PY     := $(VENV)/bin/python
SMOKE  ?= .smoke
DATA   ?= datasets/agrismart/dataset.yaml
MODEL  ?= web/models/model_int8.onnx
PORT   ?= 8000

.DEFAULT_GOAL := help
.PHONY: help setup setup-train test kb smoke check catalog labelme web web-dev demo-pack robustness recapture report clean

help:
	@echo "make setup         Tạo môi trường ảo $(VENV), cài thư viện Phần 1 + máy chủ web (web/requirements.txt)"
	@echo "make setup-train   Cài thêm ultralytics + PyTorch (ai/requirements.txt) – nặng, chỉ cần khi huấn luyện trên máy"
	@echo "make test          Chạy kiểm thử Python (pytest: ai/tests + web/tests)"
	@echo "make kb            Kiểm tra cơ sở tri thức IPM (knowledge_base/ipm_rules.json)"
	@echo "make smoke         Chạy thử s01 -> s04 trên dữ liệu tổng hợp (thư mục $(SMOKE))"
	@echo "make check         test + kb + smoke"
	@echo "make catalog       Liệt kê các nguồn dữ liệu (data_sources/catalog.yaml)"
	@echo "make labelme       Cài và mở Labelme trên thư mục data_raw/can_gan_nhan"
	@echo "make web           Chạy trang web ở cổng $(PORT) (mô hình trong web/models/, không có thì chế độ thử)"
	@echo "make web-dev       Như make web nhưng tự khởi động lại khi sửa mã"
	@echo "make demo-pack     Tạo bộ ảnh mẫu có đáp án demo/samples từ tập test ($(DATA))"
	@echo "make robustness    Kiểm thử độ bền 10 điều kiện chụp mô phỏng ($(MODEL))"
	@echo "make recapture     Đánh giá ảnh chụp lại trong demo/chup_lai"
	@echo "make report        Gộp kết quả thành reports/bao_cao_kiem_thu.md"
	@echo "make clean         Xóa dữ liệu chạy thử và bộ nhớ đệm"

# Mọi lệnh đều cần thư viện Phần 1 + máy chủ web: tệp đánh dấu $(STAMP) bảo đảm đã cài (và cài lại khi
# requirements thay đổi). Trước đây các lệnh chỉ tạo .venv rỗng nên "make web" lần đầu báo thiếu uvicorn.
STAMP  := $(VENV)/.da_cai_thu_vien

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

$(STAMP): web/requirements.txt ai/requirements-core.txt | $(PY)
	$(PY) -m pip install -r web/requirements.txt
	touch $(STAMP)

setup: $(STAMP)

setup-train: setup
	$(PY) -m pip install -r ai/requirements.txt

test: $(STAMP)
	$(PY) -m pytest -q

kb: $(STAMP)
	$(PY) knowledge_base/validate_kb.py

smoke: $(STAMP)
	rm -rf $(SMOKE)
	$(PY) ai/scripts/make_synthetic_dataset.py --out $(SMOKE)/raw --n 30
	$(PY) ai/scripts/s01_convert_masks_to_yolo.py --images $(SMOKE)/raw/images --lesion-masks $(SMOKE)/raw/masks --mask-mode binary --class-map ai/configs/class_map.yaml --bootstrap-leaf --out $(SMOKE)/all
	$(PY) ai/scripts/s02_split_dataset.py --src $(SMOKE)/all --out $(SMOKE)/ds --class-map ai/configs/class_map.yaml --group-regex '^.*?__(plant\d+)_'
	$(PY) ai/scripts/s03_augment_train.py --data $(SMOKE)/ds --copies 2
	$(PY) ai/scripts/s04_check_dataset.py --data-yaml $(SMOKE)/ds/dataset.yaml

check: test kb smoke

catalog: $(STAMP)
	$(PY) ai/scripts/datasets_catalog.py list

labelme: $(STAMP)
	$(PY) -m pip install labelme
	mkdir -p data_raw/can_gan_nhan
	$(VENV)/bin/labelme data_raw/can_gan_nhan --labels ai/configs/labels/labelme_labels.txt

web: $(STAMP)
	$(PY) web/run.py --port $(PORT)

web-dev: $(STAMP)
	$(PY) web/run.py --port $(PORT) --reload

demo-pack: $(STAMP)
	$(PY) ai/scripts/s10_demo_pack.py --data $(DATA) --split test --n 24 --source-map plantseg_=plantseg

robustness: $(STAMP)
	$(PY) ai/scripts/s09_robustness.py --model $(MODEL) --data $(DATA) --split test --out reports/robustness

recapture: $(STAMP)
	$(PY) ai/scripts/s09_robustness.py --model $(MODEL) --recapture-dir demo/chup_lai --manifest demo/samples/manifest.json --out reports/recapture

report: $(STAMP)
	$(PY) ai/scripts/s11_test_report.py --reports reports --out reports/bao_cao_kiem_thu.md

clean:
	rm -rf $(SMOKE) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
