"""BƯỚC 2 (bổ sung) – Đánh giá mô hình trên bộ dữ liệu độc lập chỉ có HỘP BAO LÁ (PlantDoc…).

Kết quả: độ chính xác nhận dạng bệnh ở mức lá, độ nhạy/độ chính xác theo bệnh, tỷ lệ báo nhầm trên lá khỏe.
Chạy được với best.pt và model_quant_int8.tflite.

Ví dụ (PlantDoc, tập TEST):
  python ai/scripts/s06b_eval_bbox.py --weights runs/segment/agrismart_v8n/weights/best.pt \
      --csv data_raw/plantdoc/test_labels.csv --images data_raw/plantdoc/TEST \
      --map ai/configs/plantdoc_map.yaml --out reports/eval_plantdoc.json

Lưu ý: ảnh PlantDoc trùng với tập huấn luyện làm kết quả bị thổi phồng. Chạy trước:
  python ai/scripts/datasets_catalog.py dedup --a data_raw/plantdoc/TEST --b datasets/agrismart/images/train \
      --out reports/trung_plantdoc_train.csv
rồi truyền --exclude-list reports/trung_plantdoc_train.csv để bỏ các ảnh trùng.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.bbox_eval import BoxEvalAccumulator, label_in_box, read_box_csv, scale_box  # noqa: E402
from agrismart.io_utils import imread  # noqa: E402
from agrismart.severity import MaskInstance  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-lesion-frac", type=float, default=0.005)
    ap.add_argument("--exclude-list", type=Path, help="CSV của datasets_catalog.py dedup (cột image_a): ảnh bị loại")
    ap.add_argument("--out", type=Path, default=Path("reports/eval_bbox.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    cmap = yaml.safe_load(args.map.read_text(encoding="utf-8"))["class_map"]
    boxes = read_box_csv(args.csv, cmap)
    excluded = set()
    if args.exclude_list:
        with open(args.exclude_list, encoding="utf-8") as f:
            excluded = {Path(r["image_a"]).name for r in csv.DictReader(f)}
        boxes = {k: v for k, v in boxes.items() if k not in excluded}
    model = YOLO(args.weights, task="segment")
    names = {int(k): v for k, v in model.names.items()}
    unknown = sorted({t for t in cmap.values() if t != "healthy"} - set(names.values()))
    if unknown:
        print("LỖI: lớp trong file ánh xạ không có trong mô hình:", unknown)
        return 1

    acc = BoxEvalAccumulator()
    missing = 0
    for fname, items in boxes.items():
        path = args.images / fname
        im = imread(path)
        if im is None:
            missing += 1
            continue
        r = model.predict(im, imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)[0]
        insts = []
        if r.masks is not None:
            for c, p, m in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.masks.data.cpu().numpy()):
                insts.append(MaskInstance(names[int(c)], float(p), m > 0.5))
        real_wh = (im.shape[1], im.shape[0])
        for it in items:
            pred, _ = label_in_box(insts, scale_box(it["box"], it["csv_wh"], real_wh), args.min_lesion_frac)
            acc.update(it["label"], pred)

    report = {"weights": str(args.weights), "csv": str(args.csv), "images_missing": missing,
              "images_excluded_as_duplicates": len(excluded), **acc.result()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
