"""BƯỚC 1 (mở rộng) – Nhập nhãn TỰ GÁN (ảnh chụp tại địa phương, PlantDoc đã gán thêm…) vào bộ dữ liệu YOLOv8-seg.

Định dạng đầu vào:
  --format coco     : file JSON xuất từ CVAT ("COCO 1.0"), Roboflow ("COCO Segmentation") …
  --format labelme  : thư mục chứa file .json của Labelme (gán nhãn offline trên máy tính)
  --format via      : file JSON của VGG Image Annotator 1.x/2.x (ví dụ bộ BRACOT); --via-label-attr là tên
                      thuộc tính chứa nhãn (bỏ trống nếu mỗi vùng chỉ có 1 thuộc tính)

Nhãn trong công cụ phải trùng tên lớp trong class_map.yaml (leaf, <crop>__<disease>); nếu khác, khai báo
ánh xạ ở mục label_aliases của class_map.yaml. Ảnh chỉ có nhãn "leaf" (lá khỏe) vẫn được nhập – đây là
mẫu âm giúp mô hình bớt báo nhầm.

Ví dụ:
  python ai/scripts/s01b_import_annotations.py --format labelme --labelme-dir data_raw/tu_gan_nhan \
      --class-map ai/configs/class_map.yaml --out datasets/agrismart_all --prefix dn_
  python ai/scripts/s01b_import_annotations.py --format coco --coco-json data_raw/cvat/annotations/instances_default.json \
      --images data_raw/cvat/images --class-map ai/configs/class_map.yaml --out datasets/agrismart_all --prefix cvat_

Ghi CÙNG thư mục --out với s01 (PlantSeg) để gộp thành một bộ dữ liệu, rồi chạy s02 như bình thường.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart import LEAF_CLASS  # noqa: E402
from agrismart.annotation_import import ImportStats, read_coco, read_labelme_dir, read_via, to_instances  # noqa: E402
from agrismart.leaf_bootstrap import bootstrap_leaf_mask  # noqa: E402
from agrismart.yolo_io import SegInstance, mask_to_polygons, normalize_polygon, polygon_to_mask, write_label_file  # noqa: E402
from agrismart.io_utils import imread, imwrite  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", choices=["coco", "labelme", "via"], required=True)
    ap.add_argument("--coco-json", type=Path)
    ap.add_argument("--images", type=Path, help="thư mục ảnh gốc (định dạng coco, via)")
    ap.add_argument("--labelme-dir", type=Path)
    ap.add_argument("--via-json", type=Path)
    ap.add_argument("--via-label-attr", default=None, help="tên thuộc tính vùng chứa nhãn trong VIA")
    ap.add_argument("--class-map", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="own_", help="tiền tố tên file, tránh trùng với ảnh PlantSeg")
    ap.add_argument("--bootstrap-leaf", action="store_true", help="ảnh thiếu nhãn leaf: sinh mask lá sơ bộ (ExG)")
    ap.add_argument("--include-unreviewed", action="store_true",
                    help="nhập cả nhãn nháp tự động CHƯA rà soát (cờ da_ra_soat=false) – không khuyến nghị")
    args = ap.parse_args()

    cmap = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    name_to_id = {v: int(k) for k, v in cmap["target_names"].items()}
    aliases = cmap.get("label_aliases") or {}
    leaf_id = name_to_id[LEAF_CLASS]
    stats = ImportStats()

    if args.format == "coco":
        if not (args.coco_json and args.images):
            ap.error("--format coco cần --coco-json và --images")
        items = read_coco(args.coco_json, args.images, stats)
    elif args.format == "via":
        if not (args.via_json and args.images):
            ap.error("--format via cần --via-json và --images")
        items = read_via(args.via_json, args.images, stats, args.via_label_attr)
    else:
        if not args.labelme_dir:
            ap.error("--format labelme cần --labelme-dir")
        items = read_labelme_dir(args.labelme_dir, stats, include_unreviewed=args.include_unreviewed)

    out_img, out_lbl = args.out / "images", args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    review = []
    for it in items:
        insts = to_instances(it, name_to_id, aliases, stats)
        has_leaf = any(i.cls_id == leaf_id for i in insts)
        lesions = [i for i in insts if i.cls_id != leaf_id]
        if not insts:
            stats.add("skip_no_valid_label")
            continue

        image = None
        if it.image_path is not None and it.image_path.exists():
            image = imread(it.image_path, cv2.IMREAD_COLOR)
        elif it.image_bytes:
            image = cv2.imdecode(np.frombuffer(it.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            stats.add("skip_image_missing")
            continue
        if image.shape[:2] != (it.height, it.width):
            stats.add("warn_size_mismatch")  # kích thước trong nhãn khác ảnh: tọa độ đã chuẩn hóa theo nhãn

        if not has_leaf:
            if not args.bootstrap_leaf:
                stats.add("skip_no_leaf")
                continue
            union = np.zeros(image.shape[:2], np.uint8)
            for les in lesions:
                union |= polygon_to_mask(les.polygon, *image.shape[:2])
            leaf_mask, needs_review = bootstrap_leaf_mask(image, union)
            for poly in mask_to_polygons(leaf_mask):
                insts.append(SegInstance(leaf_id, normalize_polygon(poly, image.shape[1], image.shape[0])))
            stats.add("leaf_bootstrapped")
            if needs_review:
                review.append(it.key)

        stem = args.prefix + it.key.replace("\\", "/").replace("/", "__").replace(" ", "_")
        ext = it.image_path.suffix.lower() if it.image_path is not None else ".jpg"
        if it.image_path is not None and it.image_path.exists():
            shutil.copy2(it.image_path, out_img / f"{stem}{ext}")
        else:
            imwrite(out_img / f"{stem}.jpg", image)
        write_label_file(out_lbl / f"{stem}.txt", insts)
        stats.add("images_ok")
        stats.add("images_healthy" if not lesions else "images_diseased")
        for i in insts:
            stats.add(f"instances_{cmap['target_names'][i.cls_id]}")

    report = dict(sorted(stats.counts.items()))
    (args.out / "import_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("skip_nhan_nhap_chua_ra_soat"):
        print(f"Bỏ qua {report['skip_nhan_nhap_chua_ra_soat']} tệp nhãn nháp chưa rà soát (cờ da_ra_soat chưa tích trên Labelme).")
    if report.get("nhan_nhap_chua_ra_soat_van_nhap"):
        print(f"CẢNH BÁO: đã nhập {report['nhan_nhap_chua_ra_soat_van_nhap']} nhãn nháp CHƯA rà soát – ghi rõ trong báo cáo.")
    unknown = [k for k in report if k.startswith("unknown_label:")]
    if unknown:
        print("CẢNH BÁO: có nhãn không thuộc class_map – thêm lớp bằng s00 hoặc khai báo label_aliases:",
              ", ".join(k.split(":", 1)[1] for k in unknown))
    if review:
        print(f"Ảnh có mask lá sinh tự động cần rà soát: {len(review)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
