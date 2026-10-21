"""BƯỚC 1.1 – Chuyển mask vết bệnh (PlantSeg hoặc tự gán nhãn) sang nhãn polygon YOLOv8-seg + lớp "leaf".

Ví dụ:
    python scripts/s01_convert_masks_to_yolo.py \
        --images data_raw/plantseg/images --lesion-masks data_raw/plantseg/annotations \
        --mask-mode value --class-map configs/class_map.yaml \
        --leaf-masks data_raw/leaf_masks          # nếu đã có mask lá gán tay / SAM
        --bootstrap-leaf                          # nếu chưa có: sinh mask lá sơ bộ bằng ExG
        --out datasets/agrismart_all

Đầu ra:
    <out>/images/*.jpg, <out>/labels/*.txt, <out>/convert_report.json, <out>/leaf_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.leaf_bootstrap import bootstrap_leaf_mask  # noqa: E402
from agrismart.yolo_io import list_images, masks_to_instances, write_label_file  # noqa: E402
from agrismart.io_utils import imread  # noqa: E402


def binary_lesion(raw: np.ndarray, invert: bool = False) -> np.ndarray:
    """Mask nhị phân từ mask nguồn.

    - Mask 0/1: pixel > 0 là vết bệnh.
    - Mask 0/255 (PNG, JPG): ngưỡng 127. Với mask JPG, nén làm viền bị nhiễu; lấy mọi pixel > 0 sẽ
      phóng đại diện tích vết bệnh (thử trên mask tròn chuẩn 1.257 px: > 0 cho 1.606 px, > 127 cho 1.257 px).
    """
    m = raw > (127 if int(raw.max()) > 1 else 0)
    return (~m if invert else m).astype(np.uint8)


def find_mask(mask_root: Path, rel: Path) -> Path | None:
    for ext in (".png", ".bmp", ".tif", ".jpg"):
        p = (mask_root / rel).with_suffix(ext)
        if p.exists():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--lesion-masks", required=True, type=Path)
    ap.add_argument("--mask-mode", choices=["value", "binary"], default="value")
    ap.add_argument("--class-map", required=True, type=Path)
    ap.add_argument("--leaf-masks", type=Path, default=None)
    ap.add_argument("--bootstrap-leaf", action="store_true")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-area", type=float, default=16.0, help="bỏ đa giác nhỏ hơn N pixel (nhiễu)")
    ap.add_argument("--invert-masks", action="store_true",
                    help="(chế độ binary) mask nguồn có vết bệnh màu ĐEN trên nền trắng")
    ap.add_argument("--prefix", default="", help="tiền tố tên file đầu ra, ví dụ 'riceseg_' để truy nguồn")
    args = ap.parse_args()

    cmap = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    name_to_id = {v: int(k) for k, v in cmap["target_names"].items()}
    leaf_id = name_to_id["leaf"]
    value_to_target = {int(k): v for k, v in (cmap.get("value_to_target") or {}).items()}
    folder_to_target = cmap.get("folder_to_target") or {}

    out_img, out_lbl = args.out / "images", args.out / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    stats: Counter = Counter()
    review_rows = []
    for img_path in list_images(args.images):
        rel = img_path.relative_to(args.images)
        mask_path = find_mask(args.lesion_masks, rel)
        if mask_path is None:
            stats["skip_no_mask"] += 1
            continue
        image = imread(img_path, cv2.IMREAD_COLOR)
        lesion_raw = imread(mask_path, cv2.IMREAD_UNCHANGED)
        if image is None or lesion_raw is None:
            stats["skip_unreadable"] += 1
            continue
        if lesion_raw.ndim == 3:
            lesion_raw = lesion_raw[..., 0]
        if lesion_raw.shape[:2] != image.shape[:2]:
            lesion_raw = cv2.resize(lesion_raw, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # ---- mask vết bệnh theo lớp đích
        class_masks: dict[int, np.ndarray] = {}
        if args.mask_mode == "value":
            if mask_path.suffix.lower() in {".jpg", ".jpeg"}:
                stats["warn_value_mode_jpg_mask"] += 1  # nén JPG làm sai giá trị mã lớp ở viền vết bệnh
            for v in np.unique(lesion_raw):
                if v == 0:
                    continue
                target = value_to_target.get(int(v))
                if target is None:
                    stats[f"unmapped_value_{int(v)}"] += 1
                    continue
                cid = name_to_id[target]
                class_masks[cid] = class_masks.get(cid, 0) | (lesion_raw == v).astype(np.uint8)
        else:
            target = folder_to_target.get(img_path.parent.name)
            if target is None:
                stats["skip_unmapped_folder"] += 1
                continue
            class_masks[name_to_id[target]] = binary_lesion(lesion_raw, args.invert_masks)
        if not class_masks:
            stats["skip_no_mapped_lesion"] += 1
            continue
        lesion_union = np.zeros(image.shape[:2], np.uint8)
        for m in class_masks.values():
            lesion_union |= m

        # ---- mask phiến lá
        leaf_mask = None
        needs_review = False
        if args.leaf_masks is not None:
            lp = find_mask(args.leaf_masks, rel)
            if lp is not None:
                lm = imread(lp, cv2.IMREAD_GRAYSCALE)
                leaf_mask = binary_lesion(cv2.resize(lm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)) > 0
                leaf_mask = (leaf_mask | (lesion_union > 0)).astype(np.uint8)  # lá phải chứa vết bệnh
        if leaf_mask is None and args.bootstrap_leaf:
            leaf_mask, needs_review = bootstrap_leaf_mask(image, lesion_union)
            stats["leaf_bootstrapped"] += 1
        if leaf_mask is None:
            stats["skip_no_leaf_mask"] += 1
            continue
        class_masks[leaf_id] = leaf_mask

        instances = masks_to_instances(class_masks, min_area_px=args.min_area)
        if not any(i.cls_id != leaf_id for i in instances):
            stats["skip_lesions_too_small"] += 1
            continue

        stem = args.prefix + "__".join(rel.with_suffix("").parts).replace(" ", "_")
        dst_img = out_img / f"{stem}{img_path.suffix.lower()}"
        shutil.copy2(img_path, dst_img)
        write_label_file(out_lbl / f"{stem}.txt", instances)
        for inst in instances:
            stats[f"instances_{cmap['target_names'][inst.cls_id]}"] += 1
        stats["images_ok"] += 1
        if needs_review:
            review_rows.append({"image": dst_img.name, "reason": "leaf_bootstrap_uncertain"})

    with open(args.out / "leaf_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "reason"])
        w.writeheader()
        w.writerows(review_rows)
    (args.out / "convert_report.json").write_text(json.dumps(dict(stats), ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print(f"Ảnh cần rà soát mask lá: {len(review_rows)} (xem {args.out / 'leaf_review.csv'})")


if __name__ == "__main__":
    main()
