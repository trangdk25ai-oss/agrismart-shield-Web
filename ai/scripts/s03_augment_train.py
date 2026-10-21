"""BƯỚC 1.2 – Tăng cường dữ liệu (offline) bằng albumentations, CHỈ cho tập train.

Biến đổi đúng theo quy trình:
    - RandomBrightnessContrast : nắng gắt / bóng râm nhiệt đới
    - GaussianBlur / MotionBlur : rung tay khi chụp
    - Rotate ±30°, HorizontalFlip : biến đổi hình học

Điểm kỹ thuật quan trọng: albumentations KHÔNG biến đổi trực tiếp polygon. Script này
    polygon -> mask từng instance -> biến đổi đồng thời ảnh + masks -> trích lại polygon
nên nhãn luôn khớp ảnh sau khi xoay/lật. Instance bị xoay ra ngoài khung (diện tích < min-area) bị loại.

Ví dụ:
    python scripts/s03_augment_train.py --data datasets/agrismart --copies 2 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.io_utils import imread, imwrite  # noqa: E402
from agrismart.yolo_io import (  # noqa: E402
    SegInstance,
    list_images,
    mask_to_polygons,
    normalize_polygon,
    polygon_to_mask,
    read_label_file,
    write_label_file,
)

AUG_TAG = "__aug"


def build_pipeline(seed: int) -> A.Compose:
    return A.Compose(
        [
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                ],
                p=0.4,
            ),
            A.Rotate(limit=30, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.7),
            A.HorizontalFlip(p=0.5),
        ],
        seed=seed,
    )


def augment_one(image: np.ndarray, instances: list[SegInstance], tf: A.Compose, min_area: float):
    h, w = image.shape[:2]
    masks = [polygon_to_mask(inst.polygon, h, w) for inst in instances]
    res = tf(image=image, masks=masks)
    img2, masks2 = res["image"], res["masks"]
    h2, w2 = img2.shape[:2]
    out: list[SegInstance] = []
    for inst, m in zip(instances, masks2):
        for poly in mask_to_polygons(np.asarray(m), min_area_px=min_area):
            out.append(SegInstance(inst.cls_id, normalize_polygon(poly, w2, h2)))
    return img2, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="thư mục dữ liệu đã chia (đầu ra s02)")
    ap.add_argument("--copies", type=int, default=2, help="số bản tăng cường cho mỗi ảnh train")
    ap.add_argument("--min-area", type=float, default=16.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir, lbl_dir = args.data / "images" / "train", args.data / "labels" / "train"
    sources = [p for p in list_images(img_dir) if AUG_TAG not in p.stem]  # chạy lại không nhân bản chồng
    tf = build_pipeline(args.seed)
    n_written = n_skipped = 0
    for img_path in sources:
        image = imread(img_path, cv2.IMREAD_COLOR)
        instances = read_label_file(lbl_dir / f"{img_path.stem}.txt")
        if image is None or not instances:
            continue
        for k in range(1, args.copies + 1):
            img2, inst2 = augment_one(image, instances, tf, args.min_area)
            if not inst2:
                n_skipped += 1
                continue
            stem = f"{img_path.stem}{AUG_TAG}{k}"
            imwrite(img_dir / f"{stem}.jpg", img2, [cv2.IMWRITE_JPEG_QUALITY, 95])
            write_label_file(lbl_dir / f"{stem}.txt", inst2)
            n_written += 1
    print(f"Ảnh gốc train: {len(sources)} | ảnh tăng cường đã ghi: {n_written} | bỏ qua (mất hết nhãn): {n_skipped}")


if __name__ == "__main__":
    main()
