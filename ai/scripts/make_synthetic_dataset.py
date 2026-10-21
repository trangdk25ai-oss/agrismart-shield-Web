"""Sinh bộ dữ liệu TỔNG HỢP (lá elip xanh + đốm bệnh nâu) để chạy thử toàn bộ pipeline trong CI
khi chưa có dữ liệu thật. KHÔNG dùng để huấn luyện mô hình thật.

Đầu ra có cùng cấu trúc với dữ liệu thật trước bước s01:
    <out>/images/<folder>/*.jpg     (folder = tên bệnh, dùng cho --mask-mode binary)
    <out>/masks/<folder>/*.png      (mask vết bệnh nhị phân, dùng với --mask-mode binary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.io_utils import imwrite  # noqa: E402

DISEASES = ["Tomato early blight", "Cucumber downy mildew", "Citrus anthracnose"]


def make_sample(rng: np.random.Generator, size: int = 320):
    img = np.full((size, size, 3), (60, 90, 120), np.uint8)  # nền đất nâu xám (BGR)
    img = (img + rng.normal(0, 8, img.shape)).clip(0, 255).astype(np.uint8)
    leaf = np.zeros((size, size), np.uint8)
    c = (int(size * rng.uniform(0.4, 0.6)), int(size * rng.uniform(0.4, 0.6)))
    axes = (int(size * rng.uniform(0.28, 0.4)), int(size * rng.uniform(0.15, 0.25)))
    cv2.ellipse(leaf, c, axes, float(rng.uniform(0, 180)), 0, 360, 1, -1)
    green = np.array([40, int(rng.uniform(140, 190)), 50], np.uint8)
    img[leaf > 0] = green
    lesion = np.zeros_like(leaf)
    ys, xs = np.nonzero(leaf)
    for _ in range(int(rng.integers(2, 7))):
        k = int(rng.integers(0, len(xs)))
        r = int(rng.integers(4, 14))
        cv2.circle(lesion, (int(xs[k]), int(ys[k])), r, 1, -1)
    lesion &= leaf
    img[lesion > 0] = (30, 60, 110)
    return img, leaf, lesion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=30, help="số ảnh mỗi bệnh")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    for name in DISEASES:
        for sub in ("images", "masks"):
            (args.out / sub / name).mkdir(parents=True, exist_ok=True)
        for i in range(args.n):
            img, _leaf, lesion = make_sample(rng)
            stem = f"plant{i // 3:03d}_shot{i % 3}"
            imwrite(args.out / "images" / name / f"{stem}.jpg", img)
            imwrite(args.out / "masks" / name / f"{stem}.png", lesion * 255)
    print(f"Đã sinh {args.n * len(DISEASES)} ảnh tại {args.out}")


if __name__ == "__main__":
    main()
