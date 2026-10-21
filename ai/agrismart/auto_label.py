"""Sinh NHÃN NHÁP (phiến lá + vết bệnh) cho ảnh chỉ có nhãn cấp ảnh, để người gán nhãn rà soát trên Labelme.

Dùng cho các nguồn phân loại (bộ dưa chuột, bộ chanh…): biết ảnh thuộc bệnh nào (theo thư mục) nhưng không có
vùng vết bệnh. Khác với PlantSeg (mask vết bệnh do người gán), nhãn ở đây do MÁY sinh nên bắt buộc rà soát.

Cách làm:
  - Phiến lá: ảnh nền trắng (đã tách nền) -> lá = vùng không phải màu trắng, lấp lỗ;
    ảnh chụp ngoài vườn -> ExG + Otsu (leaf_bootstrap.py).
  - Vết bệnh: điểm ảnh trong phiến lá có sắc độ ngoài dải xanh lá (HSV, H ∉ [33; 95] thang OpenCV) hoặc độ
    bão hòa thấp, trừ vùng trắng (lóa sáng, lỗ thủng); bỏ dải viền mỏng ở mép lá; bỏ đốm quá nhỏ.
  - Tên bệnh lấy theo thư mục chứa ảnh (data_raw/can_gan_nhan/<lớp_đích>/).

Giới hạn: lá vàng úa, cháy nắng, bóng đổ, vết côn trùng cũng bị tính là "vết bệnh"; ảnh ngoài vườn có nhiều lá
chồng nhau thì mask lá có thể gộp nhiều lá. Vì vậy mỗi tệp nhãn nháp mang cờ "da_ra_soat": false và s01b bỏ qua
cho đến khi người gán nhãn rà soát, sửa và tích cờ đó trên Labelme.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import LEAF_CLASS
from .leaf_bootstrap import bootstrap_leaf_mask, fill_holes
from .yolo_io import mask_to_polygons

REVIEW_FLAG = "da_ra_soat"
GREEN_HUE = (33, 95)  # lá khỏe H ≈ 40–60; nâu, vàng nâu của mô hoại tử H < 30
MIN_GREEN_SAT = 40


def near_white(image_bgr: np.ndarray, min_level: int = 200, max_spread: int = 30) -> np.ndarray:
    """Điểm ảnh gần trắng/xám rất sáng: kênh tối nhất >= min_level và chênh lệch giữa các kênh <= max_spread."""
    lo = image_bgr.min(axis=2).astype(np.int16)
    hi = image_bgr.max(axis=2).astype(np.int16)
    return (lo >= min_level) & ((hi - lo) <= max_spread)


def bluish(image_bgr: np.ndarray, min_sat: int = 25) -> np.ndarray:
    """Sắc độ xanh dương/tím (H ∈ [95; 150] thang OpenCV): không phải mô lá; thường là bóng đổ do tách nền hoặc nền."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 95) & (hsv[..., 0] <= 150) & (hsv[..., 1] >= min_sat)


def is_white_background(image_bgr: np.ndarray, band_frac: float = 0.03, min_ratio: float = 0.6) -> bool:
    """Ảnh đã tách nền: phần lớn dải biên ảnh là màu trắng."""
    h, w = image_bgr.shape[:2]
    b = max(2, int(band_frac * min(h, w)))
    nw = near_white(image_bgr)
    band = np.concatenate([nw[:b].ravel(), nw[-b:].ravel(), nw[:, :b].ravel(), nw[:, -b:].ravel()])
    return bool(band.mean() >= min_ratio)


def _keep_large(mask: np.ndarray, min_frac: float) -> np.ndarray:
    h, w = mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_frac * h * w] or [int(np.argmax(areas)) + 1]
    return np.isin(labels, keep).astype(np.uint8)


def leaf_mask_auto(image_bgr: np.ndarray, min_component_frac: float = 0.03) -> tuple[np.ndarray, str]:
    """Trả về (mask phiến lá 0/1, phương pháp: 'nen_trang' | 'exg_otsu')."""
    h, w = image_bgr.shape[:2]
    if is_white_background(image_bgr):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        light_grey = (hsv[..., 1] < 30) & (hsv[..., 2] > 150)  # quầng xám sáng quanh lá khi tách nền
        fg = (~near_white(image_bgr) & ~bluish(image_bgr) & ~light_grey).astype(np.uint8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(0.01 * min(h, w))) | 1,) * 2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
        return _keep_large(fill_holes(fg), min_component_frac), "nen_trang"
    leaf, _ = bootstrap_leaf_mask(image_bgr, min_component_frac=min_component_frac)
    return leaf, "exg_otsu"


def lesion_mask_auto(image_bgr: np.ndarray, leaf: np.ndarray, edge_frac: float) -> np.ndarray:
    """Vùng đổi màu trong phiến lá (đã bỏ dải viền rộng edge_frac × kích thước lá và vùng trắng)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    green = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) & (sat >= MIN_GREEN_SAT)
    out = np.zeros(leaf.shape, np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(leaf.astype(np.uint8), connectivity=8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = near_white(image_bgr)
    blue = bluish(image_bgr)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        r = max(2, round(edge_frac * min(bw, bh)))
        inner = cv2.erode(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
        spots = (inner & ~green & ~white & ~blue).astype(np.uint8)
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, k3)
        m, lab2, st2, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
        min_area = max(16, 0.0005 * stats[i, cv2.CC_STAT_AREA])
        for j in range(1, m):
            if st2[j, cv2.CC_STAT_AREA] >= min_area:
                out[lab2 == j] = 1
    return out


def auto_label(image_bgr: np.ndarray) -> dict:
    """Nhãn nháp cho một ảnh: các mask lá, mask vết bệnh, phương pháp, tỷ lệ tổn thương ước tính."""
    leaf, method = leaf_mask_auto(image_bgr)
    # ảnh nền trắng: mask lá sát biên thật -> chỉ bỏ dải viền mỏng; ảnh ngoài vườn: mask lá sơ bộ hay lấn nền
    lesion = lesion_mask_auto(image_bgr, leaf, edge_frac=0.01 if method == "nen_trang" else 0.02) if leaf.any() else leaf
    leaf_px, lesion_px = int(leaf.sum()), int((lesion & leaf).sum())
    return {
        "leaf": leaf,
        "lesion": lesion,
        "method": method,
        "leaf_frac": leaf_px / leaf.size,
        "severity_pct": 100.0 * lesion_px / leaf_px if leaf_px else 0.0,
    }


def labelme_json(result: dict, lesion_label: str, image_name: str, height: int, width: int,
                 min_area_px: float = 16.0) -> dict:
    """Tệp nhãn Labelme (định dạng đọc được bởi Labelme 5.x–7.x và s01b) với cờ chưa rà soát."""

    def shapes_of(mask: np.ndarray, label: str) -> list[dict]:
        out = []
        for poly in mask_to_polygons(mask, min_area_px=min_area_px):
            if len(poly) < 3:
                continue
            out.append({"label": label, "points": [[round(float(x), 1), round(float(y), 1)] for x, y in poly],
                        "group_id": None, "description": "nhãn nháp tự động", "shape_type": "polygon",
                        "flags": {}, "mask": None})
        return out

    shapes = shapes_of(result["leaf"], LEAF_CLASS) + shapes_of(result["lesion"], lesion_label)
    return {
        "version": "5.5.0",
        "flags": {REVIEW_FLAG: False},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
        "auto_label": {"tool": "s00c_auto_label", "method": result["method"],
                       "severity_pct_estimate": round(result["severity_pct"], 2)},
    }
