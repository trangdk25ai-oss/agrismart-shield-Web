"""Sinh mask PHIẾN LÁ sơ bộ (bootstrap) cho bộ dữ liệu chỉ có nhãn vết bệnh.

Vì sao cần: PlantSeg chỉ có mask vết bệnh, PlantDoc chỉ có bounding box. Thuật toán Severity Index
(Bước 4) bắt buộc phải có mask phiến lá (Mask_Leaf) => phải bổ sung lớp "leaf" vào nhãn.

Cách làm: chỉ số Excess Green (ExG = 2g - r - b trên tọa độ màu chuẩn hóa) + ngưỡng Otsu,
hợp với mask vết bệnh (vết bệnh nâu/đen không "xanh" nhưng vẫn thuộc phiến lá), đóng hình thái,
lấp lỗ, giữ các thành phần lớn.

GIỚI HẠN (bắt buộc rà soát thủ công): nền cỏ/lá khác cùng màu xanh sẽ bị gộp; lá vàng úa có ExG thấp.
Kết quả chỉ dùng làm nhãn KHỞI TẠO để người gán nhãn chỉnh sửa trên CVAT/Roboflow, hoặc thay bằng
SAM/SAM2 (prompt bằng điểm/hộp) nếu có GPU.
"""

from __future__ import annotations

import cv2
import numpy as np


def excess_green_mask(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr.astype(np.float32)
    s = img.sum(axis=2) + 1e-6
    b, g, r = img[..., 0] / s, img[..., 1] / s, img[..., 2] / s
    exg = 2.0 * g - r - b
    exg8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(exg8, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return m.astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    m = np.pad((mask > 0).astype(np.uint8), 1)
    background = m.copy()
    # Tô nền nối với biên ảnh thành 2; phần còn lại có giá trị 0 chính là "lỗ" bên trong vật thể
    cv2.floodFill(background, None, (0, 0), 2)
    holes = background == 0
    filled = (m > 0) | holes
    return filled[1:-1, 1:-1].astype(np.uint8)


def bootstrap_leaf_mask(
    image_bgr: np.ndarray,
    lesion_mask: np.ndarray | None = None,
    min_component_frac: float = 0.05,
    close_frac: float = 0.02,
) -> tuple[np.ndarray, bool]:
    """Trả về (mask_lá uint8 0/1, cần_rà_soát: bool)."""
    h, w = image_bgr.shape[:2]
    m = excess_green_mask(image_bgr)
    if lesion_mask is not None:
        m |= (lesion_mask > 0).astype(np.uint8)
    k = max(3, int(close_frac * min(h, w))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = fill_holes(m)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros((h, w), np.uint8), True
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_component_frac * h * w]
    if not keep:
        keep = [int(np.argmax(areas)) + 1]
    out = np.isin(labels, keep).astype(np.uint8)

    frac = out.mean()
    needs_review = bool(frac < 0.10 or frac > 0.95 or len(keep) > 3)
    if lesion_mask is not None and lesion_mask.any():
        covered = (out[lesion_mask > 0] > 0).mean()
        needs_review |= bool(covered < 0.95)  # vết bệnh phải nằm trong lá
    return out, needs_review
