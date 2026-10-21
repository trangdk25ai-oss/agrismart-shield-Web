"""Kiểm tra độ nét ảnh (Blur Check) bằng phương sai toán tử Laplace.

Ảnh mờ (rung tay, lệch nét) có ít cạnh sắc -> đáp ứng Laplace nhỏ -> phương sai thấp.
Ngưỡng mặc định 100 là điểm khởi đầu phổ biến; PHẢI hiệu chỉnh lại vì giá trị phụ thuộc độ phân giải,
kết cấu bề mặt lá và camera: s09_robustness.py in phân bố điểm của ảnh gốc và ảnh làm mờ để chọn ngưỡng
(xem docs/ke_hoach_kiem_thu.md). Máy chủ web đọc ngưỡng từ biến môi trường AGS_BLUR_THRESHOLD.
Ảnh được thu về cạnh dài 512 px trước khi tính để giá trị ít phụ thuộc độ phân giải camera.
Nhân Laplace 3x3 [0,1,0;1,-4,1;0,1,0] (cv2.Laplacian ksize=1).
"""

from __future__ import annotations

import cv2
import numpy as np

DEFAULT_BLUR_THRESHOLD = 100.0


def laplacian_variance(image_bgr: np.ndarray, max_side: int = 512) -> float:
    h, w = image_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())


def is_blurry(image_bgr: np.ndarray, threshold: float = DEFAULT_BLUR_THRESHOLD) -> tuple[bool, float]:
    v = laplacian_variance(image_bgr)
    return v < threshold, v
