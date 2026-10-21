"""Chế độ THỬ KHÔNG CÓ MÔ HÌNH: tách lá và vùng đổi màu bằng luật màu sắc.

Mục đích DUY NHẤT: chạy thử toàn bộ luồng trang web (tải ảnh -> lớp phủ -> Severity -> phác đồ -> lịch)
và kiểm thử tự động TRƯỚC khi có mô hình huấn luyện ở Phần 2. KHÔNG phải chẩn đoán:
    - không nhận biết được loại bệnh (người dùng tự chọn bệnh, giao diện ghi rõ "chế độ thử");
    - vùng "vết bệnh" chỉ là vùng không còn màu xanh trong phiến lá (cháy nắng, vàng úa, bóng tối,
      vết côn trùng… đều bị tính).

Cách làm: phiến lá = ExG (2g − r − b) + Otsu (leaf_bootstrap.py); vùng đổi màu = điểm ảnh trong phiến lá
có sắc độ ngoài dải xanh lá (HSV, H ∉ [25; 95] theo thang OpenCV 0–180) hoặc độ bão hòa thấp; bỏ qua dải
viền lá rộng ≈2% kích thước lá (mask lá sơ bộ hay lấn ra nền ở mép) nên vết bệnh sát mép bị tính thiếu.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from . import LEAF_CLASS
from .leaf_bootstrap import bootstrap_leaf_mask
from .seg_runtime import Prediction
from .severity import MaskInstance

GREEN_HUE = (25, 95)
MIN_GREEN_SAT = 40
MIN_LEAF_FRAC = 0.02  # thành phần lá nhỏ hơn 2% khung bị bỏ
MIN_LESION_FRAC_OF_LEAF = 0.0005


class ColorRuleModel:
    backend = "color-rule"
    path = None

    def __init__(self, lesion_class: str = "unknown__lesion"):
        self.names = [LEAF_CLASS, lesion_class]
        self.lesion_class = lesion_class

    def info(self) -> dict:
        return {"file": None, "backend": self.backend, "names": None, "size_mb": None}

    def predict(self, image_bgr: np.ndarray, lesion_class: str | None = None, **_) -> Prediction:
        t0 = time.perf_counter()
        name = lesion_class or self.lesion_class
        h, w = image_bgr.shape[:2]
        leaf_all, _ = bootstrap_leaf_mask(image_bgr, min_component_frac=MIN_LEAF_FRAC)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hue, sat = hsv[..., 0], hsv[..., 1]
        green = (hue >= GREEN_HUE[0]) & (hue <= GREEN_HUE[1]) & (sat >= MIN_GREEN_SAT)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        instances: list[MaskInstance] = []
        n, labels, stats, _ = cv2.connectedComponentsWithStats(leaf_all, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_LEAF_FRAC * h * w:
                continue
            leaf = labels == i
            x, y, bw, bh = stats[i, :4]
            instances.append(MaskInstance(LEAF_CLASS, 1.0, leaf, (float(x), float(y), float(x + bw), float(y + bh))))
            # bỏ dải viền lá (≈2% kích thước lá): mask lá sơ bộ thường lấn ra nền ở mép, dễ báo nhầm vết bệnh
            r = max(2, round(0.02 * min(bw, bh)))
            inner = cv2.erode(leaf.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            spots = (inner.astype(bool) & ~green).astype(np.uint8)
            spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, k)
            m, lab2, st2, _ = cv2.connectedComponentsWithStats(spots, connectivity=8)
            min_area = max(4, MIN_LESION_FRAC_OF_LEAF * stats[i, cv2.CC_STAT_AREA])
            for j in range(1, m):
                if st2[j, cv2.CC_STAT_AREA] < min_area:
                    continue
                sx, sy, sw, sh = st2[j, :4]
                instances.append(MaskInstance(name, 1.0, lab2 == j,
                                              (float(sx), float(sy), float(sx + sw), float(sy + sh))))
        return Prediction(instances, {"inference": (time.perf_counter() - t0) * 1000})
