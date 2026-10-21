"""Chỉ số đánh giá bổ sung ngoài Box/Mask mAP50-95 của ultralytics.

- Mean IoU (ngữ nghĩa, theo lớp): ultralytics KHÔNG xuất sẵn chỉ số này dù quy trình yêu cầu.
  Cách tính: cộng dồn giao/hợp theo từng lớp trên TOÀN tập rồi mới chia (micro theo pixel), tránh
  ảnh nhỏ/ít pixel làm méo kết quả.
- Sai số Severity Index (MAE, RMSE) và độ chính xác phân mức (Khởi phát/Nhẹ/Nặng): đây mới là chỉ số
  phản ánh trực tiếp chất lượng khuyến nghị cho nông dân.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


class IoUAccumulator:
    def __init__(self, names: dict[int, str]):
        self.names = names
        n = max(names) + 1
        self.inter = np.zeros(n, dtype=np.int64)
        self.union = np.zeros(n, dtype=np.int64)

    def update(self, gt: np.ndarray, pred: np.ndarray) -> None:
        """gt, pred: (C, H, W) bool."""
        assert gt.shape == pred.shape, (gt.shape, pred.shape)
        self.inter += (gt & pred).reshape(gt.shape[0], -1).sum(1)
        self.union += (gt | pred).reshape(gt.shape[0], -1).sum(1)

    def result(self) -> dict:
        per_class = {}
        for cid, name in self.names.items():
            if self.union[cid] > 0:
                per_class[name] = float(self.inter[cid] / self.union[cid])
        lesion = [v for k, v in per_class.items() if k != "leaf"]
        return {
            "per_class_iou": per_class,
            "mean_iou_all": float(np.mean(list(per_class.values()))) if per_class else 0.0,
            "mean_iou_lesions": float(np.mean(lesion)) if lesion else 0.0,
        }


@dataclass
class SeverityErrorAccumulator:
    abs_err: list[float] = field(default_factory=list)
    sq_err: list[float] = field(default_factory=list)
    confusion: Counter = field(default_factory=Counter)

    def update(self, gt_pct: float, pred_pct: float, gt_level: str, pred_level: str) -> None:
        e = pred_pct - gt_pct
        self.abs_err.append(abs(e))
        self.sq_err.append(e * e)
        self.confusion[(gt_level, pred_level)] += 1

    def result(self) -> dict:
        n = len(self.abs_err)
        if n == 0:
            return {"n": 0}
        correct = sum(v for (g, p), v in self.confusion.items() if g == p)
        return {
            "n": n,
            "mae_pct_points": float(np.mean(self.abs_err)),
            "rmse_pct_points": float(math.sqrt(np.mean(self.sq_err))),
            "p90_abs_err": float(np.percentile(self.abs_err, 90)),
            "level_accuracy": correct / n,
            "confusion": {f"{g}->{p}": v for (g, p), v in sorted(self.confusion.items())},
        }
