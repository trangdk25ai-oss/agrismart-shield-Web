"""Đánh giá mô hình phân đoạn trên bộ dữ liệu chỉ có HỘP BAO LÁ + tên bệnh (PlantDoc, bộ cà chua Bangladesh…).

Với mỗi hộp lá trong nhãn gốc:
  - lấy các pixel vết bệnh mô hình dự đoán nằm trong hộp, gộp theo lớp (phép hợp);
  - tỷ lệ vết bệnh = |vết bệnh trong hộp| / |(lá ∪ vết bệnh) trong hộp| (không có mask lá thì chia diện tích hộp);
  - tỷ lệ < min_lesion_frac  -> dự đoán "healthy"; ngược lại -> lớp bệnh có diện tích lớn nhất trong hộp.
So với nhãn gốc của lá để tính độ chính xác, độ nhạy theo bệnh và tỷ lệ báo nhầm trên lá khỏe.

Đây là đánh giá MỨC LÁ (đúng bệnh hay không), không đánh giá được độ chính xác % diện tích vì nguồn không có mask.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import LEAF_CLASS
from .severity import MaskInstance

HEALTHY = "healthy"


def label_in_box(instances: list[MaskInstance], box: tuple[float, float, float, float],
                 min_lesion_frac: float = 0.005, leaf_class: str = LEAF_CLASS) -> tuple[str, float]:
    if not instances:
        return HEALTHY, 0.0
    h, w = instances[0].mask.shape[:2]
    x1, y1, x2, y2 = box
    xa, ya = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    xb, yb = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    if xb <= xa or yb <= ya:
        return HEALTHY, 0.0
    leaf = np.zeros((yb - ya, xb - xa), dtype=bool)
    by_class: dict[str, np.ndarray] = {}
    for inst in instances:
        crop = inst.mask[ya:yb, xa:xb].astype(bool)
        if not crop.any():
            continue
        if inst.cls_name == leaf_class:
            leaf |= crop
        else:
            by_class.setdefault(inst.cls_name, np.zeros_like(leaf))
            by_class[inst.cls_name] |= crop
    lesion = np.zeros_like(leaf)
    for m in by_class.values():
        lesion |= m
    denom = int((leaf | lesion).sum()) if leaf.any() else leaf.size
    frac = float(lesion.sum()) / max(denom, 1)
    if frac < min_lesion_frac or not by_class:
        return HEALTHY, frac
    dominant = max(by_class, key=lambda k: int(by_class[k].sum()))
    return dominant, frac


@dataclass
class BoxEvalAccumulator:
    confusion: Counter = field(default_factory=Counter)

    def update(self, gt: str, pred: str) -> None:
        self.confusion[(gt, pred)] += 1

    def result(self) -> dict:
        n = sum(self.confusion.values())
        if n == 0:
            return {"n_boxes": 0}
        labels = sorted({g for g, _ in self.confusion} | {p for _, p in self.confusion})
        per_class = {}
        for c in labels:
            tp = self.confusion[(c, c)]
            fn = sum(v for (g, p), v in self.confusion.items() if g == c and p != c)
            fp = sum(v for (g, p), v in self.confusion.items() if p == c and g != c)
            per_class[c] = {
                "support": tp + fn,
                "recall": tp / (tp + fn) if tp + fn else None,
                "precision": tp / (tp + fp) if tp + fp else None,
            }
        healthy_total = sum(v for (g, _), v in self.confusion.items() if g == HEALTHY)
        healthy_fp = sum(v for (g, p), v in self.confusion.items() if g == HEALTHY and p != HEALTHY)
        return {
            "n_boxes": n,
            "accuracy": sum(v for (g, p), v in self.confusion.items() if g == p) / n,
            "healthy_false_alarm_rate": healthy_fp / healthy_total if healthy_total else None,
            "per_class": per_class,
            "confusion": {f"{g}->{p}": v for (g, p), v in sorted(self.confusion.items())},
        }


def read_box_csv(csv_path: str | Path, class_map: dict[str, str]) -> dict[str, list[dict]]:
    """Đọc CSV dạng PlantDoc: filename,width,height,class,xmin,ymin,xmax,ymax.

    Chỉ giữ các lớp có trong class_map (tên lớp nguồn -> lớp đích hoặc "healthy").
    Giữ width/height ghi trong CSV để quy đổi tọa độ khi kích thước ảnh thật khác CSV.
    """
    out: dict[str, list[dict]] = defaultdict(list)
    with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.DictReader(f):
            target = class_map.get(r["class"])
            if target is None:
                continue
            out[r["filename"]].append({
                "label": target,
                "box": tuple(float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")),
                "csv_wh": (float(r["width"]), float(r["height"])),
            })
    return dict(out)


def scale_box(box: tuple[float, float, float, float], csv_wh: tuple[float, float],
              real_wh: tuple[int, int]) -> tuple[float, float, float, float]:
    sx = real_wh[0] / csv_wh[0] if csv_wh[0] > 0 else 1.0
    sy = real_wh[1] / csv_wh[1] if csv_wh[1] > 0 else 1.0
    x1, y1, x2, y2 = box
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy
