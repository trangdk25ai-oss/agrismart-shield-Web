"""Đọc/ghi nhãn YOLOv8-seg và chuyển đổi qua lại giữa polygon và binary mask.

Định dạng một dòng nhãn YOLOv8-seg (tọa độ chuẩn hóa 0..1 theo chiều rộng/cao ảnh):
    <class_id> x1 y1 x2 y2 ... xn yn
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SegInstance:
    """Một đối tượng (instance) trong nhãn YOLOv8-seg."""

    cls_id: int
    polygon: np.ndarray  # (k, 2) float32, tọa độ chuẩn hóa [0, 1]

    def to_line(self, precision: int = 6) -> str:
        coords = " ".join(f"{v:.{precision}f}" for v in self.polygon.reshape(-1))
        return f"{self.cls_id} {coords}"


# --------------------------------------------------------------------------------------
# Polygon <-> mask
# --------------------------------------------------------------------------------------
def mask_to_polygons(
    mask: np.ndarray,
    min_area_px: float = 16.0,
    epsilon_frac: float = 0.002,
) -> list[np.ndarray]:
    """Tách binary mask thành danh sách polygon (tọa độ pixel), mỗi thành phần liên thông = 1 polygon.

    - RETR_EXTERNAL: chỉ lấy biên ngoài -> lỗ thủng bên trong được lấp (phù hợp với nhãn "phiến lá"
      vì phiến lá phải BAO GỒM cả vùng vết bệnh nằm bên trong).
    - approxPolyDP với epsilon tỷ lệ theo chu vi để giảm số đỉnh nhưng giữ hình dạng.
    """
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return []
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < min_area_px:
            continue
        eps = max(epsilon_frac * cv2.arcLength(c, True), 0.5)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append(approx.astype(np.float32))
    return polys


def normalize_polygon(poly_px: np.ndarray, width: int, height: int) -> np.ndarray:
    p = poly_px.astype(np.float32).copy()
    p[:, 0] /= float(width)
    p[:, 1] /= float(height)
    return np.clip(p, 0.0, 1.0)


def denormalize_polygon(poly_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    p = poly_norm.astype(np.float32).copy()
    p[:, 0] *= float(width)
    p[:, 1] *= float(height)
    return p


def polygon_to_mask(poly_norm: np.ndarray, height: int, width: int) -> np.ndarray:
    """Vẽ polygon chuẩn hóa thành binary mask (uint8 0/1) kích thước (height, width)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.round(denormalize_polygon(poly_norm, width, height)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def instances_to_class_masks(
    instances: list[SegInstance], height: int, width: int, num_classes: int
) -> np.ndarray:
    """Hợp (union) các instance theo lớp -> mảng (num_classes, H, W) bool. Dùng để tính IoU ngữ nghĩa."""
    out = np.zeros((num_classes, height, width), dtype=bool)
    for inst in instances:
        if 0 <= inst.cls_id < num_classes:
            out[inst.cls_id] |= polygon_to_mask(inst.polygon, height, width).astype(bool)
    return out


# --------------------------------------------------------------------------------------
# Đọc / ghi file nhãn
# --------------------------------------------------------------------------------------
class LabelFormatError(ValueError):
    pass


def parse_label_line(line: str, line_no: int = 0) -> SegInstance:
    parts = line.strip().split()
    if len(parts) < 7:
        raise LabelFormatError(f"Dòng {line_no}: cần class_id + tối thiểu 3 điểm (6 số), nhận {len(parts)} giá trị")
    try:
        cls_id = int(parts[0])
    except ValueError as exc:
        raise LabelFormatError(f"Dòng {line_no}: class_id '{parts[0]}' không phải số nguyên") from exc
    coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
    if coords.size % 2 != 0:
        raise LabelFormatError(f"Dòng {line_no}: số tọa độ lẻ ({coords.size})")
    if coords.min() < -1e-6 or coords.max() > 1 + 1e-6:
        raise LabelFormatError(f"Dòng {line_no}: tọa độ nằm ngoài [0,1] (min={coords.min():.4f}, max={coords.max():.4f})")
    return SegInstance(cls_id=cls_id, polygon=coords.reshape(-1, 2))


def read_label_file(path: str | Path) -> list[SegInstance]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            out.append(parse_label_line(line, i))
    return out


def write_label_file(path: str | Path, instances: list[SegInstance]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(inst.to_line() for inst in instances) + ("\n" if instances else ""), encoding="utf-8")


def masks_to_instances(
    class_masks: dict[int, np.ndarray], min_area_px: float = 16.0, epsilon_frac: float = 0.002
) -> list[SegInstance]:
    """{class_id: binary mask} -> danh sách SegInstance (mỗi thành phần liên thông là một instance)."""
    out: list[SegInstance] = []
    for cls_id, mask in class_masks.items():
        h, w = mask.shape[:2]
        for poly in mask_to_polygons(mask, min_area_px=min_area_px, epsilon_frac=epsilon_frac):
            out.append(SegInstance(cls_id=int(cls_id), polygon=normalize_polygon(poly, w, h)))
    return out


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path_for(image_path: Path, images_root: Path, labels_root: Path) -> Path:
    rel = image_path.relative_to(images_root).with_suffix(".txt")
    return labels_root / rel
