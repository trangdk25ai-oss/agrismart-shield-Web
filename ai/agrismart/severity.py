"""Thuật toán tính Chỉ số mức độ nghiêm trọng (Severity Index) – BẢN THAM CHIẾU.

Công thức gốc trong quy trình (Bước 4.1):
    Severity Index = Area_Lesions / Area_Leaf * 100%

Bản cài đặt này giữ nguyên ý nghĩa công thức nhưng xử lý 4 tình huống thực tế mà công thức
"đếm pixel thô" sẽ cho kết quả sai:

1. Các mask vết bệnh CHỒNG LẤN nhau  -> dùng phép HỢP (union) thay vì cộng diện tích, tránh đếm 2 lần.
2. Vết bệnh nằm ở MÉP lá, mask tràn ra ngoài viền lá -> mẫu số dùng |Leaf ∪ Lesions| nên tỷ lệ luôn ≤ 100%.
3. Mask lá bị "thủng" đúng tại vết bệnh (xảy ra khi huấn luyện với overlap_mask=True) -> vẫn đúng nhờ (2).
4. Ảnh có NHIỀU lá -> mỗi vết bệnh được gán cho đúng 1 lá (lá có độ chồng lấn lớn nhất), sau đó chọn
   "lá chính" (lá chứa tâm khung ngắm, hoặc lá lớn nhất) để báo cáo.

Máy chủ trang web (web/app/service.py), các script đánh giá (s06, s09) đều gọi CHUNG hàm này,
nên con số người dùng thấy trên trang web và con số trong báo cáo đánh giá là cùng một thuật toán.
Kiểm thử: ai/tests/test_severity.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import LEAF_CLASS

# Trạng thái kết quả
STATUS_OK = "OK"
STATUS_HEALTHY = "HEALTHY"  # có lá, không phát hiện vết bệnh
STATUS_NO_LEAF = "NO_LEAF"  # không phát hiện phiến lá
STATUS_NO_LEAF_WITH_LESIONS = "NO_LEAF_WITH_LESIONS"  # có vết bệnh nhưng không xác định được lá -> không tính được %
STATUS_LEAF_TOO_SMALL = "LEAF_TOO_SMALL"  # lá chiếm diện tích quá nhỏ trong khung -> độ phân giải mask không đủ tin cậy


@dataclass
class MaskInstance:
    """Một đối tượng do mô hình dự đoán (hoặc lấy từ nhãn gốc) trên toàn khung ảnh."""

    cls_name: str
    conf: float
    mask: np.ndarray  # bool/uint8 (H, W)
    box: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2 (pixel) – có khi lấy từ mô hình


@dataclass
class LeafSeverity:
    leaf_index: int
    leaf_conf: float
    leaf_area_px: int  # |Leaf ∪ Lesions_assigned|
    lesion_area_px: int  # |∪ Lesions_assigned|
    severity_pct: float
    n_lesions: int
    lesion_area_by_class: dict[str, int] = field(default_factory=dict)
    lesion_conf_by_class: dict[str, float] = field(default_factory=dict)
    dominant_class: str | None = None
    centroid_xy: tuple[float, float] = (0.0, 0.0)


@dataclass
class SeverityReport:
    status: str
    primary: LeafSeverity | None
    leaves: list[LeafSeverity]
    unassigned_lesions: int
    image_area_px: int

    def to_dict(self) -> dict:
        return asdict(self)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))  # nhân vuông
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def compute_severity(
    instances: list[MaskInstance],
    leaf_class: str = LEAF_CLASS,
    assoc_min_overlap: float = 0.5,
    dilate_px: int = 3,
    min_leaf_frac: float = 0.02,
    guide_center: tuple[float, float] = (0.5, 0.5),
) -> SeverityReport:
    """Tính Severity Index cho từng lá và chọn lá chính.

    Args:
        instances: danh sách đối tượng (lá + vết bệnh) cùng kích thước khung (H, W).
        leaf_class: tên lớp phiến lá.
        assoc_min_overlap: tỷ lệ tối thiểu |lesion ∩ dilate(leaf)| / |lesion| để gán vết bệnh vào lá.
        dilate_px: bán kính giãn mask lá (bù sai số biên mask khi vết bệnh nằm sát mép lá).
        min_leaf_frac: lá chính chiếm < tỷ lệ này của khung -> cảnh báo LEAF_TOO_SMALL.
        guide_center: tâm khung ngắm (tỷ lệ 0..1) để ưu tiên chọn lá người dùng đang ngắm.
    """
    if not instances:
        return SeverityReport(STATUS_NO_LEAF, None, [], 0, 0)

    h, w = instances[0].mask.shape[:2]
    image_area = h * w
    leaves = [(i, inst) for i, inst in enumerate(instances) if inst.cls_name == leaf_class]
    lesions = [inst for inst in instances if inst.cls_name != leaf_class]
    leaf_masks = [inst.mask.astype(bool) for _, inst in leaves]
    lesion_masks = [inst.mask.astype(bool) for inst in lesions]

    if not leaves:
        status = STATUS_NO_LEAF_WITH_LESIONS if lesions else STATUS_NO_LEAF
        return SeverityReport(status, None, [], len(lesions), image_area)

    # --- 1) Gán mỗi vết bệnh cho đúng 1 lá ---------------------------------------------------------
    dilated = [_dilate(m, dilate_px) for m in leaf_masks]
    assigned: list[list[int]] = [[] for _ in leaves]
    unassigned = 0
    for j, lm in enumerate(lesion_masks):
        area = int(lm.sum())
        if area == 0:
            continue
        overlaps = [float((lm & d).sum()) / area for d in dilated]
        k = int(np.argmax(overlaps))
        if overlaps[k] >= assoc_min_overlap:
            assigned[k].append(j)
        else:
            unassigned += 1

    # --- 2) Tính diện tích theo phép hợp cho từng lá ---------------------------------------------
    results: list[LeafSeverity] = []
    for k, (leaf_idx, leaf_inst) in enumerate(leaves):
        lesion_union = np.zeros((h, w), dtype=bool)
        by_class: dict[str, np.ndarray] = {}
        conf_acc: dict[str, list[float]] = {}
        for j in assigned[k]:
            lesion_union |= lesion_masks[j]
            name = lesions[j].cls_name
            by_class.setdefault(name, np.zeros((h, w), dtype=bool))
            by_class[name] |= lesion_masks[j]
            conf_acc.setdefault(name, []).append(float(lesions[j].conf))

        leaf_region = leaf_masks[k] | lesion_union
        leaf_area = int(leaf_region.sum())
        lesion_area = int(lesion_union.sum())
        severity = 100.0 * lesion_area / leaf_area if leaf_area > 0 else 0.0
        area_by_class = {n: int(m.sum()) for n, m in by_class.items()}
        dominant = max(area_by_class, key=area_by_class.get) if area_by_class else None
        ys, xs = np.nonzero(leaf_masks[k])
        centroid = (float(xs.mean() / w), float(ys.mean() / h)) if xs.size else (0.0, 0.0)
        results.append(
            LeafSeverity(
                leaf_index=leaf_idx,
                leaf_conf=float(leaf_inst.conf),
                leaf_area_px=leaf_area,
                lesion_area_px=lesion_area,
                severity_pct=severity,
                n_lesions=len(assigned[k]),
                lesion_area_by_class=area_by_class,
                lesion_conf_by_class={n: float(np.mean(v)) for n, v in conf_acc.items()},
                dominant_class=dominant,
                centroid_xy=centroid,
            )
        )

    # --- 3) Chọn lá chính ----------------------------------------------------------------------------
    gx = min(max(int(guide_center[0] * w), 0), w - 1)
    gy = min(max(int(guide_center[1] * h), 0), h - 1)
    containing = [k for k in range(len(leaves)) if leaf_masks[k][gy, gx]]
    if containing:
        pk = max(containing, key=lambda k: results[k].leaf_area_px)
    else:
        pk = max(range(len(leaves)), key=lambda k: results[k].leaf_area_px * results[k].leaf_conf)
    primary = results[pk]

    if primary.leaf_area_px < min_leaf_frac * image_area:
        status = STATUS_LEAF_TOO_SMALL
    elif primary.lesion_area_px == 0:
        status = STATUS_HEALTHY
    else:
        status = STATUS_OK
    return SeverityReport(status, primary, results, unassigned, image_area)


def severity_from_label(
    instances, names: dict[int, str], height: int, width: int, **kwargs
) -> SeverityReport:
    """Tính severity "chuẩn" (ground truth) từ nhãn polygon – dùng khi đánh giá sai số Severity."""
    from .yolo_io import polygon_to_mask

    masks = [
        MaskInstance(cls_name=names[inst.cls_id], conf=1.0, mask=polygon_to_mask(inst.polygon, height, width))
        for inst in instances
    ]
    return compute_severity(masks, **kwargs)
