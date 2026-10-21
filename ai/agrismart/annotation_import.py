"""Đọc nhãn tự gán từ công cụ gán nhãn phổ biến và chuyển thành instance YOLOv8-seg.

Hỗ trợ:
- COCO instance segmentation (xuất từ CVAT "COCO 1.0", Roboflow "COCO Segmentation", labelme2coco…):
  polygon và RLE không nén; RLE nén cần cài pycocotools.
- Labelme JSON (công cụ gán nhãn chạy offline trên máy tính): polygon, circle và mask (AI-Mask).
- VGG Image Annotator (VIA) 1.x/2.x: file dự án (_via_img_metadata) hoặc file xuất nhãn JSON;
  vùng polygon, circle, ellipse (circle/ellipse được xấp xỉ thành polygon). Dùng cho BRACOT.

Nhãn trong công cụ phải là tên lớp đích (leaf, <crop>__<disease>) hoặc được ánh xạ qua label_aliases.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np

from .io_utils import image_size
from .yolo_io import IMG_EXTS, SegInstance, mask_to_polygons, normalize_polygon


@dataclass
class Shape:
    label: str
    polygon: np.ndarray | None = None  # (k, 2) pixel
    mask: np.ndarray | None = None  # (H, W) 0/1


@dataclass
class AnnotatedImage:
    key: str  # đường dẫn tương đối dùng đặt tên file đầu ra
    image_path: Path | None
    width: int
    height: int
    shapes: list[Shape]
    image_bytes: bytes | None = None  # Labelme có thể nhúng ảnh (imageData)


@dataclass
class ImportStats:
    counts: dict = field(default_factory=dict)

    def add(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


# --------------------------------------------------------------------------------------------- RLE
def decode_rle(seg: dict, height: int, width: int) -> np.ndarray | None:
    """Giải mã RLE của COCO. counts dạng danh sách (không nén) tự giải; dạng chuỗi (nén) cần pycocotools."""
    counts = seg.get("counts")
    h, w = seg.get("size", [height, width])
    if isinstance(counts, list):
        flat = np.zeros(h * w, dtype=np.uint8)
        pos, val = 0, 0
        for c in counts:
            if val:
                flat[pos:pos + c] = 1
            pos += c
            val ^= 1
        return flat.reshape((w, h)).T.copy()  # COCO lưu theo cột (Fortran order)
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except ImportError:
        return None
    rle = dict(seg)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(np.uint8)


# --------------------------------------------------------------------------------------------- readers
def read_coco(coco_json: str | Path, images_root: str | Path, stats: ImportStats) -> list[AnnotatedImage]:
    data = json.loads(Path(coco_json).read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    if not cats:
        raise ValueError("File COCO không có danh sách categories (không biết tên nhãn)")
    by_img: dict[int, list[dict]] = {}
    for a in data.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    out = []
    for im in data.get("images", []):
        h, w = int(im["height"]), int(im["width"])
        shapes: list[Shape] = []
        for a in by_img.get(im["id"], []):
            label = cats.get(a["category_id"], f"?{a['category_id']}")
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                for poly in seg:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    if len(pts) >= 3:
                        shapes.append(Shape(label, polygon=pts))
            elif isinstance(seg, dict):
                m = decode_rle(seg, h, w)
                if m is None:
                    stats.add("skip_rle_need_pycocotools")
                    continue
                shapes.append(Shape(label, mask=m))
            else:
                stats.add("skip_annotation_without_segmentation")
        rel = Path(im["file_name"])
        out.append(AnnotatedImage(str(rel.with_suffix("")), Path(images_root) / rel, w, h, shapes))
    return out


REVIEW_FLAG = "da_ra_soat"  # cờ của nhãn nháp do s00c_auto_label.py sinh; người gán nhãn tích sau khi rà soát


def read_labelme_dir(root: str | Path, stats: ImportStats, include_unreviewed: bool = False) -> list[AnnotatedImage]:
    """Đọc mọi tệp Labelme trong thư mục (đệ quy).

    Tệp nhãn nháp tự động có cờ "da_ra_soat": false bị bỏ qua (trừ khi include_unreviewed) – chỉ nhập khi người
    gán nhãn đã rà soát và tích cờ. Tệp vẽ tay (không có cờ này) luôn được nhập.
    """
    root = Path(root)
    out = []
    for jp in sorted(root.rglob("*.json")):
        try:
            d = json.loads(jp.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            stats.add("skip_bad_json")
            continue
        if "shapes" not in d:
            continue  # không phải file Labelme
        if (d.get("flags") or {}).get(REVIEW_FLAG) is False:
            if not include_unreviewed:
                stats.add("skip_nhan_nhap_chua_ra_soat")
                continue
            stats.add("nhan_nhap_chua_ra_soat_van_nhap")
        h, w = int(d["imageHeight"]), int(d["imageWidth"])
        shapes = []
        for s in d["shapes"]:
            st = s.get("shape_type") or "polygon"
            pts = np.asarray(s.get("points", []), dtype=np.float32)
            if st == "polygon" and len(pts) >= 3:
                shapes.append(Shape(s["label"], polygon=pts))
            elif st == "circle" and len(pts) == 2:  # tâm + một điểm trên đường tròn
                r = float(np.hypot(*(pts[1] - pts[0])))
                poly = cv2.ellipse2Poly((round(pts[0][0]), round(pts[0][1])), (max(1, round(r)),) * 2, 0, 0, 360, 10)
                shapes.append(Shape(s["label"], polygon=poly.astype(np.float32)))
            elif st == "mask" and s.get("mask") and len(pts) == 2:  # vùng do công cụ AI-Mask của Labelme tạo
                m = _labelme_mask(s["mask"], pts, h, w)
                if m is not None:
                    shapes.append(Shape(s["label"], mask=m))
            else:
                stats.add(f"skip_shape_{st}")
        # imagePath có thể mang dấu "\\" khi gán nhãn trên Windows
        rel_img = PureWindowsPath(d["imagePath"]).as_posix() if d.get("imagePath") else None
        img_path = (jp.parent / rel_img) if rel_img else None
        if img_path is None or not img_path.exists():
            img_path = next((jp.with_suffix(e) for e in IMG_EXTS if jp.with_suffix(e).exists()), None)
        img_bytes = base64.b64decode(d["imageData"]) if (img_path is None and d.get("imageData")) else None
        out.append(AnnotatedImage(str(jp.relative_to(root).with_suffix("")), img_path, w, h, shapes, img_bytes))
    return out


def _labelme_mask(b64png: str, pts: np.ndarray, h: int, w: int) -> np.ndarray | None:
    """Đặt mảng mask (PNG base64) của shape 'mask' vào toàn khung ảnh, góc trên trái tại points[0]."""
    patch = cv2.imdecode(np.frombuffer(base64.b64decode(b64png), np.uint8), cv2.IMREAD_GRAYSCALE)
    if patch is None:
        return None
    (x1, y1), (x2, y2) = np.round(pts).astype(int)
    ph, pw = patch.shape
    if np.array_equal(pts, np.trunc(pts)):  # cùng quy ước với Labelme: biên nguyên -> cắt theo biên
        ph, pw = min(ph, y2 - y1 + 1), min(pw, x2 - x1 + 1)
    out = np.zeros((h, w), np.uint8)
    ys, ye, xs, xe = max(y1, 0), min(y1 + ph, h), max(x1, 0), min(x1 + pw, w)
    if ys >= ye or xs >= xe:
        return None
    out[ys:ye, xs:xe] = patch[ys - y1:ye - y1, xs - x1:xe - x1] > 0
    return out


def _via_label(attrs: dict, label_attr: str | None) -> str | None:
    """Lấy tên nhãn từ region_attributes của VIA (giá trị chuỗi, hoặc dict {lựa chọn: true} của checkbox)."""
    if not attrs:
        return None
    if label_attr is None:
        if len(attrs) != 1:
            return None  # nhiều thuộc tính mà không chỉ rõ thuộc tính nhãn
        label_attr = next(iter(attrs))
    v = attrs.get(label_attr)
    if isinstance(v, dict):
        chosen = [k for k, on in v.items() if on]
        return chosen[0] if len(chosen) == 1 else None
    return str(v).strip() if v not in (None, "") else None


def _via_shape_to_polygon(sa: dict) -> np.ndarray | None:
    name = sa.get("name")
    if name == "polygon":
        xs, ys = sa.get("all_points_x", []), sa.get("all_points_y", [])
        return np.stack([xs, ys], axis=1).astype(np.float32) if len(xs) >= 3 and len(xs) == len(ys) else None
    if name in ("circle", "ellipse"):
        cx, cy = float(sa["cx"]), float(sa["cy"])
        rx = float(sa.get("rx", sa.get("r", 0)))
        ry = float(sa.get("ry", sa.get("r", 0)))
        theta = float(np.degrees(sa.get("theta", 0.0)))
        pts = cv2.ellipse2Poly((round(cx), round(cy)), (max(1, round(rx)), max(1, round(ry))), round(theta), 0, 360, 10)
        return pts.astype(np.float32)
    return None


def read_via(via_json: str | Path, images_root: str | Path, stats: ImportStats,
             label_attr: str | None = None) -> list[AnnotatedImage]:
    """Đọc nhãn VIA 1.x/2.x. VIA không lưu kích thước ảnh nên phải mở ảnh để lấy (rộng, cao)."""
    data = json.loads(Path(via_json).read_text(encoding="utf-8"))
    if "_via_img_metadata" in data:
        meta = data["_via_img_metadata"]
    elif "metadata" in data and "file" in data and "project" in data:
        raise ValueError("Đây là định dạng VIA 3.x – hãy mở trong VIA 2 hoặc xuất sang COCO rồi dùng --format coco")
    else:
        meta = data
    out = []
    for entry in meta.values():
        if not isinstance(entry, dict) or "filename" not in entry:
            continue
        img_path = Path(images_root) / entry["filename"]
        wh = image_size(img_path)
        if wh is None:
            stats.add("skip_image_missing")
            continue
        regions = entry.get("regions", [])
        if isinstance(regions, dict):  # VIA 1.x lưu regions dạng {"0": {...}}
            regions = list(regions.values())
        shapes = []
        for r in regions:
            sa = r.get("shape_attributes", {})
            label = _via_label(r.get("region_attributes", {}), label_attr)
            if label is None:
                stats.add("skip_region_without_label")
                continue
            poly = _via_shape_to_polygon(sa)
            if poly is None:
                stats.add(f"skip_shape_{sa.get('name', '?')}")
                continue
            shapes.append(Shape(label, polygon=poly))
        out.append(AnnotatedImage(str(Path(entry["filename"]).with_suffix("")), img_path, wh[0], wh[1], shapes))
    return out


# --------------------------------------------------------------------------------------------- convert
def to_instances(img: AnnotatedImage, name_to_id: dict[str, int], aliases: dict[str, str],
                 stats: ImportStats, min_area_px: float = 16.0) -> list[SegInstance]:
    out: list[SegInstance] = []
    for s in img.shapes:
        target = aliases.get(s.label, s.label)
        if target not in name_to_id:
            stats.add(f"unknown_label:{s.label}")
            continue
        cid = name_to_id[target]
        if s.polygon is not None:
            out.append(SegInstance(cid, normalize_polygon(s.polygon, img.width, img.height)))
        elif s.mask is not None:
            for poly in mask_to_polygons(s.mask, min_area_px=min_area_px):
                out.append(SegInstance(cid, normalize_polygon(poly, img.width, img.height)))
    return out
