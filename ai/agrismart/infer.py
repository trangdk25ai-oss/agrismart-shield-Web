"""Luồng chẩn đoán đầu-cuối DÙNG CHUNG cho máy chủ trang web và dòng lệnh.

    python -m agrismart.infer --model web/models/model_int8.onnx --image la_ca_chua.jpg --crop tomato --out out/

Luồng: Blur Check -> YOLOv8-seg -> Severity Index -> Knowledge Engine -> JSON (+ ảnh lớp phủ).
Mô hình: .onnx / .tflite / .pt (seg_runtime.load_model) hoặc chế độ thử không mô hình (color_rule).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from . import LEAF_CLASS
from .blur import DEFAULT_BLUR_THRESHOLD, laplacian_variance
from .io_utils import imread, imwrite
from .knowledge_engine import KnowledgeEngine
from .severity import (
    STATUS_HEALTHY,
    STATUS_LEAF_TOO_SMALL,
    STATUS_NO_LEAF,
    STATUS_NO_LEAF_WITH_LESIONS,
    STATUS_OK,
    MaskInstance,
    compute_severity,
)

ROOT = Path(__file__).resolve().parents[2]
KB_PATH = ROOT / "knowledge_base/ipm_rules.json"
KB_SCHEMA = ROOT / "knowledge_base/ipm_rules.schema.json"
STATUS_BLURRY = "BLURRY"
RETAKE_MESSAGES = {
    STATUS_NO_LEAF: "Không nhận diện được phiến lá. Đặt một lá vào giữa khung và chụp lại.",
    STATUS_NO_LEAF_WITH_LESIONS: "Thấy vết bệnh nhưng không xác định được phiến lá nên không tính được tỷ lệ. Chụp lại cả lá.",
    STATUS_LEAF_TOO_SMALL: "Lá quá nhỏ trong khung hình. Đưa máy lại gần để lá chiếm phần lớn khung.",
    STATUS_BLURRY: "Ảnh có thể bị mờ (rung tay hoặc lệch nét). Nên chụp lại; vẫn có thể phân tích nếu chắc ảnh rõ.",
}
# Bảng màu lớp phủ (BGR) – lá luôn màu xanh, các lớp bệnh lần lượt theo thứ tự lớp của mô hình
LEAF_COLOR = (80, 175, 76)
LESION_COLORS = [(0, 0, 220), (0, 140, 255), (200, 0, 200), (255, 120, 0), (0, 200, 200), (120, 60, 220),
                 (40, 40, 160), (160, 90, 0)]


def class_color(name: str, names: list[str]) -> tuple[int, int, int]:
    if name == LEAF_CLASS:
        return LEAF_COLOR
    lesions = [n for n in names if n != LEAF_CLASS]
    i = lesions.index(name) if name in lesions else len(lesions)
    return LESION_COLORS[i % len(LESION_COLORS)]


def mask_box(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def draw_overlay(image: np.ndarray, instances: list[MaskInstance], names: list[str]) -> np.ndarray:
    """Ảnh BGR có lớp phủ (dùng cho dòng lệnh / báo cáo)."""
    out = image.copy()
    for inst in sorted(instances, key=lambda i: i.cls_name != LEAF_CLASS):  # vẽ lá trước, vết bệnh đè lên
        color = class_color(inst.cls_name, names)
        alpha = 0.25 if inst.cls_name == LEAF_CLASS else 0.55
        m = inst.mask.astype(bool)
        out[m] = (out[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def overlay_rgba(h: int, w: int, instances: list[MaskInstance], names: list[str]) -> np.ndarray:
    """Lớp phủ trong suốt BGRA (để trang web chồng lên ảnh và bật/tắt được)."""
    out = np.zeros((h, w, 4), np.uint8)
    for inst in sorted(instances, key=lambda i: i.cls_name != LEAF_CLASS):
        b, g, r = class_color(inst.cls_name, names)
        a = 70 if inst.cls_name == LEAF_CLASS else 150
        m = inst.mask.astype(bool)
        out[m] = (b, g, r, a)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (b, g, r, 255), max(2, round(max(h, w) / 400)))
    return out


def analyze(
    model,
    image_bgr: np.ndarray,
    crop_code: str,
    engine: KnowledgeEngine,
    *,
    tank_code: str = "TANK_16L",
    start_date: date | None = None,
    conf: float = 0.25,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    force: bool = True,
    guide_center: tuple[float, float] = (0.5, 0.5),
    lesion_class: str | None = None,
) -> tuple[dict, list[MaskInstance]]:
    """Chạy toàn bộ luồng trên MỘT ảnh. Trả về (kết quả JSON được, các đối tượng để vẽ lớp phủ).

    force=False: ảnh mờ thì dừng ở trạng thái BLURRY (không chạy mô hình) để người dùng chụp lại.
    lesion_class: chỉ dùng ở chế độ thử không mô hình (bệnh do người dùng chọn).
    """
    t0 = time.perf_counter()
    h, w = image_bgr.shape[:2]
    blur_score = laplacian_variance(image_bgr)
    blurry = blur_score < blur_threshold
    t_blur = (time.perf_counter() - t0) * 1000
    result: dict = {
        "image_size": [w, h],
        "blur": {"score": round(blur_score, 1), "threshold": blur_threshold, "is_blurry": blurry},
        "timings_ms": {"blur_check": round(t_blur, 1)},
        "detections": [],
        "leaves": [],
    }
    if blurry and not force:
        result["status"] = STATUS_BLURRY
        result["message"] = RETAKE_MESSAGES[STATUS_BLURRY]
        return result, []

    if getattr(model, "backend", "") == "color-rule":
        pred = model.predict(image_bgr, lesion_class=lesion_class)
    else:
        pred = model.predict(image_bgr, conf=conf)
    t1 = time.perf_counter()
    rep = compute_severity(pred.instances, guide_center=guide_center)
    t2 = time.perf_counter()

    for inst in pred.instances:
        x1, y1, x2, y2 = inst.box or mask_box(inst.mask)
        result["detections"].append({"class": inst.cls_name, "conf": round(inst.conf, 3),
                                     "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]})
    primary_idx = rep.primary.leaf_index if rep.primary else None
    for lf in rep.leaves:
        x1, y1, x2, y2 = mask_box(pred.instances[lf.leaf_index].mask)
        dom_conf = lf.lesion_conf_by_class.get(lf.dominant_class) if lf.dominant_class else None
        result["leaves"].append({
            "index": lf.leaf_index,
            "primary": lf.leaf_index == primary_idx,
            "box": [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)],
            "severity_pct": round(lf.severity_pct, 2),
            "leaf_px": lf.leaf_area_px,
            "lesion_px": lf.lesion_area_px,
            "n_lesions": lf.n_lesions,
            "dominant_class": lf.dominant_class,
            "dominant_conf": round(dom_conf, 3) if dom_conf is not None else None,
            "by_class_px": lf.lesion_area_by_class,
        })
    result["status"] = rep.status
    result["unassigned_lesions"] = rep.unassigned_lesions
    if rep.primary is not None and rep.status in (STATUS_OK, STATUS_HEALTHY):
        p = rep.primary
        dom_conf = p.lesion_conf_by_class.get(p.dominant_class) if p.dominant_class else None
        result["recommendation"] = engine.recommend(
            crop_code, p.dominant_class, p.severity_pct, has_lesion=p.lesion_area_px > 0,
            tank_code=tank_code, start_date=start_date, detection_conf=dom_conf)
    else:
        result["message"] = RETAKE_MESSAGES.get(rep.status, "Không đủ cơ sở để kết luận – đề nghị chụp lại ảnh.")
    t3 = time.perf_counter()
    result["timings_ms"].update({k: round(v, 1) for k, v in pred.timings_ms.items()})
    result["timings_ms"]["model_total"] = round((t1 - t0) * 1000 - t_blur, 1)
    result["timings_ms"]["severity"] = round((t2 - t1) * 1000, 1)
    result["timings_ms"]["total"] = round((t3 - t0) * 1000, 1)
    return result, pred.instances


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help=".onnx / .tflite / .pt; bỏ trống = chế độ thử không mô hình (cần --lesion-class)")
    ap.add_argument("--weights", dest="model", help=argparse.SUPPRESS)  # tên cũ
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--crop", required=True, help="tomato | cucumber | citrus")
    ap.add_argument("--tank", default="TANK_16L")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--lesion-class", default=None, help="chế độ thử: lớp bệnh giả định, ví dụ tomato__early_blight")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    image = imread(a.image)
    if image is None:
        raise SystemExit(f"Không đọc được ảnh {a.image}")
    if a.model:
        from .seg_runtime import load_model

        model = load_model(a.model)
    else:
        from .color_rule import ColorRuleModel

        if not a.lesion_class:
            raise SystemExit("Chế độ thử không mô hình cần --lesion-class")
        model = ColorRuleModel(a.lesion_class)
    engine = KnowledgeEngine.from_file(KB_PATH, KB_SCHEMA)
    result, instances = analyze(model, image, a.crop, engine, tank_code=a.tank, conf=a.conf, lesion_class=a.lesion_class)
    result["image"] = str(a.image)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        imwrite(a.out / f"{a.image.stem}_overlay.jpg", draw_overlay(image, instances, model.names))
        (a.out / f"{a.image.stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
