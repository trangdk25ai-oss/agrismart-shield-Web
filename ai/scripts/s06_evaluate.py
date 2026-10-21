"""BƯỚC 2 – Đánh giá mô hình: Box mAP50-95, Mask mAP50-95, Mean IoU và sai số Severity Index.

Dùng được cho trọng số PyTorch (.pt) lẫn mô hình đã xuất (.onnx INT8 cho web, .tflite) => cũng là công cụ đo
"mức suy giảm độ chính xác sau lượng tử hóa" ở Bước 3.2.

Ví dụ:
    python scripts/s06_evaluate.py --weights runs/segment/agrismart_v8n/weights/best.pt \
        --data datasets/agrismart/dataset.yaml --split test --out reports/eval_fp32.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.knowledge_engine import KnowledgeEngine  # noqa: E402
from agrismart.metrics import IoUAccumulator, SeverityErrorAccumulator  # noqa: E402
from agrismart.severity import MaskInstance, compute_severity, severity_from_label  # noqa: E402
from agrismart.yolo_io import instances_to_class_masks, list_images, read_label_file  # noqa: E402
from agrismart.io_utils import imread  # noqa: E402

KB_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "ipm_rules.json"


def predictions_to_instances(result, names: dict[int, str]) -> list[MaskInstance]:
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    masks = result.masks.data.cpu().numpy() > 0.5  # (N, H, W) – retina_masks=True => kích thước ảnh gốc
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    return [MaskInstance(names[int(c)], float(p), m) for c, p, m in zip(cls, conf, masks)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="ngưỡng tin cậy khi tính IoU/Severity (giống app)")
    ap.add_argument("--out", type=Path, default=Path("reports/eval.json"))
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    model = YOLO(args.weights, task="segment")

    # ---- (A) chỉ số chuẩn của ultralytics
    m = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, plots=False, verbose=False)
    report = {
        "weights": str(args.weights),
        "split": args.split,
        "box_map50_95": float(m.box.map),
        "box_map50": float(m.box.map50),
        "mask_map50_95": float(m.seg.map),
        "mask_map50": float(m.seg.map50),
    }

    # ---- (B) Mean IoU + sai số Severity trên từng ảnh
    engine = KnowledgeEngine.from_file(KB_PATH)
    iou = IoUAccumulator(names)
    sev = SeverityErrorAccumulator()
    img_dir = Path(cfg["path"]) / cfg[args.split]
    lbl_dir = Path(cfg["path"]) / cfg[args.split].replace("images", "labels", 1)
    for img_path in list_images(img_dir):
        im = imread(img_path)
        if im is None:
            continue
        h, w = im.shape[:2]
        gt_insts = read_label_file(lbl_dir / f"{img_path.stem}.txt")
        res = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf, retina_masks=True, verbose=False)[0]
        pred = predictions_to_instances(res, names)

        gt_cls = instances_to_class_masks(gt_insts, h, w, max(names) + 1)
        pr_cls = np.zeros_like(gt_cls)
        for p in pred:
            cid = next(k for k, v in names.items() if v == p.cls_name)
            pr_cls[cid] |= p.mask
        iou.update(gt_cls, pr_cls)

        gt_rep = severity_from_label(gt_insts, names, h, w)
        if gt_rep.primary is None:
            continue
        pr_rep = compute_severity(pred)
        gt_pct = gt_rep.primary.severity_pct
        pr_pct = pr_rep.primary.severity_pct if pr_rep.primary else 0.0
        gt_lv = engine.classify(gt_pct, gt_rep.primary.lesion_area_px > 0)
        pr_lv = engine.classify(pr_pct, bool(pr_rep.primary and pr_rep.primary.lesion_area_px > 0))
        sev.update(gt_pct, pr_pct, gt_lv, pr_lv)

    report["iou"] = iou.result()
    report["severity"] = sev.result()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
