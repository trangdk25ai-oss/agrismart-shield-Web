"""BƯỚC 1 – Kiểm tra chất lượng bộ dữ liệu trước khi huấn luyện (chạy được trong CI).

Kiểm tra:
    1. Mọi ảnh có file nhãn; mọi dòng nhãn đúng định dạng YOLOv8-seg (class + ≥3 điểm, tọa độ trong [0,1]).
    2. class_id nằm trong danh sách names của dataset.yaml.
    3. Mỗi ảnh có ít nhất 1 instance "leaf" (bắt buộc cho thuật toán Severity Index).
    4. Vết bệnh nằm trong lá: ≥ 90% diện tích vết bệnh thuộc mask lá (phát hiện nhãn lá bị thiếu/lệch).
    5. Mọi lớp bệnh trong dataset.yaml có quy tắc tương ứng trong ipm_rules.json (đồng bộ AI ↔ tri thức).
Trả mã thoát 1 nếu có lỗi nghiêm trọng.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.yolo_io import LabelFormatError, list_images, polygon_to_mask, read_label_file  # noqa: E402
from agrismart.io_utils import imread  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-yaml", required=True, type=Path)
    ap.add_argument("--kb", type=Path, default=Path(__file__).resolve().parents[2] / "knowledge_base/ipm_rules.json")
    ap.add_argument("--containment", type=float, default=0.90)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.data_yaml.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    names = {int(k): v for k, v in cfg["names"].items()}
    leaf_ids = {k for k, v in names.items() if v == "leaf"}
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Counter] = {}

    kb = json.loads(args.kb.read_text(encoding="utf-8"))
    kb_classes = {r["model_class"] for r in kb["rules"]}
    kb_pending = {r["model_class"] for r in kb["rules"] if r.get("rule_status") == "PENDING"}
    for cid, n in names.items():
        if n != "leaf" and n not in kb_classes:
            errors.append(f"Lớp '{n}' không có quy tắc trong {args.kb.name} "
                          "(chạy knowledge_base/kb_add_rules.py để tạo quy tắc chờ nhập)")
        elif n in kb_pending:
            warnings.append(f"Lớp '{n}' mới có quy tắc CHỜ NHẬP – cần chuyên gia BVTV bổ sung phác đồ")

    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        img_dir = root / cfg[split]
        # cùng quy tắc với ultralytics: thay lần xuất hiện CUỐI của /images/ bằng /labels/
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
        lbl_dir = Path(sb.join(f"{img_dir}{os.sep}".rsplit(sa, 1)))
        c: Counter = Counter()
        for img_path in list_images(img_dir):
            lbl = lbl_dir / f"{img_path.stem}.txt"
            if not lbl.exists():
                errors.append(f"[{split}] thiếu nhãn: {img_path.name}")
                continue
            try:
                insts = read_label_file(lbl)
            except LabelFormatError as e:
                errors.append(f"[{split}] {lbl.name}: {e}")
                continue
            c["images"] += 1
            bad = [i.cls_id for i in insts if i.cls_id not in names]
            if bad:
                errors.append(f"[{split}] {lbl.name}: class_id ngoài danh sách {sorted(set(bad))}")
                continue
            for i in insts:
                c[names[i.cls_id]] += 1
            if not any(i.cls_id in leaf_ids for i in insts):
                errors.append(f"[{split}] {lbl.name}: không có instance 'leaf'")
                continue
            im = imread(img_path)
            if im is None:
                errors.append(f"[{split}] không đọc được ảnh {img_path.name}")
                continue
            h, w = im.shape[:2]
            leaf = np.zeros((h, w), bool)
            lesion = np.zeros((h, w), bool)
            for i in insts:
                m = polygon_to_mask(i.polygon, h, w).astype(bool)
                if i.cls_id in leaf_ids:
                    leaf |= m
                else:
                    lesion |= m
            if lesion.any():
                ratio = float((lesion & leaf).sum() / lesion.sum())
                if ratio < args.containment:
                    warnings.append(f"[{split}] {lbl.name}: chỉ {ratio:.0%} diện tích vết bệnh nằm trong lá")
        counts[split] = c

    print(json.dumps({s: dict(c) for s, c in counts.items()}, ensure_ascii=False, indent=2))
    for wmsg in warnings[:50]:
        print("CẢNH BÁO:", wmsg)
    for e in errors[:100]:
        print("LỖI:", e)
    print(f"Tổng: {len(errors)} lỗi, {len(warnings)} cảnh báo")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
