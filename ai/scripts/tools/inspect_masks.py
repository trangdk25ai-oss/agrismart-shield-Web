"""Kiểm tra nhanh thư mục mask của một nguồn dữ liệu mới trước khi chạy s01.

Cho biết: định dạng (PNG/JPG), mask là MÃ LỚP hay NHỊ PHÂN, vết bệnh màu trắng hay đen (cần --invert-masks).
Từ đó chọn đúng --mask-mode cho s01.

  python ai/scripts/tools/inspect_masks.py data_raw/plantseg/annotations --n 50 --class-map ai/configs/class_map.yaml
  python ai/scripts/tools/inspect_masks.py data_raw/riceseg/masks --n 50

Cách nhận biết:
  - mask JPG (nén mất dữ liệu) hoặc mỗi mask có rất nhiều mức xám -> nhị phân hóa (--mask-mode binary);
  - giá trị khác 0 chỉ là 255 (hoặc chỉ là 1) -> mask nhị phân (--mask-mode binary);
  - mỗi mask chỉ vài giá trị nguyên (vd. 0 và 98) -> mask theo mã lớp (--mask-mode value), ví dụ PlantSeg
    (nền = 0, vết bệnh = mã lớp 1–115; thường mỗi ảnh chỉ một bệnh nên mỗi mask chỉ có 2 giá trị).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agrismart.io_utils import imread  # noqa: E402
from agrismart.yolo_io import list_images  # noqa: E402

# mask theo mã lớp chỉ có vài giá trị/ảnh; nhiều hơn ngưỡng này coi là ảnh xám (khử răng cưa, nén, mask mềm)
MAX_CODES_PER_MASK = 16


def inspect_one(m: np.ndarray) -> dict:
    if m.ndim == 3:
        m = m[..., 0]
    vals = np.unique(m)
    return {
        "values": [int(v) for v in vals],
        "n_values": int(vals.size),
        "nonzero_frac": float((m > 0).mean()),
        "bright_frac": float((m > 127).mean()),
    }


def classify(rows: list[dict], exts: Counter) -> tuple[str, Counter]:
    """Trả về (loại mask, số mask chứa mỗi giá trị khác 0). Loại: 'jpg' | 'gray' | 'binary255' | 'binary1' | 'value'."""
    per_value: Counter = Counter()
    for r in rows:
        for v in r["values"]:
            if v > 0:
                per_value[v] += 1
    if ".jpg" in exts or ".jpeg" in exts:
        return "jpg", per_value
    if max(r["n_values"] for r in rows) > MAX_CODES_PER_MASK:
        return "gray", per_value
    nonzero = set(per_value)
    if nonzero and nonzero <= {255}:
        return "binary255", per_value
    if nonzero and nonzero <= {1}:
        return "binary1", per_value
    return "value", per_value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks", type=Path)
    ap.add_argument("--n", type=int, default=50, help="số mask lấy ngẫu nhiên để xem")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--class-map", type=Path, help="class_map.yaml: đối chiếu mã lớp gặp được với value_to_target")
    args = ap.parse_args()
    files = list_images(args.masks)
    if not files:
        print("Không thấy file mask trong", args.masks)
        return 1
    random.Random(args.seed).shuffle(files)
    exts, rows = Counter(), []
    for f in files[: args.n]:
        m = imread(f, cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        exts[f.suffix.lower()] += 1
        rows.append(inspect_one(m))
    if not rows:
        print("Không đọc được mask nào")
        return 1

    kind, per_value = classify(rows, exts)
    nvals = max(r["n_values"] for r in rows)
    print(f"Đã xem {len(rows)}/{len(files)} mask, định dạng: {dict(exts)}")
    print(f"Số giá trị pixel khác nhau trong 1 mask (lớn nhất): {nvals}; ví dụ: {rows[0]['values'][:12]}")
    print(f"Tỷ lệ pixel khác 0 (trung vị): {float(np.median([r['nonzero_frac'] for r in rows])):.1%}")

    if kind == "jpg":
        print("-> Mask JPG (nén mất dữ liệu): dùng --mask-mode binary (s01 nhị phân hóa ở ngưỡng 127). Không dùng --mask-mode value.")
    elif kind == "gray":
        print(f"-> Mỗi mask có tới {nvals} mức xám (khử răng cưa/mask mềm): dùng --mask-mode binary.")
    elif kind in ("binary255", "binary1"):
        print("-> Mask nhị phân (giá trị khác 0 chỉ là " + ("255" if kind == "binary255" else "1")
              + "): dùng --mask-mode binary, khai báo folder_to_target trong class_map.yaml.")
        frac_key = "bright_frac" if kind == "binary255" else "nonzero_frac"
        if float(np.median([r[frac_key] for r in rows])) > 0.6:
            print("-> Vùng khác 0 chiếm phần lớn ảnh: có thể vết bệnh màu ĐEN trên nền trắng -> thêm --invert-masks"
                  " (mở vài mask để xác nhận trước).")
    else:
        top = ", ".join(f"{v} ({c} mask)" for v, c in sorted(per_value.items(), key=lambda t: (-t[1], t[0]))[:20])
        print("-> Mask theo MÃ LỚP (mỗi mask chỉ vài giá trị nguyên): dùng --mask-mode value, khai báo value_to_target.")
        print(f"   Mã lớp gặp trong {len(rows)} mask đã xem: {top}")

    if args.class_map and kind == "value":
        cmap = yaml.safe_load(args.class_map.read_text(encoding="utf-8")) or {}
        v2t = {int(k): v for k, v in (cmap.get("value_to_target") or {}).items()}
        if not v2t:
            print(f"   {args.class_map} chưa có value_to_target – chạy s00 trước.")
        else:
            seen = {v: per_value.get(v, 0) for v in sorted(v2t)}
            print("   Mã lớp đã khai báo trong value_to_target và số mask có mã đó (trong mẫu đã xem): "
                  + ", ".join(f"{v}={v2t[v]}: {c}" for v, c in seen.items()))
            if not any(seen.values()):
                print("   (Mẫu ngẫu nhiên chưa gặp mã nào đã khai báo – bình thường nếu các bệnh đã chọn chiếm ít ảnh;"
                      " s01 sẽ quét toàn bộ và báo số ảnh được giữ.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
