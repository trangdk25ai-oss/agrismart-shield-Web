"""BƯỚC 1.1 – Chia tập 70% Train / 20% Validation / 10% Test, phân tầng theo bệnh, KHÔNG rò rỉ dữ liệu.

- Phân tầng (stratify) theo "lớp bệnh chủ đạo" của từng ảnh (lớp vết bệnh có tổng diện tích lớn nhất)
  để tỷ lệ các bệnh trong 3 tập tương đương nhau.
- Chia theo NHÓM (--group-regex): các ảnh cùng một cây/lá/lần chụp phải nằm cùng một tập, nếu không
  mô hình "học thuộc" cây đó và chỉ số trên tập Test bị thổi phồng.
- Chạy TRƯỚC bước augmentation; augmentation chỉ áp dụng cho tập train (s03).

Ví dụ:
    python scripts/s02_split_dataset.py --src datasets/agrismart_all --out datasets/agrismart \
        --class-map configs/class_map.yaml --ratios 0.7 0.2 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.yolo_io import list_images, read_label_file  # noqa: E402

SPLITS = ("train", "val", "test")


def polygon_area(poly: np.ndarray) -> float:
    """Công thức Shoelace trên tọa độ chuẩn hóa."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def dominant_class(label_path: Path, names: dict[int, str]) -> str:
    areas: Counter = Counter()
    for inst in read_label_file(label_path):
        name = names.get(inst.cls_id, "?")
        if name != "leaf":
            areas[name] += polygon_area(inst.polygon)
    return areas.most_common(1)[0][0] if areas else "healthy"


def assign_groups(groups: list[tuple[str, int]], ratios: tuple[float, float, float], rng: random.Random) -> dict:
    """Gán nhóm vào tập có 'thiếu hụt' lớn nhất so với tỷ lệ mục tiêu (tham lam theo số ảnh)."""
    rng.shuffle(groups)
    groups.sort(key=lambda g: -g[1])  # nhóm lớn xếp trước để tỷ lệ cuối sát mục tiêu
    total = sum(n for _, n in groups)
    counts = dict.fromkeys(SPLITS, 0)
    out = {}
    for key, n in groups:
        deficits = {s: ratios[i] * total - counts[s] for i, s in enumerate(SPLITS)}
        s = max(deficits, key=deficits.get)
        out[key] = s
        counts[s] += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="thư mục có images/ và labels/ (đầu ra s01)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--class-map", required=True, type=Path)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--group-regex", default=r"^(.*)$",
                    help="regex trên tên file (không đuôi); nhóm 1 là khóa nhóm. VD: '^(plant\\d+)_' ")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("Tổng tỷ lệ phải bằng 1")

    cmap = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cmap["target_names"].items()}
    rx = re.compile(args.group_regex)
    rng = random.Random(args.seed)

    images = list_images(args.src / "images")
    group_images: dict[str, list[Path]] = defaultdict(list)
    group_class: dict[str, Counter] = defaultdict(Counter)
    for img in images:
        m = rx.match(img.stem)
        key = m.group(1) if m else img.stem
        group_images[key].append(img)
        group_class[key][dominant_class(args.src / "labels" / f"{img.stem}.txt", names)] += 1

    by_stratum: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key, imgs in group_images.items():
        by_stratum[group_class[key].most_common(1)[0][0]].append((key, len(imgs)))

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        assignment.update(assign_groups(by_stratum[stratum], tuple(args.ratios), rng))

    report: dict = {s: Counter() for s in SPLITS}
    for key, imgs in group_images.items():
        split = assignment[key]
        for img in imgs:
            for sub in ("images", "labels"):
                (args.out / sub / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, args.out / "images" / split / img.name)
            lbl = args.src / "labels" / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, args.out / "labels" / split / lbl.name)
            report[split][dominant_class(lbl, names)] += 1

    data_yaml = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    (args.out / "dataset.yaml").write_text(yaml.safe_dump(data_yaml, allow_unicode=True, sort_keys=False), "utf-8")
    summary = {s: {"total": sum(c.values()), **dict(c)} for s, c in report.items()}
    (args.out / "split_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Đã ghi {args.out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
