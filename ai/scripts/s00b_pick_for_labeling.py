"""BƯỚC 1 (nguồn bổ sung) – Chọn ảnh từ bộ dữ liệu PHÂN LOẠI để gán polygon (Labelme/CVAT).

Dùng cho các nguồn chỉ có nhãn phân loại theo thư mục (Cucumber Disease Recognition, BDLemonLeaf,
Large-Scale Lemon Leaf, sầu riêng Việt Nam…). Script:
  1. lấy ảnh trong các thư mục lớp được chỉ định (--map "Tên thư mục=lớp_đích");
  2. bỏ ảnh tăng cường có sẵn (--exclude-regex, mặc định bỏ tên chứa "aug");
  3. bỏ ảnh trùng / gần trùng (dHash) với nhau và với các thư mục đã có (--dedup-against: tập huấn luyện,
     các lần chọn trước, PlantSeg…);
  4. chọn ngẫu nhiên có hạt giống, chép sang <out>/<lớp_đích>/ cho tới khi đủ --per-class ảnh mỗi lớp
     (chạy lại hoặc chạy với nguồn khác chỉ bổ sung phần còn thiếu, không chép trùng);
  5. ghi manifest.csv (ảnh, lớp gợi ý, nguồn, DOI, giấy phép) để truy nguồn và trích dẫn.

Ví dụ:
  python ai/scripts/s00b_pick_for_labeling.py --source-id cucumber_recognition \
      --src "data_raw/cucumber_recognition/Original Images" \
      --map "Downy Mildew=cucumber__downy_mildew" --map "Anthracnose=cucumber__anthracnose" \
      --per-class 150 --dedup-against datasets/agrismart_all/images --out data_raw/can_gan_nhan

Tên thư mục lớp trong --map là VÍ DỤ – mở bộ dữ liệu đã tải để lấy đúng tên thư mục.
Sau đó gán nhãn theo docs/huong_dan_gan_nhan.md và nhập bằng s01b (--format labelme --labelme-dir <out>).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.class_registry import CLASS_NAME_RE  # noqa: E402
from agrismart.image_hash import NearDuplicateIndex, dhash, dhash_file  # noqa: E402
from agrismart.io_utils import imread  # noqa: E402
from agrismart.sources import load_catalog  # noqa: E402
from agrismart.yolo_io import list_images  # noqa: E402

MANIFEST_FIELDS = ["file", "target_class_hint", "source_id", "source_class", "source_path", "dhash", "doi", "license"]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def parse_map(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--map '{it}' phải có dạng 'Tên thư mục=lớp_đích'")
        folder, target = (x.strip() for x in it.split("=", 1))
        if not CLASS_NAME_RE.match(target):
            raise SystemExit(f"Lớp đích '{target}' sai quy ước <cây>__<bệnh>")
        out[folder] = target
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-id", required=True, help="id trong data_sources/catalog.yaml")
    ap.add_argument("--src", required=True, type=Path, help="thư mục gốc của bộ dữ liệu phân loại")
    ap.add_argument("--map", action="append", default=[], required=True, help="'Tên thư mục lớp=lớp_đích', lặp lại được")
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--exclude-regex", default=r"(?i)aug", help="bỏ ảnh có đường dẫn khớp regex (ảnh tăng cường)")
    ap.add_argument("--dedup-against", nargs="*", type=Path, default=[], help="thư mục ảnh đã có, để loại ảnh trùng")
    ap.add_argument("--hamming", type=int, default=5, help="ngưỡng khoảng cách dHash coi là trùng")
    ap.add_argument("--min-side", type=int, default=256, help="bỏ ảnh có cạnh ngắn nhỏ hơn N pixel")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    catalog = load_catalog()
    if args.source_id not in catalog:
        print(f"LỖI: '{args.source_id}' không có trong danh mục. Xem: python ai/scripts/datasets_catalog.py list")
        return 1
    src = catalog[args.source_id]
    if not src.license_verified:
        print(f"CẢNH BÁO: giấy phép của '{src.id}' chưa được xác minh – kiểm tra {src.main_url} trước khi dùng.")
    folder_map = parse_map(args.map)
    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None

    # 1) thu thập ứng viên theo lớp (khớp tên thư mục ở bất kỳ cấp nào trong đường dẫn tương đối)
    cands: dict[str, list[Path]] = defaultdict(list)
    stats: Counter = Counter()
    for p in list_images(args.src):
        rel = p.relative_to(args.src)
        if exclude and exclude.search(str(rel)):
            stats["skip_augmented"] += 1
            continue
        folder = next((part for part in reversed(rel.parts[:-1]) if part in folder_map), None)
        if folder is None:
            continue
        cands[folder].append(p)
    if not cands:
        print("LỖI: không tìm thấy ảnh nào trong các thư mục:", list(folder_map))
        print("Các thư mục con hiện có:", sorted({q.name for q in args.src.rglob('*') if q.is_dir()})[:30])
        return 1

    # 2) chỉ mục ảnh đã có để loại trùng (gồm cả ảnh đã chọn ở các lần chạy trước trong --out)
    index = NearDuplicateIndex(args.hamming)
    for d in [*args.dedup_against, *([args.out] if args.out.exists() else [])]:
        for q in list_images(d):
            h = dhash_file(q)
            if h is not None:
                index.add(str(q), h)
    print(f"Chỉ mục loại trùng: {len(index)} ảnh có sẵn")

    # 3) chọn ngẫu nhiên, bỏ trùng, chép
    rng = random.Random(args.seed)
    manifest_path = args.out / "manifest.csv"
    new_manifest = not manifest_path.exists()
    args.out.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        if new_manifest:
            w.writeheader()
        for folder, paths in sorted(cands.items()):
            target = folder_map[folder]
            rng.shuffle(paths)
            already = len(list_images(args.out / target)) if (args.out / target).exists() else 0
            picked = already  # --per-class là TỔNG số ảnh mỗi lớp trong --out, chạy lại chỉ bổ sung phần thiếu
            for p in paths:
                if picked >= args.per_class:
                    break
                gray = imread(p, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    stats["skip_unreadable"] += 1
                    continue
                if min(gray.shape[:2]) < args.min_side:
                    stats["skip_too_small"] += 1
                    continue
                h = dhash(gray)
                dup = index.find(h)
                if dup is not None:
                    stats["skip_near_duplicate_or_already_picked"] += 1
                    continue
                name = f"{src.id}__{safe_name(folder)}__{safe_name(p.stem)}{p.suffix.lower()}"
                dst = args.out / target / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                index.add(str(dst), h)
                w.writerow({"file": str(dst.relative_to(args.out)).replace("\\", "/"), "target_class_hint": target,
                            "source_id": src.id, "source_class": folder,
                            "source_path": str(p.relative_to(args.src)).replace("\\", "/"),
                            "dhash": f"{h:016x}", "doi": src.doi or "", "license": src.license})
                picked += 1
            stats[f"picked_new_{target}"] += picked - already
            stats[f"total_in_out_{target}"] = picked
            stats[f"available_{target}"] += len(paths)

    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print(f"Ảnh đã chọn nằm trong {args.out}/<lớp_đích>/ ; nguồn ghi tại {manifest_path}")
    print("Bước tiếp: gán polygon 'leaf' + vết bệnh theo docs/huong_dan_gan_nhan.md, rồi chạy s01b --format labelme.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
