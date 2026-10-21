"""Tra cứu danh mục nguồn dữ liệu, sinh trích dẫn và kiểm tra ảnh trùng giữa các nguồn.

  python ai/scripts/datasets_catalog.py list                     # toàn bộ nguồn
  python ai/scripts/datasets_catalog.py list --crop cucumber      # theo cây trồng
  python ai/scripts/datasets_catalog.py list --gap citrus__anthracnose   # nguồn có ảnh cho lớp còn thiếu
  python ai/scripts/datasets_catalog.py show riceseg_5932
  python ai/scripts/datasets_catalog.py cite --ids plantseg plantdoc --manifest data_raw/can_gan_nhan/manifest.csv
  python ai/scripts/datasets_catalog.py dedup --a data_raw/plantdoc/TEST --b datasets/agrismart/images/train \
      --out reports/trung_plantdoc_train.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.image_hash import NearDuplicateIndex, dhash_file  # noqa: E402
from agrismart.sources import filter_sources, load_catalog  # noqa: E402
from agrismart.yolo_io import list_images  # noqa: E402


def cmd_list(cat, args) -> int:
    rows = filter_sources(cat, args.crop, args.level, args.gap)
    for s in rows:
        lic = s.license if s.license_verified else "CHƯA XÁC MINH"
        print(f"[{s.level}] {s.id:<32} {s.pipeline:<16} {lic:<22} {s.main_url}")
    print(f"{len(rows)} nguồn")
    return 0


def cmd_show(cat, args) -> int:
    s = cat.get(args.id)
    if s is None:
        print("Không có nguồn", args.id)
        return 1
    for k, v in s.__dict__.items():
        if v not in (None, "", [], {}):
            print(f"{k:>17}: {v}")
    return 0


def cmd_cite(cat, args) -> int:
    ids = list(dict.fromkeys(args.ids or []))
    for m in args.manifest or []:
        with open(m, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["source_id"] not in ids:
                    ids.append(r["source_id"])
    missing = [i for i in ids if i not in cat]
    if missing:
        print("Không có trong danh mục:", missing)
        return 1
    for n, i in enumerate(ids, 1):
        print(f"[{n}] {cat[i].citation()}")
    unverified = [i for i in ids if not cat[i].license_verified]
    if unverified:
        print("\nCẢNH BÁO: chưa xác minh giấy phép:", ", ".join(unverified))
    return 0


def cmd_dedup(cat, args) -> int:
    index = NearDuplicateIndex(args.hamming)
    for p in list_images(args.b):
        h = dhash_file(p)
        if h is not None:
            index.add(str(p), h)
    pairs = []
    for p in list_images(args.a):
        h = dhash_file(p)
        if h is None:
            continue
        hit = index.find(h)
        if hit:
            pairs.append((str(p), hit[0], hit[1]))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_a", "image_b", "hamming"])
            w.writerows(pairs)
    print(f"{len(pairs)} ảnh trong {args.a} trùng/gần trùng với {args.b} (ngưỡng Hamming {args.hamming})")
    for a, b, d in pairs[:20]:
        print(f"  {d:>2}  {a}  ~  {b}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("list")
    a.add_argument("--crop")
    a.add_argument("--level")
    a.add_argument("--gap", help="lớp đích còn thiếu dữ liệu, ví dụ cucumber__downy_mildew")
    b = sub.add_parser("show")
    b.add_argument("id")
    c = sub.add_parser("cite")
    c.add_argument("--ids", nargs="*")
    c.add_argument("--manifest", nargs="*", help="manifest.csv do s00b sinh ra")
    d = sub.add_parser("dedup")
    d.add_argument("--a", required=True, type=Path, help="tập cần kiểm tra (ví dụ tập kiểm tra ngoài)")
    d.add_argument("--b", required=True, type=Path, help="tập đối chiếu (ví dụ ảnh huấn luyện)")
    d.add_argument("--hamming", type=int, default=5)
    d.add_argument("--out", type=Path)
    args = ap.parse_args()
    cat = load_catalog()
    return {"list": cmd_list, "show": cmd_show, "cite": cmd_cite, "dedup": cmd_dedup}[args.cmd](cat, args)


if __name__ == "__main__":
    sys.exit(main())
