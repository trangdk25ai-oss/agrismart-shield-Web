"""BƯỚC 1 (mở rộng) – Chọn loại bệnh cần nhận dạng và sinh class_map.yaml + danh sách nhãn cho công cụ gán nhãn.

Hai nguồn lớp bệnh có thể kết hợp:
  (1) Bệnh có sẵn mask trong PlantSeg  -> chọn theo tên (--plantseg-diseases) hoặc theo cây trồng (--plantseg-crops)
  (2) Bệnh tự chụp, tự gán nhãn         -> khai báo tên lớp <crop>__<disease> (--extra)

Ví dụ:
  # Xem PlantSeg có những bệnh nào của dưa chuột
  python ai/scripts/s00_build_class_map.py --list cucumber

  # 3 bệnh cà chua + phấn trắng dưa chuột + loét cây có múi (PlantSeg)
  # + 3 bệnh tự gán nhãn tại địa phương
  python ai/scripts/s00_build_class_map.py \
      --plantseg-diseases "tomato early blight" "tomato late blight" "tomato septoria leaf spot" \
                          "cucumber powdery mildew" "citrus canker" \
      --extra cucumber__downy_mildew cucumber__anthracnose citrus__anthracnose \
      --out ai/configs/class_map.yaml --labels-dir ai/configs/labels

Sau bước này: gán nhãn ảnh tự chụp theo docs/huong_dan_gan_nhan.md, chạy s01 (PlantSeg) và s01b (nhãn tự gán)
vào CÙNG một thư mục đầu ra, rồi chạy knowledge_base/kb_add_rules.py để tạo quy tắc IPM chờ nhập cho bệnh mới.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.class_registry import (  # noqa: E402
    ClassRegistryError,
    build_class_map,
    carry_over_manual_mappings,
    load_plantseg,
    plantseg_target_name,
    select_plantseg,
    write_annotation_labels,
    write_class_map,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", nargs="*", metavar="CROP", help="liệt kê bệnh PlantSeg (lọc theo cây trồng nếu có)")
    ap.add_argument("--plantseg-diseases", nargs="*", default=[])
    ap.add_argument("--plantseg-crops", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[], help="lớp tự gán nhãn, dạng <crop>__<disease>")
    ap.add_argument("--out", type=Path, default=Path("ai/configs/class_map.yaml"))
    ap.add_argument("--labels-dir", type=Path, default=Path("ai/configs/labels"))
    args = ap.parse_args()

    entries = load_plantseg()
    if args.list is not None:
        crops = {c.lower().replace(" ", "") for c in args.list}
        for e in entries:
            if not crops or e["crop"] in crops:
                vi = f"{e['crop_vi']} – {e['disease_vi']}" if e["disease_vi"] else e["crop_vi"]
                print(f"{e['id']:>4}  {e['name']:<42} -> {plantseg_target_name(e):<40} ({vi})")
        return 0

    try:
        selected = select_plantseg(entries, args.plantseg_diseases, args.plantseg_crops)
        cmap = build_class_map(selected, args.extra)
    except ClassRegistryError as e:
        print("LỖI:", e)
        return 1
    # giữ ánh xạ thư mục / bí danh nhãn đã khai báo tay trong tệp cũ (nếu lớp đích vẫn còn)
    old = yaml.safe_load(args.out.read_text(encoding="utf-8")) if args.out.exists() else None
    kept, dropped = carry_over_manual_mappings(cmap, old)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_class_map(cmap, args.out)
    files = write_annotation_labels(cmap, args.labels_dir)
    print(f"Đã ghi {args.out} với {len(cmap['target_names'])} lớp:")
    for i, n in cmap["target_names"].items():
        src = next((f"PlantSeg mã {v}" for v, t in cmap["value_to_target"].items() if t == n), "tự gán nhãn" if i else "phiến lá")
        print(f"  {i:>2}  {n:<40} {src}")
    if kept:
        print(f"Giữ lại {len(kept)} ánh xạ khai báo tay từ tệp cũ (folder_to_target/label_aliases).")
    for d in dropped:
        print("  Bỏ ánh xạ vì lớp đích không còn trong danh sách:", d)
    print("Danh sách nhãn cho công cụ gán nhãn:", ", ".join(str(f) for f in files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
