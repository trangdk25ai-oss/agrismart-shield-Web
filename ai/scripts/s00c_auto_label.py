"""BƯỚC 1 (hỗ trợ gán nhãn) – Sinh NHÃN NHÁP Labelme cho ảnh đã chọn bằng s00b, để rà soát thay vì vẽ từ đầu.

Với mỗi ảnh trong <thư mục>/<lớp_đích>/ chưa có tệp .json: tự tách phiến lá và vùng đổi màu (agrismart/auto_label.py),
ghi tệp nhãn Labelme cạnh ảnh với cờ "da_ra_soat": false. Tên bệnh lấy theo tên thư mục (phải là lớp trong
class_map.yaml). Ảnh không tách được lá hoặc không thấy vùng đổi màu thì KHÔNG ghi tệp, để gán nhãn tay.

QUAN TRỌNG – nhãn do máy sinh, không phải nhãn chuẩn: mở Labelme, sửa/xóa vùng sai, rồi tích cờ "da_ra_soat"
ở khung Flags. s01b chỉ nhập tệp đã tích cờ (hoặc tệp vẽ tay); tệp nháp chưa rà soát bị bỏ qua.

  python ai/scripts/s00c_auto_label.py --dir data_raw/can_gan_nhan --class-map ai/configs/class_map.yaml
  labelme data_raw/can_gan_nhan --labels ai/configs/labels/labelme_labels.txt --flags da_ra_soat
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart import LEAF_CLASS  # noqa: E402
from agrismart.auto_label import REVIEW_FLAG, auto_label, labelme_json  # noqa: E402
from agrismart.yolo_io import IMG_EXTS  # noqa: E402


def load_oriented_bgr(path: Path) -> np.ndarray:
    """Đọc ảnh và xoay theo EXIF như Labelme, để tọa độ nhãn khớp với ảnh Labelme hiển thị."""
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    return np.ascontiguousarray(rgb[..., ::-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path("data_raw/can_gan_nhan"),
                    help="thư mục ảnh cần gán nhãn, mỗi lớp một thư mục con (đầu ra của s00b)")
    ap.add_argument("--class-map", type=Path, default=Path("ai/configs/class_map.yaml"))
    ap.add_argument("--classes", nargs="*", help="chỉ xử lý các lớp này (mặc định: mọi thư mục con là lớp bệnh)")
    ap.add_argument("--overwrite", action="store_true",
                    help="ghi đè tệp .json đã có (MẤT nhãn đã sửa) – mặc định giữ nguyên")
    ap.add_argument("--max-severity", type=float, default=60.0,
                    help="tỷ lệ vùng đổi màu ước tính lớn hơn ngưỡng này -> đánh dấu nghi ngờ trong báo cáo")
    ap.add_argument("--report", type=Path, help="tệp CSV báo cáo (mặc định <dir>/auto_label_report.csv)")
    args = ap.parse_args()

    if not args.dir.is_dir():
        print(f"LỖI: không thấy thư mục {args.dir}")
        return 1
    cmap = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    targets = {v for v in cmap["target_names"].values() if v != LEAF_CLASS}
    folders = sorted(p for p in args.dir.iterdir() if p.is_dir())
    unknown = [p.name for p in folders if p.name not in targets]
    folders = [p for p in folders if p.name in targets and (not args.classes or p.name in args.classes)]
    if unknown:
        print("Bỏ qua thư mục không phải lớp bệnh trong class_map:", ", ".join(unknown))
    if not folders:
        print("LỖI: không có thư mục lớp bệnh nào. Lớp hợp lệ:", ", ".join(sorted(targets)))
        return 1

    stats: Counter = Counter()
    rows = []
    for folder in folders:
        for img_path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS):
            js = img_path.with_suffix(".json")
            row = {"anh": str(img_path.relative_to(args.dir)).replace("\\", "/"), "lop": folder.name,
                   "ket_qua": "", "phuong_phap": "", "ty_le_lon_uoc_tinh": "", "so_vung_benh": "", "ghi_chu": ""}
            if js.exists() and not args.overwrite:
                stats["giu_nguyen_json_da_co"] += 1
                row["ket_qua"] = "giu_nguyen_json_da_co"
                rows.append(row)
                continue
            try:
                img = load_oriented_bgr(img_path)
            except OSError as e:
                stats["loi_doc_anh"] += 1
                row.update(ket_qua="loi_doc_anh", ghi_chu=str(e))
                rows.append(row)
                continue
            h, w = img.shape[:2]
            res = auto_label(img)
            row.update(phuong_phap=res["method"], ty_le_lon_uoc_tinh=f"{res['severity_pct']:.2f}")
            if not res["leaf"].any():
                stats["khong_tach_duoc_la"] += 1
                row["ket_qua"] = "khong_tach_duoc_la"
                rows.append(row)
                continue
            if not res["lesion"].any():
                stats["khong_thay_vung_doi_mau"] += 1
                row["ket_qua"] = "khong_thay_vung_doi_mau"
                rows.append(row)
                continue
            data = labelme_json(res, folder.name, img_path.name, h, w)
            n_les = sum(1 for s in data["shapes"] if s["label"] != LEAF_CLASS)
            n_leaf = len(data["shapes"]) - n_les
            if n_les == 0 or n_leaf == 0:
                stats["vung_qua_nho"] += 1
                row["ket_qua"] = "vung_qua_nho"
                rows.append(row)
                continue
            js.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            notes = []
            if res["severity_pct"] > args.max_severity:
                notes.append(f"vùng đổi màu > {args.max_severity:g}% lá: có thể lá vàng/khô, kiểm tra kỹ")
            if n_leaf > 1:
                notes.append(f"{n_leaf} vùng lá: kiểm tra có tách đúng từng lá không")
            if res["leaf_frac"] < 0.05:
                notes.append("lá chiếm < 5% ảnh")
            stats["da_ghi_nhan_nhap"] += 1
            if notes:
                stats["nghi_ngo_can_xem_ky"] += 1
            row.update(ket_qua="da_ghi_nhan_nhap", so_vung_benh=n_les, ghi_chu="; ".join(notes))
            rows.append(row)

    report = args.report or args.dir / "auto_label_report.csv"
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["anh"])
        wr.writeheader()
        wr.writerows(rows)
    print(json.dumps(dict(sorted(stats.items())), ensure_ascii=False, indent=2))
    print(f"Báo cáo từng ảnh: {report}")
    print("Bước tiếp: mở Labelme để rà soát, sửa vùng sai, tích cờ '" + REVIEW_FLAG + "' rồi bấm Next:")
    print(f"  labelme {args.dir.as_posix()} --labels ai/configs/labels/labelme_labels.txt --flags {REVIEW_FLAG}")
    print("Ảnh 'khong_thay_vung_doi_mau' / 'khong_tach_duoc_la' chưa có tệp nhãn: gán tay hoặc bỏ qua.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
