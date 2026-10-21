"""BƯỚC 6 (thay thực địa) – Tạo bộ ảnh mẫu có đáp án cho Demo và tờ in để kiểm thử "chụp lại".

Chọn ảnh từ tập KIỂM TRA (không dùng khi huấn luyện), rải đều theo bệnh × mức độ (Khởi phát / Nhẹ / Nặng /
Khỏe), tính đáp án Severity từ nhãn polygon bằng đúng thuật toán của hệ thống, rồi ghi:
    demo/samples/sNN.jpg               ảnh mẫu (trang web hiển thị ở thẻ "Ảnh mẫu")
    demo/samples/manifest.json         đáp án + nguồn + giấy phép của từng ảnh
    demo/samples/to_in_A4.html         tờ in A4 (mở bằng trình duyệt -> In) để chụp lại bằng điện thoại

    python ai/scripts/s10_demo_pack.py --data datasets/agrismart/dataset.yaml --split test --n 24 \
        --source-map plantseg_=plantseg --source-map lbl_=tu_gan_nhan

Giấy phép: ảnh PlantSeg theo CC BY-NC-ND 4.0 – được dùng nguyên bản, phi thương mại, ghi nguồn; KHÔNG đưa
thư mục demo/samples lên kho mã công khai (đã có trong .gitignore).
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.io_utils import imread, imwrite  # noqa: E402
from agrismart.knowledge_engine import KnowledgeEngine  # noqa: E402
from agrismart.severity import severity_from_label  # noqa: E402
from agrismart.sources import load_catalog  # noqa: E402
from agrismart.yolo_io import list_images, read_label_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEVEL_ORDER = ["HEALTHY", "ONSET", "LIGHT", "HEAVY"]


def pick(cands: list[dict], n: int, seed: int) -> list[dict]:
    """Chọn n ảnh rải đều theo (bệnh, mức); trong mỗi nhóm lấy ngẫu nhiên có hạt giống cố định."""
    rng = np.random.default_rng(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cands:
        groups[(c["gt_class"] or "", c["gt_level_code"])].append(c)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=lambda k: (k[0], LEVEL_ORDER.index(k[1]) if k[1] in LEVEL_ORDER else 9))
    out: list[dict] = []
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def source_of(name: str, mapping: dict[str, str]) -> str | None:
    for prefix, sid in sorted(mapping.items(), key=lambda kv: -len(kv[0])):
        if name.startswith(prefix):
            return sid
    return None


def print_sheet(items: list[dict], folder: Path) -> str:
    cells = "".join(
        f'<figure><img src="{html.escape(it["file"])}" alt=""><figcaption><b>{html.escape(it["id"])}</b> · '
        f'{html.escape(it["gt_disease_name"] or "Lá khỏe")} · đáp án {it["gt_severity_pct"]:.2f}%</figcaption></figure>'
        for it in items)
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Tờ in ảnh mẫu AgriSmart Shield</title>
<style>
@page {{ size: A4; margin: 12mm; }}
body {{ font: 11pt system-ui, Arial, sans-serif; margin: 0; }}
.note {{ font-size: 10pt; margin: 0 0 6mm; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }}
figure {{ margin: 0; break-inside: avoid; border: 1px solid #999; padding: 3mm; }}
img {{ width: 100%; height: 78mm; object-fit: contain; display: block; }}
figcaption {{ font-size: 10pt; margin-top: 2mm; }}
@media print {{ .note {{ display: none; }} }}
</style></head><body>
<p class="note">In tờ này (A4, màu, không co giãn). Chụp lại từng ảnh bằng điện thoại ở 3 điều kiện: ánh sáng phòng, gần cửa sổ,
nghiêng khoảng 20°. Đặt tên tệp <code>&lt;mã&gt;_&lt;lần&gt;.jpg</code> (ví dụ <code>s07_1.jpg</code>), chép vào
<code>demo/chup_lai/</code> rồi chạy <code>s09_robustness.py --recapture-dir demo/chup_lai</code>.
Có thể chụp màn hình máy tính thay cho tờ in (ghi rõ trong báo cáo).</p>
<div class="grid">{cells}</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--source-map", action="append", default=[], metavar="TIỀN_TỐ=ID_NGUỒN",
                    help="tiền tố tên ảnh -> id trong data_sources/catalog.yaml, ví dụ plantseg_=plantseg")
    ap.add_argument("--out", type=Path, default=ROOT / "demo/samples")
    args = ap.parse_args()

    engine = KnowledgeEngine.from_file(ROOT / "knowledge_base/ipm_rules.json")
    catalog = load_catalog()
    mapping = dict(m.split("=", 1) for m in args.source_map)
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    img_dir = Path(cfg["path"]) / cfg[args.split]
    lbl_dir = Path(cfg["path"]) / cfg[args.split].replace("images", "labels", 1)

    cands = []
    for p in list_images(img_dir):
        img = imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        rep = severity_from_label(read_label_file(lbl_dir / f"{p.stem}.txt"), names, h, w)
        if rep.primary is None or rep.status not in ("OK", "HEALTHY"):
            continue
        pr = rep.primary
        cls = pr.dominant_class
        rule = engine.rule_for_class(cls or "")
        level = engine.classify(pr.severity_pct, pr.lesion_area_px > 0, rule)
        lv = engine.kb["severity_levels"][level]
        crop = rule["crop_code"] if rule else (cls.split("__")[0] if cls else None)
        cands.append({"path": p, "gt_class": cls, "gt_disease_name": rule["disease_name"] if rule else None,
                      "gt_severity_pct": round(pr.severity_pct, 2), "gt_level_code": level, "gt_level_label": lv["label"],
                      "crop_code": crop})
    if not cands:
        raise SystemExit(f"Không có ảnh có nhãn hợp lệ trong {img_dir}")
    chosen = pick([c for c in cands if c["crop_code"]], args.n, args.seed)

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    items = []
    for i, c in enumerate(chosen, 1):
        sid = f"s{i:02d}"
        img = imread(c["path"])
        h, w = img.shape[:2]
        s = args.max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        imwrite(args.out / f"{sid}.jpg", img)
        src = source_of(c["path"].name, mapping)
        cat = catalog.get(src) if src else None
        items.append({
            "id": sid, "file": f"{sid}.jpg", "crop_code": c["crop_code"], "gt_class": c["gt_class"],
            "gt_disease_name": c["gt_disease_name"], "gt_severity_pct": c["gt_severity_pct"],
            "gt_level_code": c["gt_level_code"], "gt_level_label": c["gt_level_label"],
            "origin": c["path"].name, "source_id": src, "license": cat.license if cat else None,
        })
    manifest = {"generated_at": datetime.now().isoformat(timespec="seconds"), "dataset": str(args.data),
                "split": args.split, "note": "Đáp án tính từ nhãn polygon bằng agrismart.severity (cùng thuật toán với hệ thống).",
                "items": items}
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    (args.out / "to_in_A4.html").write_text(print_sheet(items, args.out), "utf-8")
    groups = defaultdict(int)
    for it in items:
        groups[(it["gt_disease_name"] or "Lá khỏe", it["gt_level_label"])] += 1
    print(f"Đã chọn {len(items)}/{len(cands)} ảnh vào {args.out}")
    for (d, lv), n in sorted(groups.items()):
        print(f"  {d:32s} {lv:14s} {n}")
    no_src = sum(1 for it in items if not it["source_id"])
    if no_src:
        print(f"Lưu ý: {no_src} ảnh chưa xác định nguồn – thêm --source-map <tiền_tố>=<id_nguồn> để ghi giấy phép.")


if __name__ == "__main__":
    main()
