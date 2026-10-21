"""BƯỚC 6 (thay thực địa) – Kiểm thử độ bền và kiểm thử "chụp lại ảnh in".

(A) Độ bền với điều kiện chụp mô phỏng, trên tập kiểm tra có nhãn (đáp án tính từ nhãn polygon):
    python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx \
        --data datasets/agrismart/dataset.yaml --split test --out reports/robustness

(B) Chụp lại ảnh mẫu đã in / hiển thị trên màn hình bằng điện thoại (ảnh đặt tên <id ảnh mẫu>_<số>.jpg,
    ví dụ s07_1.jpg, chép vào demo/chup_lai/), đáp án lấy từ demo/samples/manifest.json do s10 sinh:
    python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx \
        --recapture-dir demo/chup_lai --manifest demo/samples/manifest.json --out reports/recapture

(C) Chạy trên chính bộ ảnh mẫu để so với đánh giá của chuyên gia (s11 ghép với demo/danh_gia_chuyen_gia.csv):
    python ai/scripts/s09_robustness.py --model web/models/model_int8.onnx \
        --recapture-dir demo/samples --condition-name anh_mau --out reports/samples

Không có mô hình (chỉ để chạy thử script): thay --model bằng --color-rule.
Đầu ra: <out>.json (tổng hợp theo điều kiện + gợi ý ngưỡng Blur), <out>.csv (từng ảnh), <out>.md (bảng).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.io_utils import imread  # noqa: E402
from agrismart.knowledge_engine import KnowledgeEngine  # noqa: E402
from agrismart.robustness import CONDITIONS, Truth, evaluate_one, suggest_blur_threshold, summarize, to_markdown  # noqa: E402
from agrismart.severity import severity_from_label  # noqa: E402
from agrismart.yolo_io import list_images, read_label_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def load_engine() -> KnowledgeEngine:
    return KnowledgeEngine.from_file(ROOT / "knowledge_base/ipm_rules.json")


def truth_from_labels(img_path: Path, lbl_dir: Path, names: dict[int, str], engine, h: int, w: int) -> Truth | None:
    gt = severity_from_label(read_label_file(lbl_dir / f"{img_path.stem}.txt"), names, h, w)
    if gt.primary is None:
        return None
    p = gt.primary
    cls = p.dominant_class
    return Truth(p.severity_pct, engine.classify(p.severity_pct, p.lesion_area_px > 0, engine.rule_for_class(cls or "")), cls)


def load_model(args):
    if args.color_rule:
        from agrismart.color_rule import ColorRuleModel

        return ColorRuleModel()
    from agrismart.seg_runtime import load_model as _load

    return _load(args.model)


def run_dataset(args, model, engine) -> list[dict]:
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = {int(k): v for k, v in cfg["names"].items()}
    img_dir = Path(cfg["path"]) / cfg[args.split]
    lbl_dir = Path(cfg["path"]) / cfg[args.split].replace("images", "labels", 1)
    conds = args.conditions or list(CONDITIONS)
    unknown = [c for c in conds if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"Điều kiện không có: {unknown}. Có: {list(CONDITIONS)}")
    images = list_images(img_dir)[: args.limit or None]
    rows = []
    t0 = time.time()
    for i, path in enumerate(images, 1):
        img = imread(path)
        if img is None:
            continue
        truth = truth_from_labels(path, lbl_dir, names, engine, *img.shape[:2])
        if truth is None:
            continue
        for c in conds:
            row = evaluate_one(model, engine, CONDITIONS[c][1](img), truth, c, conf=args.conf,
                               blur_threshold=args.blur_threshold)
            row["image"] = path.name
            rows.append(row)
        if i % 20 == 0:
            print(f"  {i}/{len(images)} ảnh – {time.time() - t0:.0f} s", flush=True)
    return rows


def run_recapture(args, model, engine) -> list[dict]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    items = {it["id"]: it for it in manifest["items"]}
    ids = sorted(items, key=len, reverse=True)  # khớp id dài nhất trước (s1 vs s10)
    rows, skipped = [], []
    for path in list_images(args.recapture_dir):
        sid = next((k for k in ids if path.stem == k or path.stem.startswith(k + "_")), None)
        img = imread(path)
        if sid is None or img is None:
            skipped.append(path.name)
            continue
        it = items[sid]
        truth = Truth(it["gt_severity_pct"], it["gt_level_code"], it.get("gt_class"))
        row = evaluate_one(model, engine, img, truth, args.condition_name, conf=args.conf, blur_threshold=args.blur_threshold)
        row.update(image=path.name, sample_id=sid)
        rows.append(row)
    if skipped:
        print(f"Bỏ qua {len(skipped)} ảnh không khớp id ảnh mẫu (đặt tên <id>_<số>.jpg): {skipped[:5]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="mô hình .onnx / .tflite / .pt")
    src.add_argument("--color-rule", action="store_true", help="chế độ thử không mô hình (chỉ để kiểm tra script)")
    ap.add_argument("--data", help="dataset.yaml (chế độ A)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--conditions", nargs="*", help=f"mặc định tất cả: {' '.join(CONDITIONS)}")
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N ảnh đầu (chạy thử nhanh)")
    ap.add_argument("--recapture-dir", type=Path, help="thư mục ảnh chụp lại (chế độ B)")
    ap.add_argument("--manifest", type=Path, default=ROOT / "demo/samples/manifest.json")
    ap.add_argument("--condition-name", default="chup_lai",
                    help="tên điều kiện ở chế độ B; dùng 'anh_mau' khi chạy trên chính demo/samples (để so với chuyên gia)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--blur-threshold", type=float, default=100.0)
    ap.add_argument("--out", type=Path, default=Path("reports/robustness"), help="tiền tố tệp kết quả (không đuôi)")
    args = ap.parse_args()
    if not args.data and not args.recapture_dir:
        ap.error("cần --data (kiểm thử độ bền) hoặc --recapture-dir (chụp lại ảnh in)")

    engine = load_engine()
    model = load_model(args)
    rows = run_recapture(args, model, engine) if args.recapture_dir else run_dataset(args, model, engine)
    if not rows:
        raise SystemExit("Không có ảnh nào có đáp án để đánh giá")
    summary = summarize(rows)
    blur = suggest_blur_threshold(rows) if not args.recapture_dir else {}
    title = {"chup_lai": "Kiểm thử chụp lại ảnh in / màn hình", "anh_mau": "Kết quả trên bộ ảnh mẫu"}.get(
        args.condition_name, args.condition_name) if args.recapture_dir else "Kiểm thử độ bền (điều kiện chụp mô phỏng)"
    info = model.info() if hasattr(model, "info") else {}
    report = {"title": title, "model": info.get("file"), "backend": info.get("backend"),
              "mode": args.condition_name if args.recapture_dir else "robustness", "summary": summary, "blur": blur}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    keys = ["image", "sample_id", "condition", "gt_class", "gt_pct", "gt_level", "status", "pr_class", "pr_pct",
            "pr_level", "blur_score", "is_blurry", "infer_ms"]
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    md = to_markdown(summary, blur, title)
    args.out.with_suffix(".md").write_text(md, "utf-8")
    print(md)
    print(f"Đã ghi {args.out.with_suffix('.json')}, .csv, .md")


if __name__ == "__main__":
    main()
