"""BƯỚC 3.2 – Benchmark: độ trễ trên máy chạy Demo và mức suy giảm mAP sau lượng tử hóa INT8.

1) Độ trễ TOÀN LUỒNG trên máy tính chạy máy chủ web (tiêu chí < 100 ms cho bước suy luận):
    python ai/scripts/s08_benchmark.py latency --model web/models/model_int8.onnx \
        --images datasets/agrismart/images/test --runs 30 --out reports/latency.json
   Đo bằng đúng bộ chạy của máy chủ web (seg_runtime): tiền xử lý, suy luận, hậu xử lý (giải mã mask)
   trên ảnh thật. Không có --images thì dùng ảnh ngẫu nhiên (chỉ đo suy luận). Ghi lại cấu hình máy
   (CPU, RAM) vào báo cáo – độ trễ phụ thuộc phần cứng.

2) So sánh độ chính xác FP32 vs INT8 (2 báo cáo JSON do s06_evaluate.py sinh ra):
    python ai/scripts/s08_benchmark.py compare --fp32 reports/eval_fp32.json --int8 reports/eval_int8.json \
        --max-drop 2.0 --mode absolute --out reports/compare.json
   Quy trình ghi "mAP giảm không quá 2%" nhưng không nói rõ 2% TUYỆT ĐỐI (điểm phần trăm) hay
   TƯƠNG ĐỐI. Script báo cả hai; tiêu chí đạt/không đạt theo --mode (khuyến nghị nêu rõ trong hồ sơ).

3) (Tùy chọn, nếu sau này làm ứng dụng di động) đo mô hình LiteRT trên điện thoại bằng công cụ
   benchmark_model chính thức qua adb; xem tài liệu LiteRT "Benchmark tools".
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"mean": round(statistics.mean(xs), 2), "p50": round(xs[len(xs) // 2], 2),
            "p90": round(xs[max(0, int(len(xs) * 0.9) - 1)], 2), "max": round(xs[-1], 2)}


def machine_info() -> dict:
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(), "python": platform.python_version()}


def cmd_latency(args) -> None:
    from agrismart.io_utils import imread
    from agrismart.seg_runtime import load_model
    from agrismart.yolo_io import list_images

    model = load_model(args.model, threads=args.threads)
    if args.images:
        imgs = [im for im in (imread(p) for p in list_images(args.images)[: args.max_images]) if im is not None]
    else:
        rng = np.random.default_rng(0)
        imgs = [rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)]
    if not imgs:
        raise SystemExit("Không đọc được ảnh nào")
    for i in range(args.warmup):
        model.predict(imgs[i % len(imgs)])
    rec: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": [], "total": []}
    for i in range(args.runs):
        t = model.predict(imgs[i % len(imgs)]).timings_ms
        for k in ("preprocess", "inference", "postprocess"):
            rec[k].append(t.get(k, 0.0))
        rec["total"].append(sum(t.get(k, 0.0) for k in ("preprocess", "inference", "postprocess")))
    res = {"model": str(args.model), "backend": model.backend, "size_mb": model.info()["size_mb"],
           "threads": args.threads, "runs": args.runs, "images": len(imgs),
           "ms": {k: _stats(v) for k, v in rec.items()}, "machine": machine_info(),
           "criterion": "suy luận (inference) trung bình < 100 ms",
           "pass": _stats(rec["inference"])["mean"] < 100.0}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")


def cmd_compare(args) -> int:
    fp32 = json.loads(Path(args.fp32).read_text(encoding="utf-8"))
    int8 = json.loads(Path(args.int8).read_text(encoding="utf-8"))
    rows = {}
    ok = True
    for key in ("mask_map50_95", "box_map50_95"):
        a, b = fp32[key], int8[key]
        abs_drop = (a - b) * 100  # điểm phần trăm
        rel_drop = (a - b) / a * 100 if a > 0 else 0.0
        drop = abs_drop if args.mode == "absolute" else rel_drop
        passed = drop <= args.max_drop
        ok &= passed if key == "mask_map50_95" else True  # tiêu chí chính: Mask mAP
        rows[key] = {"fp32": a, "int8": b, "drop_abs_points": round(abs_drop, 3),
                     "drop_relative_pct": round(rel_drop, 3), "pass": passed}
    if "severity" in fp32 and "severity" in int8 and fp32["severity"].get("n"):
        rows["severity_mae"] = {"fp32": fp32["severity"]["mae_pct_points"], "int8": int8["severity"]["mae_pct_points"]}
    res = {"criterion": f"{args.mode} drop <= {args.max_drop}", "result": "ĐẠT" if ok else "KHÔNG ĐẠT",
           "pass": ok, "details": rows}
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), "utf-8")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("latency")
    a.add_argument("--model", required=True, type=Path)
    a.add_argument("--images", type=Path, default=None, help="thư mục ảnh thật (khuyến nghị)")
    a.add_argument("--max-images", type=int, default=30)
    a.add_argument("--runs", type=int, default=30)
    a.add_argument("--warmup", type=int, default=3)
    a.add_argument("--threads", type=int, default=None)
    a.add_argument("--out", type=Path, default=None)
    b = sub.add_parser("compare")
    b.add_argument("--fp32", required=True)
    b.add_argument("--int8", required=True)
    b.add_argument("--max-drop", type=float, default=2.0)
    b.add_argument("--mode", choices=["absolute", "relative"], default="absolute")
    b.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.cmd == "latency":
        cmd_latency(args)
    else:
        raise SystemExit(cmd_compare(args))


if __name__ == "__main__":
    main()
