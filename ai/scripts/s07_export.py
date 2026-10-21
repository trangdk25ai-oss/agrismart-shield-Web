"""BƯỚC 3.1 – Lượng tử hóa INT8 và xuất mô hình cho máy chủ trang web, kèm model_meta.json.

Câu lệnh trong quy trình gốc:
    yolo export model=runs/segment/train/weights/best.pt format=tflite int8=True data=dataset.yaml
Bước 5 nay là TRANG WEB (máy chủ Python trên laptop), nên định dạng chính đổi sang ONNX INT8:
    - ultralytics 8.4.176 hỗ trợ format=onnx, quantize=8: lượng tử hóa tĩnh bằng ONNX Runtime, hiệu chỉnh
      (calibration) trên ảnh của dataset.yaml; chỉ lượng tử hóa Conv/Gemm/MatMul, phần giải mã đầu ra giữ float.
    - onnxruntime chạy được trên Windows, Linux, macOS (LiteRT/TFLite chỉ chạy trên Linux x86 và macOS).
Đường LiteRT INT8 cũ vẫn giữ (--format litert-int8) cho trường hợp làm ứng dụng di động sau này
(ở bản mới format=tflite -> litert, int8=True -> quantize=8).

    python ai/scripts/s07_export.py --weights runs/segment/agrismart_v8n/weights/best.pt \
        --data datasets/agrismart/dataset.yaml --copy-to web/models

Kết quả: model_int8.onnx (+ model_meta.json) – máy chủ web tự nạp tệp này khi đặt trong web/models/.
Lưu ý tốc độ: trên CPU, bản INT8 dạng QDQ không chắc nhanh hơn FP32 (tùy CPU). Xuất thêm --format onnx-fp32 và
đo cả hai bằng s08_benchmark.py latency trên máy chạy Demo trước khi chọn.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MAX_SIZE_MB = 10.0


def inspect_onnx(path: Path) -> dict:
    import onnxruntime as ort

    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    d = lambda t: {"name": t.name, "shape": list(t.shape), "dtype": t.type}  # noqa: E731
    meta = s.get_modelmeta().custom_metadata_map
    return {"inputs": [d(t) for t in s.get_inputs()], "outputs": [d(t) for t in s.get_outputs()],
            "input_layout": "NCHW", "has_names_metadata": "names" in meta}


def ensure_onnx_metadata(path: Path, names: dict[int, str], imgsz: int) -> None:
    """Bảo đảm tệp ONNX tự chứa danh sách lớp (một số phiên bản bước lượng tử hóa làm rơi metadata)."""
    import onnx

    m = onnx.load(str(path))
    keys = {p.key for p in m.metadata_props}
    changed = False
    for k, v in {"names": str(names), "imgsz": str([imgsz, imgsz]), "task": "segment"}.items():
        if k not in keys:
            p = m.metadata_props.add()
            p.key, p.value = k, v
            changed = True
    if changed:
        onnx.save(m, str(path))


def inspect_tflite(path: Path) -> dict:
    """Đọc shape/dtype/quantization của input/output để bộ chạy không phải 'đoán'."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # môi trường cũ
        from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
    itp = Interpreter(model_path=str(path))
    itp.allocate_tensors()

    def desc(d):
        scale, zp = d.get("quantization", (0.0, 0))
        return {"name": d["name"], "shape": [int(x) for x in d["shape"]], "dtype": d["dtype"].__name__,
                "quant_scale": float(scale), "quant_zero_point": int(zp)}

    inp = [desc(d) for d in itp.get_input_details()]
    out = [desc(d) for d in itp.get_output_details()]
    layout = "NHWC" if inp[0]["shape"][-1] == 3 else "NCHW"
    return {"inputs": inp, "outputs": out, "input_layout": layout}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--data", required=True, help="dataset.yaml – ảnh dùng để hiệu chỉnh (calibration) INT8")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--fraction", type=float, default=1.0, help="tỷ lệ ảnh dùng hiệu chỉnh INT8")
    ap.add_argument("--format", choices=["onnx-int8", "onnx-fp32", "litert-int8"], default="onnx-int8")
    ap.add_argument("--copy-to", type=Path, default=None, help="thư mục mô hình của máy chủ web (web/models)")
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT

    ver = ultralytics.__version__
    model = YOLO(str(args.weights))
    names = {int(k): v for k, v in model.names.items()}
    new_api = "quantize" in DEFAULT_CFG_DICT  # nhận biết theo TÍNH NĂNG của bản đang cài, không theo số phiên bản

    if args.format.startswith("onnx"):
        if args.format == "onnx-int8" and not new_api:
            raise SystemExit(f"ultralytics {ver} chưa hỗ trợ xuất ONNX INT8; cài bản trong ai/requirements.txt")
        kw = {"quantize": 8, "data": args.data, "fraction": args.fraction} if args.format == "onnx-int8" else {}
        print(f"ultralytics {ver}: export format=onnx {kw or '(FP32)'}")
        exported = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=False, **kw))
        final = exported.with_name("model_int8.onnx" if args.format == "onnx-int8" else "model_fp32.onnx")
        shutil.copy2(exported, final)
        ensure_onnx_metadata(final, names, args.imgsz)
        io = inspect_onnx(final)
        boxes_normalized = False
    else:
        from ultralytics.engine.exporter import export_formats

        fmt = "litert" if "litert" in list(export_formats()["Argument"]) else "tflite"
        quant = {"quantize": 8} if new_api else {"int8": True}
        print(f"ultralytics {ver}: export format={fmt}, {quant}")
        exported = Path(model.export(format=fmt, data=args.data, imgsz=args.imgsz, fraction=args.fraction, **quant))
        if exported.is_dir():  # một số phiên bản cũ trả về thư mục *_saved_model
            cands = sorted(exported.glob("*int8*.tflite"))
            if not cands:
                raise SystemExit(f"Không tìm thấy file *int8*.tflite trong {exported}")
            exported = cands[0]
        final = exported.with_name("model_quant_int8.tflite")
        shutil.copy2(exported, final)
        io = inspect_tflite(final)
        boxes_normalized = True

    size_mb = final.stat().st_size / 1024 / 1024
    meta = {"model_file": final.name, "format": args.format, "ultralytics_version": ver, "task": "segment",
            "imgsz": args.imgsz, "names": names, "leaf_class": "leaf", "boxes_normalized": boxes_normalized,
            "size_mb": round(size_mb, 2), **io}
    meta_path = final.with_name("model_meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    status = "ĐẠT" if size_mb < MAX_SIZE_MB else "KHÔNG ĐẠT"
    print(f"[{status}] {final} = {size_mb:.2f} MB (mục tiêu < {MAX_SIZE_MB:g} MB)")
    if args.format.startswith("onnx"):
        print("Đo tốc độ trên máy chạy Demo: python ai/scripts/s08_benchmark.py latency --model", final)
    print(json.dumps(io, ensure_ascii=False, indent=2))

    if args.copy_to:
        args.copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, args.copy_to / final.name)
        shutil.copy2(meta_path, args.copy_to / meta_path.name)
        print(f"Đã chép mô hình và model_meta.json vào {args.copy_to}")


if __name__ == "__main__":
    main()
