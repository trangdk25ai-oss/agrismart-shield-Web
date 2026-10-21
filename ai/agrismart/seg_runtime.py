"""Chạy mô hình YOLOv8-seg ĐÃ XUẤT (ONNX, LiteRT) mà không cần PyTorch/ultralytics.

Dùng cho máy chủ trang web (Bước 5) và script kiểm thử độ bền (Bước 6). Cài đặt máy chủ chỉ cần
numpy + opencv + onnxruntime (~ vài chục MB) thay vì PyTorch (hàng trăm MB đến vài GB).

Tiền xử lý và hậu xử lý bám theo ultralytics 8.4.176 để kết quả khớp `model.predict(retina_masks=True)`:
    - LetterBox: co giữ tỷ lệ, đệm 114, phần đệm chia đôi với round(d - 0.1) / round(d + 0.1)
    - non_max_suppression: lọc conf > ngưỡng (lấy lớp có điểm cao nhất), NMS theo từng lớp, IoU 0.7
    - process_mask_native: logit = hệ số · prototype -> phóng về kích thước input -> bỏ phần đệm
      -> phóng về kích thước ảnh gốc -> ngưỡng logit > 0 (tức sigmoid > 0,5) -> cắt theo hộp bao
Đã đối chiếu với ultralytics trên mô hình thật (xem ai/tests/test_seg_runtime.py và mục Xác minh).

Ba loại mô hình:
    *.onnx    – onnxruntime (Windows/Linux/macOS). Bản INT8 sinh bởi s07: model_int8.onnx
    *.tflite  – ai-edge-litert (chỉ Linux/macOS). Đầu vào NCHW hoặc NHWC, float hoặc int8 đều đọc được
    *.pt      – ultralytics (cần PyTorch); dùng khi chưa xuất mô hình
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import LEAF_CLASS
from .severity import MaskInstance

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.7  # mặc định khi predict của ultralytics
MAX_DET = 300
MAX_NMS = 30000


# ----------------------------------------------------------------------------------------- tiền xử lý
@dataclass(frozen=True)
class LetterboxInfo:
    src_w: int
    src_h: int
    input_w: int
    input_h: int
    gain: float
    new_w: int
    new_h: int
    left: int
    top: int


def letterbox_info(src_w: int, src_h: int, input_w: int, input_h: int) -> LetterboxInfo:
    gain = min(input_h / src_h, input_w / src_w)
    new_w, new_h = round(src_w * gain), round(src_h * gain)
    dw, dh = (input_w - new_w) / 2, (input_h - new_h) / 2
    return LetterboxInfo(src_w, src_h, input_w, input_h, gain, new_w, new_h, round(dw - 0.1), round(dh - 0.1))


def letterbox(image_bgr: np.ndarray, input_w: int, input_h: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Trả về (ảnh RGB float32 [0,1] kích thước input_h x input_w x 3, thông số letterbox)."""
    h, w = image_bgr.shape[:2]
    lb = letterbox_info(w, h, input_w, input_h)
    img = image_bgr
    if (w, h) != (lb.new_w, lb.new_h):
        img = cv2.resize(image_bgr, (lb.new_w, lb.new_h), interpolation=cv2.INTER_LINEAR)
    bottom = input_h - lb.new_h - lb.top
    right = input_w - lb.new_w - lb.left
    img = cv2.copyMakeBorder(img, lb.top, bottom, lb.left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, lb


# ----------------------------------------------------------------------------------------- hậu xử lý
def _nms_per_class(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float, max_det: int) -> list[int]:
    """NMS tham lam theo từng lớp; boxes đã sắp theo điểm giảm dần."""
    keep: list[int] = []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        rest = np.arange(i + 1, len(boxes))
        rest = rest[(classes[rest] == classes[i]) & ~suppressed[rest]]
        if rest.size == 0:
            continue
        iw = (np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest])).clip(0)
        ih = (np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest])).clip(0)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        suppressed[rest[iou > iou_thr]] = True
    return keep


def decode_yolo_seg(
    det: np.ndarray,
    protos: np.ndarray,
    names: list[str],
    lb: LetterboxInfo,
    conf: float = DEFAULT_CONF,
    iou: float = DEFAULT_IOU,
    boxes_normalized: bool | None = None,
    max_det: int = MAX_DET,
) -> list[MaskInstance]:
    """Giải mã 2 đầu ra YOLOv8-seg thành các MaskInstance trên khung ảnh gốc.

    det:    [1, 4 + nc + nm, N] hoặc chuyển vị [1, N, 4 + nc + nm]
    protos: [1, nm, Hp, Wp] (NCHW) hoặc [1, Hp, Wp, nm] (NHWC)
    boxes_normalized: None = tự nhận (giá trị tọa độ ≤ 1,5 => đã chuẩn hóa [0,1], như bản xuất LiteRT)
    """
    nc = len(names)
    det = np.asarray(det, dtype=np.float32)
    protos = np.asarray(protos, dtype=np.float32)
    if protos.ndim != 4 or det.ndim != 3:
        raise ValueError(f"Cần det 3 chiều và protos 4 chiều, nhận {det.shape} và {protos.shape}")
    p = protos[0]
    if p.shape[0] > p.shape[2]:  # NHWC -> NCHW (số kênh nm=32 nhỏ hơn kích thước lưới)
        p = p.transpose(2, 0, 1)
    nm, mh, mw = p.shape
    c = 4 + nc + nm
    d = det[0]
    if d.shape[0] == c:
        d = d.T  # -> [N, c]
    elif d.shape[1] != c:
        raise ValueError(f"Số kênh {det.shape} không khớp 4 + {nc} lớp + {nm} hệ số mask. "
                         "Kiểm tra danh sách lớp (names) có đúng với mô hình không.")

    cls_scores = d[:, 4:4 + nc]
    best = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(d)), best]
    keep = score > conf
    if not keep.any():
        return []
    d, best, score = d[keep], best[keep], score[keep]
    order = np.argsort(-score, kind="stable")[:MAX_NMS]
    d, best, score = d[order], best[order], score[order]

    xywh = d[:, :4].copy()
    if boxes_normalized is None:
        boxes_normalized = float(np.abs(xywh[:, :2]).max()) <= 1.5
    if boxes_normalized:
        xywh *= np.array([lb.input_w, lb.input_h, lb.input_w, lb.input_h], dtype=np.float32)
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    idx = _nms_per_class(boxes, score, best, iou, max_det)
    boxes, best, score, coefs = boxes[idx], best[idx], score[idx], d[idx, 4 + nc:]

    # hộp về tọa độ ảnh gốc (scale_boxes của ultralytics)
    gx, gy = lb.new_w / lb.src_w, lb.new_h / lb.src_h
    src_boxes = boxes.copy()
    src_boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - lb.left) / gx).clip(0, lb.src_w)
    src_boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - lb.top) / gy).clip(0, lb.src_h)

    # mask: logit trên lưới prototype -> kích thước input -> bỏ đệm -> kích thước ảnh gốc
    logits = (coefs @ p.reshape(nm, -1)).reshape(-1, mh, mw)
    out: list[MaskInstance] = []
    xs = np.arange(lb.src_w)
    ys = np.arange(lb.src_h)
    for k in range(len(idx)):
        up = cv2.resize(logits[k], (lb.input_w, lb.input_h), interpolation=cv2.INTER_LINEAR)
        up = up[lb.top:lb.top + lb.new_h, lb.left:lb.left + lb.new_w]
        full = cv2.resize(up, (lb.src_w, lb.src_h), interpolation=cv2.INTER_LINEAR) > 0.0
        x1, y1, x2, y2 = src_boxes[k]
        full &= ((xs >= x1) & (xs < x2))[None, :] & ((ys >= y1) & (ys < y2))[:, None]
        out.append(MaskInstance(names[int(best[k])], float(score[k]), full,
                                (float(x1), float(y1), float(x2), float(y2))))
    return out


def dequantize(arr: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if arr.dtype == np.float32 or not scale:
        return arr.astype(np.float32)
    return (arr.astype(np.float32) - zero_point) * scale


# ----------------------------------------------------------------------------------------- mô hình
def _parse_names(raw) -> list[str]:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    if isinstance(raw, dict):
        ids = sorted(int(k) for k in raw)
        if ids != list(range(len(ids))):
            raise ValueError(f"names phải liên tục 0..n-1, nhận {ids}")
        return [str(raw[i] if i in raw else raw[str(i)]) for i in ids]
    return [str(x) for x in raw]


@dataclass
class Prediction:
    instances: list[MaskInstance]
    timings_ms: dict[str, float] = field(default_factory=dict)


class SegModel:
    """Giao diện chung: predict(ảnh BGR) -> Prediction (mask ở kích thước ảnh gốc)."""

    backend = "?"
    names: list[str]
    path: Path

    def predict(self, image_bgr: np.ndarray, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU) -> Prediction:
        raise NotImplementedError

    def info(self) -> dict:
        return {"file": self.path.name, "backend": self.backend, "names": self.names,
                "size_mb": round(self.path.stat().st_size / 1024 / 1024, 2) if self.path.is_file() else None}


class _TensorModel(SegModel):
    input_w = input_h = 640
    nchw = True
    boxes_normalized: bool | None = None

    def _run(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(self, image_bgr: np.ndarray, conf: float = DEFAULT_CONF, iou: float = DEFAULT_IOU) -> Prediction:
        t0 = time.perf_counter()
        rgb, lb = letterbox(image_bgr, self.input_w, self.input_h)
        x = rgb.transpose(2, 0, 1)[None] if self.nchw else rgb[None]
        x = np.ascontiguousarray(x)
        t1 = time.perf_counter()
        det, protos = self._run(x)
        t2 = time.perf_counter()
        inst = decode_yolo_seg(det, protos, self.names, lb, conf, iou, self.boxes_normalized)
        t3 = time.perf_counter()
        return Prediction(inst, {"preprocess": (t1 - t0) * 1000, "inference": (t2 - t1) * 1000,
                                 "postprocess": (t3 - t2) * 1000})


def _split_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    det = next((o for o in outputs if o.ndim == 3), None)
    protos = next((o for o in outputs if o.ndim == 4), None)
    if det is None or protos is None:
        raise ValueError(f"Mô hình phân đoạn cần 2 đầu ra (3 chiều và 4 chiều), nhận {[o.shape for o in outputs]}. "
                         "Có thể đã xuất nhầm mô hình detect.")
    return det, protos


class OnnxSegModel(_TensorModel):
    backend = "onnxruntime"

    def __init__(self, path: Path, names: list[str] | None = None, threads: int | None = None):
        import onnxruntime as ort

        self.path = Path(path)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        self.names = names or (_parse_names(meta["names"]) if "names" in meta else None)
        if not self.names:
            raise ValueError(f"{self.path.name} không có metadata 'names'; cần model_meta.json đi kèm")
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        shape = [s if isinstance(s, int) else None for s in inp.shape]
        if len(shape) != 4:
            raise ValueError(f"Đầu vào phải 4 chiều, nhận {inp.shape}")
        self.nchw = shape[1] == 3
        hw = shape[2:4] if self.nchw else shape[1:3]
        if None in hw:  # mô hình xuất với dynamic=True
            imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
            hw = imgsz if isinstance(imgsz, (list, tuple)) else [imgsz, imgsz]
        self.input_h, self.input_w = int(hw[0]), int(hw[1])
        self.boxes_normalized = False  # ONNX của ultralytics trả tọa độ pixel theo kích thước input

    def _run(self, x):
        return _split_outputs(self.session.run(None, {self.input_name: x}))


class LiteRTSegModel(_TensorModel):
    backend = "ai-edge-litert"

    def __init__(self, path: Path, names: list[str], boxes_normalized: bool | None = True, threads: int | None = None):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # môi trường cũ
            from tensorflow.lite.python.interpreter import Interpreter  # type: ignore
        self.path = Path(path)
        self.names = names
        self.itp = Interpreter(model_path=str(self.path), num_threads=threads)
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()[0]
        shape = list(self.inp["shape"])
        self.nchw = shape[1] == 3
        self.input_h, self.input_w = (shape[2], shape[3]) if self.nchw else (shape[1], shape[2])
        self.boxes_normalized = boxes_normalized

    def _run(self, x):
        d = self.inp
        if d["dtype"] != np.float32:
            scale, zp = d["quantization"]
            info = np.iinfo(d["dtype"])
            x = np.clip(np.round(x / scale + zp), info.min, info.max).astype(d["dtype"])
        self.itp.set_tensor(d["index"], x)
        self.itp.invoke()
        outs = []
        for o in self.itp.get_output_details():
            scale, zp = o["quantization"]
            outs.append(dequantize(self.itp.get_tensor(o["index"]), scale, zp))
        return _split_outputs(outs)


class UltralyticsSegModel(SegModel):
    backend = "ultralytics"

    def __init__(self, path: Path, imgsz: int = 640):
        from ultralytics import YOLO

        self.path = Path(path)
        self.model = YOLO(str(self.path), task="segment")
        self.names = [self.model.names[i] for i in sorted(self.model.names)]
        self.imgsz = imgsz

    def predict(self, image_bgr, conf=DEFAULT_CONF, iou=DEFAULT_IOU):
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, iou=iou, retina_masks=True, verbose=False)[0]
        inst: list[MaskInstance] = []
        if r.masks is not None:
            for c, p, m, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                  r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                inst.append(MaskInstance(self.names[int(c)], float(p), m > 0.5, tuple(float(v) for v in b)))
        return Prediction(inst, {k: float(v) for k, v in r.speed.items()})


def load_model(path: str | Path, meta_path: str | Path | None = None, threads: int | None = None) -> SegModel:
    """Chọn bộ chạy theo đuôi tệp. model_meta.json (do s07 sinh) cung cấp names khi tệp không tự chứa."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không thấy mô hình {path}")
    meta = {}
    mp = Path(meta_path) if meta_path else path.with_name("model_meta.json")
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    names = _parse_names(meta["names"]) if "names" in meta else None
    suffix = path.suffix.lower()
    if suffix == ".onnx":
        model: SegModel = OnnxSegModel(path, names, threads)
    elif suffix == ".tflite":
        if not names:
            raise ValueError("Mô hình .tflite cần model_meta.json (do s07 sinh) để biết tên lớp")
        model = LiteRTSegModel(path, names, meta.get("boxes_normalized", True), threads)
    elif suffix == ".pt":
        model = UltralyticsSegModel(path, int(meta.get("imgsz", 640)))
    else:
        raise ValueError(f"Chưa hỗ trợ định dạng {suffix} (dùng .onnx, .tflite hoặc .pt)")
    if LEAF_CLASS not in model.names:
        raise ValueError(f"Mô hình thiếu lớp '{LEAF_CLASS}' – không tính được Severity Index. Lớp: {model.names}")
    return model
