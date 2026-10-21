"""Kiểm thử độ bền thay cho thực địa (Bước 6): mô phỏng điều kiện chụp ngoài ruộng trên ảnh tập kiểm tra.

Không đi thực địa được thì không có ảnh "khó" thật; nhưng các yếu tố làm kết quả ngoài ruộng kém đi
(rung tay, lệch nét, thiếu/thừa sáng, chụp xa, nén ảnh, nghiêng máy, nhiễu cảm biến) mô phỏng được bằng
phép biến đổi ảnh. Vì các phép này KHÔNG làm đổi tỷ lệ diện tích vết bệnh/lá, đáp án Severity tính từ
nhãn gốc vẫn dùng được cho ảnh đã biến đổi -> đo được mức suy giảm của từng điều kiện.

Giới hạn (phải nêu trong hồ sơ): mô phỏng không thay thế được khác biệt về giống cây, giai đoạn bệnh,
nền ruộng, lá ướt, nhiều lá chồng nhau… của ảnh thật tại Đà Nẵng.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .blur import DEFAULT_BLUR_THRESHOLD, laplacian_variance
from .severity import compute_severity


def _gamma(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _contrast(img: np.ndarray, k: float) -> np.ndarray:
    mean = img.reshape(-1, 3).mean(0)
    return np.clip((img.astype(np.float32) - mean) * k + mean, 0, 255).astype(np.uint8)


def _motion(img: np.ndarray, size: int) -> np.ndarray:
    k = np.zeros((size, size), np.float32)
    k[size // 2, :] = 1.0 / size
    return cv2.filter2D(img, -1, k)


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _far(img: np.ndarray, s: float) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """Xoay và nới khung để không cắt mất phần lá (tỷ lệ vết bệnh/lá giữ nguyên)."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    c, s = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(128, 128, 128))


def _noise(img: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


# mã điều kiện -> (mô tả tiếng Việt, hàm biến đổi)
CONDITIONS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "goc": ("Ảnh gốc", lambda im: im),
    "mo_nhe": ("Lệch nét nhẹ (Gauss σ = 1,5)", lambda im: cv2.GaussianBlur(im, (0, 0), 1.5)),
    "mo_nang": ("Lệch nét nặng (Gauss σ = 3)", lambda im: cv2.GaussianBlur(im, (0, 0), 3.0)),
    "rung_tay": ("Rung tay (nhòe chuyển động 9 px)", lambda im: _motion(im, 9)),
    "thieu_sang": ("Thiếu sáng / bóng râm (độ sáng × 0,55)", lambda im: _gamma(im, 0.55)),
    "thua_sang": ("Nắng gắt (độ sáng × 1,5)", lambda im: _gamma(im, 1.5)),
    "tuong_phan_thap": ("Tương phản thấp (× 0,6)", lambda im: _contrast(im, 0.6)),
    "chup_xa": ("Chụp xa / ảnh nhỏ (thu 40% rồi phóng lại)", lambda im: _far(im, 0.4)),
    "nen_jpeg": ("Nén JPEG mạnh (chất lượng 30)", lambda im: _jpeg(im, 30)),
    "nghieng": ("Nghiêng máy 20°", lambda im: _rotate(im, 20)),
    "nhieu": ("Nhiễu cảm biến (σ = 12)", lambda im: _noise(im, 12)),
}
BLUR_CONDITIONS = ("mo_nang", "rung_tay")
OTHER_LABELS = {"chup_lai": "Chụp lại ảnh in / màn hình bằng điện thoại", "anh_mau": "Bộ ảnh mẫu (ảnh gốc)"}


@dataclass
class Truth:
    pct: float
    level: str
    cls: str | None  # lớp bệnh chiếm diện tích lớn nhất trên lá chính; None = lá khỏe


def evaluate_one(model, engine, image: np.ndarray, truth: Truth, condition: str, *, conf: float = 0.25,
                 blur_threshold: float = DEFAULT_BLUR_THRESHOLD, lesion_class: str | None = None) -> dict:
    """Chạy mô hình trên MỘT ảnh (đã biến đổi) và so với đáp án."""
    blur = laplacian_variance(image)
    if getattr(model, "backend", "") == "color-rule":
        pred = model.predict(image, lesion_class=lesion_class or truth.cls or "unknown__lesion")
    else:
        pred = model.predict(image, conf=conf)
    rep = compute_severity(pred.instances)
    row = {"condition": condition, "gt_pct": round(truth.pct, 2), "gt_level": truth.level, "gt_class": truth.cls,
           "status": rep.status, "blur_score": round(blur, 1), "is_blurry": blur < blur_threshold,
           "infer_ms": round(pred.timings_ms.get("inference", float("nan")), 1)}
    if rep.primary is None or rep.status not in ("OK", "HEALTHY"):
        row.update(pr_pct=None, pr_level=None, pr_class=None)
        return row
    p = rep.primary
    cls = p.dominant_class
    rule = engine.rule_for_class(cls or truth.cls or "")
    row.update(pr_pct=round(p.severity_pct, 2), pr_class=cls,
               pr_level=engine.classify(p.severity_pct, p.lesion_area_px > 0, rule))
    return row


def summarize(rows: list[dict]) -> dict:
    """Gộp theo điều kiện: tỷ lệ ra kết quả, MAE/RMSE, đúng mức, đúng bệnh, tỷ lệ bị cảnh báo mờ."""
    out: dict[str, dict] = {}
    for cond in dict.fromkeys(r["condition"] for r in rows):
        rs = [r for r in rows if r["condition"] == cond]
        done = [r for r in rs if r["pr_pct"] is not None]
        err = [r["pr_pct"] - r["gt_pct"] for r in done]
        sick = [r for r in done if r["gt_class"]]
        times = [r["infer_ms"] for r in rs if r["infer_ms"] == r["infer_ms"]]  # bỏ NaN
        out[cond] = {
            "label": CONDITIONS[cond][0] if cond in CONDITIONS else OTHER_LABELS.get(cond, cond),
            "n": len(rs),
            "result_rate": len(done) / len(rs) if rs else 0.0,
            "mae": float(np.mean(np.abs(err))) if err else None,
            "rmse": float(math.sqrt(np.mean(np.square(err)))) if err else None,
            "level_accuracy": sum(r["pr_level"] == r["gt_level"] for r in rs) / len(rs) if rs else 0.0,
            "class_accuracy": (sum(r["pr_class"] == r["gt_class"] for r in sick) / len(sick)) if sick else None,
            "blur_flag_rate": sum(r["is_blurry"] for r in rs) / len(rs) if rs else 0.0,
            "infer_ms_mean": float(np.mean(times)) if times else None,
        }
    base = out.get("goc", {}).get("mae")
    for v in out.values():
        v["mae_increase"] = (v["mae"] - base) if (base is not None and v["mae"] is not None) else None
    return out


def suggest_blur_threshold(rows: list[dict]) -> dict:
    """Gợi ý ngưỡng Blur Check: giữa phân bố điểm của ảnh gốc và ảnh làm mờ nặng."""
    sharp = np.array([r["blur_score"] for r in rows if r["condition"] == "goc"])
    blurred = np.array([r["blur_score"] for r in rows if r["condition"] in BLUR_CONDITIONS])
    if sharp.size == 0 or blurred.size == 0:
        return {}
    lo, hi = float(np.percentile(blurred, 90)), float(np.percentile(sharp, 10))
    res = {"sharp_p10": round(hi, 1), "blurred_p90": round(lo, 1), "sharp_median": round(float(np.median(sharp)), 1),
           "blurred_median": round(float(np.median(blurred)), 1)}
    if lo < hi:
        t = round((lo + hi) / 2, 1)
        res.update(suggested=t, false_alarm_rate=float((sharp < t).mean()), detection_rate=float((blurred < t).mean()))
    else:
        res["note"] = "Hai phân bố chồng lấn: không có ngưỡng tách sạch, nên giữ ngưỡng thấp để tránh báo nhầm ảnh nét"
    return res


def _fmt(v, pct=False, nd=2):
    if v is None:
        return "–"
    s = f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}"
    return s.replace(".", ",")


def to_markdown(summary: dict, blur: dict | None = None, title: str = "Kiểm thử độ bền") -> str:
    lines = [f"## {title}", "",
             "| Điều kiện | n | Ra kết quả | MAE (điểm %) | Tăng MAE | Đúng mức | Đúng bệnh | Bị cảnh báo mờ | Suy luận (ms) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in summary.values():
        lines.append(f"| {v['label']} | {v['n']} | {_fmt(v['result_rate'], True)} | {_fmt(v['mae'])} | "
                     f"{_fmt(v['mae_increase'])} | {_fmt(v['level_accuracy'], True)} | {_fmt(v['class_accuracy'], True)} | "
                     f"{_fmt(v['blur_flag_rate'], True)} | {_fmt(v['infer_ms_mean'], nd=0)} |")
    if blur:
        lines += ["", f"Điểm nét (Blur) – ảnh gốc: trung vị {_fmt(blur.get('sharp_median'), nd=1)}, phân vị 10% "
                      f"{_fmt(blur.get('sharp_p10'), nd=1)}; ảnh làm mờ: trung vị {_fmt(blur.get('blurred_median'), nd=1)}, "
                      f"phân vị 90% {_fmt(blur.get('blurred_p90'), nd=1)}."]
        if "suggested" in blur:
            lines.append(f"Ngưỡng gợi ý: **{_fmt(blur['suggested'], nd=1)}** (báo nhầm ảnh nét "
                         f"{_fmt(blur['false_alarm_rate'], True)}, phát hiện ảnh mờ {_fmt(blur['detection_rate'], True)}).")
        elif "note" in blur:
            lines.append(blur["note"] + ".")
    return "\n".join(lines) + "\n"
