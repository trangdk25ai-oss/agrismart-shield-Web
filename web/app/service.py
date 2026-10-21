"""Lớp nghiệp vụ của máy chủ: nạp mô hình + tri thức MỘT lần, phục vụ chẩn đoán / khuyến nghị / lịch.

Không có logic nông học trong tệp này: Severity Index (agrismart.severity), luồng chẩn đoán (agrismart.infer)
và Knowledge Engine (agrismart.knowledge_engine) được dùng nguyên vẹn – cùng mã với các script đánh giá.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date

import numpy as np

from agrismart import LEAF_CLASS, __version__
from agrismart.color_rule import ColorRuleModel
from agrismart.infer import analyze, class_color, overlay_rgba
from agrismart.knowledge_engine import KnowledgeEngine
from agrismart.seg_runtime import load_model

from .calendar_ics import build_ics
from .config import Settings
from .imaging import bgr_hex, to_jpeg_b64, to_png_b64
from .samples import SampleStore

log = logging.getLogger("agrismart.web")
MODE_MODEL = "model"
MODE_COLOR_RULE = "color_rule"


class ServiceError(ValueError):
    pass


class DiagnosisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = KnowledgeEngine.from_file(settings.kb_path, settings.kb_schema)
        self.samples = SampleStore(settings.samples_dir)
        self.lock = threading.Lock()  # mô hình dùng chung giữa các yêu cầu: chạy lần lượt cho an toàn
        self.model_error: str | None = None
        self.model = None
        if settings.model_path:
            try:
                self.model = load_model(settings.model_path, threads=settings.threads)
            except Exception as e:  # không dừng máy chủ: báo lỗi rõ trên giao diện và chuyển chế độ thử
                self.model_error = f"Không nạp được mô hình {settings.model_path}: {e}"
                log.error(self.model_error)
        self.mode = MODE_MODEL if self.model else MODE_COLOR_RULE
        if self.model is None:
            self.model = ColorRuleModel()

    # ------------------------------------------------------------------ thông tin
    def label_of(self, cls: str) -> str:
        if cls == LEAF_CLASS:
            return "Phiến lá"
        rule = self.engine.rule_for_class(cls)
        return f"{rule['disease_name']} ({rule['crop_name'].lower()})" if rule else cls

    def model_classes(self) -> list[str]:
        if self.mode == MODE_MODEL:
            return [n for n in self.model.names if n != LEAF_CLASS]
        return [r["model_class"] for r in self.engine.kb["rules"]]

    def meta(self) -> dict:
        classes = self.model_classes()
        crops = []
        for code, name in self.engine.crops():
            cl = [{"class": r["model_class"], "disease_name": r["disease_name"], "rule_status": r.get("rule_status", "ACTIVE")}
                  for r in self.engine.kb["rules"] if r["crop_code"] == code and r["model_class"] in classes]
            if cl:
                crops.append({"code": code, "name": name, "classes": cl})
        kb = self.engine.kb
        return {
            "version": __version__,
            "mode": self.mode,
            "model": self.model.info(),
            "model_error": self.model_error,
            "classes_without_rule": [c for c in classes if self.engine.rule_for_class(c) is None],
            "kb": {"version": kb["kb_version"], "review_status": kb.get("review_status"), "updated_at": kb.get("updated_at")},
            "crops": crops,
            "tanks": kb["tank_presets"],
            "blur_threshold": self.settings.blur_threshold,
            "conf": self.settings.conf,
            "max_upload_mb": self.settings.max_upload_mb,
            "samples": len(self.samples),
            "lan_urls": self.settings.lan_urls,
        }

    # ------------------------------------------------------------------ chẩn đoán
    def diagnose(self, image: np.ndarray, crop_code: str, *, tank_code: str = "TANK_16L", start_date: date | None = None,
                 force: bool = False, lesion_class: str | None = None, sample_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        if self.mode == MODE_COLOR_RULE:
            rule = self.engine.rule_for_class(lesion_class or "")
            if rule is None:
                raise ServiceError("Chế độ thử (chưa có mô hình): hãy chọn loại bệnh giả định.")
            crop_code = rule["crop_code"]
        if crop_code not in dict(self.engine.crops()):
            raise ServiceError(f"Cây trồng không hợp lệ: {crop_code}")
        if tank_code not in self.engine.tanks:
            raise ServiceError(f"Loại bình không hợp lệ: {tank_code}")
        with self.lock:
            res, instances = analyze(
                self.model, image, crop_code, self.engine, tank_code=tank_code, start_date=start_date,
                conf=self.settings.conf, blur_threshold=self.settings.blur_threshold, force=force,
                lesion_class=lesion_class)
        h, w = image.shape[:2]
        names = self.model.names if self.mode == MODE_MODEL else [LEAF_CLASS, lesion_class]
        t1 = time.perf_counter()
        present = sorted({i.cls_name for i in instances}, key=lambda c: (c != LEAF_CLASS, c))
        res.update({
            "mode": self.mode,
            "crop_code": crop_code,
            "image_jpeg_b64": to_jpeg_b64(image),
            "overlay_png_b64": to_png_b64(overlay_rgba(h, w, instances, names)) if instances else None,
            "legend": [{"class": c, "label": self.label_of(c), "color": bgr_hex(class_color(c, names))} for c in present],
        })
        for d in res["detections"]:
            d["label"] = self.label_of(d["class"])
        for lf in res["leaves"]:
            lf["dominant_label"] = self.label_of(lf["dominant_class"]) if lf["dominant_class"] else None
        res["timings_ms"]["encode"] = round((time.perf_counter() - t1) * 1000, 1)
        res["timings_ms"]["server_total"] = round((time.perf_counter() - t0) * 1000, 1)
        if sample_id:
            res["ground_truth"] = self.ground_truth(sample_id, res)
        return res

    def ground_truth(self, sample_id: str, res: dict) -> dict | None:
        it = self.samples.get(sample_id)
        if not it:
            return None
        rec = res.get("recommendation") or {}
        pred_pct = rec.get("severity_pct")
        gt = {k: it.get(k) for k in ("gt_class", "gt_disease_name", "gt_severity_pct", "gt_level_code", "gt_level_label",
                                      "source_id", "license", "origin")}
        gt["abs_error_pct"] = round(abs(pred_pct - it["gt_severity_pct"]), 2) if pred_pct is not None and it.get("gt_severity_pct") is not None else None
        gt["same_level"] = (rec.get("level_code") == it.get("gt_level_code")) if rec else None
        gt["same_class"] = (rec.get("model_class") == it.get("gt_class")) if rec.get("model_class") else None
        return gt

    def recommend(self, crop_code: str, model_class: str | None, severity_pct: float, has_lesion: bool,
                  tank_code: str, start_date: date | None, detection_conf: float | None = None) -> dict:
        if tank_code not in self.engine.tanks:
            raise ServiceError(f"Loại bình không hợp lệ: {tank_code}")
        if not 0 <= severity_pct <= 100:
            raise ServiceError("Tỷ lệ tổn thương phải trong khoảng 0–100%")
        return self.engine.recommend(crop_code, model_class, severity_pct, has_lesion=has_lesion, tank_code=tank_code,
                                     start_date=start_date, detection_conf=detection_conf)

    def calendar(self, **kw) -> str:
        rec = self.recommend(**kw)
        if not rec.get("spray_schedule"):
            raise ServiceError("Không có lịch phun cho kết quả này (lá khỏe, mức khởi phát hoặc phác đồ chưa nhập).")
        return build_ics(rec)
