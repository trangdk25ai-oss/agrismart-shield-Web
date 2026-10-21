"""Kiểm thử luồng chẩn đoán dùng chung (infer.analyze) và chế độ thử không mô hình (color_rule)."""

from datetime import date
from pathlib import Path

import cv2
import numpy as np

from agrismart.color_rule import ColorRuleModel
from agrismart.infer import STATUS_BLURRY, analyze, overlay_rgba
from agrismart.knowledge_engine import KnowledgeEngine
from agrismart.seg_runtime import Prediction
from agrismart.severity import MaskInstance

ROOT = Path(__file__).resolve().parents[2]
ENGINE = KnowledgeEngine.from_file(ROOT / "knowledge_base/ipm_rules.json")


def leaf_image(lesion_radius=20, size=400, texture=True):
    rng = np.random.default_rng(1)
    img = np.full((size, size, 3), (70, 95, 125), np.uint8)
    if texture:
        img = (img + rng.normal(0, 12, img.shape)).clip(0, 255).astype(np.uint8)
    leaf = np.zeros((size, size), np.uint8)
    cv2.ellipse(leaf, (size // 2, size // 2), (150, 90), 20, 0, 360, 1, -1)
    img[leaf > 0] = (45, 165, 60)
    if texture:  # gân lá để ảnh có cạnh sắc (không bị coi là mờ)
        for k in range(-140, 141, 20):
            cv2.line(img, (size // 2 + k, size // 2 - 60), (size // 2 + k + 15, size // 2 + 60), (30, 120, 40), 1)
        img[leaf == 0] = (img[leaf == 0].astype(int) + rng.normal(0, 10, (int((leaf == 0).sum()), 3))).clip(0, 255)
    lesion = np.zeros_like(leaf)
    if lesion_radius:
        cv2.circle(lesion, (size // 2 - 40, size // 2), lesion_radius, 1, -1)
        img[lesion > 0] = (35, 70, 120)
    return img, leaf, lesion


def test_color_rule_severity_close_to_truth():
    img, leaf, lesion = leaf_image()
    pred = ColorRuleModel().predict(img, lesion_class="tomato__early_blight")
    leaves = [i for i in pred.instances if i.cls_name == "leaf"]
    spots = [i for i in pred.instances if i.cls_name == "tomato__early_blight"]
    assert len(leaves) == 1 and len(spots) == 1
    truth = 100 * lesion.sum() / leaf.sum()
    est = 100 * spots[0].mask.sum() / (leaves[0].mask | spots[0].mask).sum()
    assert abs(est - truth) < 0.6, (est, truth)


def test_analyze_ok_returns_recommendation_and_timings():
    img, leaf, lesion = leaf_image(lesion_radius=28)
    res, inst = analyze(ColorRuleModel(), img, "tomato", ENGINE, lesion_class="tomato__early_blight",
                        start_date=date(2026, 10, 19))
    assert res["status"] == "OK" and len(inst) == 2
    rec = res["recommendation"]
    assert rec["status"] == "OK" and rec["model_class"] == "tomato__early_blight"
    assert rec["spray_schedule"][0]["date"] == "2026-10-19"
    lf = res["leaves"][0]
    assert lf["primary"] and 0 < lf["severity_pct"] < 20
    assert all(0 <= v <= 1 for v in lf["box"])
    assert {"blur_check", "inference", "severity", "total"} <= set(res["timings_ms"])


def test_analyze_healthy_leaf():
    img, *_ = leaf_image(lesion_radius=0)
    res, _ = analyze(ColorRuleModel(), img, "tomato", ENGINE, lesion_class="tomato__early_blight")
    assert res["status"] == "HEALTHY" and res["recommendation"]["status"] == "HEALTHY"


def test_analyze_blurry_stops_before_model_unless_forced():
    img, *_ = leaf_image()
    blurred = cv2.GaussianBlur(img, (0, 0), 6)

    class Boom:
        backend = "x"

        def predict(self, *a, **k):
            raise AssertionError("không được chạy mô hình khi ảnh mờ")

    res, inst = analyze(Boom(), blurred, "tomato", ENGINE, force=False)
    assert res["status"] == STATUS_BLURRY and res["blur"]["is_blurry"] and inst == []
    res, _ = analyze(ColorRuleModel(), blurred, "tomato", ENGINE, force=True, lesion_class="tomato__early_blight")
    assert res["status"] != STATUS_BLURRY


def test_analyze_no_leaf_gives_retake_message():
    class Empty:
        backend = "x"

        def predict(self, *a, **k):
            return Prediction([])

    img, *_ = leaf_image()
    res, _ = analyze(Empty(), img, "tomato", ENGINE)
    assert res["status"] == "NO_LEAF" and "chụp lại" in res["message"]
    assert "recommendation" not in res


def test_analyze_crop_mismatch_from_model():
    m = np.zeros((100, 100), bool)
    m[10:90, 10:90] = True
    s = np.zeros_like(m)
    s[40:50, 40:50] = True

    class Fake:
        backend = "x"
        names = ["leaf", "citrus__anthracnose"]

        def predict(self, *a, **k):
            return Prediction([MaskInstance("leaf", 0.9, m), MaskInstance("citrus__anthracnose", 0.8, s)])

    res, _ = analyze(Fake(), np.zeros((100, 100, 3), np.uint8), "tomato", ENGINE)
    assert res["recommendation"]["status"] == "CROP_MISMATCH"


def test_overlay_rgba_transparent_outside_masks():
    m = np.zeros((50, 60), bool)
    m[10:40, 10:50] = True
    ov = overlay_rgba(50, 60, [MaskInstance("leaf", 1.0, m)], ["leaf"])
    assert ov.shape == (50, 60, 4) and ov[0, 0, 3] == 0 and ov[25, 30, 3] > 0
