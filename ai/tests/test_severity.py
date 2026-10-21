"""Kiểm thử thuật toán Severity Index – bản cài đặt duy nhất, dùng chung cho máy chủ web và các script đánh giá."""

import numpy as np
import pytest

from agrismart.severity import (
    STATUS_HEALTHY,
    STATUS_LEAF_TOO_SMALL,
    STATUS_NO_LEAF,
    STATUS_NO_LEAF_WITH_LESIONS,
    STATUS_OK,
    MaskInstance,
    compute_severity,
)

H = W = 100


def rect(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def leaf(m, conf=0.9):
    return MaskInstance("leaf", conf, m)


def lesion(m, name="tomato__early_blight", conf=0.8):
    return MaskInstance(name, conf, m)


def test_basic_ratio():
    # lá 80x50 = 4000 px; vết bệnh 10x20 = 200 px -> 5%
    rep = compute_severity([leaf(rect(10, 25, 90, 75)), lesion(rect(20, 30, 30, 50))])
    assert rep.status == STATUS_OK
    assert rep.primary.leaf_area_px == 4000
    assert rep.primary.lesion_area_px == 200
    assert rep.primary.severity_pct == pytest.approx(5.0)
    assert rep.primary.dominant_class == "tomato__early_blight"


def test_overlapping_lesions_not_double_counted():
    a = rect(20, 30, 40, 50)  # 400
    b = rect(30, 40, 50, 60)  # 400, giao a 100
    rep = compute_severity([leaf(rect(10, 25, 90, 75)), lesion(a), lesion(b)])
    assert rep.primary.lesion_area_px == 700  # không phải 800
    assert rep.primary.severity_pct == pytest.approx(700 / 4000 * 100)


def test_leaf_mask_with_holes_gives_same_result():
    full = rect(10, 25, 90, 75)
    les = rect(20, 30, 30, 50)
    holed = full & ~les  # mô phỏng mô hình huấn luyện với overlap_mask=True
    r1 = compute_severity([leaf(full), lesion(les)])
    r2 = compute_severity([leaf(holed), lesion(les)])
    assert r1.primary.severity_pct == pytest.approx(r2.primary.severity_pct)


def test_lesion_spilling_outside_leaf_is_capped():
    lf = rect(10, 10, 50, 50)  # 1600
    les = rect(45, 20, 55, 30)  # 100, một nửa nằm ngoài lá
    rep = compute_severity([leaf(lf), lesion(les)], assoc_min_overlap=0.5, dilate_px=0)
    # mẫu số = |leaf ∪ lesion| = 1600 + 50
    assert rep.primary.leaf_area_px == 1650
    assert rep.primary.severity_pct == pytest.approx(100 / 1650 * 100)
    assert rep.primary.severity_pct <= 100


def test_lesion_assigned_to_correct_leaf_and_primary_by_guide_center():
    left = rect(0, 0, 40, 100)  # 4000, không chứa tâm (50,50)
    center = rect(45, 30, 75, 70)  # 1200, chứa tâm
    les_left = rect(5, 5, 15, 15)  # 100 -> thuộc lá trái
    les_center = rect(50, 40, 56, 50)  # 60 -> thuộc lá giữa
    rep = compute_severity([leaf(left), leaf(center), lesion(les_left), lesion(les_center)])
    assert rep.primary.leaf_area_px == 1200
    assert rep.primary.lesion_area_px == 60
    assert rep.primary.severity_pct == pytest.approx(5.0)
    assert sorted(lv.n_lesions for lv in rep.leaves) == [1, 1]


def test_unassigned_lesion_far_from_leaf():
    rep = compute_severity([leaf(rect(0, 0, 30, 30)), lesion(rect(80, 80, 90, 90))])
    assert rep.unassigned_lesions == 1
    assert rep.status == STATUS_HEALTHY


def test_healthy_and_no_leaf_states():
    assert compute_severity([leaf(rect(10, 10, 90, 90))]).status == STATUS_HEALTHY
    assert compute_severity([lesion(rect(10, 10, 20, 20))]).status == STATUS_NO_LEAF_WITH_LESIONS
    assert compute_severity([]).status == STATUS_NO_LEAF


def test_leaf_too_small():
    rep = compute_severity([leaf(rect(0, 0, 10, 10)), lesion(rect(2, 2, 4, 4))], min_leaf_frac=0.02)
    assert rep.status == STATUS_LEAF_TOO_SMALL


def test_dominant_class_by_area():
    rep = compute_severity([
        leaf(rect(0, 0, 100, 100)),
        lesion(rect(0, 0, 10, 10), "tomato__early_blight"),  # 100
        lesion(rect(50, 50, 70, 70), "tomato__late_blight"),  # 400
    ])
    assert rep.primary.dominant_class == "tomato__late_blight"
    assert rep.primary.lesion_area_by_class == {"tomato__early_blight": 100, "tomato__late_blight": 400}
