import numpy as np
import pytest

from agrismart.blur import laplacian_variance
from agrismart.metrics import IoUAccumulator, SeverityErrorAccumulator


def test_iou_accumulates_over_dataset():
    acc = IoUAccumulator({0: "leaf", 1: "tomato__early_blight"})
    gt = np.zeros((2, 10, 10), bool)
    pr = np.zeros_like(gt)
    gt[0, :5] = True
    pr[0, :5] = True  # IoU lá = 1
    gt[1, 0, :4] = True
    pr[1, 0, 2:6] = True  # giao 2, hợp 6
    acc.update(gt, pr)
    r = acc.result()
    assert r["per_class_iou"]["leaf"] == pytest.approx(1.0)
    assert r["per_class_iou"]["tomato__early_blight"] == pytest.approx(2 / 6)
    assert r["mean_iou_lesions"] == pytest.approx(2 / 6)


def test_severity_error():
    acc = SeverityErrorAccumulator()
    acc.update(5.0, 7.0, "LIGHT", "LIGHT")
    acc.update(20.0, 14.0, "HEAVY", "LIGHT")
    r = acc.result()
    assert r["mae_pct_points"] == pytest.approx(4.0)
    assert r["level_accuracy"] == pytest.approx(0.5)
    assert r["confusion"]["HEAVY->LIGHT"] == 1


def test_blur_metric_orders_sharp_vs_blurred():
    rng = np.random.default_rng(0)
    sharp = (rng.random((256, 256, 3)) * 255).astype(np.uint8)
    import cv2

    blurred = cv2.GaussianBlur(sharp, (0, 0), 3)
    assert laplacian_variance(sharp) > 10 * laplacian_variance(blurred)
