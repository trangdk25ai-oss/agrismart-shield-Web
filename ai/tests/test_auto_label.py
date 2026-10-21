"""Nhãn nháp tự động (s00c): tách lá/vùng đổi màu, ghi tệp Labelme có cờ chưa rà soát; s01b bỏ qua cho đến khi tích cờ."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from agrismart.auto_label import REVIEW_FLAG, auto_label, is_white_background
from agrismart.yolo_io import read_label_file

ROOT = Path(__file__).resolve().parents[2]
CLASS_MAP = ROOT / "ai/configs/class_map.yaml"


def _leaf_image(white_bg: bool, spots: list[tuple[int, int, int]], size=(480, 360)):
    w, h = size
    img = np.full((h, w, 3), (245, 246, 244) if white_bg else (60, 90, 120), np.uint8)  # BGR: trắng / nền đất
    leaf = np.zeros((h, w), np.uint8)
    cv2.ellipse(leaf, (w // 2, h // 2), (int(w * 0.38), int(h * 0.36)), 20, 0, 360, 1, -1)
    img[leaf > 0] = (40, 150, 50)  # xanh lá
    lesion = np.zeros((h, w), np.uint8)
    for cx, cy, r in spots:
        cv2.circle(lesion, (cx, cy), r, 1, -1)
    lesion &= leaf
    img[lesion > 0] = (60, 120, 170)  # nâu vàng
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    return img, leaf, lesion


def _iou(a, b):
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum())


def test_white_background_leaf_and_lesions():
    img, leaf, lesion = _leaf_image(True, [(200, 150, 18), (300, 200, 12), (360, 170, 15)])
    assert is_white_background(img)
    res = auto_label(img)
    assert res["method"] == "nen_trang"
    assert _iou(res["leaf"], leaf) > 0.95
    assert _iou(res["lesion"], lesion) > 0.7
    true_pct = 100 * lesion.sum() / leaf.sum()
    assert abs(res["severity_pct"] - true_pct) < 1.5


def test_field_background_uses_exg():
    img, leaf, lesion = _leaf_image(False, [(220, 160, 16), (280, 200, 14)])
    assert not is_white_background(img)
    res = auto_label(img)
    assert res["method"] == "exg_otsu"
    assert _iou(res["leaf"], leaf) > 0.9
    assert _iou(res["lesion"], lesion) > 0.6


def _run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_s00c_writes_drafts_and_s01b_waits_for_review(tmp_path):
    d = tmp_path / "can_gan_nhan"
    (d / "citrus__anthracnose").mkdir(parents=True)
    (d / "khac").mkdir()
    sick, _, _ = _leaf_image(True, [(200, 150, 18), (300, 200, 12)])
    healthy, _, _ = _leaf_image(True, [])
    cv2.imwrite(str(d / "citrus__anthracnose" / "a1.jpg"), sick)
    cv2.imwrite(str(d / "citrus__anthracnose" / "a2.jpg"), healthy)
    manual = d / "citrus__anthracnose" / "a3.jpg"
    cv2.imwrite(str(manual), sick)
    manual_json = {"version": "7.9.0", "flags": {}, "imagePath": "a3.jpg", "imageData": None, "imageHeight": 360,
                   "imageWidth": 480, "shapes": [
                       {"label": "leaf", "points": [[60, 60], [420, 60], [420, 300], [60, 300]], "shape_type": "polygon"},
                       {"label": "citrus__anthracnose", "points": [[190, 140], [210, 140], [210, 160]], "shape_type": "polygon"}]}
    manual.with_suffix(".json").write_text(json.dumps(manual_json), encoding="utf-8")

    out = _run("ai/scripts/s00c_auto_label.py", "--dir", d, "--class-map", CLASS_MAP)
    assert '"da_ghi_nhan_nhap": 1' in out and '"khong_thay_vung_doi_mau": 1' in out and '"giu_nguyen_json_da_co": 1' in out
    draft = json.loads((d / "citrus__anthracnose" / "a1.json").read_text(encoding="utf-8"))
    assert draft["flags"] == {REVIEW_FLAG: False}
    assert {s["label"] for s in draft["shapes"]} == {"leaf", "citrus__anthracnose"}
    assert draft["imageHeight"] == 360 and draft["imageWidth"] == 480 and draft["imageData"] is None
    assert not (d / "citrus__anthracnose" / "a2.json").exists()  # không thấy vùng đổi màu -> không ghi
    assert (d / "auto_label_report.csv").exists()

    s01b = ("ai/scripts/s01b_import_annotations.py", "--format", "labelme", "--labelme-dir", d,
            "--class-map", CLASS_MAP, "--prefix", "lbl_")
    out1 = _run(*s01b, "--out", tmp_path / "o1")
    assert '"skip_nhan_nhap_chua_ra_soat": 1' in out1 and '"images_ok": 1' in out1  # chỉ tệp vẽ tay

    draft["flags"][REVIEW_FLAG] = True  # người gán nhãn tích cờ trên Labelme
    (d / "citrus__anthracnose" / "a1.json").write_text(json.dumps(draft), encoding="utf-8")
    out2 = _run(*s01b, "--out", tmp_path / "o2")
    assert '"images_ok": 2' in out2 and "skip_nhan_nhap_chua_ra_soat" not in out2
    insts = read_label_file(tmp_path / "o2" / "labels" / "lbl_citrus__anthracnose__a1.txt")
    assert {i.cls_id for i in insts} == {0, 6}  # leaf + citrus__anthracnose
