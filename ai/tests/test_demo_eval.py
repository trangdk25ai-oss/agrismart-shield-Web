"""Kiểm thử phương án thay thực địa (Bước 6): biến đổi ảnh, tổng hợp số liệu, và chuỗi script s10 -> s09 -> s11."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from agrismart.color_rule import ColorRuleModel
from agrismart.robustness import CONDITIONS, suggest_blur_threshold, summarize
from agrismart.severity import compute_severity
from test_pipeline import leaf_image

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def test_conditions_keep_image_valid_and_rotation_keeps_ratio():
    img, *_ = leaf_image(lesion_radius=28)
    for code, (label, fn) in CONDITIONS.items():
        out = fn(img)
        assert out.dtype == np.uint8 and out.ndim == 3 and label
        if code != "nghieng":
            assert out.shape == img.shape, code
    model = ColorRuleModel()
    sev = lambda im: compute_severity(model.predict(im, lesion_class="tomato__early_blight").instances).primary.severity_pct  # noqa: E731
    assert abs(sev(CONDITIONS["nghieng"][1](img)) - sev(img)) < 1.0


def test_summarize_and_blur_threshold():
    rows = []
    for cond, blur, pr in [("goc", 300, 10.0), ("goc", 250, 6.0), ("mo_nang", 20, 4.0), ("rung_tay", 40, None)]:
        rows.append({"condition": cond, "gt_pct": 8.0, "gt_level": "LIGHT", "gt_class": "tomato__early_blight",
                     "pr_pct": pr, "pr_level": None if pr is None else ("LIGHT" if pr >= 5 else "ONSET"),
                     "pr_class": None if pr is None else "tomato__early_blight", "blur_score": blur,
                     "is_blurry": blur < 100, "infer_ms": 12.0})
    s = summarize(rows)
    assert s["goc"]["mae"] == pytest.approx(2.0) and s["goc"]["level_accuracy"] == 1.0
    assert s["mo_nang"]["mae_increase"] == pytest.approx(2.0)
    assert s["rung_tay"]["result_rate"] == 0.0 and s["rung_tay"]["level_accuracy"] == 0.0
    b = suggest_blur_threshold(rows)
    assert 40 < b["suggested"] < 250 and b["detection_rate"] == 1.0 and b["false_alarm_rate"] == 0.0


def run(*args, cwd=ROOT):
    r = subprocess.run([PY, *map(str, args)], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                       env={**__import__("os").environ, "PYTHONUTF8": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    t = tmp_path_factory.mktemp("ds")
    run("ai/scripts/make_synthetic_dataset.py", "--out", t / "raw", "--n", "30")
    run("ai/scripts/s01_convert_masks_to_yolo.py", "--images", t / "raw/images", "--lesion-masks", t / "raw/masks",
        "--mask-mode", "binary", "--class-map", "ai/configs/class_map.yaml", "--bootstrap-leaf", "--out", t / "all",
        "--prefix", "syn_")
    run("ai/scripts/s02_split_dataset.py", "--src", t / "all", "--out", t / "ds", "--class-map", "ai/configs/class_map.yaml",
        "--group-regex", r"^.*?__(plant\d+)_")
    return t


def test_demo_chain_s10_s09_s11(dataset, tmp_path):
    data = dataset / "ds/dataset.yaml"
    samples = tmp_path / "samples"
    out = run("ai/scripts/s10_demo_pack.py", "--data", data, "--split", "test", "--n", "4", "--out", samples,
              "--source-map", "syn_=plantseg")
    m = json.loads((samples / "manifest.json").read_text(encoding="utf-8"))
    assert 1 <= len(m["items"]) <= 4 and "Đã chọn" in out
    it = m["items"][0]
    assert (samples / it["file"]).is_file() and it["source_id"] == "plantseg" and it["license"]
    assert it["gt_level_code"] in ("HEALTHY", "ONSET", "LIGHT", "HEAVY")
    assert "s01" in (samples / "to_in_A4.html").read_text(encoding="utf-8")

    rep = tmp_path / "reports"
    run("ai/scripts/s09_robustness.py", "--color-rule", "--data", data, "--split", "test", "--limit", "3",
        "--conditions", "goc", "mo_nang", "nghieng", "--out", rep / "robustness")
    r = json.loads((rep / "robustness.json").read_text(encoding="utf-8"))
    assert list(r["summary"]) == ["goc", "mo_nang", "nghieng"]
    assert (rep / "robustness.md").read_text(encoding="utf-8").startswith("## Kiểm thử độ bền")

    # chụp lại: giả lập bằng cách nén JPEG ảnh mẫu, đặt tên <id>_<lần>.jpg
    rc = tmp_path / "chup_lai"
    rc.mkdir()
    img = cv2.imread(str(samples / it["file"]))
    cv2.imwrite(str(rc / f"{it['id']}_1.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 40])
    cv2.imwrite(str(rc / "la_khong_ro.jpg"), img)
    out = run("ai/scripts/s09_robustness.py", "--color-rule", "--recapture-dir", rc, "--manifest", samples / "manifest.json",
              "--out", rep / "recapture")
    assert "Bỏ qua 1 ảnh" in out
    run("ai/scripts/s09_robustness.py", "--color-rule", "--recapture-dir", samples, "--manifest", samples / "manifest.json",
        "--condition-name", "anh_mau", "--out", rep / "samples")
    rows = list(csv.DictReader((rep / "samples.csv").open(encoding="utf-8-sig")))
    assert {r_["sample_id"] for r_ in rows} == {i["id"] for i in m["items"]}

    expert = tmp_path / "chuyen_gia.csv"
    expert.write_text("sample_id,chuyen_gia_benh,chuyen_gia_severity_pct,chuyen_gia_muc_do,nguoi_danh_gia,ghi_chu\n"
                      f"{it['id']},,\"{it['gt_severity_pct'] + 1:.1f}\",Nhiễm nhẹ,Cán bộ A,\n", encoding="utf-8")
    run("ai/scripts/s11_test_report.py", "--reports", rep, "--expert", expert, "--out", rep / "bao_cao.md")
    text = (rep / "bao_cao.md").read_text(encoding="utf-8")
    assert "Chỉ tiêu nghiệm thu" in text and "Chưa có số liệu" in text  # chưa có eval/latency -> không suy đoán
    assert "Chụp lại ảnh in" in text and "So với chuyên gia" in text
