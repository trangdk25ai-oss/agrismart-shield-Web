"""inspect_masks phải nhận đúng mask theo mã lớp kiểu PlantSeg (0 + một mã bệnh) – không được báo nhầm là JPG/nhị phân."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "ai/scripts/tools/inspect_masks.py"


def _write_masks(folder: Path, values: list[int], ext: str = ".png", soft: bool = False) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(8):
        m = np.zeros((128, 128), np.uint8)
        v = values[i % len(values)]
        for _ in range(3):
            x, y = (int(t) for t in rng.integers(15, 110, 2))
            cv2.circle(m, (x, y), int(rng.integers(5, 15)), v, -1)
        if soft:
            m = cv2.GaussianBlur(m, (7, 7), 0)
        cv2.imwrite(str(folder / f"m{i}{ext}"), m)


def _run(folder: Path, *extra: str) -> str:
    r = subprocess.run([sys.executable, str(SCRIPT), str(folder), "--n", "50", *extra],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_plantseg_style_value_masks(tmp_path):
    _write_masks(tmp_path / "annotations" / "train", [98, 99, 102, 5])
    cm = tmp_path / "class_map.yaml"
    cm.write_text("value_to_target:\n  98: tomato__early_blight\n  99: tomato__late_blight\n", encoding="utf-8")
    out = _run(tmp_path / "annotations", "--class-map", str(cm))
    assert "MÃ LỚP" in out and "--mask-mode value" in out
    assert "Không dùng --mask-mode value" not in out
    assert "98=tomato__early_blight: 2" in out


@pytest.mark.parametrize("values,ext,soft,expect", [
    ([255], ".png", False, "Mask nhị phân"),
    ([1], ".png", False, "Mask nhị phân"),
    ([255], ".jpg", False, "Mask JPG"),
    ([255], ".png", True, "mức xám"),
])
def test_binary_and_lossy_masks(tmp_path, values, ext, soft, expect):
    _write_masks(tmp_path / "m", values, ext, soft)
    out = _run(tmp_path / "m")
    assert expect in out
    assert "MÃ LỚP" not in out
