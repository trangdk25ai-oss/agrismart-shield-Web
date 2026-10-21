"""Kiểm thử phần tích hợp nguồn dữ liệu bổ sung: danh mục, dHash, chọn ảnh gán nhãn, mask JPG, VIA, đánh giá hộp bao."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from agrismart.annotation_import import ImportStats, read_via, to_instances
from agrismart.bbox_eval import HEALTHY, BoxEvalAccumulator, label_in_box, read_box_csv, scale_box
from agrismart.image_hash import NearDuplicateIndex, dhash, hamming_to_many
from agrismart.io_utils import imread, imwrite
from agrismart.severity import MaskInstance
from agrismart.sources import CatalogError, filter_sources, load_catalog

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(*args):
    r = subprocess.run([PY, *map(str, args)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


# ------------------------------------------------------------------ danh mục nguồn
def test_catalog_loads_and_filters():
    cat = load_catalog()
    assert len(cat) == 19
    assert {"plantseg", "plantdoc", "riceseg_5932", "cucumber_recognition"} <= set(cat)
    gap = {s.id for s in filter_sources(cat, gap="citrus__anthracnose")}
    assert gap == {"bdlemonleaf", "lemon_large_bd"}
    assert {s.id for s in filter_sources(cat, level="A")} == {"plantseg", "riceseg_5932"}
    assert "CHƯA" not in cat["plantseg"].citation() and "cần kiểm tra" in cat["bdlemonleaf"].citation()
    assert cat["plantdoc"].citation().startswith("PlantDoc") and "doi.org/10.1145/3371158.3371196" in cat["plantdoc"].citation()


def test_catalog_rejects_bad_entries(tmp_path):
    data = yaml.safe_load((ROOT / "data_sources/catalog.yaml").read_text(encoding="utf-8"))
    data["sources"][0]["level"] = "Z"
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    with pytest.raises(CatalogError):
        load_catalog(p)


# ------------------------------------------------------------------ đọc/ghi ảnh đường dẫn có dấu
def test_unicode_path_roundtrip(tmp_path):
    p = tmp_path / "Thắng" / "lá cà chua.png"
    img = np.full((20, 30, 3), 77, np.uint8)
    assert imwrite(p, img)
    back = imread(p)
    assert back is not None and back.shape == (20, 30, 3) and int(back[0, 0, 0]) == 77


# ------------------------------------------------------------------ dHash
def _textured(seed, size=200):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur((rng.random((size, size, 3)) * 255).astype(np.uint8), (0, 0), 6)


def test_dhash_detects_resized_recompressed_copy():
    a = _textured(1)
    b = cv2.resize(a, (137, 137))
    ok, buf = cv2.imencode(".jpg", b, [cv2.IMWRITE_JPEG_QUALITY, 60])
    b = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    c = _textured(2)
    ha, hb, hc = dhash(a), dhash(b), dhash(c)
    arr = np.array([hb, hc], dtype=np.uint64)
    d = hamming_to_many(ha, arr)
    assert d[0] <= 5 < d[1]
    idx = NearDuplicateIndex(5)
    for i in range(1500):  # vượt kích thước bộ đệm ban đầu
        idx.add(f"x{i}", (i * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
    idx.add("a", ha)
    assert idx.find(hb)[0] == "a" and len(idx) == 1501


# ------------------------------------------------------------------ s00b chọn ảnh để gán nhãn
def test_pick_for_labeling_end_to_end(tmp_path):
    src = tmp_path / "cucumber" / "Original Images"
    for folder, seeds in {"Downy Mildew": range(0, 6), "Anthracnose": range(10, 14), "Belly Rot": range(20, 22)}.items():
        for s in seeds:
            imwrite(src / folder / f"img{s}.jpg", _textured(s, 300))
    imwrite(src / "Downy Mildew" / "img0_aug1.jpg", _textured(0, 300))  # ảnh tăng cường -> bỏ
    imwrite(src / "Downy Mildew" / "copy_of_1.jpg", cv2.resize(_textured(1, 300), (280, 280)))  # gần trùng -> bỏ
    imwrite(src / "Anthracnose" / "tiny.jpg", _textured(99, 100))  # quá nhỏ -> bỏ
    existing = tmp_path / "train"
    imwrite(existing / "da_co.jpg", _textured(11, 300))  # đã có trong tập huấn luyện -> bỏ img11
    out = tmp_path / "can_gan_nhan"
    log = run(ROOT / "ai/scripts/s00b_pick_for_labeling.py", "--source-id", "cucumber_recognition", "--src", src,
              "--map", "Downy Mildew=cucumber__downy_mildew", "--map", "Anthracnose=cucumber__anthracnose",
              "--per-class", "4", "--dedup-against", existing, "--out", out)
    assert "CẢNH BÁO: giấy phép" in log
    dm = sorted(p.name for p in (out / "cucumber__downy_mildew").iterdir())
    an = sorted(p.name for p in (out / "cucumber__anthracnose").iterdir())
    assert len(dm) == 4 and not any("aug" in n for n in dm)
    assert len(an) == 3 and not any("img11" in n or "tiny" in n for n in an)
    assert not any("copy_of_1" in n and "img1." in " ".join(dm) for n in dm)
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert len(rows) == 7 and {r["doi"] for r in rows} == {"10.17632/y6d3z6f8z9.1"}
    # chạy lại: không chép trùng
    run(ROOT / "ai/scripts/s00b_pick_for_labeling.py", "--source-id", "cucumber_recognition", "--src", src,
        "--map", "Downy Mildew=cucumber__downy_mildew", "--per-class", "4", "--out", out)
    assert len(list((out / "cucumber__downy_mildew").iterdir())) == 4
    cite = run(ROOT / "ai/scripts/datasets_catalog.py", "cite", "--ids", "plantseg", "--manifest", out / "manifest.csv")
    assert "[1] PlantSeg" in cite and "[2] Cucumber Disease Recognition Dataset" in cite


def test_dedup_command(tmp_path):
    imwrite(tmp_path / "a" / "x.jpg", _textured(5, 256))
    imwrite(tmp_path / "a" / "y.jpg", _textured(6, 256))
    imwrite(tmp_path / "b" / "x_small.jpg", cv2.resize(_textured(5, 256), (128, 128)))
    log = run(ROOT / "ai/scripts/datasets_catalog.py", "dedup", "--a", tmp_path / "a", "--b", tmp_path / "b",
              "--out", tmp_path / "d.csv")
    assert log.startswith("1 ảnh")
    rows = list(csv.DictReader(open(tmp_path / "d.csv", encoding="utf-8")))
    assert Path(rows[0]["image_a"]).name == "x.jpg"


# ------------------------------------------------------------------ mask JPG trong s01
def test_s01_binary_mode_thresholds_jpg_masks(tmp_path):
    img = np.full((200, 200, 3), (40, 170, 50), np.uint8)
    m = np.zeros((200, 200), np.uint8)
    cv2.circle(m, (100, 100), 20, 255, -1)
    imwrite(tmp_path / "img" / "Blast" / "r1.jpg", img)
    imwrite(tmp_path / "mask" / "Blast" / "r1.jpg", m, [cv2.IMWRITE_JPEG_QUALITY, 60])
    leaf = np.full((200, 200), 255, np.uint8)
    imwrite(tmp_path / "leaf" / "Blast" / "r1.png", leaf)
    cmap = {"target_names": {0: "leaf", 1: "rice__blast"}, "folder_to_target": {"Blast": "rice__blast"}}
    (tmp_path / "cm.yaml").write_text(yaml.safe_dump(cmap), encoding="utf-8")
    run(ROOT / "ai/scripts/s01_convert_masks_to_yolo.py", "--images", tmp_path / "img", "--lesion-masks", tmp_path / "mask",
        "--mask-mode", "binary", "--class-map", tmp_path / "cm.yaml", "--leaf-masks", tmp_path / "leaf",
        "--out", tmp_path / "out", "--prefix", "riceseg_")
    from agrismart.yolo_io import polygon_to_mask, read_label_file
    insts = read_label_file(tmp_path / "out/labels/riceseg_Blast__r1.txt")
    lesion = sum(polygon_to_mask(i.polygon, 200, 200).sum() for i in insts if i.cls_id == 1)
    true_area = int((m > 0).sum())
    assert abs(int(lesion) - true_area) / true_area < 0.08  # lấy mọi pixel > 0 sẽ lệch ~ +28%


# ------------------------------------------------------------------ VIA
def _via_fixture(tmp_path, v1=False):
    imwrite(tmp_path / "img" / "c1.jpg", np.zeros((100, 160, 3), np.uint8))
    regions = [
        {"shape_attributes": {"name": "polygon", "all_points_x": [10, 150, 150, 10], "all_points_y": [10, 10, 90, 90]},
         "region_attributes": {"label": "leaf"}},
        {"shape_attributes": {"name": "ellipse", "cx": 50, "cy": 50, "rx": 12, "ry": 8, "theta": 0.3},
         "region_attributes": {"label": "coffee__rust"}},
        {"shape_attributes": {"name": "rect", "x": 1, "y": 1, "width": 5, "height": 5},
         "region_attributes": {"label": "leaf"}},
        {"shape_attributes": {"name": "circle", "cx": 120, "cy": 40, "r": 6},
         "region_attributes": {"label": {"coffee__rust": True, "leaf": False}}},
    ]
    if v1:
        regions = {str(i): r for i, r in enumerate(regions)}
    meta = {"c1.jpg123": {"filename": "c1.jpg", "size": 123, "regions": regions, "file_attributes": {}}}
    return meta


@pytest.mark.parametrize("wrap,v1", [(True, False), (False, False), (False, True)])
def test_via_reader_versions(tmp_path, wrap, v1):
    meta = _via_fixture(tmp_path, v1)
    data = {"_via_settings": {}, "_via_img_metadata": meta} if wrap else meta
    p = tmp_path / "via.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    stats = ImportStats()
    (img,) = read_via(p, tmp_path / "img", stats)
    assert (img.width, img.height) == (160, 100)
    assert stats.counts["skip_shape_rect"] == 1
    insts = to_instances(img, {"leaf": 0, "coffee__rust": 1}, {}, stats)
    assert sorted(i.cls_id for i in insts) == [0, 1, 1]


def test_s01b_via_end_to_end(tmp_path):
    meta = _via_fixture(tmp_path)
    (tmp_path / "via.json").write_text(json.dumps({"_via_img_metadata": meta}), encoding="utf-8")
    cmap = {"target_names": {0: "leaf", 1: "coffee__rust"}, "label_aliases": {}}
    (tmp_path / "cm.yaml").write_text(yaml.safe_dump(cmap), encoding="utf-8")
    run(ROOT / "ai/scripts/s01b_import_annotations.py", "--format", "via", "--via-json", tmp_path / "via.json",
        "--images", tmp_path / "img", "--class-map", tmp_path / "cm.yaml", "--out", tmp_path / "out", "--prefix", "bracot_")
    rep = json.loads((tmp_path / "out/import_report.json").read_text(encoding="utf-8"))
    assert rep["images_ok"] == 1 and rep["instances_coffee__rust"] == 2


# ------------------------------------------------------------------ đánh giá theo hộp bao
def _rect(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_label_in_box():
    H = W = 100
    insts = [MaskInstance("leaf", 0.9, _rect(H, W, 0, 0, 50, 100)),
             MaskInstance("tomato__early_blight", 0.8, _rect(H, W, 10, 10, 20, 20)),   # 100 px trong lá trái
             MaskInstance("tomato__late_blight", 0.8, _rect(H, W, 10, 30, 14, 34)),    # 16 px
             MaskInstance("tomato__late_blight", 0.7, _rect(H, W, 70, 70, 72, 72))]    # 4 px trong hộp phải
    lab, frac = label_in_box(insts, (0, 0, 50, 100))
    assert lab == "tomato__early_blight" and frac == pytest.approx(116 / 5000)
    lab2, frac2 = label_in_box(insts, (55, 55, 100, 100))  # không có lá: chia diện tích hộp
    assert lab2 == HEALTHY and frac2 == pytest.approx(4 / 2025)
    assert label_in_box([], (0, 0, 10, 10))[0] == HEALTHY


def test_box_accumulator_and_csv(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("filename,width,height,class,xmin,ymin,xmax,ymax\n"
                 "a.jpg,200,100,Tomato Early blight leaf,10,10,50,50\n"
                 "a.jpg,200,100,Tomato leaf,60,10,90,50\n"
                 "b.jpg,200,100,Apple leaf,1,1,5,5\n", encoding="utf-8")
    boxes = read_box_csv(p, {"Tomato Early blight leaf": "tomato__early_blight", "Tomato leaf": "healthy"})
    assert list(boxes) == ["a.jpg"] and len(boxes["a.jpg"]) == 2
    assert scale_box((10, 10, 50, 50), (200, 100), (400, 300)) == (20, 30, 100, 150)
    acc = BoxEvalAccumulator()
    acc.update("tomato__early_blight", "tomato__early_blight")
    acc.update("tomato__early_blight", HEALTHY)
    acc.update(HEALTHY, "tomato__late_blight")
    acc.update(HEALTHY, HEALTHY)
    r = acc.result()
    assert r["accuracy"] == 0.5 and r["healthy_false_alarm_rate"] == 0.5
    assert r["per_class"]["tomato__early_blight"]["recall"] == 0.5


# ------------------------------------------------------------------ Labelme 6/7: mask AI, circle, đường dẫn Windows
def test_labelme_mask_circle_and_windows_path(tmp_path):
    import base64

    from agrismart.annotation_import import read_labelme_dir

    (tmp_path / "anh").mkdir()
    imwrite(tmp_path / "anh" / "la1.jpg", np.zeros((80, 120, 3), np.uint8))
    patch = np.zeros((10, 20), np.uint8)
    patch[2:8, 5:15] = 255  # 6 x 10 = 60 px
    b64 = base64.b64encode(cv2.imencode(".png", patch)[1].tobytes()).decode()
    doc = {"shapes": [
        {"label": "leaf", "shape_type": "polygon", "points": [[1, 1], [118, 1], [118, 78], [1, 78]]},
        {"label": "cucumber__downy_mildew", "shape_type": "mask", "points": [[30, 40], [49, 49]], "mask": b64},
        {"label": "cucumber__downy_mildew", "shape_type": "circle", "points": [[90, 20], [96, 20]]},
        {"label": "leaf", "shape_type": "rectangle", "points": [[0, 0], [5, 5]]},
    ], "imagePath": "..\\anh\\la1.jpg", "imageData": None, "imageHeight": 80, "imageWidth": 120}
    (tmp_path / "nhan").mkdir()
    (tmp_path / "nhan" / "la1.json").write_text(json.dumps(doc), encoding="utf-8")
    stats = ImportStats()
    (img,) = read_labelme_dir(tmp_path / "nhan", stats)
    assert img.image_path is not None and img.image_path.resolve() == (tmp_path / "anh" / "la1.jpg").resolve()
    assert stats.counts["skip_shape_rectangle"] == 1
    m = next(s.mask for s in img.shapes if s.mask is not None)
    assert int(m.sum()) == 60 and m[42, 35] == 1 and m[41, 35] == 0
    insts = to_instances(img, {"leaf": 0, "cucumber__downy_mildew": 1}, {}, stats)
    assert sorted(i.cls_id for i in insts) == [0, 1, 1]
