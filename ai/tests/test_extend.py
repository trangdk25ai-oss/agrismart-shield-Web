"""Kiểm thử cơ chế mở rộng loại bệnh: danh mục PlantSeg, sinh class_map, nhập nhãn tự gán, quy tắc CHỜ NHẬP."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from agrismart.annotation_import import ImportStats, decode_rle, read_coco, read_labelme_dir, to_instances
from agrismart.class_registry import (
    ClassRegistryError,
    build_class_map,
    load_plantseg,
    select_plantseg,
    write_annotation_labels,
)
from agrismart.knowledge_engine import STATUS_PENDING_RULE, KnowledgeBaseError, KnowledgeEngine
from agrismart.yolo_io import read_label_file

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "knowledge_base" / "ipm_rules.json"
SCHEMA = ROOT / "knowledge_base" / "ipm_rules.schema.json"


# ------------------------------------------------------------------ danh mục & class_map
def test_plantseg_registry_matches_known_ids():
    e = {x["name"]: x for x in load_plantseg()}
    assert len(e) == 115
    assert (e["tomato early blight"]["id"], e["tomato late blight"]["id"], e["tomato septoria leaf spot"]["id"]) == (98, 99, 102)
    assert e["bell pepper bacterial spot"]["crop"] == "bellpepper"  # mã cây trồng không có gạch dưới
    assert e["grapevine leafroll disease"]["crop"] == "grape"


def test_build_class_map_plantseg_plus_own_labels(tmp_path):
    entries = load_plantseg()
    sel = select_plantseg(entries, ["tomato early blight", "cucumber powdery mildew"], ["citrus"])
    cmap = build_class_map(sel, ["cucumber__downy_mildew"])
    names = list(cmap["target_names"].values())
    assert names[0] == "leaf"
    assert names[1:] == ["citrus__canker", "citrus__greening_disease", "cucumber__powdery_mildew",
                         "tomato__early_blight", "cucumber__downy_mildew"]
    assert cmap["value_to_target"][98] == "tomato__early_blight"
    files = write_annotation_labels(cmap, tmp_path)
    cvat = json.loads(files[0].read_text(encoding="utf-8"))
    assert [c["name"] for c in cvat] == names
    assert files[1].read_text(encoding="utf-8").split() == names


@pytest.mark.parametrize("bad", ["Cucumber__x", "bell_pepper__spot", "cucumber_downy", "cucumber__"])
def test_extra_name_validation(bad):
    with pytest.raises(ClassRegistryError):
        build_class_map([], [bad])


def test_s00_rerun_keeps_manual_folder_mappings(tmp_path):
    """Chạy lại s00 không được xóa folder_to_target khai báo tay (dữ liệu tổng hợp chạy thử dựa vào nó)."""
    out = tmp_path / "class_map.yaml"
    out.write_text(yaml.safe_dump({"folder_to_target": {"Tomato early blight": "tomato__early_blight",
                                                        "Rice blast": "rice__blast"},
                                   "label_aliases": {"suong mai": "cucumber__downy_mildew"}}), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "ai/scripts/s00_build_class_map.py"),
                        "--plantseg-diseases", "tomato early blight", "--extra", "cucumber__downy_mildew",
                        "--out", str(out), "--labels-dir", str(tmp_path / "labels")],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    cm = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert cm["folder_to_target"] == {"Tomato early blight": "tomato__early_blight"}
    assert cm["label_aliases"] == {"suong mai": "cucumber__downy_mildew"}
    assert cm["value_to_target"] == {98: "tomato__early_blight"}
    assert "rice__blast" in r.stdout  # mục bị bỏ được báo ra


def test_unknown_plantseg_name_gives_suggestion():
    with pytest.raises(ClassRegistryError, match="cucumber powdery mildew"):
        select_plantseg(load_plantseg(), ["cucumber downy mildew"])


# ------------------------------------------------------------------ nhập nhãn
def _rle_encode_colmajor(mask):
    flat = mask.T.reshape(-1)  # theo cột như COCO
    counts, cur, run = [], 0, 0
    for v in flat:
        if v == cur:
            run += 1
        else:
            counts.append(run)
            cur, run = v, 1
    counts.append(run)
    return counts


def test_rle_roundtrip():
    m = np.zeros((7, 11), np.uint8)
    m[2:5, 3:9] = 1
    m[6, 0] = 1
    back = decode_rle({"counts": _rle_encode_colmajor(m), "size": [7, 11]}, 7, 11)
    assert np.array_equal(back, m)


def _write_labelme(folder: Path, name: str, shapes, w=200, h=160):
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.imwrite(str(folder / f"{name}.jpg"), img)
    (folder / f"{name}.json").write_text(json.dumps({
        "version": "5.4.1", "shapes": shapes, "imagePath": f"{name}.jpg", "imageData": None,
        "imageHeight": h, "imageWidth": w}), encoding="utf-8")


def _poly(label, x0, y0, x1, y1, shape_type="polygon"):
    return {"label": label, "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "shape_type": shape_type}


def _labelme_set(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    _write_labelme(folder, "benh", [_poly("leaf", 20, 20, 180, 140), _poly("cucumber__downy_mildew", 50, 50, 80, 80),
                                    _poly("suong mai", 100, 60, 120, 90), _poly("leaf", 0, 0, 5, 5, "rectangle")])
    _write_labelme(folder, "khoe", [_poly("leaf", 10, 10, 190, 150)])
    _write_labelme(folder, "la_khac", [_poly("tomato__unknown", 10, 10, 50, 50)])


def test_labelme_reader_aliases_and_unknown(tmp_path):
    _labelme_set(tmp_path / "lm")
    stats = ImportStats()
    items = {i.key: i for i in read_labelme_dir(tmp_path / "lm", stats)}
    assert stats.counts.get("skip_shape_rectangle") == 1
    name_to_id = {"leaf": 0, "cucumber__downy_mildew": 1}
    insts = to_instances(items["benh"], name_to_id, {"suong mai": "cucumber__downy_mildew"}, stats)
    assert sorted(i.cls_id for i in insts) == [0, 1, 1]
    to_instances(items["la_khac"], name_to_id, {}, stats)
    assert stats.counts.get("unknown_label:tomato__unknown") == 1


def test_coco_reader_polygon_and_rle(tmp_path):
    m = np.zeros((40, 60), np.uint8)
    m[10:20, 30:50] = 1
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 60, "height": 40}],
        "categories": [{"id": 1, "name": "leaf"}, {"id": 2, "name": "citrus__anthracnose"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[1, 1, 58, 1, 58, 38, 1, 38]], "iscrowd": 0},
            {"id": 2, "image_id": 1, "category_id": 2, "iscrowd": 1,
             "segmentation": {"counts": _rle_encode_colmajor(m), "size": [40, 60]}},
        ],
    }
    p = tmp_path / "c.json"
    p.write_text(json.dumps(coco), encoding="utf-8")
    stats = ImportStats()
    (img,) = read_coco(p, tmp_path, stats)
    insts = to_instances(img, {"leaf": 0, "citrus__anthracnose": 1}, {}, stats)
    assert sorted(i.cls_id for i in insts) == [0, 1]


def test_s01b_end_to_end(tmp_path):
    _labelme_set(tmp_path / "lm")
    cmap = build_class_map([], ["cucumber__downy_mildew"])
    cmap["label_aliases"] = {"suong mai": "cucumber__downy_mildew"}
    cm = tmp_path / "class_map.yaml"
    cm.write_text(yaml.safe_dump(cmap, allow_unicode=True), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "ai/scripts/s01b_import_annotations.py"), "--format", "labelme",
                        "--labelme-dir", str(tmp_path / "lm"), "--class-map", str(cm), "--out", str(tmp_path / "out"),
                        "--prefix", "dn_"], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr
    report = json.loads((tmp_path / "out/import_report.json").read_text(encoding="utf-8"))
    assert report["images_ok"] == 2 and report["images_healthy"] == 1 and report["images_diseased"] == 1
    assert report["skip_no_valid_label"] == 1  # ảnh chỉ có nhãn lạ -> bỏ qua, có thống kê
    labels = read_label_file(tmp_path / "out/labels/dn_benh.txt")
    assert sorted(i.cls_id for i in labels) == [0, 1, 1]
    assert (tmp_path / "out/images/dn_khoe.jpg").exists()


# ------------------------------------------------------------------ quy tắc CHỜ NHẬP
def test_kb_add_rules_and_pending_recommendation(tmp_path):
    kb_copy = tmp_path / "ipm_rules.json"
    kb_copy.write_text(KB.read_text(encoding="utf-8"), encoding="utf-8")
    cm = tmp_path / "class_map.yaml"
    cm.write_text(yaml.safe_dump(build_class_map(select_plantseg(load_plantseg(), ["cucumber powdery mildew"]),
                                                 ["tomato__early_blight"])), encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "knowledge_base/kb_add_rules.py"), "--class-map", str(cm),
                        "--kb", str(kb_copy), "--schema", str(SCHEMA)], capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    eng = KnowledgeEngine.from_file(kb_copy, SCHEMA)
    assert eng.pending_rules() == ["cucumber__powdery_mildew"]
    rule = eng.rule_for_class("cucumber__powdery_mildew")
    assert rule["crop_name"] == "Dưa chuột" and rule["disease_name"].startswith("Phấn trắng")
    rec = eng.recommend("cucumber", "cucumber__powdery_mildew", 12.0)
    assert rec["status"] == STATUS_PENDING_RULE
    assert rec["level_label"] == "Nhiễm nhẹ" and rec["dosage"] == [] and "chưa được cán bộ" in rec["message"]
    # chạy lại không thêm trùng
    r2 = subprocess.run([sys.executable, str(ROOT / "knowledge_base/kb_add_rules.py"), "--class-map", str(cm),
                         "--kb", str(kb_copy), "--schema", str(SCHEMA)], capture_output=True, text=True, encoding="utf-8")
    assert "không thay đổi" in r2.stdout


def test_active_rule_must_have_actions():
    kb = json.loads(KB.read_text(encoding="utf-8"))
    bad = copy.deepcopy(kb)
    bad["rules"][0]["actions"]["LIGHT"] = []
    with pytest.raises(KnowledgeBaseError):
        KnowledgeEngine(bad)
    bad["rules"][0]["rule_status"] = "PENDING"
    KnowledgeEngine(bad)  # PENDING được phép để trống
