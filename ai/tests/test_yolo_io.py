import cv2
import numpy as np
import pytest

from agrismart.leaf_bootstrap import bootstrap_leaf_mask, fill_holes
from agrismart.yolo_io import (
    LabelFormatError,
    SegInstance,
    mask_to_polygons,
    masks_to_instances,
    normalize_polygon,
    parse_label_line,
    polygon_to_mask,
    read_label_file,
    write_label_file,
)


def iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / (a | b).sum()


def test_mask_polygon_roundtrip():
    m = np.zeros((200, 300), np.uint8)
    cv2.ellipse(m, (150, 100), (90, 50), 30, 0, 360, 1, -1)
    polys = mask_to_polygons(m)
    assert len(polys) == 1
    back = polygon_to_mask(normalize_polygon(polys[0], 300, 200), 200, 300)
    assert iou(m, back) > 0.97


def test_components_become_instances_and_small_noise_dropped():
    m = np.zeros((100, 100), np.uint8)
    m[10:30, 10:30] = 1
    m[60:80, 60:80] = 1
    m[95:97, 95:97] = 1  # 4 px nhiễu
    insts = masks_to_instances({3: m}, min_area_px=16)
    assert len(insts) == 2 and all(i.cls_id == 3 for i in insts)


def test_label_write_read(tmp_path):
    inst = SegInstance(1, np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.4]], np.float32))
    p = tmp_path / "a.txt"
    write_label_file(p, [inst])
    back = read_label_file(p)
    assert back[0].cls_id == 1 and np.allclose(back[0].polygon, inst.polygon, atol=1e-6)


@pytest.mark.parametrize("line", ["0 0.1 0.2 0.3", "a 0.1 0.1 0.2 0.2 0.3 0.3", "0 0.1 0.1 0.2 0.2 0.3 1.5",
                                  "0 0.1 0.1 0.2 0.2 0.3 0.3 0.4"])
def test_bad_label_lines(line):
    with pytest.raises(LabelFormatError):
        parse_label_line(line)


def test_fill_holes():
    m = np.zeros((50, 50), np.uint8)
    m[10:40, 10:40] = 1
    m[20:30, 20:30] = 0
    assert fill_holes(m)[25, 25] == 1 and fill_holes(m)[0, 0] == 0


def test_bootstrap_leaf_covers_lesion():
    img = np.full((200, 200, 3), (60, 90, 120), np.uint8)
    leaf = np.zeros((200, 200), np.uint8)
    cv2.ellipse(leaf, (100, 100), (70, 40), 0, 0, 360, 1, -1)
    img[leaf > 0] = (40, 170, 50)
    lesion = np.zeros_like(leaf)
    cv2.circle(lesion, (100, 100), 10, 1, -1)
    img[lesion > 0] = (30, 60, 110)  # vết bệnh nâu – không "xanh"
    mask, review = bootstrap_leaf_mask(img, lesion)
    assert iou(mask, leaf) > 0.95
    assert mask[100, 100] == 1  # lá bao trùm vết bệnh
    assert not review
