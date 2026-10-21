"""Kiểm thử bộ chạy mô hình đã xuất (seg_runtime) bằng "mô hình giả": tensor đầu ra dựng tay,
không cần tệp mô hình thật. Đối chiếu với ultralytics trên mô hình thật: xem mục Xác minh của tài liệu.
"""

import numpy as np
import pytest

from agrismart.seg_runtime import decode_yolo_seg, letterbox, letterbox_info, load_model
from fake_models import NAMES, fake_outputs, write_tiny_onnx


def test_letterbox_matches_ultralytics_rounding():
    lb = letterbox_info(320, 240, 640, 640)
    assert (lb.new_w, lb.new_h, lb.left, lb.top) == (640, 480, 0, 80)
    lb = letterbox_info(1001, 700, 640, 640)  # phần đệm lẻ: round(d - 0.1)
    assert lb.new_w == 640 and lb.new_h == round(700 * 640 / 1001)
    assert lb.top == round((640 - lb.new_h) / 2 - 0.1)
    img = np.zeros((240, 320, 3), np.uint8)
    rgb, _ = letterbox(img, 640, 640)
    assert rgb.shape == (640, 640, 3) and rgb.dtype == np.float32
    assert np.isclose(rgb[0, 0, 0], 114 / 255) and np.isclose(rgb[320, 320, 0], 0.0)


def test_nms_conf_and_mask_on_square_input():
    # 3 anchor: a0 giữ lại; a1 trùng a0 (bị NMS); a2 dưới ngưỡng tin cậy
    det, protos = fake_outputs([(32, 32, 32, 32, 0, 0.9), (33, 32, 32, 32, 0, 0.8), (10, 10, 8, 8, 1, 0.1)])
    lb = letterbox_info(64, 64, 64, 64)
    out = decode_yolo_seg(det, protos, NAMES, lb, conf=0.25)
    assert len(out) == 1
    o = out[0]
    assert o.cls_name == "leaf" and o.conf == pytest.approx(0.9)
    assert o.box == pytest.approx((16, 16, 48, 48))
    assert int(o.mask.sum()) == 32 * 32  # prototype dương khắp nơi -> mask = đúng hộp bao


def test_letterboxed_boxes_and_masks_map_back_to_source():
    # ảnh gốc 128x64 -> input 64x64: gain 0,5, nội dung 64x32, đệm trên 16
    lb = letterbox_info(128, 64, 64, 64)
    assert (lb.gain, lb.new_w, lb.new_h, lb.top) == (0.5, 64, 32, 16)
    det, protos = fake_outputs([(32, 32, 32, 16, 1, 0.7)])
    out = decode_yolo_seg(det, protos, NAMES, lb)
    o = out[0]
    assert o.mask.shape == (64, 128)
    assert o.box == pytest.approx((32, 16, 96, 48))
    assert int(o.mask.sum()) == 64 * 32


def test_layouts_and_normalized_boxes_give_same_result():
    lb = letterbox_info(64, 64, 64, 64)
    det, protos = fake_outputs([(32, 32, 32, 32, 0, 0.9), (20, 40, 10, 10, 1, 0.6)], nm=2)
    ref = decode_yolo_seg(det, protos, NAMES, lb)
    variants = [
        (det.transpose(0, 2, 1), protos, None),  # det chuyển vị [1, N, c]
        (det, protos.transpose(0, 2, 3, 1), None),  # protos NHWC
    ]
    det_n = det.copy()
    det_n[0, :4] /= 64.0  # tọa độ chuẩn hóa như bản xuất LiteRT
    variants.append((det_n, protos, None))
    variants.append((det_n, protos, True))
    for d, p, norm in variants:
        out = decode_yolo_seg(d, p, NAMES, lb, boxes_normalized=norm)
        assert [o.cls_name for o in out] == [o.cls_name for o in ref]
        for a, b in zip(out, ref):
            assert a.box == pytest.approx(b.box)
            assert np.array_equal(a.mask, b.mask)


def test_negative_prototype_gives_empty_mask_and_wrong_channels_raise():
    lb = letterbox_info(64, 64, 64, 64)
    det, protos = fake_outputs([(32, 32, 32, 32, 0, 0.9)], proto_value=-3.0)
    out = decode_yolo_seg(det, protos, NAMES, lb)
    assert len(out) == 1 and int(out[0].mask.sum()) == 0
    with pytest.raises(ValueError, match="không khớp"):
        decode_yolo_seg(det, protos, NAMES + ["extra"], lb)


def test_load_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "khong_co.onnx")
    p = tmp_path / "m.bin"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="Chưa hỗ trợ"):
        load_model(p)


def test_onnx_model_reads_metadata_and_predicts(tmp_path):
    pytest.importorskip("onnxruntime")
    det, protos = fake_outputs([(32, 32, 32, 32, 0, 0.9), (40, 40, 8, 8, 1, 0.8)])
    path = tmp_path / "fake.onnx"
    write_tiny_onnx(path, det, protos, NAMES)
    model = load_model(path)
    assert model.backend == "onnxruntime" and model.names == NAMES
    assert (model.input_w, model.input_h, model.nchw) == (64, 64, True)
    pred = model.predict(np.zeros((64, 64, 3), np.uint8))
    assert [i.cls_name for i in pred.instances] == ["leaf", "tomato__early_blight"]
    assert set(pred.timings_ms) == {"preprocess", "inference", "postprocess"}


def test_onnx_without_leaf_class_is_rejected(tmp_path):
    pytest.importorskip("onnxruntime")
    det, protos = fake_outputs([(32, 32, 32, 32, 0, 0.9)])
    path = tmp_path / "noleaf.onnx"
    write_tiny_onnx(path, det, protos, ["la", "benh"])
    with pytest.raises(ValueError, match="thiếu lớp 'leaf'"):
        load_model(path)
