"""Mô hình giả dùng chung cho kiểm thử: tensor đầu ra YOLOv8-seg dựng tay và tệp ONNX hằng số."""

import numpy as np
import pytest

NAMES = ["leaf", "tomato__early_blight"]


def fake_outputs(anchors, nm=1, ph=16, pw=16, proto_value=5.0):
    """anchors: list (cx, cy, w, h, cls, score) theo pixel của input. Trả về det [1, c, N], protos [1, nm, ph, pw]."""
    nc = len(NAMES)
    c = 4 + nc + nm
    det = np.zeros((1, c, len(anchors)), np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(anchors):
        det[0, :4, i] = (cx, cy, w, h)
        det[0, 4 + cls, i] = score
        det[0, 4 + nc:, i] = 1.0
    protos = np.full((1, nm, ph, pw), proto_value, np.float32)
    return det, protos


def write_tiny_onnx(path, det, protos, names):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    nodes = [
        helper.make_node("Constant", [], ["output0"], value=numpy_helper.from_array(det, "c0")),
        helper.make_node("Constant", [], ["output1"], value=numpy_helper.from_array(protos, "c1")),
    ]
    graph = helper.make_graph(
        nodes, "fake", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(det.shape)),
         helper.make_tensor_value_info("output1", TensorProto.FLOAT, list(protos.shape))])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    for k, v in {"names": str(dict(enumerate(names))), "imgsz": "[64, 64]", "task": "segment"}.items():
        m = model.metadata_props.add()
        m.key, m.value = k, v
    onnx.save(model, str(path))
