"""Kiểm thử API máy chủ web bằng TestClient (không cần mở cổng mạng, không cần mô hình thật)."""

import io
import json

import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from fake_models import fake_outputs, write_tiny_onnx  # noqa: E402
from test_pipeline import leaf_image  # noqa: E402
from web.app.config import Settings  # noqa: E402
from web.app.main import create_app  # noqa: E402


def png_bytes(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(Settings(model_path=None, samples_dir=tmp_path / "none")))


def post_image(client, data, **form):
    files = {"image": ("la.png", data, "image/png")}
    return client.post("/api/diagnose", files=files, data=form)


def test_index_meta_and_health_in_test_mode(client):
    r = client.get("/")
    assert r.status_code == 200 and "AgriSmart Shield" in r.text
    assert client.get("/static/app.js").status_code == 200
    m = client.get("/api/meta").json()
    assert m["mode"] == "color_rule" and m["model"]["file"] is None
    assert [c["code"] for c in m["crops"]] == ["tomato", "cucumber", "citrus"]
    assert client.get("/api/health").json()["status"] == "ok"


def test_diagnose_upload_color_mode(client):
    img, *_ = leaf_image(lesion_radius=28)
    r = post_image(client, png_bytes(img), crop_code="tomato", lesion_class="tomato__early_blight",
                   start_date="2026-10-19", tank_code="TANK_2L")
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "OK" and res["mode"] == "color_rule"
    assert res["image_jpeg_b64"] and res["overlay_png_b64"]
    assert [l["label"] for l in res["legend"]] == ["Phiến lá", "Bệnh đốm vòng (cà chua)"]
    rec = res["recommendation"]
    assert rec["status"] == "OK" and rec["spray_schedule"][0]["date"] == "2026-10-19"
    assert all(d["tank_code"] == "TANK_2L" for d in rec["dosage"])
    assert res["timings_ms"]["server_total"] > 0


def test_color_mode_requires_disease_and_rejects_bad_input(client):
    img, *_ = leaf_image()
    r = post_image(client, png_bytes(img), crop_code="tomato")
    assert r.status_code == 400 and "chọn loại bệnh" in r.json()["detail"]
    r = post_image(client, b"khong phai anh", lesion_class="tomato__early_blight")
    assert r.status_code == 400 and "Không đọc được" in r.json()["detail"]
    assert client.post("/api/diagnose", data={"crop_code": "tomato"}).status_code == 400
    r = post_image(client, png_bytes(img), lesion_class="tomato__early_blight", tank_code="TANK_99L")
    assert r.status_code == 400


def test_upload_size_limit(tmp_path):
    c = TestClient(create_app(Settings(model_path=None, max_upload_mb=0.01, samples_dir=tmp_path)))
    img, *_ = leaf_image()
    r = post_image(c, png_bytes(img), lesion_class="tomato__early_blight")
    assert r.status_code == 413


def test_blurry_image_asks_to_retake_unless_forced(client):
    img, *_ = leaf_image()
    blurred = png_bytes(cv2.GaussianBlur(img, (0, 0), 6))
    r = post_image(client, blurred, lesion_class="tomato__early_blight").json()
    assert r["status"] == "BLURRY" and "image_jpeg_b64" in r and r["overlay_png_b64"] is None
    r = post_image(client, blurred, lesion_class="tomato__early_blight", force="true").json()
    assert r["status"] != "BLURRY"


def test_exif_orientation_is_applied(client):
    img, *_ = leaf_image(size=400)
    img = img[:, :300]  # 300 x 400 (rộng x cao)
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    exif = Image.Exif()
    exif[0x0112] = 6  # xoay 90° khi hiển thị
    buf = io.BytesIO()
    pil.save(buf, "JPEG", exif=exif, quality=95)
    r = client.post("/api/diagnose", files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")},
                    data={"lesion_class": "tomato__early_blight", "force": "true"}).json()
    assert r["image_size"] == [400, 300]


def test_recommend_scales_with_tank_and_calendar(client):
    body = {"crop_code": "tomato", "model_class": "tomato__early_blight", "severity_pct": 8.3, "tank_code": "TANK_16L",
            "start_date": "2026-10-19"}
    big = client.post("/api/recommend", json=body).json()
    small = client.post("/api/recommend", json={**body, "tank_code": "TANK_2L"}).json()
    assert big["dosage"][0]["amount"] == pytest.approx(small["dosage"][0]["amount"] * 8, rel=0.05)
    assert client.post("/api/recommend", json={**body, "severity_pct": 120}).status_code == 422
    r = client.get("/api/calendar.ics", params={k: v for k, v in body.items()})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/calendar")
    text = r.content.decode("utf-8")
    assert text.count("BEGIN:VEVENT") == len(big["spray_schedule"]) and "\r\n" in text
    assert "DTSTART;VALUE=DATE:20261019" in text
    assert all(len(line.encode("utf-8")) <= 75 for line in text.split("\r\n"))
    assert "attachment" in r.headers["content-disposition"]
    # bệnh chưa có phác đồ -> không có lịch phun -> 400 kèm lý do
    r = client.get("/api/calendar.ics", params={**body, "model_class": "tomato__unknown"})
    assert r.status_code == 400 and "Không có lịch phun" in r.json()["detail"]


def make_samples(folder):
    folder.mkdir(parents=True, exist_ok=True)
    img, leaf, lesion = leaf_image(lesion_radius=28)
    cv2.imwrite(str(folder / "s01.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    gt = round(100 * lesion.sum() / leaf.sum(), 2)
    manifest = {"items": [
        {"id": "s01", "file": "s01.jpg", "crop_code": "tomato", "gt_class": "tomato__early_blight",
         "gt_disease_name": "Bệnh đốm vòng", "gt_severity_pct": gt, "gt_level_code": "LIGHT", "gt_level_label": "Nhiễm nhẹ",
         "source_id": "synthetic", "license": "–"},
        {"id": "../bad", "file": "s01.jpg"},
        {"id": "s02", "file": "khong_co.jpg"},
    ]}
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
    return gt


def test_samples_gallery_and_ground_truth(tmp_path):
    gt = make_samples(tmp_path / "samples")
    c = TestClient(create_app(Settings(model_path=None, samples_dir=tmp_path / "samples")))
    items = c.get("/api/samples").json()["items"]
    assert [i["id"] for i in items] == ["s01"]  # id lạ và tệp thiếu bị bỏ
    t = c.get("/api/samples/s01/thumb")
    assert t.status_code == 200 and t.headers["content-type"] == "image/jpeg"
    assert c.get("/api/samples/nope/thumb").status_code == 404
    res = c.post("/api/diagnose", data={"sample_id": "s01", "force": "true"}).json()
    g = res["ground_truth"]
    assert g["gt_severity_pct"] == gt and g["abs_error_pct"] < 1.0 and g["same_class"] is True
    assert c.post("/api/diagnose", data={"sample_id": "../bad"}).status_code == 404


def test_model_mode_with_onnx(tmp_path):
    pytest.importorskip("onnxruntime")
    det, protos = fake_outputs([(32, 32, 48, 48, 0, 0.9), (30, 30, 10, 10, 1, 0.8)], proto_value=5.0)
    path = tmp_path / "model_int8.onnx"
    write_tiny_onnx(path, det, protos, ["leaf", "tomato__early_blight"])
    c = TestClient(create_app(Settings(model_path=path, samples_dir=tmp_path)))
    m = c.get("/api/meta").json()
    assert m["mode"] == "model" and m["model"]["backend"] == "onnxruntime"
    assert [cr["code"] for cr in m["crops"]] == ["tomato"]  # chỉ cây có bệnh trong mô hình
    img, *_ = leaf_image(size=64, texture=False)
    r = c.post("/api/diagnose", files={"image": ("a.png", png_bytes(img), "image/png")},
               data={"crop_code": "tomato", "force": "true"}).json()
    assert [d["class"] for d in r["detections"]] == ["leaf", "tomato__early_blight"]
    assert r["status"] == "OK" and r["recommendation"]["model_class"] == "tomato__early_blight"
    assert {"preprocess", "inference", "postprocess"} <= set(r["timings_ms"])


def test_broken_model_falls_back_to_test_mode_with_error(tmp_path):
    bad = tmp_path / "hong.onnx"
    bad.write_bytes(b"not an onnx model")
    m = TestClient(create_app(Settings(model_path=bad, samples_dir=tmp_path))).get("/api/meta").json()
    assert m["mode"] == "color_rule" and "Không nạp được mô hình" in m["model_error"]


def test_ics_folding_handles_long_vietnamese_lines():
    from web.app.calendar_ics import build_ics

    rec = {"rule_id": "R", "disease_name": "Bệnh đốm vòng", "crop_name": "Cà chua", "level_label": "Nhiễm nặng",
           "severity_pct": 20.0, "actions": ["Ngắt bỏ lá bệnh, " * 20], "dosage": [], "warnings": [],
           "spray_schedule": [{"application": 1, "date": "2026-10-19"}]}
    text = build_ics(rec)
    lines = text.split("\r\n")
    assert all(len(x.encode("utf-8")) <= 75 for x in lines)
    unfolded = text.replace("\r\n ", "")
    assert "Ngắt bỏ lá bệnh\\, " in unfolded
