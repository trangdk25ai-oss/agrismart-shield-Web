"""Máy chủ trang web AgriSmart Shield (FastAPI). Chạy: python web/run.py  (xem web/README.md).

API (JSON, tiếng Việt):
    GET  /api/health                 tình trạng máy chủ
    GET  /api/meta                   cây trồng, bệnh, loại bình, chế độ (mô hình / thử), địa chỉ cho điện thoại
    POST /api/diagnose               multipart: image | sample_id, crop_code, tank_code, start_date, force, lesion_class
    POST /api/recommend              JSON: tính lại phác đồ khi đổi bình hoặc chọn lá khác (không chạy lại mô hình)
    GET  /api/calendar.ics           tệp lịch phun (.ics) để thêm vào ứng dụng Lịch của điện thoại
    GET  /api/samples                danh sách ảnh mẫu Demo (kèm đáp án)
    GET  /api/samples/{id}/thumb     ảnh thu nhỏ
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "ai") not in sys.path:
    sys.path.insert(0, str(REPO / "ai"))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from .config import Settings  # noqa: E402
from .imaging import BadImage, decode_upload, limit_side  # noqa: E402
from .service import DiagnosisService, ServiceError  # noqa: E402

STATIC = Path(__file__).resolve().parents[1] / "static"


class RecommendIn(BaseModel):
    crop_code: str
    model_class: str | None = None
    severity_pct: float = Field(ge=0, le=100)
    has_lesion: bool = True
    tank_code: str = "TANK_16L"
    start_date: date | None = None
    detection_conf: float | None = Field(default=None, ge=0, le=1)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise HTTPException(400, "Ngày bắt đầu phải có dạng YYYY-MM-DD") from e


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    svc = DiagnosisService(settings)
    app = FastAPI(title="AgriSmart Shield", version=svc.meta()["version"],
                  description="Chẩn đoán bệnh lá cây, tính Severity Index và khuyến nghị IPM")
    app.state.service = svc
    logging.getLogger("agrismart.web").info("Chế độ: %s – %s", svc.mode, svc.model.info())

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": svc.mode, "model": svc.model.info()["file"], "kb_version": svc.engine.kb["kb_version"]}

    @app.get("/api/meta")
    def meta():
        return svc.meta()

    @app.post("/api/diagnose")
    def diagnose(
        image: UploadFile | None = File(default=None),
        sample_id: str | None = Form(default=None),
        crop_code: str = Form(default="tomato"),
        tank_code: str = Form(default="TANK_16L"),
        start_date: str | None = Form(default=None),
        force: bool = Form(default=False),
        lesion_class: str | None = Form(default=None),
    ):
        limit = int(settings.max_upload_mb * 1024 * 1024)
        if sample_id:
            img = svc.samples.image(sample_id)
            if img is None:
                raise HTTPException(404, "Không có ảnh mẫu này")
            img = limit_side(img, settings.max_side)
            it = svc.samples.get(sample_id)
            if svc.mode == "color_rule" and not lesion_class:
                lesion_class = it.get("gt_class")
            crop_code = it.get("crop_code", crop_code)
        elif image is not None:
            data = image.file.read(limit + 1)
            if len(data) > limit:
                raise HTTPException(413, f"Ảnh lớn hơn {settings.max_upload_mb:g} MB")
            try:
                img = decode_upload(data, settings.max_side)
            except BadImage as e:
                raise HTTPException(400, str(e)) from e
        else:
            raise HTTPException(400, "Chưa có ảnh: chụp/chọn ảnh hoặc chọn một ảnh mẫu")
        try:
            return svc.diagnose(img, crop_code, tank_code=tank_code, start_date=_parse_date(start_date), force=force,
                                lesion_class=lesion_class, sample_id=sample_id)
        except ServiceError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/api/recommend")
    def recommend(body: RecommendIn):
        try:
            return svc.recommend(body.crop_code, body.model_class, body.severity_pct, body.has_lesion, body.tank_code,
                                 body.start_date, body.detection_conf)
        except ServiceError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/api/calendar.ics")
    def calendar(
        crop_code: str,
        model_class: str,
        severity_pct: float = Query(ge=0, le=100),
        tank_code: str = "TANK_16L",
        start_date: str | None = None,
    ):
        d = _parse_date(start_date) or date.today()
        try:
            ics = svc.calendar(crop_code=crop_code, model_class=model_class, severity_pct=severity_pct, has_lesion=True,
                               tank_code=tank_code, start_date=d)
        except ServiceError as e:
            raise HTTPException(400, str(e)) from e
        return Response(ics, media_type="text/calendar; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="lich_phun_{d:%Y%m%d}.ics"'})

    @app.get("/api/samples")
    def samples():
        return {"items": svc.samples.list()}

    @app.get("/api/samples/{sample_id}/thumb")
    def sample_thumb(sample_id: str):
        data = svc.samples.thumbnail(sample_id)
        if not data:
            raise HTTPException(404, "Không có ảnh mẫu này")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
