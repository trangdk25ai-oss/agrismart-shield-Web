"""Cấu hình máy chủ, đọc từ biến môi trường (web/run.py đặt sẵn từ tham số dòng lệnh)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WEB = REPO / "web"
MODEL_DIR = WEB / "models"
# Thứ tự ưu tiên khi không chỉ định mô hình: INT8 ONNX (s07) -> ONNX khác -> LiteRT -> PyTorch
MODEL_PATTERNS = ("model_int8.onnx", "*.onnx", "*.tflite", "*.pt")


def find_default_model(model_dir: Path = MODEL_DIR) -> Path | None:
    for pat in MODEL_PATTERNS:
        hits = sorted(model_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


@dataclass
class Settings:
    model_path: Path | None = None  # None = tự tìm trong web/models/, không có thì chế độ thử
    conf: float = 0.25
    blur_threshold: float = 100.0
    max_upload_mb: float = 12.0
    max_side: int = 1280  # ảnh được thu nhỏ về cạnh dài này trước khi xử lý
    samples_dir: Path = REPO / "demo/samples"
    threads: int | None = None
    lan_urls: list[str] = field(default_factory=list)
    kb_path: Path = REPO / "knowledge_base/ipm_rules.json"
    kb_schema: Path = REPO / "knowledge_base/ipm_rules.schema.json"

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.environ.get("AGS_MODEL")
        threads = os.environ.get("AGS_THREADS")
        return cls(
            model_path=Path(model) if model else find_default_model(),
            conf=_env_float("AGS_CONF", 0.25),
            blur_threshold=_env_float("AGS_BLUR_THRESHOLD", 100.0),
            max_upload_mb=_env_float("AGS_MAX_UPLOAD_MB", 12.0),
            max_side=int(_env_float("AGS_MAX_SIDE", 1280)),
            samples_dir=Path(os.environ.get("AGS_SAMPLES_DIR", REPO / "demo/samples")),
            threads=int(threads) if threads else None,
            lan_urls=[u for u in os.environ.get("AGS_LAN_URLS", "").split(",") if u],
        )
