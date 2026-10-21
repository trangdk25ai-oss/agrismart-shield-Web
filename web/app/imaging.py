"""Đọc ảnh tải lên an toàn (giới hạn dung lượng, xoay theo EXIF, thu nhỏ) và mã hóa ảnh trả về trình duyệt."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000  # chặn ảnh "bom giải nén"; ảnh điện thoại 50 MP vẫn đọc được


class BadImage(ValueError):
    pass


def decode_upload(data: bytes, max_side: int) -> np.ndarray:
    """bytes (JPEG/PNG/WebP…) -> ảnh BGR đã xoay đúng chiều theo EXIF và thu nhỏ về cạnh dài max_side."""
    if not data:
        raise BadImage("Tệp rỗng")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)  # ảnh điện thoại thường lưu chiều xoay trong EXIF
            im = im.convert("RGB")
            arr = np.asarray(im)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise BadImage("Không đọc được tệp ảnh (hỗ trợ JPEG, PNG, WebP)") from e
    img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return limit_side(img, max_side)


def limit_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1.0:
        img = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return img


def to_jpeg_b64(img_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Không mã hóa được JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("Không mã hóa được PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def bgr_hex(color: tuple[int, int, int]) -> str:
    b, g, r = color
    return f"#{r:02x}{g:02x}{b:02x}"
