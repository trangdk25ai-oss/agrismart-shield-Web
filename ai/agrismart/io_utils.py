"""Đọc/ghi ảnh an toàn với đường dẫn có dấu tiếng Việt.

cv2.imread / cv2.imwrite trên Windows không mở được đường dẫn chứa ký tự ngoài ASCII
(ví dụ C:\\Users\\Thắng\\...). Đọc/ghi qua numpy + imdecode/imencode tránh được lỗi này
trên mọi hệ điều hành.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    path = Path(path)
    ok, buf = cv2.imencode(path.suffix or ".png", image, params or [])
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))
    return True


def image_size(path: str | Path) -> tuple[int, int] | None:
    """(width, height) của ảnh, None nếu không đọc được."""
    im = imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    return int(im.shape[1]), int(im.shape[0])
