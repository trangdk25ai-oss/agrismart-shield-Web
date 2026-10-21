"""Bộ ảnh mẫu cho Demo (do ai/scripts/s10_demo_pack.py sinh): ảnh tập kiểm tra kèm đáp án Severity tính từ nhãn.

Trang web hiển thị "đáp án" bên cạnh kết quả của mô hình để người xem tự so sánh – thay cho việc
đối chiếu với chuyên gia ngoài ruộng khi không thể đi thực địa.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from agrismart.io_utils import imread

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SampleStore:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.items: dict[str, dict] = {}
        self._thumbs: dict[str, bytes] = {}
        mf = self.folder / "manifest.json"
        if mf.is_file():
            data = json.loads(mf.read_text(encoding="utf-8"))
            for it in data.get("items", []):
                if ID_RE.match(str(it.get("id", ""))) and (self.folder / it["file"]).is_file():
                    self.items[it["id"]] = it

    def __len__(self) -> int:
        return len(self.items)

    def list(self) -> list[dict]:
        return list(self.items.values())

    def get(self, sample_id: str) -> dict | None:
        return self.items.get(sample_id) if ID_RE.match(sample_id or "") else None

    def path(self, sample_id: str) -> Path | None:
        it = self.get(sample_id)
        return (self.folder / it["file"]) if it else None

    def image(self, sample_id: str) -> np.ndarray | None:
        p = self.path(sample_id)
        return imread(p) if p else None

    def thumbnail(self, sample_id: str, side: int = 320) -> bytes | None:
        if sample_id in self._thumbs:
            return self._thumbs[sample_id]
        img = self.image(sample_id)
        if img is None:
            return None
        h, w = img.shape[:2]
        s = side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self._thumbs[sample_id] = buf.tobytes() if ok else b""
        return self._thumbs[sample_id]
