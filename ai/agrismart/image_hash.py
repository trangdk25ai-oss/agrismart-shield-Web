"""Băm cảm nhận (difference hash – dHash) để phát hiện ảnh trùng / gần trùng giữa các nguồn dữ liệu.

Vì sao cần: nhiều bộ dữ liệu thu thập ảnh từ Internet hoặc từ cùng vùng trồng, cùng một ảnh có thể xuất hiện
ở PlantSeg, PlantDoc và bộ khác. Ảnh trùng nằm ở cả tập huấn luyện lẫn tập kiểm tra làm chỉ số bị thổi phồng.

dHash 64 bit: thu ảnh xám về 9×8, so sánh độ sáng hai pixel liền kề theo hàng. Hai ảnh có khoảng cách Hamming
≤ 5 (trên 64 bit) thường là cùng ảnh đã đổi kích thước / nén lại. Ngưỡng cần kiểm tra lại trên dữ liệu thật.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .io_utils import imread

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(image: np.ndarray, size: int = 8) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def dhash_file(path: str | Path) -> int | None:
    im = imread(path, cv2.IMREAD_GRAYSCALE)
    return None if im is None else dhash(im)


def hamming_to_many(h: int, many: np.ndarray) -> np.ndarray:
    """Khoảng cách Hamming từ một mã băm tới mảng mã băm (uint64)."""
    if many.size == 0:
        return np.zeros(0, dtype=np.int64)
    x = np.bitwise_xor(many, np.uint64(h))
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    return _POP8[x.view(np.uint8).reshape(-1, 8)].sum(axis=1).astype(np.int64)


class NearDuplicateIndex:
    """Chỉ mục mã băm: kiểm tra một ảnh mới có gần trùng ảnh nào đã có không."""

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self._buf = np.zeros(1024, dtype=np.uint64)  # bộ đệm tăng gấp đôi khi đầy
        self._keys: list[str] = []

    def __len__(self) -> int:
        return len(self._keys)

    def find(self, h: int) -> tuple[str, int] | None:
        d = hamming_to_many(h, self._buf[: len(self._keys)])
        if d.size == 0:
            return None
        i = int(np.argmin(d))
        return (self._keys[i], int(d[i])) if d[i] <= self.threshold else None

    def add(self, key: str, h: int) -> None:
        n = len(self._keys)
        if n == self._buf.size:
            self._buf = np.concatenate([self._buf, np.zeros(n, dtype=np.uint64)])
        self._buf[n] = np.uint64(h)
        self._keys.append(key)
