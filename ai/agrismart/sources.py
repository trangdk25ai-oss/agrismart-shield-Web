"""Danh mục nguồn dữ liệu (data_sources/catalog.yaml): đọc, kiểm tra, lọc, sinh trích dẫn.

Mục đích: mọi ảnh đưa vào bộ dữ liệu đều truy được về nguồn (id trong danh mục) để
(1) ghi nguồn đầy đủ trong hồ sơ dự thi, (2) kiểm soát giấy phép, (3) tránh dùng nhầm nguồn chưa xác minh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATALOG = Path(__file__).resolve().parents[2] / "data_sources" / "catalog.yaml"
LEVELS = {"A", "B", "C", "D", "Q"}
PIPELINES = {"s01_value", "s01_binary", "s01b_coco", "s01b_labelme", "s01b_via", "label_then_s01b",
             "eval_bbox", "reference", "out_of_scope"}


class CatalogError(ValueError):
    pass


@dataclass
class Source:
    id: str
    name: str
    crops: list[str]
    level: str
    labels: str
    license: str
    license_verified: bool
    urls: dict[str, str]
    pipeline: str
    doi: str | None = None
    size: str = ""
    classes: str = ""
    region: str = ""
    notes: str = ""
    fills_gap: list[str] = field(default_factory=list)
    pair_with: str | None = None

    @property
    def main_url(self) -> str:
        return self.urls.get("data") or self.urls.get("code") or next(iter(self.urls.values()))

    def citation(self) -> str:
        """Trích dẫn ngắn: Tên. Nơi lưu trữ. DOI/URL. Giấy phép."""
        where = f"https://doi.org/{self.doi}" if self.doi else self.main_url
        lic = self.license if self.license_verified else f"{self.license} – cần kiểm tra giấy phép"
        extra = f" Bài báo: {self.urls['paper']}." if "paper" in self.urls else ""
        return f"{self.name}. {where}.{extra} Giấy phép: {lic}."


def load_catalog(path: str | Path = CATALOG) -> dict[str, Source]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise CatalogError("catalog.yaml: schema_version phải là 1")
    out: dict[str, Source] = {}
    allowed = set(Source.__dataclass_fields__)
    for raw in data["sources"]:
        unknown = set(raw) - allowed
        if unknown:
            raise CatalogError(f"{raw.get('id')}: trường không hỗ trợ {sorted(unknown)}")
        s = Source(**raw)
        if s.id in out:
            raise CatalogError(f"id trùng: {s.id}")
        if s.level not in LEVELS:
            raise CatalogError(f"{s.id}: level '{s.level}' không hợp lệ")
        if s.pipeline not in PIPELINES:
            raise CatalogError(f"{s.id}: pipeline '{s.pipeline}' không hợp lệ")
        if not s.urls or not all(u.startswith("https://") for u in s.urls.values()):
            raise CatalogError(f"{s.id}: urls phải có ít nhất 1 đường link https://")
        out[s.id] = s
    for s in out.values():
        if s.pair_with and s.pair_with not in out:
            raise CatalogError(f"{s.id}: pair_with '{s.pair_with}' không có trong danh mục")
    return out


def filter_sources(cat: dict[str, Source], crop: str | None = None, level: str | None = None,
                   gap: str | None = None) -> list[Source]:
    res = list(cat.values())
    if crop:
        res = [s for s in res if crop in s.crops]
    if level:
        res = [s for s in res if s.level == level.upper()]
    if gap:
        res = [s for s in res if gap in s.fills_gap]
    return res
