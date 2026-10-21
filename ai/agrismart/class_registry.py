"""Danh mục lớp bệnh: chọn bệnh từ PlantSeg, thêm lớp tự gán nhãn, sinh class_map.yaml và nhãn cho công cụ gán nhãn.

Quy ước tên lớp (dùng chung cho mô hình, dataset.yaml, ipm_rules.json và ứng dụng):
    leaf                      – phiến lá (luôn là lớp 0)
    <crop>__<disease>         – vết bệnh, ví dụ cucumber__powdery_mildew
<crop> chỉ gồm chữ thường và số (không gạch dưới) để tách được mã cây trồng từ tên lớp.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import yaml

from . import LEAF_CLASS

PLANTSEG_YAML = Path(__file__).resolve().parents[1] / "configs" / "plantseg115_classes.yaml"
CLASS_NAME_RE = re.compile(r"^[a-z0-9]+__[a-z0-9_]+$")

# Màu nhãn cho công cụ gán nhãn (lá xanh lá, vết bệnh các màu tương phản)
_PALETTE = ["#e53935", "#fb8c00", "#8e24aa", "#1e88e5", "#00897b", "#6d4c41",
            "#d81b60", "#3949ab", "#c0ca33", "#00acc1", "#f4511e", "#5e35b1"]


class ClassRegistryError(ValueError):
    pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_plantseg(path: str | Path = PLANTSEG_YAML) -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["classes"]


def plantseg_target_name(entry: dict) -> str:
    return f"{entry['crop']}__{slug(entry['disease'])}"


def crop_of(class_name: str) -> str:
    return class_name.split("__", 1)[0]


def select_plantseg(entries: list[dict], diseases: list[str] | None = None,
                    crops: list[str] | None = None) -> list[dict]:
    """Chọn bệnh theo tên đầy đủ (ví dụ "cucumber powdery mildew") hoặc theo mã cây trồng (ví dụ "cucumber")."""
    by_name = {e["name"]: e for e in entries}
    chosen: dict[int, dict] = {}
    for name in diseases or []:
        key = name.strip().lower()
        if key not in by_name:
            hint = difflib.get_close_matches(key, list(by_name), n=3)
            raise ClassRegistryError(f"PlantSeg không có bệnh '{name}'. Gần đúng: {hint or 'không có'}")
        chosen[by_name[key]["id"]] = by_name[key]
    all_crops = {e["crop"] for e in entries}
    for crop in crops or []:
        c = slug(crop).replace("_", "")
        if c not in all_crops:
            hint = difflib.get_close_matches(c, sorted(all_crops), n=3)
            raise ClassRegistryError(f"PlantSeg không có cây trồng '{crop}'. Gần đúng: {hint or 'không có'}")
        for e in entries:
            if e["crop"] == c:
                chosen[e["id"]] = e
    return [chosen[k] for k in sorted(chosen)]


def build_class_map(selected: list[dict], extras: list[str] | None = None) -> dict:
    """Sinh nội dung class_map.yaml: lớp 0 = leaf, tiếp theo các bệnh PlantSeg đã chọn, rồi các lớp tự gán nhãn."""
    names: list[str] = [LEAF_CLASS]
    value_to_target: dict[int, str] = {}
    for e in selected:
        n = plantseg_target_name(e)
        if n not in names:
            names.append(n)
        value_to_target[int(e["id"])] = n
    for x in extras or []:
        x = x.strip()
        if not CLASS_NAME_RE.match(x):
            raise ClassRegistryError(f"Tên lớp '{x}' sai quy ước <crop>__<disease> (chữ thường, số, gạch dưới; crop không có gạch dưới)")
        if x not in names:
            names.append(x)
    if len(names) == 1:
        raise ClassRegistryError("Chưa chọn lớp bệnh nào")
    return {
        "target_names": {i: n for i, n in enumerate(names)},
        "value_to_target": value_to_target,
        "folder_to_target": {},
        "label_aliases": {},
    }


def carry_over_manual_mappings(new: dict, old: dict | None) -> tuple[list[str], list[str]]:
    """Giữ lại folder_to_target và label_aliases khai báo tay trong class_map cũ khi chạy lại s00.

    Chỉ giữ mục có lớp đích vẫn còn trong danh sách lớp mới. Trả về (mục đã giữ, mục bị bỏ) để báo cho người dùng.
    Không có bước này thì chạy lại s00 sẽ xóa các ánh xạ thư mục đã khai báo (vd. dữ liệu tổng hợp chạy thử,
    mask nhị phân xếp theo thư mục bệnh) và s01 --mask-mode binary bỏ qua toàn bộ ảnh.
    """
    kept: list[str] = []
    dropped: list[str] = []
    valid = set(new["target_names"].values())
    for key in ("folder_to_target", "label_aliases"):
        for src, dst in ((old or {}).get(key) or {}).items():
            if dst in valid:
                new[key].setdefault(src, dst)
                kept.append(f"{key}: {src} -> {dst}")
            else:
                dropped.append(f"{key}: {src} -> {dst}")
    return kept, dropped


def write_class_map(cmap: dict, path: str | Path) -> None:
    header = ("# Sinh tự động bởi ai/scripts/s00_build_class_map.py – thứ tự lớp quyết định chỉ số lớp của mô hình.\n"
              "# label_aliases: ánh xạ tên nhãn trong công cụ gán nhãn (nếu khác tên lớp) -> tên lớp đích.\n")
    Path(path).write_text(header + yaml.safe_dump(cmap, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_annotation_labels(cmap: dict, out_dir: str | Path) -> list[Path]:
    """Xuất danh sách nhãn cho CVAT (dán vào ô Raw của Labels) và Labelme (tham số --labels)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = [cmap["target_names"][k] for k in sorted(cmap["target_names"], key=int)]
    cvat = []
    for i, n in enumerate(names):
        color = "#43a047" if n == LEAF_CLASS else _PALETTE[(i - 1) % len(_PALETTE)]
        cvat.append({"name": n, "color": color, "type": "any", "attributes": []})
    p1 = out / "cvat_labels.json"
    p1.write_text(json.dumps(cvat, ensure_ascii=False, indent=2), encoding="utf-8")
    p2 = out / "labelme_labels.txt"
    p2.write_text("\n".join(names) + "\n", encoding="utf-8")
    return [p1, p2]
