"""Tạo quy tắc IPM "CHỜ NHẬP" (rule_status = PENDING) cho các lớp bệnh mới chưa có trong ipm_rules.json.

Quy tắc CHỜ NHẬP cho phép huấn luyện và chạy trang web ngay: hệ thống vẫn khoanh vùng, tính % tổn thương và
mức độ, nhưng KHÔNG đưa phác đồ/liều lượng, chỉ thông báo "phác đồ chưa được cán bộ chuyên môn nhập". Sau khi
chuyên gia BVTV điền actions, dosage_guide, spray_plan… thì đổi rule_status thành ACTIVE.
Máy chủ web đọc trực tiếp tệp này (không có bản sao), chỉ cần khởi động lại máy chủ sau khi sửa.

    python knowledge_base/kb_add_rules.py --class-map ai/configs/class_map.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai"))


def skeleton(model_class: str, crop_name: str, disease_name: str) -> dict:
    return {
        "rule_id": model_class.upper(),
        "model_class": model_class,
        "crop_code": model_class.split("__", 1)[0],
        "crop_name": crop_name,
        "disease_name": disease_name,
        "rule_status": "PENDING",
        "severity_thresholds": None,
        "active_ingredient": [],
        "dosage_guide": [],
        "actions": {"HEALTHY": [], "ONSET": [], "LIGHT": [], "HEAVY": []},
        "spray_plan": {lv: {"applications": 0, "interval_days": 7} for lv in ("ONSET", "LIGHT", "HEAVY")},
        "escalation": {},
    }


def main() -> int:
    from agrismart.class_registry import load_plantseg, plantseg_target_name
    from agrismart.knowledge_engine import KnowledgeEngine

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--class-map", type=Path, required=True, help="class_map.yaml hoặc dataset.yaml (có names)")
    ap.add_argument("--kb", type=Path, default=HERE / "ipm_rules.json")
    ap.add_argument("--schema", type=Path, default=HERE / "ipm_rules.schema.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.class_map.read_text(encoding="utf-8"))
    names = list((cfg.get("target_names") or cfg.get("names")).values())
    kb = json.loads(args.kb.read_text(encoding="utf-8"))
    existing = {r["model_class"] for r in kb["rules"]}
    crop_names = {r["crop_code"]: r["crop_name"] for r in kb["rules"]}
    plantseg = {plantseg_target_name(e): e for e in load_plantseg()}

    added = []
    for n in names:
        if n == "leaf" or n in existing:
            continue
        e = plantseg.get(n)
        crop = n.split("__", 1)[0]
        crop_name = crop_names.get(crop) or (e["crop_vi"] if e else crop)
        if e:
            disease = f"{e['disease_vi']} ({e['name']})" if e["disease_vi"] else e["name"]
        else:
            disease = n.split("__", 1)[1].replace("_", " ") + " (cần nhập tên tiếng Việt)"
        kb["rules"].append(skeleton(n, crop_name, disease))
        crop_names.setdefault(crop, crop_name)
        added.append(n)

    if not added:
        print("Mọi lớp đã có quy tắc – không thay đổi.")
        return 0
    # kiểm tra trước khi ghi: lỗi thì không ghi đè file đang dùng
    KnowledgeEngine(kb, json.loads(args.schema.read_text(encoding="utf-8")))
    shutil.copy2(args.kb, args.kb.with_suffix(".json.bak"))
    args.kb.write_text(json.dumps(kb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã thêm {len(added)} quy tắc CHỜ NHẬP vào {args.kb} (bản cũ: {args.kb.with_suffix('.json.bak').name}):")
    for n in added:
        print("  -", n)
    print("Việc tiếp theo: cán bộ BVTV điền tên bệnh, tác nhân, actions, dosage_guide, spray_plan rồi đổi rule_status = ACTIVE.")
    print("Khởi động lại máy chủ web (python web/run.py) để nạp tri thức mới.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
