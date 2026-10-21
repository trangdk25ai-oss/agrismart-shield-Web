"""Kiểm tra ipm_rules.json: JSON Schema + ràng buộc chéo + đồng bộ với danh sách lớp của mô hình.

    python knowledge_base/validate_kb.py --dataset-yaml ai/configs/dataset.yaml
Mã thoát khác 0 nếu có lỗi -> dùng làm cổng chặn trong CI trước khi đóng gói app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ai"))


def main() -> int:
    import yaml
    from agrismart.knowledge_engine import KnowledgeBaseError, KnowledgeEngine

    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, default=HERE / "ipm_rules.json")
    ap.add_argument("--schema", type=Path, default=HERE / "ipm_rules.schema.json")
    ap.add_argument("--dataset-yaml", type=Path, default=HERE.parent / "ai/configs/dataset.yaml")
    args = ap.parse_args()

    try:
        engine = KnowledgeEngine.from_file(args.kb, args.schema)
    except KnowledgeBaseError as e:
        print("LỖI ràng buộc:", e)
        return 1
    except Exception as e:  # jsonschema.ValidationError
        print("LỖI schema:", getattr(e, "message", e))
        return 1

    names = yaml.safe_load(args.dataset_yaml.read_text(encoding="utf-8"))["names"]
    model_classes = {v for v in names.values() if v != "leaf"}
    kb_classes = set(engine.rules_by_class)
    missing = sorted(model_classes - kb_classes)
    orphan = sorted(kb_classes - model_classes)
    unverified = [p["code"] for p in engine.kb["products"] if not p["rate_verified"]]

    print(f"kb_version={engine.kb['kb_version']}  review_status={engine.kb['review_status']}")
    print(f"Số quy tắc: {len(kb_classes)} | lớp mô hình: {len(model_classes)}")
    if orphan:
        print("CẢNH BÁO – quy tắc không có lớp tương ứng trong mô hình:", orphan)
    if unverified:
        print("CẢNH BÁO – liều lượng chưa thẩm định:", unverified)
    pending = engine.pending_rules()
    if pending:
        print("CẢNH BÁO – quy tắc CHỜ NHẬP phác đồ (app chỉ báo mức độ, không đưa phác đồ):", pending)
    if missing:
        print("LỖI – lớp mô hình chưa có quy tắc IPM:", missing)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
