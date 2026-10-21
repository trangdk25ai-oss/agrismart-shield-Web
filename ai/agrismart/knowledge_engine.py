"""Knowledge Engine – khớp bộ 3 tham số (Cây, Bệnh, Severity_Index) với ipm_rules.json (Bước 4.2).

Thiết kế:
- KHÔNG có logic nông học "cứng" trong mã: mọi ngưỡng, hành động, liều lượng nằm trong JSON để
  chuyên gia BVTV cập nhật mà không phải phát hành lại ứng dụng logic.
- Mọi liều lượng chưa thẩm định (rate_verified=false) đều sinh cảnh báo hiển thị cho người dùng.
- Phân loại mức độ dùng giá trị severity đã LÀM TRÒN 2 chữ số (bằng độ chính xác hiển thị) để
  con số trên màn hình và nhãn mức độ luôn nhất quán (ví dụ: hiển thị "5,00%" thì phải là "Nhiễm nhẹ").
- Máy chủ trang web gọi trực tiếp lớp này (web/app/service.py) – không có bản cài đặt thứ hai.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

LEVEL_HEALTHY = "HEALTHY"
LEVEL_ONSET = "ONSET"
LEVEL_LIGHT = "LIGHT"
LEVEL_HEAVY = "HEAVY"

STATUS_OK = "OK"
STATUS_HEALTHY = "HEALTHY"
STATUS_NO_RULE = "NO_RULE"
STATUS_CROP_MISMATCH = "CROP_MISMATCH"
STATUS_PENDING_RULE = "PENDING_RULE"  # đã có lớp bệnh trong mô hình nhưng phác đồ chưa được chuyên gia nhập

RULE_ACTIVE = "ACTIVE"
RULE_PENDING = "PENDING"


def rule_status(rule: dict) -> str:
    return rule.get("rule_status", RULE_ACTIVE)


class KnowledgeBaseError(ValueError):
    pass


def _fmt_num(x: float) -> str:
    """Định dạng số kiểu Việt Nam: tối đa 2 chữ số thập phân, dấu phẩy thập phân, bỏ số 0 thừa."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s.replace(".", ",")


class KnowledgeEngine:
    def __init__(self, kb: dict[str, Any], schema: dict[str, Any] | None = None):
        if schema is not None:
            import jsonschema  # chỉ cần khi kiểm tra schema

            jsonschema.validate(kb, schema)
        self.kb = kb
        self._check_consistency()
        self.rules_by_class = {r["model_class"]: r for r in kb["rules"]}
        self.products = {p["code"]: p for p in kb["products"]}
        self.tanks = {t["code"]: t for t in kb["tank_presets"]}

    # ------------------------------------------------------------------ nạp & kiểm tra
    @classmethod
    def from_file(cls, path: str | Path, schema_path: str | Path | None = None) -> "KnowledgeEngine":
        kb = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8")) if schema_path else None
        return cls(kb, schema)

    def _check_consistency(self) -> None:
        """Các ràng buộc chéo mà JSON Schema không diễn đạt được."""
        kb = self.kb
        product_codes = {p["code"] for p in kb["products"]}
        seen_classes: set[str] = set()
        self._validate_thresholds(kb["default_severity_thresholds"], "default_severity_thresholds")
        for r in kb["rules"]:
            rid = r.get("rule_id", "?")
            if r["model_class"] in seen_classes:
                raise KnowledgeBaseError(f"{rid}: model_class '{r['model_class']}' bị trùng")
            seen_classes.add(r["model_class"])
            if not r["model_class"].startswith(r["crop_code"] + "__"):
                raise KnowledgeBaseError(f"{rid}: model_class phải bắt đầu bằng '{r['crop_code']}__'")
            if r.get("severity_thresholds"):
                self._validate_thresholds(r["severity_thresholds"], rid)
            for d in r["dosage_guide"]:
                if d["product_code"] not in product_codes:
                    raise KnowledgeBaseError(f"{rid}: product_code '{d['product_code']}' không có trong products")
            if rule_status(r) == RULE_ACTIVE:
                # Quy tắc đang áp dụng phải có đủ hành động cho 3 mức; quy tắc PENDING được phép để trống
                empty = [lv for lv in (LEVEL_ONSET, LEVEL_LIGHT, LEVEL_HEAVY) if not r["actions"].get(lv)]
                if empty:
                    raise KnowledgeBaseError(f"{rid}: quy tắc ACTIVE thiếu hành động cho mức {empty} "
                                             "(đặt rule_status = PENDING nếu chưa nhập xong)")

    def pending_rules(self) -> list[str]:
        return [r["model_class"] for r in self.kb["rules"] if rule_status(r) == RULE_PENDING]

    @staticmethod
    def _validate_thresholds(t: dict, where: str) -> None:
        if not (0 < t["onset_upper_exclusive"] <= t["light_upper_inclusive"] <= 100):
            raise KnowledgeBaseError(f"{where}: yêu cầu 0 < onset_upper_exclusive <= light_upper_inclusive <= 100")

    # ------------------------------------------------------------------ tra cứu
    def crops(self) -> list[tuple[str, str]]:
        seen: dict[str, str] = {}
        for r in self.kb["rules"]:
            seen.setdefault(r["crop_code"], r["crop_name"])
        return list(seen.items())

    def rule_for_class(self, model_class: str) -> dict | None:
        return self.rules_by_class.get(model_class)

    def thresholds_for(self, rule: dict | None) -> dict:
        if rule and rule.get("severity_thresholds"):
            return rule["severity_thresholds"]
        return self.kb["default_severity_thresholds"]

    def classify(self, severity_pct: float, has_lesion: bool = True, rule: dict | None = None) -> str:
        """< onset -> ONSET ; onset..light (bao gồm 2 đầu) -> LIGHT ; > light -> HEAVY.

        Quy trình gốc ghi "< 5% / 5 - 15% / > 15%": khoảng giữa được hiểu là ĐÓNG hai đầu [5; 15].
        """
        if not has_lesion:
            return LEVEL_HEALTHY
        s = round(float(severity_pct), 2)
        t = self.thresholds_for(rule)
        if s < t["onset_upper_exclusive"]:
            return LEVEL_ONSET
        if s <= t["light_upper_inclusive"]:
            return LEVEL_LIGHT
        return LEVEL_HEAVY

    # ------------------------------------------------------------------ khuyến nghị
    def dosage_for(self, rule: dict, level: str, tank_code: str) -> list[dict]:
        tank = self.tanks.get(tank_code)
        if tank is None:
            raise KnowledgeBaseError(f"Không có loại bình '{tank_code}'")
        out = []
        for d in rule["dosage_guide"]:
            if level not in d["apply_at"]:
                continue
            p = self.products[d["product_code"]]
            rate = p["label_rate_per_liter"]
            amount = round(rate["amount"] * tank["volume_l"], 1)
            out.append(
                {
                    "product_code": p["code"],
                    "active_ingredient": p["active_ingredient"],
                    "amount": amount,
                    "unit": rate["unit"],
                    "tank_code": tank["code"],
                    "tank_label": tank["label"],
                    "volume_l": tank["volume_l"],
                    "verified": bool(p["rate_verified"]),
                    "text": (
                        f"Pha {_fmt_num(amount)} {rate['unit']} chế phẩm {p['active_ingredient']} "
                        f"cho 1 {tank['label'].lower()} ({_fmt_num(rate['amount'])} {rate['unit']}/lít)"
                    ),
                }
            )
        return out

    def spray_schedule(self, rule: dict, level: str, start: date) -> list[dict]:
        plan = rule["spray_plan"].get(level)
        if not plan or plan["applications"] <= 0:
            return []
        return [
            {"application": i + 1, "date": (start + timedelta(days=i * plan["interval_days"])).isoformat()}
            for i in range(plan["applications"])
        ]

    def recommend(
        self,
        crop_code: str,
        model_class: str | None,
        severity_pct: float,
        has_lesion: bool = True,
        tank_code: str = "TANK_16L",
        start_date: date | None = None,
        detection_conf: float | None = None,
        min_conf: float = 0.5,
    ) -> dict:
        start_date = start_date or date.today()
        warnings: list[str] = []
        if self.kb.get("review_status") != "APPROVED":
            warnings.append("Cơ sở tri thức đang ở trạng thái DỰ THẢO, chưa được cơ quan chuyên môn phê duyệt.")

        base = {
            "kb_version": self.kb["kb_version"],
            "crop_code": crop_code,
            "severity_pct": round(float(severity_pct), 2),
            "warnings": warnings,
            "disclaimer": self.kb["disclaimer"],
        }

        if not has_lesion or model_class is None:
            lv = self.kb["severity_levels"][LEVEL_HEALTHY]
            return {**base, "status": STATUS_HEALTHY, "level_code": LEVEL_HEALTHY, "level_label": lv["label"],
                    "level_color": lv["color"], "actions": [], "dosage": [], "spray_schedule": []}

        rule = self.rule_for_class(model_class)
        if rule is None:
            return {**base, "status": STATUS_NO_RULE, "model_class": model_class,
                    "message": "Chưa có phác đồ cho loại bệnh này trong cơ sở tri thức. Đề nghị hỏi cán bộ BVTV."}

        if rule["crop_code"] != crop_code:
            return {**base, "status": STATUS_CROP_MISMATCH, "model_class": model_class,
                    "detected_crop": rule["crop_name"],
                    "message": (f"Bệnh nhận dạng được thuộc cây {rule['crop_name']}, khác cây trồng đã chọn. "
                                "Đề nghị kiểm tra lại loại cây hoặc chụp lại ảnh.")}

        level = self.classify(severity_pct, True, rule)
        lv = self.kb["severity_levels"][level]
        if rule_status(rule) == RULE_PENDING:
            return {**base, "status": STATUS_PENDING_RULE, "rule_id": rule["rule_id"], "model_class": model_class,
                    "crop_name": rule["crop_name"], "disease_name": rule["disease_name"],
                    "level_code": level, "level_label": lv["label"], "level_color": lv["color"],
                    "actions": [], "dosage": [], "spray_schedule": [],
                    "message": (f"Đã nhận dạng {rule['disease_name']} nhưng phác đồ xử lý chưa được cán bộ chuyên môn "
                                "nhập vào hệ thống. Đề nghị liên hệ cán bộ BVTV.")}
        dosage = self.dosage_for(rule, level, tank_code)
        if any(not d["verified"] for d in dosage):
            warnings.append("Liều lượng là dữ liệu MẪU, chưa thẩm định: bắt buộc đối chiếu nhãn sản phẩm trước khi pha.")
        if detection_conf is not None and detection_conf < min_conf:
            warnings.append(
                f"Độ tin cậy nhận dạng thấp ({detection_conf * 100:.0f}%): nên chụp lại ảnh rõ hơn hoặc hỏi cán bộ BVTV."
            )

        return {
            **base,
            "status": STATUS_OK,
            "rule_id": rule["rule_id"],
            "model_class": model_class,
            "crop_name": rule["crop_name"],
            "disease_name": rule["disease_name"],
            "pathogen": rule.get("pathogen"),
            "level_code": level,
            "level_label": lv["label"],
            "level_color": lv["color"],
            "actions": list(rule["actions"].get(level, [])),
            "dosage": dosage,
            "spray_schedule": self.spray_schedule(rule, level, start_date),
            "escalation": (rule.get("escalation") or {}).get(level),
        }
