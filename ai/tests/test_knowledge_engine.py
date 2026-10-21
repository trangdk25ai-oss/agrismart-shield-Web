import copy
import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from agrismart.knowledge_engine import (
    LEVEL_HEALTHY,
    LEVEL_HEAVY,
    LEVEL_LIGHT,
    LEVEL_ONSET,
    STATUS_CROP_MISMATCH,
    STATUS_HEALTHY,
    STATUS_NO_RULE,
    STATUS_OK,
    KnowledgeBaseError,
    KnowledgeEngine,
)

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "knowledge_base" / "ipm_rules.json"
SCHEMA = ROOT / "knowledge_base" / "ipm_rules.schema.json"


@pytest.fixture(scope="module")
def engine():
    return KnowledgeEngine.from_file(KB, SCHEMA)


@pytest.mark.parametrize(
    "pct,expected",
    [(0.01, LEVEL_ONSET), (4.99, LEVEL_ONSET), (4.996, LEVEL_LIGHT),  # 4,996 hiển thị 5,00% -> Nhiễm nhẹ
     (5.0, LEVEL_LIGHT), (15.0, LEVEL_LIGHT), (15.004, LEVEL_LIGHT), (15.01, LEVEL_HEAVY), (80, LEVEL_HEAVY)],
)
def test_threshold_boundaries(engine, pct, expected):
    assert engine.classify(pct) == expected


def test_no_lesion_is_healthy(engine):
    assert engine.classify(0.0, has_lesion=False) == LEVEL_HEALTHY
    r = engine.recommend("tomato", None, 0.0, has_lesion=False)
    assert r["status"] == STATUS_HEALTHY and r["dosage"] == []


def test_recommend_light_16l(engine):
    r = engine.recommend("tomato", "tomato__early_blight", 8.3, tank_code="TANK_16L", start_date=date(2026, 10, 13))
    assert r["status"] == STATUS_OK
    assert r["level_code"] == LEVEL_LIGHT and r["level_label"] == "Nhiễm nhẹ"
    doses = {d["product_code"]: d for d in r["dosage"]}
    assert doses["BIO_BACILLUS_SUBTILIS"]["amount"] == pytest.approx(20.0)  # 1,25 g/L x 16 L
    assert doses["BIO_TRICHODERMA"]["amount"] == pytest.approx(48.0)  # 3 g/L x 16 L
    assert "1,25 g/lít" in doses["BIO_BACILLUS_SUBTILIS"]["text"]
    assert [s["date"] for s in r["spray_schedule"]] == ["2026-10-13", "2026-10-20"]
    assert any("MẪU" in w for w in r["warnings"])  # liều chưa thẩm định phải bị cảnh báo


def test_recommend_onset_2l_only_applicable_products(engine):
    r = engine.recommend("tomato", "tomato__early_blight", 2.0, tank_code="TANK_2L")
    assert r["level_code"] == LEVEL_ONSET
    assert [d["product_code"] for d in r["dosage"]] == ["BIO_BACILLUS_SUBTILIS"]
    assert r["dosage"][0]["amount"] == pytest.approx(2.5)


def test_heavy_has_escalation(engine):
    r = engine.recommend("cucumber", "cucumber__downy_mildew", 30.0)
    assert r["level_code"] == LEVEL_HEAVY and r["escalation"]


def test_crop_mismatch_and_unknown_class(engine):
    assert engine.recommend("citrus", "tomato__late_blight", 10)["status"] == STATUS_CROP_MISMATCH
    assert engine.recommend("tomato", "tomato__unknown", 10)["status"] == STATUS_NO_RULE


def test_low_confidence_warning(engine):
    r = engine.recommend("tomato", "tomato__early_blight", 10, detection_conf=0.31)
    assert any("Độ tin cậy" in w for w in r["warnings"])


def test_rule_level_threshold_override():
    kb = json.loads(KB.read_text(encoding="utf-8"))
    kb = copy.deepcopy(kb)
    kb["rules"][1]["severity_thresholds"] = {"onset_upper_exclusive": 1.0, "light_upper_inclusive": 5.0}
    eng = KnowledgeEngine(kb)
    assert eng.recommend("tomato", "tomato__late_blight", 3.0)["level_code"] == LEVEL_LIGHT
    assert eng.recommend("tomato", "tomato__early_blight", 3.0)["level_code"] == LEVEL_ONSET


def test_consistency_errors():
    kb = json.loads(KB.read_text(encoding="utf-8"))
    bad = copy.deepcopy(kb)
    bad["rules"][0]["dosage_guide"][0]["product_code"] = "NOT_EXIST"
    with pytest.raises(KnowledgeBaseError):
        KnowledgeEngine(bad)
    bad = copy.deepcopy(kb)
    bad["default_severity_thresholds"] = {"onset_upper_exclusive": 20, "light_upper_inclusive": 15}
    with pytest.raises(KnowledgeBaseError):
        KnowledgeEngine(bad)


def test_every_model_class_has_rule(engine):
    names = yaml.safe_load((ROOT / "ai/configs/dataset.yaml").read_text(encoding="utf-8"))["names"]
    for n in names.values():
        if n != "leaf":
            assert engine.rule_for_class(n) is not None, n
