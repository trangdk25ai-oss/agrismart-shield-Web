"""BƯỚC 6 (thay thực địa) – Gộp các kết quả kiểm thử thành MỘT báo cáo Markdown cho hồ sơ dự thi.

    python ai/scripts/s11_test_report.py --reports reports --out reports/bao_cao_kiem_thu.md

Đọc (tệp nào chưa có thì ghi "Chưa có số liệu", không suy đoán):
    reports/eval_fp32.json, reports/eval_int8.json   (s06)      reports/compare.json   (s08 compare)
    reports/eval_plantdoc.json                       (s06b)     reports/latency.json   (s08 latency)
    reports/robustness.json                          (s09 A)    reports/recapture.json (s09 B)
    reports/samples.csv + demo/danh_gia_chuyen_gia.csv          (s09 C + phiếu chuyên gia, tùy chọn)

Ngưỡng: hai chỉ tiêu < 10 MB và mAP giảm ≤ 2 điểm % lấy từ quy trình gốc (Bước 3); chỉ tiêu < 100 ms của
quy trình gốc đo trên điện thoại, nay đo trên máy chạy Demo. Các ngưỡng còn lại là ĐỀ XUẤT của nhóm,
cần thống nhất trước khi đưa vào hồ sơ.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agrismart.robustness import BLUR_CONDITIONS, to_markdown  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEVEL_WORDS = {"khoe": "HEALTHY", "khong": "HEALTHY", "khoi phat": "ONSET", "nhe": "LIGHT", "nang": "HEAVY"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def plain(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower().strip()


def level_code(text: str) -> str | None:
    t = plain(text)
    if t.upper() in ("HEALTHY", "ONSET", "LIGHT", "HEAVY"):
        return t.upper()
    for k, v in LEVEL_WORDS.items():
        if k in t:
            return v
    return None


def num(v, nd=2, pct=False):
    if v is None:
        return "–"
    return (f"{v * 100:.0f}%" if pct else f"{v:.{nd}f}").replace(".", ",")


def verdict(ok):
    return "Chưa có số liệu" if ok is None else ("**Đạt**" if ok else "**Chưa đạt**")


def expert_compare(samples_csv: Path, expert_csv: Path) -> dict | None:
    if not (samples_csv.is_file() and expert_csv.is_file()):
        return None
    sys_rows = {r["sample_id"]: r for r in csv.DictReader(samples_csv.open(encoding="utf-8-sig"))}
    pairs = []
    for r in csv.DictReader(expert_csv.open(encoding="utf-8-sig")):
        s = sys_rows.get((r.get("sample_id") or "").strip())
        try:
            e_pct = float((r.get("chuyen_gia_severity_pct") or "").replace(",", "."))
        except ValueError:
            continue
        if s is None:
            continue
        pairs.append((s, e_pct, level_code(r.get("chuyen_gia_muc_do", ""))))
    if not pairs:
        return None
    sys_err = [abs(float(s["pr_pct"]) - e) for s, e, _ in pairs if s.get("pr_pct")]
    lbl_err = [abs(float(s["gt_pct"]) - e) for s, e, _ in pairs]
    lv = [(s.get("pr_level") == lc) for s, _, lc in pairs if lc]
    return {"n": len(pairs), "mae_system_vs_expert": sum(sys_err) / len(sys_err) if sys_err else None,
            "mae_label_vs_expert": sum(lbl_err) / len(lbl_err), "level_agreement": sum(lv) / len(lv) if lv else None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--expert", type=Path, default=ROOT / "demo/danh_gia_chuyen_gia.csv")
    ap.add_argument("--out", type=Path, default=Path("reports/bao_cao_kiem_thu.md"))
    a = ap.parse_args()
    R = a.reports
    fp32, int8 = load(R / "eval_fp32.json"), load(R / "eval_int8.json")
    cmp_, lat = load(R / "compare.json"), load(R / "latency.json")
    rob, rec, pdoc = load(R / "robustness.json"), load(R / "recapture.json"), load(R / "eval_plantdoc.json")
    exp = expert_compare(R / "samples.csv", a.expert)

    ev = int8 or fp32
    sev = (ev or {}).get("severity") or {}
    rows = []
    size = lat.get("size_mb") if lat else None
    rows.append(("Kích thước mô hình INT8", "Quy trình 3.1", "< 10 MB", f"{num(size)} MB" if size else "–",
                 None if size is None else size < 10))
    drop = cmp_["details"]["mask_map50_95"]["drop_abs_points"] if cmp_ else None
    rows.append(("Mask mAP50-95 giảm sau INT8", "Quy trình 3.2", "≤ 2 điểm %", f"{num(drop)} điểm %" if drop is not None else "–",
                 None if drop is None else drop <= 2.0))
    inf = lat["ms"]["inference"]["mean"] if lat else None
    rows.append(("Thời gian suy luận (máy chạy Demo)", "Quy trình 3.2 (gốc: điện thoại)", "< 100 ms",
                 f"{num(inf, 0)} ms" if inf is not None else "–", None if inf is None else inf < 100))
    mae = sev.get("mae_pct_points")
    rows.append(("Sai số Severity trên tập kiểm tra (MAE)", "Đề xuất", "≤ 3 điểm %", num(mae), None if mae is None else mae <= 3))
    acc = sev.get("level_accuracy")
    rows.append(("Đúng mức xử lý trên tập kiểm tra", "Đề xuất", "≥ 85%", num(acc, pct=True), None if acc is None else acc >= 0.85))
    if rob:
        s = rob["summary"]
        cls_acc = s.get("goc", {}).get("class_accuracy")
        rows.append(("Nhận đúng bệnh (ảnh gốc tập kiểm tra)", "Đề xuất", "≥ 85%", num(cls_acc, pct=True),
                     None if cls_acc is None else cls_acc >= 0.85))
        hard = [v["level_accuracy"] for k, v in s.items() if k not in BLUR_CONDITIONS]
        worst = min(hard) if hard else None
        rows.append(("Độ bền: đúng mức thấp nhất (trừ ảnh mờ nặng)", "Đề xuất", "≥ 75%", num(worst, pct=True),
                     None if worst is None else worst >= 0.75))
        det = min((s[k]["blur_flag_rate"] for k in BLUR_CONDITIONS if k in s), default=None)
        fa = s.get("goc", {}).get("blur_flag_rate")
        ok = None if det is None or fa is None else (det >= 0.9 and fa <= 0.1)
        rows.append(("Blur Check: phát hiện ảnh mờ / báo nhầm ảnh nét", "Đề xuất", "≥ 90% / ≤ 10%",
                     f"{num(det, pct=True)} / {num(fa, pct=True)}", ok))
    else:
        rows.append(("Nhận đúng bệnh (ảnh gốc tập kiểm tra)", "Đề xuất", "≥ 85%", "–", None))
        rows.append(("Độ bền (điều kiện chụp mô phỏng)", "Đề xuất", "đúng mức ≥ 75%", "–", None))
    if rec:
        v = rec["summary"].get("chup_lai") or next(iter(rec["summary"].values()))
        ok = None if v["mae"] is None else (v["mae"] <= 5 and v["level_accuracy"] >= 0.7)
        rows.append(("Chụp lại ảnh in bằng điện thoại", "Đề xuất", "MAE ≤ 5; đúng mức ≥ 70%",
                     f"MAE {num(v['mae'])}; {num(v['level_accuracy'], pct=True)}", ok))
    else:
        rows.append(("Chụp lại ảnh in bằng điện thoại", "Đề xuất", "MAE ≤ 5; đúng mức ≥ 70%", "–", None))
    if exp:
        ok = None if exp["mae_system_vs_expert"] is None else exp["mae_system_vs_expert"] <= 5
        rows.append(("So với chuyên gia BVTV (ảnh mẫu)", "Đề xuất, tùy chọn", "MAE ≤ 5 điểm %",
                     f"MAE {num(exp['mae_system_vs_expert'])} (n = {exp['n']})", ok))

    lines = ["# Báo cáo kiểm thử AgriSmart Shield (thay thế kiểm thử thực địa)", "",
             f"Lập tự động ngày {datetime.now():%d/%m/%Y %H:%M} bằng `ai/scripts/s11_test_report.py`.", "",
             "## 1. Phạm vi và phương pháp", "",
             "Không thực hiện kiểm thử ngoài ruộng. Thay thế bằng: (1) đánh giá trên tập kiểm tra độc lập; "
             "(2) kiểm thử độ bền với 10 điều kiện chụp mô phỏng; (3) chụp lại ảnh mẫu đã in bằng điện thoại; "
             "(4) đo tốc độ trên máy chạy Demo; (5) tùy chọn: chuyên gia BVTV đánh giá mù bộ ảnh mẫu.", "",
             "**Giới hạn:** ảnh huấn luyện và kiểm tra chủ yếu từ bộ dữ liệu công khai nước ngoài; chưa có ảnh "
             "chụp tại ruộng ở Đà Nẵng nên kết quả chưa đại diện cho điều kiện địa phương.", "",
             "## 2. Chỉ tiêu nghiệm thu", "",
             "| Chỉ tiêu | Căn cứ ngưỡng | Ngưỡng | Kết quả | Đánh giá |", "|---|---|---|---|---|"]
    lines += [f"| {a_} | {b} | {c} | {d} | {verdict(e)} |" for a_, b, c, d, e in rows]
    lines += ["", "## 3. Chi tiết", ""]
    if ev:
        lines += [f"Tập kiểm tra (`{ev.get('weights')}`): Box mAP50-95 {num(ev.get('box_map50_95'), 3)}, "
                  f"Mask mAP50-95 {num(ev.get('mask_map50_95'), 3)}, Mean IoU vết bệnh "
                  f"{num((ev.get('iou') or {}).get('mean_iou_lesions'), 3)}, số ảnh tính Severity {sev.get('n', '–')}.", ""]
    if pdoc:
        lines += ["Kiểm tra độc lập trên PlantDoc (mức lá, hộp bao): xem `reports/eval_plantdoc.json`.", ""]
    if lat:
        m = lat["machine"]
        lines += [f"Tốc độ đo trên: {m['platform']}, {m['processor']}, {m['cpu_count']} luồng CPU; "
                  f"mô hình {lat['model']} ({lat['backend']}). Tổng tiền xử lý + suy luận + hậu xử lý trung bình "
                  f"{num(lat['ms']['total']['mean'], 0)} ms.", ""]
    if rob:
        lines += [to_markdown(rob["summary"], rob.get("blur"), "Độ bền (điều kiện chụp mô phỏng)"), ""]
    if rec:
        lines += [to_markdown(rec["summary"], None, "Chụp lại ảnh in / màn hình"), ""]
    if exp:
        lines += ["## So với chuyên gia", "",
                  f"{exp['n']} ảnh mẫu. MAE hệ thống – chuyên gia: {num(exp['mae_system_vs_expert'])} điểm %; "
                  f"MAE đáp án từ nhãn – chuyên gia: {num(exp['mae_label_vs_expert'])} điểm %; mức xử lý trùng "
                  f"{num(exp['level_agreement'], pct=True)}.", ""]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(lines), "utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
