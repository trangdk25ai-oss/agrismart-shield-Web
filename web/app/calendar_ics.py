"""Sinh tệp lịch .ics (RFC 5545) cho lịch phun, để nhập vào ứng dụng Lịch.

Thay cho thông báo nhắc lịch của ứng dụng di động: trang web không chạy nền được trên điện thoại, còn sự kiện
đã nhập vào ứng dụng Lịch thì vẫn nhắc đúng ngày kể cả khi không mở trang web. Việc mở tệp .ics trên điện thoại
phụ thuộc ứng dụng lịch đang cài (tài liệu chính thức của Google Calendar hướng dẫn nhập .ics trên máy tính) –
kiểm tra trước trên chính điện thoại dùng Demo; lịch phun luôn hiển thị trên trang dù có nhập được hay không.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone


def _escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n"))


def _fold(line: str) -> list[str]:
    """Gấp dòng dài hơn 75 octet (UTF-8), không cắt giữa ký tự nhiều byte."""
    out, cur, size = [], "", 0
    for ch in line:
        n = len(ch.encode("utf-8"))
        limit = 75 if not out else 74  # dòng tiếp theo bắt đầu bằng 1 dấu cách
        if size + n > limit:
            out.append(cur)
            cur, size = "", 0
        cur += ch
        size += n
    out.append(cur)
    return [out[0]] + [" " + x for x in out[1:]]


def build_ics(rec: dict, now: datetime | None = None) -> str:
    """rec: kết quả KnowledgeEngine.recommend() có spray_schedule."""
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    title = rec.get("disease_name") or "Phun thuốc"
    crop = rec.get("crop_name") or rec.get("crop_code", "")
    desc_lines = [f"{crop} – {title} – mức {rec.get('level_label', '')} ({rec.get('severity_pct', '')}%)"]
    desc_lines += [f"- {a}" for a in rec.get("actions", [])]
    desc_lines += [f"* {d['text']}" + ("" if d.get("verified") else " (liều MẪU, chưa thẩm định)") for d in rec.get("dosage", [])]
    desc_lines += [f"! {w}" for w in rec.get("warnings", [])]
    desc_lines.append("Tạo bởi AgriSmart Shield – đối chiếu nhãn thuốc và hỏi cán bộ BVTV trước khi phun.")
    description = "\n".join(desc_lines)

    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AgriSmart Shield//Lich phun//VI", "CALSCALE:GREGORIAN",
             "METHOD:PUBLISH"]
    schedule = rec.get("spray_schedule", [])
    for s in schedule:
        d = date.fromisoformat(s["date"])
        uid_src = f"{rec.get('rule_id')}|{s['date']}|{s['application']}|{stamp}"
        uid = hashlib.sha1(uid_src.encode("utf-8")).hexdigest()[:20] + "@agrismart-shield"
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{d:%Y%m%d}",
            f"DTEND;VALUE=DATE:{d + timedelta(days=1):%Y%m%d}",
            "SUMMARY:" + _escape(f"Phun lần {s['application']}/{len(schedule)} – {title} ({crop})"),
            f"DESCRIPTION:{_escape(description)}",
            "BEGIN:VALARM",
            "ACTION:DISPLAY",
            "DESCRIPTION:" + _escape(f"Phun lần {s['application']} – {title}"),
            "TRIGGER:PT7H",  # 7 giờ sáng của ngày phun (sự kiện cả ngày bắt đầu lúc 0 giờ)
            "END:VALARM",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    folded: list[str] = []
    for ln in lines:
        folded += _fold(ln)
    return "\r\n".join(folded) + "\r\n"
