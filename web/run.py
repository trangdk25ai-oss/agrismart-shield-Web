"""Khởi động máy chủ trang web AgriSmart Shield.

    python web/run.py                         # tự tìm mô hình trong web/models/, không có thì chạy chế độ thử
    python web/run.py --model web/models/model_int8.onnx
    python web/run.py --host 127.0.0.1        # chỉ mở trên máy này (không cho điện thoại truy cập)

Mặc định lắng nghe 0.0.0.0:8000 để điện thoại cùng mạng Wi-Fi / điểm phát sóng mở được bằng địa chỉ
in ra màn hình (ví dụ http://192.168.1.5:8000). Không cần Internet: chỉ cần máy tính và điện thoại
chung một mạng cục bộ.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "ai"))


def lan_addresses() -> list[str]:
    """Địa chỉ IPv4 của máy trong mạng cục bộ (không gửi gói tin nào ra ngoài)."""
    ips: set[str] = set()
    for target in ("10.255.255.255", "192.168.255.255", "172.31.255.255"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((target, 1))  # UDP connect chỉ chọn giao diện mạng, không gửi dữ liệu
                ips.add(s.getsockname()[0])
        except OSError:
            pass
    try:
        ips.update(a for a in socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127.") and not ip.startswith("169.254."))


def running_in_wsl() -> bool:
    """Đang chạy trong Windows Subsystem for Linux (WSL đặt biến WSL_DISTRO_NAME; nhân Linux có chữ 'microsoft')."""
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="tệp mô hình .onnx / .tflite / .pt (mặc định: tìm trong web/models/)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--conf", type=float, default=None, help="ngưỡng tin cậy (mặc định 0,25)")
    ap.add_argument("--blur-threshold", type=float, default=None, help="ngưỡng Blur Check (mặc định 100)")
    ap.add_argument("--samples", default=None, help="thư mục ảnh mẫu Demo (mặc định demo/samples)")
    ap.add_argument("--reload", action="store_true", help="tự khởi động lại khi sửa mã (khi phát triển)")
    a = ap.parse_args()

    if a.model:
        os.environ["AGS_MODEL"] = str(Path(a.model).resolve())
    if a.conf is not None:
        os.environ["AGS_CONF"] = str(a.conf)
    if a.blur_threshold is not None:
        os.environ["AGS_BLUR_THRESHOLD"] = str(a.blur_threshold)
    if a.samples:
        os.environ["AGS_SAMPLES_DIR"] = str(Path(a.samples).resolve())
    urls = [f"http://{ip}:{a.port}" for ip in lan_addresses()] if a.host == "0.0.0.0" else []
    os.environ["AGS_LAN_URLS"] = ",".join(urls)

    try:
        import uvicorn
    except ModuleNotFoundError:
        print("LỖI: môi trường Python đang dùng chưa cài thư viện máy chủ web (uvicorn, fastapi…).")
        print(f"  Python đang dùng: {sys.executable}")
        print("  Cài vào chính môi trường này:  python -m pip install -r web/requirements.txt")
        print("  (hoặc trên Ubuntu: make setup, rồi make web)")
        sys.exit(1)

    print("=" * 64)
    print(f"  AgriSmart Shield – mở trên máy này:  http://127.0.0.1:{a.port}")
    for u in urls:
        print(f"  Mở trên điện thoại (cùng Wi-Fi):     {u}")
    if a.host == "0.0.0.0" and not urls:
        print("  Chưa thấy địa chỉ mạng cục bộ: kiểm tra Wi-Fi / điểm phát sóng.")
    if a.host == "0.0.0.0" and running_in_wsl():
        print("  [WSL] Đang chạy trong WSL: trên máy này mở http://127.0.0.1:%d bằng trình duyệt Windows." % a.port)
        print("  [WSL] Điện thoại chỉ vào được khi WSL dùng mạng 'mirrored' hoặc đã chuyển cổng (netsh portproxy).")
        print("        Ở chế độ mặc định (NAT), địa chỉ in ở trên là mạng nội bộ của WSL. Xem web/README.md, mục WSL.")
    print("  Dừng máy chủ: Ctrl+C")
    print("=" * 64, flush=True)
    uvicorn.run("web.app.main:create_app", factory=True, host=a.host, port=a.port, reload=a.reload,
                reload_dirs=[str(HERE), str(HERE.parent / "ai" / "agrismart")] if a.reload else None, log_level="info")


if __name__ == "__main__":
    main()
