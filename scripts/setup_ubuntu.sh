#!/usr/bin/env bash
# AgriSmart Shield – Phần 1 trên Ubuntu 22.04 / 24.04 (không cần GPU):
# cài gói hệ thống còn thiếu, tạo .venv, cài thư viện (gồm máy chủ web), chạy kiểm thử và chạy thử
# pipeline dữ liệu tổng hợp. Sau đó chạy được ngay trang web ở chế độ thử: make web
#
#   bash scripts/setup_ubuntu.sh            # Phần 1 đầy đủ
#   bash scripts/setup_ubuntu.sh --no-smoke # bỏ bước chạy thử dữ liệu tổng hợp
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=1
[[ "${1:-}" == "--no-smoke" ]] && SMOKE=0

need_apt=()
command -v python3 >/dev/null 2>&1 || need_apt+=(python3)
# Ubuntu tách venv/ensurepip ra gói python3-venv; thiếu gói này thì "python3 -m venv" báo lỗi
python3 -c "import ensurepip, venv" >/dev/null 2>&1 || need_apt+=(python3-venv)
command -v make >/dev/null 2>&1 || need_apt+=(make)
command -v git >/dev/null 2>&1 || need_apt+=(git)
if ((${#need_apt[@]})); then
  echo "==> Cài gói hệ thống: ${need_apt[*]} (cần quyền sudo)"
  sudo apt-get update
  sudo apt-get install -y "${need_apt[@]}"
fi

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Cần Python >= 3.10, đang có {sys.version.split()[0]}. Ubuntu 22.04 trở lên đáp ứng sẵn.")
print("==> Python", sys.version.split()[0])
PY

echo "==> Tạo .venv và cài thư viện Phần 1 + trang web"
make setup
echo "==> Kiểm thử + kiểm tra tri thức IPM"
make test
make kb
if ((SMOKE)); then
  echo "==> Chạy thử pipeline dữ liệu tổng hợp (s01 -> s04)"
  make smoke
fi
echo
echo "HOÀN TẤT PHẦN 1. Kích hoạt môi trường cho các lệnh tiếp theo:  source .venv/bin/activate"
echo "Chạy thử trang web (chế độ thử, chưa có mô hình):  make web   rồi mở http://127.0.0.1:8000"
