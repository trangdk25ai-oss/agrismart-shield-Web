# AgriSmart Shield – Phần 1 trên Windows (không cần GPU):
#   tạo môi trường ảo .venv, cài thư viện (gồm máy chủ web), chạy kiểm thử, kiểm tra tri thức IPM,
#   chạy thử pipeline dữ liệu tổng hợp. Sau đó chạy được trang web: .\.venv\Scripts\python web\run.py
#
# Cách chạy (mở PowerShell tại thư mục dự án):
#   powershell -ExecutionPolicy Bypass -File scripts\phan1_windows.ps1
# Tệp lưu dạng UTF-8 có BOM để Windows PowerShell 5.1 đọc đúng tiếng Việt.
# Tham số tùy chọn:
#   -SkipSmoke    bỏ bước chạy thử dữ liệu tổng hợp
#   -Python "py -3.12"   chỉ định trình thông dịch Python

param(
    [switch]$SkipSmoke,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = "1"   # in tiếng Việt ra màn hình không lỗi mã hóa

$onWindows = ($env:OS -eq "Windows_NT") -or ($IsWindows -eq $true)
$venvPy = if ($onWindows) { Join-Path ".venv" "Scripts\python.exe" } else { Join-Path ".venv" "bin/python" }

function Invoke-Step {
    param([string]$Title, [string[]]$Cmd)
    Write-Host ""
    Write-Host "==> $Title" -ForegroundColor Green
    $exe = $Cmd[0]
    $rest = @()
    if ($Cmd.Length -gt 1) { $rest = $Cmd[1..($Cmd.Length - 1)] }
    & $exe @rest
    if ($LASTEXITCODE -ne 0) { throw "Bước '$Title' thất bại (mã lỗi $LASTEXITCODE)" }
}

function Find-Python {
    # Windows PowerShell 5.1 coi dòng stderr của lệnh ngoài là lỗi khi ErrorActionPreference = Stop
    $ErrorActionPreference = "Continue"
    if ($Python -ne "") { return ,($Python -split " ") }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10", "3.13")) {
            $null = & py "-$v" -c "import sys" 2>&1
            if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
        }
    }
    foreach ($name in @("python", "python3")) {
        if (Get-Command $name -ErrorAction SilentlyContinue) {
            # loại lối tắt "python" của Microsoft Store (chưa cài Python thật)
            $null = & $name -c "import sys" 2>&1
            if ($LASTEXITCODE -eq 0) { return ,@($name) }
        }
    }
    throw "Không tìm thấy Python. Cài Python 3.12 từ https://www.python.org/downloads/ (tích 'Add python.exe to PATH')."
}

if (-not (Test-Path $venvPy)) {
    $py = Find-Python
    Invoke-Step "Tạo môi trường ảo .venv bằng $($py -join ' ')" ($py + @("-m", "venv", ".venv"))
}
Invoke-Step "Nâng cấp pip" @($venvPy, "-m", "pip", "install", "--upgrade", "pip")
Invoke-Step "Cài thư viện Phần 1 + trang web (web/requirements.txt)" @($venvPy, "-m", "pip", "install", "-r", "web/requirements.txt")
Invoke-Step "Chạy kiểm thử (pytest)" @($venvPy, "-m", "pytest", "ai/tests", "web/tests", "-q")
Invoke-Step "Kiểm tra cơ sở tri thức IPM" @($venvPy, "knowledge_base/validate_kb.py")

if (-not $SkipSmoke) {
    $s = ".smoke"
    if (Test-Path $s) { Remove-Item -Recurse -Force $s }
    Invoke-Step "Sinh dữ liệu tổng hợp" @($venvPy, "ai/scripts/make_synthetic_dataset.py", "--out", "$s/raw", "--n", "30")
    Invoke-Step "s01 – chuyển mask sang nhãn YOLOv8-seg" @($venvPy, "ai/scripts/s01_convert_masks_to_yolo.py",
        "--images", "$s/raw/images", "--lesion-masks", "$s/raw/masks", "--mask-mode", "binary",
        "--class-map", "ai/configs/class_map.yaml", "--bootstrap-leaf", "--out", "$s/all")
    Invoke-Step "s02 – chia tập 70/20/10" @($venvPy, "ai/scripts/s02_split_dataset.py", "--src", "$s/all", "--out", "$s/ds",
        "--class-map", "ai/configs/class_map.yaml", "--group-regex", '^.*?__(plant\d+)_')
    Invoke-Step "s03 – tăng cường dữ liệu" @($venvPy, "ai/scripts/s03_augment_train.py", "--data", "$s/ds", "--copies", "2")
    Invoke-Step "s04 – kiểm tra bộ dữ liệu" @($venvPy, "ai/scripts/s04_check_dataset.py", "--data-yaml", "$s/ds/dataset.yaml")
}

Write-Host ""
Write-Host "HOÀN TẤT PHẦN 1. Kích hoạt môi trường cho các lệnh tiếp theo:" -ForegroundColor Green
if ($onWindows) { Write-Host "  .\.venv\Scripts\Activate.ps1" } else { Write-Host "  source .venv/bin/activate" }
Write-Host "Chạy thử trang web (chế độ thử, chưa có mô hình):" -ForegroundColor Green
if ($onWindows) { Write-Host "  .\.venv\Scripts\python web\run.py   rồi mở http://127.0.0.1:8000" } else { Write-Host "  .venv/bin/python web/run.py" }
