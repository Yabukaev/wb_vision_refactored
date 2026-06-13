# One-command setup for Windows PowerShell.
# Run from anywhere (clones the repo if needed):
#   irm https://raw.githubusercontent.com/Yabukaev/wb_vision_refactored/main/scripts/bootstrap.ps1 | iex
$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Yabukaev/wb_vision_refactored.git"
$RepoDir = "wb_vision_refactored"
$Branch  = "main"

function Find-Python {
    foreach ($c in @("py -3", "python", "python3")) {
        if (Get-Command $c.Split(" ")[0] -ErrorAction SilentlyContinue) { return $c }
    }
    throw "Python 3 not found. Install it from https://www.python.org/downloads/ and re-run."
}

# Locate the project: already inside it, next to this script, or clone fresh.
$root = $null
if (Test-Path "app\main.py") {
    $root = (Get-Location).Path
} elseif ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "..\app\main.py"))) {
    $root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if (-not $root) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git not found. Install Git for Windows." }
    if (-not (Test-Path $RepoDir)) { git clone --branch $Branch $RepoUrl $RepoDir }
    $root = (Resolve-Path $RepoDir).Path
}
Set-Location $root
Write-Host "Project: $root" -ForegroundColor Cyan

$py = Find-Python
if (-not (Test-Path ".venv")) { Invoke-Expression "$py -m venv .venv" }
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip | Out-Null

# Hardware detection: NVIDIA GPU via nvidia-smi -> CUDA wheels, else CPU.
$gpu = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    & nvidia-smi *> $null
    if ($LASTEXITCODE -eq 0) { $gpu = $true }
}
Write-Host ("Hardware: " + $(if ($gpu) { "NVIDIA GPU -> CUDA" } else { "no GPU -> CPU" })) -ForegroundColor Cyan

if ($gpu) {
    pip install -r requirements-gpu.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "GPU PyTorch install failed - falling back to CPU."
        pip install -r requirements-cpu.txt
    }
} else {
    pip install -r requirements-cpu.txt
}
pip install -r requirements.txt

if ((-not (Test-Path ".env")) -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example - edit RTSP_URL / MQTT before running." -ForegroundColor Yellow
}

Write-Host "Running smoke test..." -ForegroundColor Cyan
python scripts\smoke_test.py
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed - see output above." }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "1) Edit .env  (RTSP_URL, MQTT_*)" -ForegroundColor Green
Write-Host "2) Start:  cd `"$root`"; .\scripts\serve.ps1" -ForegroundColor Green
Write-Host "   Web UI:  http://localhost:8000" -ForegroundColor Green
