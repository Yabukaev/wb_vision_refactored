# Start WB Vision as a background service (web UI on http://localhost:8000).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) { & .\scripts\install.ps1 }
. .\.venv\Scripts\Activate.ps1

$pidFile = Join-Path $root ".service.pid"
if (Test-Path $pidFile) {
    $old = Get-Content $pidFile
    if (Get-Process -Id $old -ErrorAction SilentlyContinue) {
        Write-Host "Already running (PID $old). Run scripts\stop.ps1 first."
        exit 0
    }
}

$proc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "app.main", "--config", "configs/config.yaml" `
    -PassThru -WindowStyle Hidden
$proc.Id | Out-File -FilePath $pidFile -Encoding ascii

Write-Host "WB Vision started (PID $($proc.Id))"
Write-Host "Web UI: http://localhost:8000  (opens automatically)"
Write-Host "Stop with: scripts\stop.ps1"
