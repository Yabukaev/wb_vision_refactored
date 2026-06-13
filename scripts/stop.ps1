# Stop the WB Vision background service started by scripts\serve.ps1.
$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root ".service.pid"

if (Test-Path $pidFile) {
    $id = Get-Content $pidFile
    if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $id -Force
        Write-Host "WB Vision stopped (PID $id)"
    } else {
        Write-Host "Process $id not running."
    }
    Remove-Item $pidFile -Force
} else {
    Write-Host "No .service.pid found; nothing to stop."
}
