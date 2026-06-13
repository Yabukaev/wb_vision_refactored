$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
  & .\scripts\install.ps1
}

. .\.venv\Scripts\Activate.ps1
python -m compileall app
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python -m pytest tests/ -q

