#!/usr/bin/env bash
# One-command setup for Linux / macOS. Run from anywhere (clones if needed):
#   curl -fsSL https://raw.githubusercontent.com/Yabukaev/wb_vision_refactored/main/scripts/bootstrap.sh | bash
set -euo pipefail

REPO_URL="https://github.com/Yabukaev/wb_vision_refactored.git"
REPO_DIR="wb_vision_refactored"
BRANCH="main"

PY=""
for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PY="$c"; break; }; done
[ -n "$PY" ] || { echo "Python 3 not found. Install it and re-run."; exit 1; }

# Locate the project: already inside it, next to this script, or clone fresh.
if [ -f "app/main.py" ]; then
    ROOT="$(pwd)"
elif [ -n "${BASH_SOURCE:-}" ] && [ -f "$(dirname "${BASH_SOURCE[0]}")/../app/main.py" ]; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    command -v git >/dev/null 2>&1 || { echo "git not found. Install git and re-run."; exit 1; }
    [ -d "$REPO_DIR" ] || git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    ROOT="$(cd "$REPO_DIR" && pwd)"
fi
cd "$ROOT"
echo "Project: $ROOT"

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

# Hardware detection: NVIDIA GPU via nvidia-smi -> CUDA wheels, else CPU.
GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then GPU=1; fi
[ "$GPU" = "1" ] && echo "Hardware: NVIDIA GPU -> CUDA" || echo "Hardware: no GPU -> CPU"

if [ "$GPU" = "1" ]; then
    pip install -r requirements-gpu.txt || { echo "GPU PyTorch failed - CPU fallback"; pip install -r requirements-cpu.txt; }
else
    pip install -r requirements-cpu.txt
fi
pip install -r requirements.txt

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example - edit RTSP_URL / MQTT before running."
fi

echo "Running smoke test..."
python scripts/smoke_test.py

echo ""
echo "Setup complete."
echo "1) Edit .env  (RTSP_URL, MQTT_*)"
echo "2) Start:  cd \"$ROOT\" && . .venv/bin/activate && python -m app.main --config configs/config.yaml"
echo "   Web UI: http://localhost:8000"
