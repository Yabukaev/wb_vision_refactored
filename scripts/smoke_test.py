"""Post-install health check — no camera or real hardware required.

Verifies that all key dependencies import, the app modules load, and the YAML
config parses. Exits 0 on success, 1 on failure. Run from the repo root:

    python scripts/smoke_test.py
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

# Allow running from anywhere — resolve the repo root (parent of scripts/).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

_DEPS = [
    "numpy", "cv2", "yaml", "psutil", "paho.mqtt.client",
    "fastapi", "uvicorn", "ultralytics", "torch",
]
_APP_MODULES = [
    "app.config", "app.types", "app.core.latest_value",
    "app.camera.rtsp_reader", "app.vision.calibration", "app.vision.detector",
    "app.vision.tracker", "app.vision.inference_worker", "app.vision.overlay",
    "app.vision.activity_classifier", "app.mqtt.mqtt_worker", "app.mqtt.discovery",
    "app.runtime_tuning", "app.runtime_store", "app.model_registry", "app.web.server",
]


def run_smoke() -> bool:
    for m in _DEPS + _APP_MODULES:
        importlib.import_module(m)

    import torch
    cuda = bool(torch.cuda.is_available())

    # Config must parse; use a dummy RTSP if none configured yet.
    os.environ.setdefault("RTSP_URL", "rtsp://dummy/stream")
    from app.config import ConfigManager
    settings = ConfigManager("configs/config.yaml").get()
    assert settings.camera is not None
    assert settings.vision is not None
    assert settings.web.port > 0
    print(f"[smoke] device: {'CUDA GPU' if cuda else 'CPU'} (torch {torch.__version__})")
    return cuda


if __name__ == "__main__":
    try:
        run_smoke()
        print("[smoke] OK — dependencies import, app modules load, config parses.")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        print(f"[smoke] FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)
