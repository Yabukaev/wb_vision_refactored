"""Deployment smoke check: key modules import and the config parses."""
from __future__ import annotations

import importlib


def test_key_modules_import():
    for m in (
        "app.config", "app.vision.calibration", "app.vision.inference_worker",
        "app.web.server", "app.mqtt.discovery", "app.runtime_tuning",
        "app.runtime_store", "app.model_registry",
    ):
        importlib.import_module(m)


def test_config_parses_with_dummy_env(monkeypatch):
    monkeypatch.setenv("RTSP_URL", "rtsp://dummy/stream")
    from app.config import ConfigManager
    settings = ConfigManager("configs/config.yaml").get()
    assert settings.web.port > 0
    assert settings.camera is not None and settings.vision is not None
