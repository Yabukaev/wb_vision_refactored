from __future__ import annotations

import pytest

from app.config import ConfigManager, _expand_env


def test_expand_env_keeps_numeric_strings_as_strings(monkeypatch):
    monkeypatch.setenv("TEST_MQTT_PASSWORD", "12345")
    assert _expand_env("${TEST_MQTT_PASSWORD}") == "12345"


def test_expand_env_keeps_leading_zeros(monkeypatch):
    monkeypatch.setenv("TEST_PIN", "007")
    assert _expand_env("${TEST_PIN}") == "007"


def test_expand_env_uses_default_when_var_missing(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_VAR", raising=False)
    assert _expand_env("${TEST_MISSING_VAR:fallback}") == "fallback"


def test_expand_env_missing_var_without_default_is_empty(monkeypatch):
    monkeypatch.delenv("TEST_MISSING_VAR", raising=False)
    assert _expand_env("${TEST_MISSING_VAR}") == ""


def test_expand_env_recurses_into_mappings(monkeypatch):
    monkeypatch.setenv("TEST_HOST", "broker.local")
    raw = {"mqtt": {"host": "${TEST_HOST}", "port": 1883}}
    assert _expand_env(raw) == {"mqtt": {"host": "broker.local", "port": 1883}}


MINIMAL_YAML = """
camera:
  rtsp_url: "{rtsp_url}"
"""


def _write_config(tmp_path, rtsp_url: str):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(MINIMAL_YAML.format(rtsp_url=rtsp_url), encoding="utf-8")
    return cfg


def test_config_manager_rejects_empty_rtsp_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, "")
    with pytest.raises(ValueError, match="rtsp_url"):
        ConfigManager(cfg)


def test_config_manager_accepts_valid_rtsp_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(tmp_path, "rtsp://user:pass@10.0.0.2:554/stream")
    settings = ConfigManager(cfg).get()
    assert settings.camera.rtsp_url.startswith("rtsp://")
