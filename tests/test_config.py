from __future__ import annotations

import pytest

from app.config import ConfigManager, _cast, _expand_env
from dataclasses import MISSING


# ── _expand_env ──────────────────────────────────────────────────────────────

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


# ── _cast (B-11) ─────────────────────────────────────────────────────────────

def test_cast_string_int_to_int():
    assert _cast("1883", 1883) == 1883
    assert isinstance(_cast("1883", 1883), int)


def test_cast_string_float_to_float():
    result = _cast("5.0", 1.5)
    assert result == pytest.approx(5.0)
    assert isinstance(result, float)


def test_cast_bool_true_variants():
    for val in ("true", "True", "TRUE", "1", "yes", "on"):
        assert _cast(val, True) is True


def test_cast_bool_false_variants():
    for val in ("false", "False", "0", "no", "off"):
        assert _cast(val, True) is False


def test_cast_non_string_passthrough():
    assert _cast(1883, 0) == 1883
    assert _cast(None, 0) is None


def test_cast_missing_default_returns_string():
    assert _cast("hello", MISSING) == "hello"


# ── ConfigManager ─────────────────────────────────────────────────────────────

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


def test_config_unknown_keys_do_not_raise(tmp_path, monkeypatch, caplog):
    """B-12: unknown keys warn but don't crash."""
    monkeypatch.chdir(tmp_path)
    yaml_text = """
camera:
  rtsp_url: rtsp://x/stream
  unknown_field_xyz: 99
"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    import logging
    with caplog.at_level(logging.WARNING, logger="config"):
        ConfigManager(cfg)
    assert "unknown_field_xyz" in caplog.text


def test_config_env_port_cast_to_int(tmp_path, monkeypatch):
    """B-11: MQTT_PORT from env should become int, not str."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MQTT_PORT", "1884")
    yaml_text = """
camera:
  rtsp_url: rtsp://x/stream
mqtt:
  port: ${MQTT_PORT:1883}
"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml_text, encoding="utf-8")
    settings = ConfigManager(cfg).get()
    assert settings.mqtt.port == 1884
    assert isinstance(settings.mqtt.port, int)
