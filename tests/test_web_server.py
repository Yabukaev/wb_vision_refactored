"""Phase 2: FastAPI control UI endpoints (state + calibration mutations)."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.config import ConfigManager, WebSection
from app.core.latest_value import LatestValue
from app.vision.calibration import CalibrationManager
from app.web.server import build_app

CONFIG_YAML = """
camera:
  rtsp_url: rtsp://dummy/stream
calibration:
  file: {cal_file}
"""


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    calibration = CalibrationManager(ConfigManager(cfg))
    app = build_app(LatestValue(), LatestValue(), calibration, WebSection())
    c = TestClient(app)
    c.calibration = calibration  # type: ignore[attr-defined]
    return c


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "WB Vision" in r.text


def test_state_shape(client):
    r = client.get("/api/state")
    assert r.status_code == 200
    data = r.json()
    assert "calibration" in data and "status" in data and "source" in data
    assert data["calibration"]["trap_angles_deg"] == [90, 90, 90, 90]


def test_set_aim(client):
    assert client.post("/api/aim", json={"x": 111, "y": 222}).json()["ok"]
    cal = client.calibration.snapshot()
    assert cal.aim_px == 111 and cal.aim_py == 222


def test_full_trapezoid_via_api(client):
    quad = [(0, 0), (100, 0), (100, 100), (0, 100)]
    for i, (x, y) in enumerate(quad):
        client.post("/api/quad_point", json={"index": i, "x": x, "y": y})
    for i, v in enumerate([2.0, 2.0, 2.0, 2.0]):
        client.post("/api/edge", json={"index": i, "value": v})
    state = client.get("/api/state").json()["calibration"]
    assert len(state["quad_px"]) == 4
    assert state["closure_error_m"] == pytest.approx(0.0, abs=1e-6)


def test_set_value_rejects_unknown_key(client):
    r = client.post("/api/value", json={"key": "nope", "value": 1})
    assert r.status_code == 400
    assert r.json()["ok"] is False


def test_zone_add_and_delete(client):
    client.post("/api/zone/add", json={"name": "shower", "polygon_px": [[1, 1], [9, 1], [5, 9]]})
    zones = client.get("/api/state").json()["calibration"]["zones"]
    assert len(zones) == 1 and zones[0]["name"] == "shower"
    client.post("/api/zone/delete", json={"index": 0})
    assert client.get("/api/state").json()["calibration"]["zones"] == []
