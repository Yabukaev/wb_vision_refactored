"""Phase 2: FastAPI control UI endpoints (state + calibration mutations)."""
from __future__ import annotations

import threading

import pytest
from fastapi.testclient import TestClient

from app.config import (
    ActivitySection, ConfigManager, TrackerSection, VisionSection, WebSection,
)
from app.core.latest_value import LatestValue
from app.runtime_store import RuntimeStore
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
    vision, tracker = VisionSection(), TrackerSection()
    app = build_app(LatestValue(), LatestValue(), calibration, WebSection(), vision, tracker)
    c = TestClient(app)
    c.calibration = calibration  # type: ignore[attr-defined]
    c.vision = vision  # type: ignore[attr-defined]
    c.tracker = tracker  # type: ignore[attr-defined]
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


def test_state_includes_tuning(client):
    data = client.get("/api/state").json()
    assert "tuning" in data and "tuning_specs" in data
    assert data["tuning"]["inference_fps"] == client.vision.inference_fps


def test_tuning_updates_live_cfg(client):
    r = client.post("/api/tuning", json={"key": "inference_fps", "value": 8})
    assert r.json()["ok"] and client.vision.inference_fps == 8.0
    client.post("/api/tuning", json={"key": "match_distance_px", "value": 150})
    assert client.tracker.match_distance_px == 150.0


def test_tuning_clamps_and_rejects(client):
    hi = client.post("/api/tuning", json={"key": "inference_fps", "value": 9999}).json()
    assert hi["value"] == 30.0
    bad = client.post("/api/tuning", json={"key": "nope", "value": 1})
    assert bad.status_code == 400


class _InferenceStub:
    def __init__(self):
        self.pose_req = None
        self.object_req = None

    def request_pose_model(self, p):
        self.pose_req = p

    def request_object_model(self, p):
        self.object_req = p

    def current_models(self):
        return {"pose": "yolo11n-pose.pt", "object": "yolo11n.pt"}


def _client_with_inference(tmp_path):
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    calibration = CalibrationManager(ConfigManager(cfg))
    stub = _InferenceStub()
    app = build_app(LatestValue(), LatestValue(), calibration, WebSection(),
                    VisionSection(), TrackerSection(), stub)
    c = TestClient(app)
    c.stub = stub  # type: ignore[attr-defined]
    return c


def test_models_endpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _client_with_inference(tmp_path)
    data = c.get("/api/models").json()
    assert "available" in data and "current" in data
    assert data["current"]["pose"] == "yolo11n-pose.pt"


def test_set_model_requests_swap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _client_with_inference(tmp_path)
    assert c.post("/api/model", json={"kind": "pose", "path": "yolo11s-pose.pt"}).json()["ok"]
    assert c.stub.pose_req == "yolo11s-pose.pt"
    c.post("/api/model", json={"kind": "object", "path": "yolo11s.pt"})
    assert c.stub.object_req == "yolo11s.pt"


def test_set_model_rejects_bad_kind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = _client_with_inference(tmp_path)
    assert c.post("/api/model", json={"kind": "bogus", "path": "x.pt"}).status_code == 400


def test_tuning_and_model_persist_to_store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    calibration = CalibrationManager(ConfigManager(cfg))
    store = RuntimeStore(tmp_path / "runtime.json")
    stub = _InferenceStub()
    app = build_app(LatestValue(), LatestValue(), calibration, WebSection(),
                    VisionSection(), TrackerSection(), stub, ActivitySection(), store)
    c = TestClient(app)
    c.post("/api/tuning", json={"key": "det_fps", "value": 5})
    c.post("/api/model", json={"kind": "pose", "path": "models/yolo11s-pose.pt"})
    assert store.tuning()["det_fps"] == 5.0
    assert store.models()["pose"] == "models/yolo11s-pose.pt"


def test_zone_add_and_delete(client):
    client.post("/api/zone/add", json={"name": "shower", "polygon_px": [[1, 1], [9, 1], [5, 9]]})
    zones = client.get("/api/state").json()["calibration"]["zones"]
    assert len(zones) == 1 and zones[0]["name"] == "shower"
    client.post("/api/zone/delete", json={"index": 0})
    assert client.get("/api/state").json()["calibration"]["zones"] == []
