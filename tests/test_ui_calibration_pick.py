"""UIWorker floor-point calibration flow.

Points are placed by clicking the video (like zones): F / 'Set 4 floor points'
arms floor4 mode, then each video click adds a corner and the 4th finishes.
"""
from __future__ import annotations

import threading

import cv2
import pytest

from app.config import ConfigManager
from app.core.latest_value import LatestValue
from app.vision.calibration import CalibrationManager
from app.ui.ui_worker import UIWorker

CONFIG_YAML = """
camera:
  rtsp_url: rtsp://dummy/stream
calibration:
  file: {cal_file}
  room_width_m: 2.0
  room_depth_m: 2.0
"""


@pytest.fixture()
def worker(tmp_path, monkeypatch) -> UIWorker:
    monkeypatch.chdir(tmp_path)
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    settings = ConfigManager(cfg).get()
    calibration = CalibrationManager(ConfigManager(cfg))
    w = UIWorker(settings.ui, LatestValue(), LatestValue(), calibration, threading.Event())
    # Identity video geometry so screen coords == frame coords.
    w.scale = 1.0
    w.off_x = w.off_y = 0
    w.draw_w = w.draw_h = w.src_w = w.src_h = 200
    return w


def _click(worker: UIWorker, x: int, y: int) -> None:
    worker._mouse_cb(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)


def test_floor4_button_arms_mode(worker):
    worker._handle_button("floor4")
    assert worker.calib_mode == "floor4"
    assert worker.calibration.snapshot().floor_points == []


def test_four_clicks_record_floor_points(worker):
    worker._handle_button("floor4")
    for (x, y) in ((10, 10), (190, 10), (190, 190), (10, 190)):
        _click(worker, x, y)
    pts = worker.calibration.snapshot().floor_points
    assert len(pts) == 4
    assert pts[0] == [10.0, 10.0]
    assert worker.calib_mode is None  # auto-finished after the 4th


def test_aim_click_sets_aim(worker):
    worker._handle_button("aim")
    assert worker.calib_mode == "aim"
    _click(worker, 123, 45)
    cal = worker.calibration.snapshot()
    assert cal.aim_px == 123 and cal.aim_py == 45
    assert worker.calib_mode is None


def test_zone_draw_still_works(worker):
    worker._handle_button("zone_draw")
    assert worker.zone_draw_active is True
    for (x, y) in ((20, 20), (40, 20), (40, 40)):
        _click(worker, x, y)
    assert len(worker.zone_polygon_px) == 3
    worker._handle_button("zone_finish")
    worker.zone_name_buf = "shower"
    worker._save_zone()
    zones = worker.calibration.snapshot().zones
    assert len(zones) == 1 and zones[0]["name"] == "shower"
