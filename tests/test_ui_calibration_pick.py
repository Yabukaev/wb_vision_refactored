"""UIWorker calibration point-pick flow.

Regression guard for the bug where '+ Add point' started pick mode and then
immediately cancelled it (point_pick_active flipped back to False inside
_start_point_pick -> _point_entry_cancel), so clicking the video never placed
a point. Points must place by clicking (like zones) and record per mode.
"""
from __future__ import annotations

import threading

import pytest

from app.config import ConfigManager
from app.core.latest_value import LatestValue
from app.vision.calibration import CAL_MODE_XY, CAL_MODE_HYBRID, CalibrationManager
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
    return UIWorker(
        settings.ui,
        LatestValue(),
        LatestValue(),
        calibration,
        threading.Event(),
    )


def test_start_point_pick_leaves_pick_active(worker):
    worker._start_point_pick()
    assert worker.point_pick_active is True
    assert worker.pending_px is None
    assert worker.aim_mode is False


def _simulate_video_click(worker, fx: int, fy: int) -> None:
    """Mimic what _mouse_cb does for a video click in pick mode."""
    from app.vision.calibration import mode_entry_fields

    assert worker.point_pick_active is True, "pick mode must be active to place a point"
    worker.pending_px = fx
    worker.pending_py = fy
    worker.point_pick_active = False
    cal = worker.calibration.snapshot()
    worker.point_entry_fields = mode_entry_fields(cal.cal_mode)
    worker.point_entry_cursor = 0
    worker.point_entry_buf = ""
    worker.point_entry_values = {}


def test_full_point_entry_records_xy(worker):
    worker.calibration.set_cal_mode(CAL_MODE_XY)
    worker._start_point_pick()
    _simulate_video_click(worker, 100, 200)

    worker.point_entry_buf = "1.5"
    worker._point_entry_advance()  # x_m
    worker.point_entry_buf = "2.0"
    worker._point_entry_advance()  # y_m -> commits

    pts = worker.calibration.snapshot().cal_points
    assert len(pts) == 1
    assert pts[0]["px"] == 100 and pts[0]["py"] == 200
    assert pts[0]["x_m"] == pytest.approx(1.5)
    assert pts[0]["y_m"] == pytest.approx(2.0)
    assert worker.pending_px is None  # entry finished


def test_mode_switch_changes_entry_fields(worker):
    from app.vision.calibration import mode_entry_fields

    worker.calibration.set_cal_mode(CAL_MODE_XY)
    xy_fields = mode_entry_fields(worker.calibration.snapshot().cal_mode)
    worker.calibration.set_cal_mode(CAL_MODE_HYBRID)
    hybrid_fields = mode_entry_fields(worker.calibration.snapshot().cal_mode)
    assert xy_fields != hybrid_fields
    assert len(hybrid_fields) == 3
