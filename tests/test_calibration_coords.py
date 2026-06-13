"""Coordinate tests for the NEW cal_points homography path + debug logging.

The legacy floor_points path is covered by test_calibration.py. The cal_points
path (3 modes: xy / laser / hybrid) is what the live UI uses, and it had no
coordinate characterization test. These lock the px -> metres math and verify
that pixel_to_floor emits diagnostic debug logs (step 1 of the MVP fix plan in
ANALYSIS_POSITIONING.md).
"""
from __future__ import annotations

import logging

import pytest

from app.config import ConfigManager
from app.vision.calibration import CalibrationManager

CONFIG_YAML = """
camera:
  rtsp_url: rtsp://dummy/stream
calibration:
  file: {cal_file}
  room_width_m: 2.0
  room_depth_m: 2.0
"""


@pytest.fixture()
def manager(tmp_path, monkeypatch) -> CalibrationManager:
    monkeypatch.chdir(tmp_path)
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    return CalibrationManager(ConfigManager(cfg))


def _calibrate_square_xy(manager: CalibrationManager) -> None:
    """100x100 px square mapped to a 2.0x2.0 m square in XY (tape) mode."""
    manager.set_cal_mode("xy")
    for (px, py), (x_m, y_m) in (
        ((0, 0), (0.0, 0.0)),
        ((100, 0), (2.0, 0.0)),
        ((100, 100), (2.0, 2.0)),
        ((0, 100), (0.0, 2.0)),
    ):
        manager.add_cal_point(px, py, x_m=x_m, y_m=y_m)


def test_cal_points_requires_four_points(manager):
    manager.set_cal_mode("xy")
    manager.add_cal_point(0, 0, x_m=0.0, y_m=0.0)
    assert manager.pixel_to_floor(50, 50) is None


def test_cal_points_xy_maps_center(manager):
    _calibrate_square_xy(manager)
    geo = manager.pixel_to_floor(50, 50)
    assert geo is not None
    assert geo.x_m == pytest.approx(1.0, abs=1e-3)
    assert geo.y_m == pytest.approx(1.0, abs=1e-3)


def test_cal_points_xy_maps_corner(manager):
    _calibrate_square_xy(manager)
    geo = manager.pixel_to_floor(100, 100)
    assert geo is not None
    assert geo.x_m == pytest.approx(2.0, abs=1e-3)
    assert geo.y_m == pytest.approx(2.0, abs=1e-3)


def test_pixel_to_floor_emits_debug_log(manager, caplog):
    _calibrate_square_xy(manager)
    with caplog.at_level(logging.DEBUG, logger="calibration"):
        manager.pixel_to_floor(50, 50)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("pixel_to_floor" in m for m in msgs), msgs


def test_pixel_to_floor_logs_when_no_homography(manager, caplog):
    manager.set_cal_mode("xy")
    manager.add_cal_point(0, 0, x_m=0.0, y_m=0.0)  # only 1 point -> H is None
    with caplog.at_level(logging.DEBUG, logger="calibration"):
        result = manager.pixel_to_floor(50, 50)
    assert result is None
    msgs = [r.getMessage() for r in caplog.records]
    assert any("no homography" in m for m in msgs), msgs
