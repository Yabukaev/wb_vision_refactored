"""Coordinate tests for the floor-points homography + camera distance + logging.

Simple model: 4 floor pixels map to the room rectangle; the camera sits above
AIM at camera_height_m, so distance from camera is derived automatically.
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


def _calibrate_square(manager: CalibrationManager) -> None:
    """100x100 px square -> 2.0x2.0 m room, clockwise from top-left."""
    for x, y in ((0, 0), (100, 0), (100, 100), (0, 100)):
        manager.add_floor_point(x, y)


def test_requires_four_points(manager):
    manager.add_floor_point(0, 0)
    assert manager.pixel_to_floor(50, 50) is None


def test_maps_center(manager):
    _calibrate_square(manager)
    geo = manager.pixel_to_floor(50, 50)
    assert geo is not None
    assert geo.x_m == pytest.approx(1.0, abs=1e-3)
    assert geo.y_m == pytest.approx(1.0, abs=1e-3)
    assert geo.inside_room is True
    assert geo.inside_calibration_zone is True


def test_distance_from_camera(manager):
    _calibrate_square(manager)
    manager.set_aim(50, 50)              # AIM at floor centre -> (1,1) m
    manager.set_value("camera_height_m", 2.0)
    geo = manager.pixel_to_floor(50, 50)  # foot at same spot -> floor dist 0
    assert geo is not None
    assert geo.distance_m == pytest.approx(0.0, abs=1e-3)
    assert geo.distance_cam_m == pytest.approx(2.0, abs=1e-3)  # straight down: just height


def test_emits_debug_log(manager, caplog):
    _calibrate_square(manager)
    with caplog.at_level(logging.DEBUG, logger="calibration"):
        manager.pixel_to_floor(50, 50)
    assert any("pixel_to_floor" in r.getMessage() for r in caplog.records)


def test_logs_when_no_homography(manager, caplog):
    manager.add_floor_point(0, 0)  # only 1 point -> H is None
    with caplog.at_level(logging.DEBUG, logger="calibration"):
        result = manager.pixel_to_floor(50, 50)
    assert result is None
    assert any("no homography" in r.getMessage() for r in caplog.records)
