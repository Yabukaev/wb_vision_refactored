from __future__ import annotations

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
    # 100x100 px square mapped onto a 2.0 x 2.0 m room.
    for x, y in ((0, 0), (100, 0), (100, 100), (0, 100)):
        manager.add_floor_point(x, y)


def test_pixel_to_floor_requires_four_points(manager):
    manager.add_floor_point(0, 0)
    assert manager.pixel_to_floor(50, 50) is None


def test_pixel_to_floor_maps_center_to_room_center(manager):
    _calibrate_square(manager)
    geo = manager.pixel_to_floor(50, 50)
    assert geo is not None
    assert geo.x_m == pytest.approx(1.0, abs=1e-3)
    assert geo.y_m == pytest.approx(1.0, abs=1e-3)
    assert geo.inside_room is True
    assert geo.inside_calibration_zone is True


def test_pixel_to_floor_outside_zone(manager):
    _calibrate_square(manager)
    geo = manager.pixel_to_floor(500, 500)
    assert geo is not None
    assert geo.inside_room is False
    assert geo.inside_calibration_zone is False


def test_calibration_persists_to_json(manager):
    _calibrate_square(manager)
    manager.save()
    assert manager.path.exists()
