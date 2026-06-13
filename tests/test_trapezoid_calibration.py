"""Phase 1: trapezoid floor calibration.

P1 = origin (0,0). Operator clicks 4 floor corners in the image and enters the
edge lengths (AB, BC, CD, DA) and interior angles; the builder computes the 4
world coordinates and the manager builds a homography from clicked pixels to
those world points.
"""
from __future__ import annotations

import math

import pytest

from app.config import ConfigManager
from app.vision.calibration import CalibrationManager, trapezoid_world_points

CONFIG_YAML = """
camera:
  rtsp_url: rtsp://dummy/stream
calibration:
  file: {cal_file}
"""


@pytest.fixture()
def manager(tmp_path, monkeypatch) -> CalibrationManager:
    monkeypatch.chdir(tmp_path)
    cal_file = (tmp_path / "calibration.json").as_posix()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(CONFIG_YAML.format(cal_file=cal_file), encoding="utf-8")
    return CalibrationManager(ConfigManager(cfg))


# ── pure builder ────────────────────────────────────────────────────────────

def test_builder_rectangle():
    pts, err = trapezoid_world_points([2.0, 1.0, 2.0, 1.0], [90, 90, 90, 90])
    assert pts[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert pts[1] == pytest.approx((2.0, 0.0), abs=1e-6)
    assert pts[2] == pytest.approx((2.0, 1.0), abs=1e-6)
    assert pts[3] == pytest.approx((0.0, 1.0), abs=1e-6)
    assert err == pytest.approx(0.0, abs=1e-6)


def test_builder_first_edge_on_x_axis():
    pts, _ = trapezoid_world_points([3.0, 2.5, 2.0, 2.0], [80, 100, 100, 80])
    assert pts[0] == pytest.approx((0.0, 0.0), abs=1e-6)
    assert pts[1] == pytest.approx((3.0, 0.0), abs=1e-6)  # P1->P2 along +X


def test_builder_closure_error_flags_inconsistent_da():
    # rectangle geometry but DA wrong by 0.5 m -> closure error ~0.5
    _pts, err = trapezoid_world_points([2.0, 1.0, 2.0, 1.5], [90, 90, 90, 90])
    assert err == pytest.approx(0.5, abs=1e-6)


# ── manager integration ────────────────────────────────────────────────────

def _configure_square(m: CalibrationManager) -> None:
    quad = [(0, 0), (100, 0), (100, 100), (0, 100)]
    for i, (x, y) in enumerate(quad):
        m.set_quad_point(i, x, y)
    for i, v in enumerate([2.0, 2.0, 2.0, 2.0]):
        m.set_trap_edge(i, v)
    for i, v in enumerate([90, 90, 90, 90]):
        m.set_trap_angle(i, v)


def test_manager_requires_full_quad(manager):
    manager.set_quad_point(0, 0, 0)
    assert manager.pixel_to_floor(50, 50) is None


def test_manager_trapezoid_maps_corner_to_origin(manager):
    _configure_square(manager)
    geo = manager.pixel_to_floor(0, 0)
    assert geo is not None
    assert geo.x_m == pytest.approx(0.0, abs=1e-3)
    assert geo.y_m == pytest.approx(0.0, abs=1e-3)


def test_manager_trapezoid_maps_center(manager):
    _configure_square(manager)
    geo = manager.pixel_to_floor(50, 50)
    assert geo is not None
    assert geo.x_m == pytest.approx(1.0, abs=1e-3)
    assert geo.y_m == pytest.approx(1.0, abs=1e-3)


def test_manager_trapezoid_persists(manager):
    _configure_square(manager)
    manager.save()
    reloaded = CalibrationManager(manager.config)
    geo = reloaded.pixel_to_floor(50, 50)
    assert geo is not None
    assert geo.x_m == pytest.approx(1.0, abs=1e-3)
