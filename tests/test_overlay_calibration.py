"""draw_calibration must render the trapezoid quad (quad_px), not only the
legacy floor_points — otherwise web-placed corners never appear on the video.
"""
from __future__ import annotations

import numpy as np

from app.vision.calibration import CalibrationData
from app.vision.overlay import draw_calibration


def _blank():
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_draws_quad_px():
    frame = _blank()
    cal = CalibrationData(aim_px=10, aim_py=10,
                          quad_px=[[20, 20], [180, 20], [180, 180], [20, 180]])
    draw_calibration(frame, cal, scale=1.0)
    assert frame.sum() > 0  # something was drawn for the quad


def test_draws_partial_quad():
    frame = _blank()
    cal = CalibrationData(aim_px=10, aim_py=10, quad_px=[[20, 20], [180, 20]])
    draw_calibration(frame, cal, scale=1.0)
    assert frame.sum() > 0


def test_falls_back_to_floor_points():
    frame = _blank()
    cal = CalibrationData(aim_px=10, aim_py=10,
                          floor_points=[[20, 20], [180, 20], [180, 180], [20, 180]])
    draw_calibration(frame, cal, scale=1.0)
    assert frame.sum() > 0
