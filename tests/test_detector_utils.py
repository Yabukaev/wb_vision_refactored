from __future__ import annotations

import numpy as np
import pytest

from app.types import Detection
from app.vision.detector import box_iou, foot_from_pose, state_by_pose, suppress_duplicates


def _det(box: tuple[int, int, int, int], conf: float) -> Detection:
    x1, y1, x2, y2 = box
    foot = (int((x1 + x2) / 2), y2)
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return Detection(box=box, conf=conf, foot=foot, center=center, state="standing")


def test_box_iou_identical_boxes():
    assert box_iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0


def test_box_iou_disjoint_boxes():
    assert box_iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_suppress_duplicates_keeps_higher_confidence():
    strong = _det((100, 100, 200, 300), conf=0.9)
    weak = _det((105, 102, 205, 302), conf=0.5)
    kept = suppress_duplicates([weak, strong], foot_dist_px=70.0, iou_threshold=0.35)
    assert kept == [strong]


def test_suppress_duplicates_keeps_distant_detections():
    a = _det((0, 0, 60, 200), conf=0.9)
    b = _det((600, 0, 660, 200), conf=0.8)
    kept = suppress_duplicates([a, b], foot_dist_px=70.0, iou_threshold=0.35)
    assert kept == [a, b]


# ── foot_from_pose (B-05) ─────────────────────────────────────────────────────

def _make_keypoints_xyc(overrides: dict[int, tuple[float, float, float]]) -> np.ndarray:
    """Build a (17, 3) keypoint array with given per-idx (x, y, conf) overrides."""
    kp = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y, c) in overrides.items():
        kp[idx] = [x, y, c]
    return kp


def test_foot_uses_ankle_keypoints_with_high_confidence():
    box = (100, 50, 200, 300)
    kp = _make_keypoints_xyc({15: (140.0, 290.0, 0.9), 16: (160.0, 295.0, 0.85)})
    fx, fy = foot_from_pose(kp, box)
    assert fx == 150  # average of 140, 160
    assert fy == 295  # max ankle y


def test_foot_ignores_low_confidence_ankle():
    """B-05: ankle with conf < threshold should be ignored; fall back to bbox bottom."""
    box = (100, 50, 200, 300)
    kp = _make_keypoints_xyc({15: (140.0, 200.0, 0.05), 16: (160.0, 200.0, 0.05)})
    fx, fy = foot_from_pose(kp, box)
    # both ankles filtered out → fall back to bbox centre-bottom
    assert fx == 150  # (100 + 200) / 2
    assert fy == 300  # y2


def test_foot_falls_back_when_no_keypoints():
    box = (100, 50, 200, 300)
    fx, fy = foot_from_pose(None, box)
    assert fx == 150
    assert fy == 300


# ── state_by_pose (B-06) ─────────────────────────────────────────────────────

def test_state_standing_from_keypoints():
    box = (50, 50, 120, 250)
    # shoulders high, hips mid, ankles low → normal vertical body
    kp = _make_keypoints_xyc({
        5: (70.0, 80.0, 0.9), 6: (100.0, 80.0, 0.9),   # shoulders
        11: (70.0, 150.0, 0.9), 12: (100.0, 150.0, 0.9), # hips
        15: (70.0, 240.0, 0.9), 16: (100.0, 240.0, 0.9), # ankles
    })
    assert state_by_pose(kp, box) == "standing"


def test_state_lying_from_keypoints():
    box = (50, 100, 300, 140)  # wide, shallow bounding box
    # shoulders and ankles are at similar y → body horizontal
    kp = _make_keypoints_xyc({
        5: (70.0, 115.0, 0.9), 6: (100.0, 115.0, 0.9),
        15: (270.0, 120.0, 0.9), 16: (280.0, 120.0, 0.9),
    })
    assert state_by_pose(kp, box) == "lying"


def test_state_fallback_bbox_standing():
    """No keypoints → fall back to bbox ratio (tall narrow → standing)."""
    box = (50, 50, 100, 250)  # w=50, h=200 → ratio=0.25
    assert state_by_pose(None, box) == "standing"


def test_state_fallback_bbox_lying():
    box = (0, 100, 300, 160)  # w=300, h=60 → ratio=5.0
    assert state_by_pose(None, box) == "lying"
