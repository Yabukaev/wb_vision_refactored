from __future__ import annotations

from app.types import Detection
from app.vision.detector import box_iou, suppress_duplicates


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
