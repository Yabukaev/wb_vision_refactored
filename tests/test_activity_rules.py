"""Activity association: objects map to a person by their bounding box, not a
fixed pixel radius (which broke on high-resolution frames).
"""
from __future__ import annotations

from app.types import TrackSnapshot
from app.vision.activity_classifier import ActivityRules


def _track(box):
    cx = (box[0] + box[2]) // 2
    return TrackSnapshot(
        track_id=1, box=box, conf=0.9, foot=(cx, box[3]),
        center=(cx, (box[1] + box[3]) // 2), state="standing",
        last_seen=0.0, hits=3, age_sec=1.0,
    )


def _obj(label, cx, cy):
    return {"label": label, "cx": float(cx), "cy": float(cy), "conf": 0.5}


def test_object_inside_bbox_maps_to_activity():
    r = ActivityRules()
    track = _track((100, 100, 200, 400))
    assert r.classify(track, [_obj("cell phone", 150, 250)], margin_ratio=0.6) == "on phone"


def test_far_object_ignored():
    r = ActivityRules()
    track = _track((100, 100, 200, 400))
    assert r.classify(track, [_obj("laptop", 2000, 250)], margin_ratio=0.6) == ""


def test_relative_threshold_scales_with_person_size():
    # Same object offset is "near" for a large (close) person, "far" for a tiny one.
    r = ActivityRules()
    big = _track((0, 0, 600, 1200))      # large person
    small = _track((0, 0, 60, 120))      # distant person
    obj = [_obj("laptop", 750, 600)]
    assert r.classify(big, obj, margin_ratio=0.6) == "at computer"
    assert r.classify(small, obj, margin_ratio=0.6) == ""


def test_priority_order():
    r = ActivityRules()
    track = _track((100, 100, 200, 400))
    objs = [_obj("cup", 150, 250), _obj("toilet", 160, 260)]
    assert r.classify(track, objs) == "in bathroom"  # higher priority than drinking
