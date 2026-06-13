from __future__ import annotations

from app.config import TrackerSection
from app.types import Detection
from app.vision.tracker import StableTracker


def _det(x1: int, y1: int, x2: int, y2: int, conf: float = 0.9) -> Detection:
    foot = (int((x1 + x2) / 2), y2)
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return Detection(box=(x1, y1, x2, y2), conf=conf, foot=foot, center=center, state="standing")


def _tracker(**overrides) -> StableTracker:
    cfg = TrackerSection(min_hits=1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return StableTracker(cfg)


def test_same_detection_keeps_stable_id():
    tracker = _tracker()
    first = tracker.update([_det(100, 100, 160, 300)], now=1.0)
    second = tracker.update([_det(104, 102, 164, 302)], now=1.2)
    assert len(first) == len(second) == 1
    assert first[0].track_id == second[0].track_id


def test_distant_detections_create_separate_tracks():
    tracker = _tracker()
    tracks = tracker.update([_det(0, 0, 60, 200), _det(800, 0, 860, 200)], now=1.0)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_track_expires_after_keep_sec():
    tracker = _tracker(keep_sec=1.0)
    tracker.update([_det(100, 100, 160, 300)], now=1.0)
    tracks = tracker.update([], now=3.0)
    assert tracks == []


def test_min_hits_hides_unconfirmed_tracks():
    tracker = _tracker(min_hits=2)
    first = tracker.update([_det(100, 100, 160, 300)], now=1.0)
    assert first == []  # single hit: not confirmed yet
    second = tracker.update([_det(102, 101, 162, 301)], now=1.2)
    assert len(second) == 1
    assert second[0].hits == 2


def test_one_detection_updates_only_one_track():
    tracker = _tracker()
    tracker.update([_det(0, 0, 60, 200), _det(70, 0, 130, 200)], now=1.0)
    tracks = tracker.update([_det(30, 0, 90, 200)], now=1.2)
    assert sum(tr.hits for tr in tracks) == 3


def test_snapshot_foot_is_integer_tuple():
    """B-16: internal float smoothing must still export integer coordinates."""
    tracker = _tracker(smoothing=0.65)
    tracker.update([_det(100, 100, 160, 300)], now=1.0)
    tracks = tracker.update([_det(104, 102, 164, 302)], now=1.2)
    fx, fy = tracks[0].foot
    assert isinstance(fx, int)
    assert isinstance(fy, int)


def test_snapshot_box_is_integer_tuple():
    """B-16: internal float smoothing must still export integer box."""
    tracker = _tracker(smoothing=0.65)
    tracks = tracker.update([_det(100, 100, 160, 300)], now=1.0)
    for v in tracks[0].box:
        assert isinstance(v, int)
