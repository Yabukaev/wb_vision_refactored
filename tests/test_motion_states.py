from __future__ import annotations

from unittest.mock import MagicMock

from app.config import ActivitySection, TrackerSection
from app.types import Detection, TrackSnapshot
from app.vision.activity_classifier import ActivityRules
from app.vision.tracker import StableTracker


def _det(x1: int, y1: int, x2: int, y2: int, state: str = "standing", conf: float = 0.9) -> Detection:
    foot = (int((x1 + x2) / 2), y2)
    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    return Detection(box=(x1, y1, x2, y2), conf=conf, foot=foot, center=center, state=state)


def _tracker(**overrides) -> StableTracker:
    cfg = TrackerSection(min_hits=1)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return StableTracker(cfg)


# ── walking detection ────────────────────────────────────────────────────────

def test_walking_detection():
    """Fast lateral movement triggers walking label."""
    t = _tracker(walking_px_s=10.0, still_px_s=5.0)
    t.update([_det(200, 100, 260, 300, "standing")], now=0.0)
    # Move 60px in 1.5s → ~40 px/s > walking threshold
    t.update([_det(230, 100, 290, 300, "standing")], now=0.5)
    t.update([_det(250, 100, 310, 300, "standing")], now=1.0)
    tracks = t.update([_det(260, 100, 320, 300, "standing")], now=1.5)
    assert len(tracks) == 1
    assert tracks[0].motion == "walking"


def test_stationary_standing():
    """Standing without movement stays stationary."""
    t = _tracker(walking_px_s=10.0, still_px_s=5.0)
    for i in range(6):
        tracks = t.update([_det(200, 100, 260, 300, "standing")], now=float(i) * 0.5)
    assert tracks[0].motion == "stationary"


# ── fallen detection ─────────────────────────────────────────────────────────

def test_fallen_detection():
    """Rapid transition from standing to lying triggers fallen."""
    t = _tracker(fallen_window_sec=3.0, fallen_persist_sec=10.0, walking_px_s=10.0, still_px_s=5.0)
    # Standing phase
    t.update([_det(200, 100, 260, 300, "standing")], now=0.0)
    t.update([_det(200, 100, 260, 300, "standing")], now=0.5)
    # Sudden collapse to lying within fallen_window_sec
    t.update([_det(100, 200, 400, 260, "lying")], now=1.0)
    t.update([_det(100, 200, 400, 260, "lying")], now=1.5)
    t.update([_det(100, 200, 400, 260, "lying")], now=2.0)
    # Need enough state_history votes (maxlen=7) to flip state to "lying"
    t.update([_det(100, 200, 400, 260, "lying")], now=2.5)
    tracks = t.update([_det(100, 200, 400, 260, "lying")], now=3.0)
    assert len(tracks) == 1
    assert tracks[0].motion == "fallen"


def test_fallen_expires():
    """Fallen status expires after fallen_persist_sec; track must be kept alive."""
    t = _tracker(
        keep_sec=999.0,
        fallen_window_sec=3.0,
        fallen_persist_sec=5.0,
        sleep_still_sec=999.0,
        walking_px_s=10.0,
        still_px_s=5.0,
    )
    t.update([_det(200, 100, 260, 300, "standing")], now=0.0)
    t.update([_det(200, 100, 260, 300, "standing")], now=0.5)
    for ts in [1.0, 1.5, 2.0, 2.5, 3.0]:
        t.update([_det(100, 200, 400, 260, "lying")], now=ts)
    # Continue feeding updates so the track stays alive
    for ts in range(4, 20):
        t.update([_det(100, 200, 400, 260, "lying")], now=float(ts))
    # fallen_until_ts ≈ 3.0 + 5.0 = 8.0, so at t=20 it must have expired
    tracks = t.update([_det(100, 200, 400, 260, "lying")], now=20.0)
    assert tracks[0].motion in ("lying", "sleeping")


# ── sleeping detection ───────────────────────────────────────────────────────

def test_sleeping_detection():
    """Lying without movement for sleep_still_sec → sleeping."""
    t = _tracker(keep_sec=999.0, sleep_still_sec=5.0, walking_px_s=10.0, still_px_s=5.0, fallen_window_sec=0.1)
    # Fill state_history with "lying" and keep track alive
    for i in range(30):
        t.update([_det(100, 200, 400, 260, "lying")], now=float(i) * 0.5)
    # At t=14.5, last_moved_ts should be early (feet at same coords → velocity=0)
    # Advance beyond sleep_still_sec (5s) without movement
    tracks = t.update([_det(100, 200, 400, 260, "lying")], now=20.0)
    assert tracks[0].motion == "sleeping"


def test_stationary_while_lying():
    """Lying but recently moved is just 'lying', not sleeping."""
    t = _tracker(sleep_still_sec=30.0, walking_px_s=10.0, still_px_s=5.0, fallen_window_sec=0.1)
    # Fill state_history with lying
    for i in range(8):
        t.update([_det(100 + i * 5, 200, 400 + i * 5, 260, "lying")], now=float(i) * 0.5)
    # Moved recently (within sleep_still_sec window)
    tracks = t.update([_det(150, 200, 450, 260, "lying")], now=5.0)
    assert tracks[0].motion in ("lying", "stationary")
    assert tracks[0].motion != "sleeping"


# ── activity rules ───────────────────────────────────────────────────────────

def _snap(cx: int = 230, fy: int = 300) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=1,
        box=(200, 100, 260, 300),
        conf=0.9,
        foot=(cx, fy),
        center=(cx, 200),
        state="standing",
        last_seen=1.0,
        hits=5,
        age_sec=2.0,
    )


def test_activity_rules_phone():
    rules = ActivityRules()
    objects = [{"label": "cell phone", "cx": 235.0, "cy": 195.0, "conf": 0.85}]
    assert rules.classify(_snap(), objects) == "с телефоном"


def test_activity_rules_laptop():
    rules = ActivityRules()
    objects = [{"label": "laptop", "cx": 230.0, "cy": 198.0, "conf": 0.80}]
    assert rules.classify(_snap(), objects) == "у компьютера"


def test_activity_rules_no_nearby_objects():
    rules = ActivityRules()
    objects = [{"label": "cell phone", "cx": 900.0, "cy": 900.0, "conf": 0.85}]
    assert rules.classify(_snap(), objects) == ""


def test_activity_rules_priority_toilet_over_phone():
    """Toilet takes priority over phone in priority list."""
    rules = ActivityRules()
    objects = [
        {"label": "toilet", "cx": 232.0, "cy": 195.0, "conf": 0.80},
        {"label": "cell phone", "cx": 228.0, "cy": 200.0, "conf": 0.85},
    ]
    assert rules.classify(_snap(), objects) == "в туалете"


def test_activity_rules_empty_objects():
    rules = ActivityRules()
    assert rules.classify(_snap(), []) == ""
