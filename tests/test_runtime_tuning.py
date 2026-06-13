"""Phase 3: live tuning of inference FPS and tracker params (no restart)."""
from __future__ import annotations

import pytest

from app.config import TrackerSection, VisionSection
from app.runtime_tuning import TUNABLES, apply_tuning, get_tuning, tuning_specs


def _cfgs():
    return VisionSection(), TrackerSection()


def test_apply_updates_vision_fps():
    v, t = _cfgs()
    out = apply_tuning(v, t, "inference_fps", 8)
    assert out == 8.0 and v.inference_fps == 8.0


def test_apply_updates_tracker_param():
    v, t = _cfgs()
    apply_tuning(v, t, "match_distance_px", 150)
    assert t.match_distance_px == 150.0


def test_apply_clamps_to_range():
    v, t = _cfgs()
    hi = apply_tuning(v, t, "inference_fps", 9999)
    lo = apply_tuning(v, t, "inference_fps", -5)
    assert hi == TUNABLES["inference_fps"].hi
    assert lo == TUNABLES["inference_fps"].lo


def test_int_params_are_cast():
    v, t = _cfgs()
    apply_tuning(v, t, "min_hits", 3.0)
    assert t.min_hits == 3 and isinstance(t.min_hits, int)


def test_unknown_key_raises():
    v, t = _cfgs()
    with pytest.raises(KeyError):
        apply_tuning(v, t, "nope", 1)


def test_get_tuning_roundtrip():
    v, t = _cfgs()
    apply_tuning(v, t, "smoothing", 0.5)
    d = get_tuning(v, t)
    assert d["smoothing"] == 0.5 and "inference_fps" in d and "keep_sec" in d


def test_specs_have_bounds():
    s = tuning_specs()
    assert s["inference_fps"]["hi"] == TUNABLES["inference_fps"].hi
    assert "label" in s["match_distance_px"]
