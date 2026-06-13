"""Live-tunable runtime parameters (inference FPS + tracker settings).

These mutate the shared VisionSection / TrackerSection dataclasses in place.
Workers read the fields on each loop/update, so changes take effect without a
restart. Scalar assignment is atomic under the GIL, so no extra locking is
needed for single-value writes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TuneSpec:
    section: str   # "vision" | "tracker" | "activity"
    attr: str
    kind: str      # "float" | "int"
    lo: float
    hi: float
    label: str
    step: float


TUNABLES: dict[str, TuneSpec] = {
    "inference_fps":     TuneSpec("vision",  "inference_fps",     "float", 0.5, 30.0, "Inference FPS", 0.5),
    "conf":             TuneSpec("vision",  "conf",              "float", 0.05, 0.9, "Detect conf",   0.01),
    "iou":              TuneSpec("vision",  "iou",               "float", 0.1, 0.95, "Detect IoU",    0.01),
    "match_distance_px": TuneSpec("tracker", "match_distance_px", "float", 10.0, 400.0, "Match dist px", 1.0),
    "iou_match":         TuneSpec("tracker", "iou_match",         "float", 0.0, 1.0,  "Track IoU",     0.01),
    "smoothing":         TuneSpec("tracker", "smoothing",         "float", 0.0, 0.95, "Smoothing",     0.01),
    "min_hits":          TuneSpec("tracker", "min_hits",          "int",   1, 10,     "Min hits",      1),
    "keep_sec":          TuneSpec("tracker", "keep_sec",          "float", 0.5, 15.0, "Keep sec",      0.5),
    "walking_px_s":      TuneSpec("tracker", "walking_px_s",      "float", 1.0, 100.0, "Walking px/s", 1.0),
    "still_px_s":        TuneSpec("tracker", "still_px_s",        "float", 1.0, 100.0, "Still px/s",   1.0),
    # Classification (activity) model — how often the action request runs:
    "det_fps":           TuneSpec("activity", "det_fps",          "float", 0.2, 10.0, "Class FPS",     0.1),
    "det_conf":          TuneSpec("activity", "det_conf",         "float", 0.05, 0.9, "Class conf",    0.01),
}


def _target(vision, tracker, activity, spec: TuneSpec):
    if spec.section == "vision":
        return vision
    if spec.section == "tracker":
        return tracker
    return activity


def apply_tuning(vision, tracker, key: str, value, activity=None) -> float | int:
    if key not in TUNABLES:
        raise KeyError(f"Unknown tunable: {key}")
    spec = TUNABLES[key]
    target = _target(vision, tracker, activity, spec)
    if target is None:
        raise KeyError(f"Section unavailable for: {key}")
    v = max(spec.lo, min(spec.hi, float(value)))
    cast: float | int = int(round(v)) if spec.kind == "int" else float(v)
    setattr(target, spec.attr, cast)
    return cast


def get_tuning(vision, tracker, activity=None) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for k, s in TUNABLES.items():
        target = _target(vision, tracker, activity, s)
        if target is not None:
            out[k] = getattr(target, s.attr)
    return out


def tuning_specs() -> dict[str, dict]:
    return {
        k: {"label": s.label, "lo": s.lo, "hi": s.hi, "step": s.step, "kind": s.kind}
        for k, s in TUNABLES.items()
    }
