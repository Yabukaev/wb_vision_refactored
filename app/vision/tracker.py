from __future__ import annotations

import math
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.config import TrackerSection
from app.types import Detection, GeoPoint, TrackSnapshot
from app.vision.detector import box_iou


def _smooth_point(old: tuple[int, int], new: tuple[int, int], alpha: float) -> tuple[int, int]:
    return (int(old[0] * alpha + new[0] * (1.0 - alpha)), int(old[1] * alpha + new[1] * (1.0 - alpha)))


def _smooth_box(old: tuple[int, int, int, int], new: tuple[int, int, int, int], alpha: float) -> tuple[int, int, int, int]:
    return tuple(int(old[i] * alpha + new[i] * (1.0 - alpha)) for i in range(4))  # type: ignore[return-value]


@dataclass
class _Track:
    track_id: int
    box: tuple[int, int, int, int]
    conf: float
    foot: tuple[int, int]
    center: tuple[int, int]
    state: str
    keypoints: Optional[np.ndarray]
    first_seen: float
    last_seen: float
    hits: int = 1
    history: deque[tuple[int, int]] = field(default_factory=deque)
    state_history: deque[str] = field(default_factory=lambda: deque(maxlen=7))


class StableTracker:
    """Stable greedy multi-object tracker.

    Matching uses both foot distance and bbox IoU. It also prevents several
    detections from updating the same track during one frame.
    """

    def __init__(self, cfg: TrackerSection) -> None:
        self.cfg = cfg
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], now: Optional[float] = None, geo_fn=None) -> list[TrackSnapshot]:
        now = time.time() if now is None else now
        self._drop_expired(now)

        candidates: list[tuple[float, int, int]] = []
        for det_i, det in enumerate(detections):
            for tid, tr in self._tracks.items():
                d = math.hypot(det.foot[0] - tr.foot[0], det.foot[1] - tr.foot[1])
                iou = box_iou(det.box, tr.box)
                if d < self.cfg.match_distance_px or iou > self.cfg.iou_match:
                    candidates.append((d - iou * 100.0, det_i, tid))

        candidates.sort(key=lambda x: x[0])
        used_dets: set[int] = set()
        used_tracks: set[int] = set()
        assignments: dict[int, int] = {}
        for _cost, det_i, tid in candidates:
            if det_i in used_dets or tid in used_tracks:
                continue
            assignments[det_i] = tid
            used_dets.add(det_i)
            used_tracks.add(tid)

        for det_i, det in enumerate(detections):
            if det_i in assignments:
                self._update_track(assignments[det_i], det, now)
            else:
                self._create_track(det, now)

        self._drop_expired(now)
        return self.snapshots(now=now, geo_fn=geo_fn)

    def snapshots(self, now: Optional[float] = None, geo_fn=None) -> list[TrackSnapshot]:
        now = time.time() if now is None else now
        out: list[TrackSnapshot] = []
        for tid in sorted(self._tracks):
            tr = self._tracks[tid]
            if tr.hits < self.cfg.min_hits:
                continue
            geo: Optional[GeoPoint] = geo_fn(tr.foot[0], tr.foot[1]) if geo_fn else None
            out.append(
                TrackSnapshot(
                    track_id=tr.track_id,
                    box=tr.box,
                    conf=tr.conf,
                    foot=tr.foot,
                    center=tr.center,
                    state=tr.state,
                    last_seen=tr.last_seen,
                    hits=tr.hits,
                    age_sec=max(0.0, now - tr.first_seen),
                    history=list(tr.history),
                    keypoints=tr.keypoints,
                    geo=geo,
                )
            )
        return out

    def _create_track(self, det: Detection, now: float) -> None:
        tid = self._next_id
        self._next_id += 1
        tr = _Track(
            track_id=tid,
            box=det.box,
            conf=det.conf,
            foot=det.foot,
            center=det.center,
            state=det.state,
            keypoints=det.keypoints,
            first_seen=now,
            last_seen=now,
        )
        tr.history = deque([det.foot], maxlen=int(self.cfg.max_history))
        tr.state_history.append(det.state)
        self._tracks[tid] = tr

    def _update_track(self, tid: int, det: Detection, now: float) -> None:
        tr = self._tracks[tid]
        alpha = float(self.cfg.smoothing)
        tr.box = _smooth_box(tr.box, det.box, alpha)
        tr.foot = _smooth_point(tr.foot, det.foot, alpha)
        tr.center = _smooth_point(tr.center, det.center, alpha)
        tr.conf = float(alpha * tr.conf + (1.0 - alpha) * det.conf)
        tr.keypoints = det.keypoints
        tr.last_seen = now
        tr.hits += 1
        tr.history.append(tr.foot)
        tr.state_history.append(det.state)
        tr.state = Counter(tr.state_history).most_common(1)[0][0]

    def _drop_expired(self, now: float) -> None:
        keep_sec = float(self.cfg.keep_sec)
        for tid in list(self._tracks):
            if now - self._tracks[tid].last_seen > keep_sec:
                del self._tracks[tid]

