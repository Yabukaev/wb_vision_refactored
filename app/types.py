from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    ts: float
    image: np.ndarray

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass(slots=True)
class Detection:
    box: tuple[int, int, int, int]
    conf: float
    foot: tuple[int, int]
    center: tuple[int, int]
    state: str
    keypoints: Optional[np.ndarray] = None


@dataclass(slots=True)
class GeoPoint:
    x_m: float
    y_m: float
    distance_m: float           # floor distance from AIM
    inside_room: bool
    inside_calibration_zone: bool
    distance_cam_m: float = 0.0  # 3D distance from camera lens


@dataclass(slots=True)
class TrackSnapshot:
    track_id: int
    box: tuple[int, int, int, int]
    conf: float
    foot: tuple[int, int]
    center: tuple[int, int]
    state: str
    last_seen: float
    hits: int
    age_sec: float
    history: list[tuple[int, int]] = field(default_factory=list)
    keypoints: Optional[np.ndarray] = None
    geo: Optional[GeoPoint] = None
    motion: str = "stationary"
    activity: str = ""


@dataclass(slots=True)
class VisionPacket:
    frame_id: int
    ts: float
    infer_ms: float
    inference_fps: float
    source_width: int
    source_height: int
    tracks: list[TrackSnapshot]
    detections_count: int
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    reader_fps: float = 0.0


@dataclass(slots=True)
class MqttMessage:
    topic: str
    value: Any
    retain: bool = False

