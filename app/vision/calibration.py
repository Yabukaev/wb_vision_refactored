from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import ConfigManager
from app.types import GeoPoint


@dataclass(slots=True)
class CalibrationData:
    room_width_m: float = 2.5
    room_depth_m: float = 2.5

    aim_px: int = 320
    aim_py: int = 240
    floor_points: list[list[float]] | None = None
    world_points: list[list[float]] | None = None

    camera_height_m: float = 2.5
    camera_pitch_deg: float = 45.0
    camera_yaw_deg: float = 0.0
    camera_roll_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 55.0
    rotation_deg: float = 0.0
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0

    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.floor_points is None:
            self.floor_points = []
        if self.world_points is None:
            self.world_points = []


class CalibrationManager:
    """Homography-based floor calibration plus editable camera geometry metadata."""

    NUMERIC_FIELDS = {
        "room_width_m",
        "room_depth_m",
        "camera_height_m",
        "camera_pitch_deg",
        "camera_yaw_deg",
        "camera_roll_deg",
        "hfov_deg",
        "vfov_deg",
        "rotation_deg",
        "lens_distortion_k1",
        "lens_distortion_k2",
    }

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._H: Optional[np.ndarray] = None
        self._version = 0
        self.path = self.config.resolve_path(self.config.get().calibration.file)
        self.data = self._load()

    def _load(self) -> CalibrationData:
        cfg = self.config.get().calibration
        data = CalibrationData(
            room_width_m=float(cfg.room_width_m),
            room_depth_m=float(cfg.room_depth_m),
            aim_px=int(cfg.aim_px),
            aim_py=int(cfg.aim_py),
            floor_points=[list(map(float, p)) for p in cfg.floor_points],
            camera_height_m=float(cfg.camera_height_m),
            camera_pitch_deg=float(cfg.camera_pitch_deg),
            camera_yaw_deg=float(cfg.camera_yaw_deg),
            camera_roll_deg=float(cfg.camera_roll_deg),
            hfov_deg=float(cfg.hfov_deg),
            vfov_deg=float(cfg.vfov_deg),
            rotation_deg=float(cfg.rotation_deg),
            lens_distortion_k1=float(cfg.lens_distortion_k1),
            lens_distortion_k2=float(cfg.lens_distortion_k2),
            created_at=time.time(),
            updated_at=time.time(),
        )

        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
            floor_points = raw.get("floor_points", raw.get("image_points", data.floor_points))

            data.room_width_m = float(raw.get("room_width_m", raw.get("floor_width_m", data.room_width_m)))
            data.room_depth_m = float(raw.get("room_depth_m", raw.get("floor_depth_m", data.room_depth_m)))
            data.aim_px = int(raw.get("aim_px", data.aim_px))
            data.aim_py = int(raw.get("aim_py", data.aim_py))
            data.floor_points = [list(map(float, p)) for p in floor_points]
            data.world_points = [list(map(float, p)) for p in raw.get("world_points", [])]

            for key in self.NUMERIC_FIELDS:
                if key in raw:
                    setattr(data, key, float(raw[key]))

            data.created_at = float(raw.get("created_at", data.created_at))
            data.updated_at = float(raw.get("updated_at", raw.get("created_at", data.updated_at)))

        return data

    def snapshot(self) -> CalibrationData:
        with self._lock:
            return CalibrationData(**asdict(self.data))

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.data.world_points = self._world_points_locked()
            self.data.updated_at = time.time()
            if not self.data.created_at:
                self.data.created_at = self.data.updated_at
            self.path.write_text(json.dumps(asdict(self.data), ensure_ascii=False, indent=2), encoding="utf-8")

    def set_value(self, key: str, value: float) -> None:
        if key not in self.NUMERIC_FIELDS:
            raise KeyError(f"Unknown calibration field: {key}")

        with self._lock:
            setattr(self.data, key, float(value))
            if key in {"room_width_m", "room_depth_m"}:
                self._invalidate_locked()
            self.save()

    def set_room_size(self, width_m: float, depth_m: float) -> None:
        with self._lock:
            self.data.room_width_m = float(width_m)
            self.data.room_depth_m = float(depth_m)
            self._invalidate_locked()
            self.save()

    def set_aim(self, x: int, y: int) -> None:
        with self._lock:
            self.data.aim_px = int(x)
            self.data.aim_py = int(y)
            self.save()

    def clear_floor_points(self) -> None:
        with self._lock:
            self.data.floor_points = []
            self._invalidate_locked()
            self.save()

    def add_floor_point(self, x: int, y: int) -> int:
        with self._lock:
            if len(self.data.floor_points or []) >= 4:
                self.data.floor_points = []
            self.data.floor_points.append([float(x), float(y)])
            if len(self.data.floor_points) > 4:
                self.data.floor_points = self.data.floor_points[:4]
            self._invalidate_locked()
            self.save()
            return len(self.data.floor_points)

    def pixel_to_floor(self, px: float, py: float) -> Optional[GeoPoint]:
        with self._lock:
            H = self._homography_locked()
            pts = np.array(self.data.floor_points or [], dtype=np.float32)
            width = float(self.data.room_width_m)
            depth = float(self.data.room_depth_m)

        if H is None or len(pts) != 4:
            return None

        p = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, H)[0][0]
        x_m, y_m = float(out[0]), float(out[1])
        dist = math.sqrt(x_m * x_m + y_m * y_m)
        inside_room = 0 <= x_m <= width and 0 <= y_m <= depth
        inside_cal = cv2.pointPolygonTest(pts.astype(np.int32), (float(px), float(py)), False) >= 0

        return GeoPoint(
            x_m=x_m,
            y_m=y_m,
            distance_m=dist,
            inside_room=inside_room,
            inside_calibration_zone=inside_cal,
        )

    def _world_points_locked(self) -> list[list[float]]:
        return [
            [0.0, 0.0],
            [float(self.data.room_width_m), 0.0],
            [float(self.data.room_width_m), float(self.data.room_depth_m)],
            [0.0, float(self.data.room_depth_m)],
        ]

    def _homography_locked(self) -> Optional[np.ndarray]:
        if self._H is not None:
            return self._H
        if len(self.data.floor_points or []) != 4:
            return None

        src = np.array(self.data.floor_points, dtype=np.float32)
        dst = np.array(self._world_points_locked(), dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        self._H = H
        return self._H

    def _invalidate_locked(self) -> None:
        self._H = None
        self._version += 1
