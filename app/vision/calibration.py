from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import cv2
import numpy as np

from app.config import ConfigManager
from app.types import GeoPoint

# Calibration modes
CAL_MODE_XY = "xy"            # user enters real-world X,Y per point (tape measure)
CAL_MODE_LASER = "laser_angle" # user enters laser distance + compass angle per point
CAL_MODE_HYBRID = "hybrid"    # user enters X,Y + optional measured distance for validation
CAL_MODES = (CAL_MODE_XY, CAL_MODE_LASER, CAL_MODE_HYBRID)

_MODE_LABELS = {
    CAL_MODE_XY: "XY-координаты",
    CAL_MODE_LASER: "Лазер + угол",
    CAL_MODE_HYBRID: "Гибрид",
}

# Fields required per mode for point entry (in order shown to user)
_MODE_ENTRY_FIELDS: dict[str, list[tuple[str, str]]] = {
    CAL_MODE_XY:     [("x_m", "X от AIM, м"), ("y_m", "Y от AIM, м")],
    CAL_MODE_LASER:  [("dist_m", "Дист от AIM, м"), ("angle_deg", "Угол, °")],
    CAL_MODE_HYBRID: [("x_m", "X от AIM, м"), ("y_m", "Y от AIM, м"), ("dist_m", "Дист (контроль), м")],
}


def mode_entry_fields(mode: str) -> list[tuple[str, str]]:
    return _MODE_ENTRY_FIELDS.get(mode, _MODE_ENTRY_FIELDS[CAL_MODE_XY])


def mode_label(mode: str) -> str:
    return _MODE_LABELS.get(mode, mode)


@dataclass(slots=True)
class CalibrationData:
    room_width_m: float = 2.5
    room_depth_m: float = 2.5

    aim_px: int = 320
    aim_py: int = 240
    floor_points: list | None = None    # legacy: 4 pixel corners
    world_points: list | None = None    # legacy: 4 world corners

    camera_height_m: float = 2.5
    camera_pitch_deg: float = 45.0
    camera_yaw_deg: float = 0.0
    camera_roll_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 55.0
    rotation_deg: float = 0.0
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0

    # New calibration system
    cal_mode: str = CAL_MODE_XY
    cam_to_aim_m: float = 0.0          # laser: AIM point → camera lens (metres)
    camera_floor_x_m: float = 0.0     # camera floor-projection X offset from AIM (0 = above AIM)
    camera_floor_y_m: float = 0.0     # camera floor-projection Y offset from AIM
    cal_points: list | None = None     # list of dicts {px,py,x_m,y_m,dist_m,angle_deg}

    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.floor_points is None:
            self.floor_points = []
        if self.world_points is None:
            self.world_points = []
        if self.cal_points is None:
            self.cal_points = []


class CalibrationManager:
    """Floor calibration: 3 modes (XY / laser-angle / hybrid) + legacy homography fallback."""

    NUMERIC_FIELDS = {
        "room_width_m", "room_depth_m",
        "camera_height_m", "camera_pitch_deg", "camera_yaw_deg", "camera_roll_deg",
        "hfov_deg", "vfov_deg", "rotation_deg",
        "lens_distortion_k1", "lens_distortion_k2",
        "cam_to_aim_m", "camera_floor_x_m", "camera_floor_y_m",
    }

    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._H: Optional[np.ndarray] = None
        self._version = 0
        self.path = self.config.resolve_path(self.config.get().calibration.file)
        self.data = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

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

            data.cal_mode = str(raw.get("cal_mode", data.cal_mode))
            data.cal_points = list(raw.get("cal_points", []))
            data.created_at = float(raw.get("created_at", data.created_at))
            data.updated_at = float(raw.get("updated_at", raw.get("created_at", data.updated_at)))

        return data

    def snapshot(self) -> CalibrationData:
        with self._lock:
            return CalibrationData(**asdict(self.data))

    def save(self) -> None:
        with self._lock:
            self.data.updated_at = time.time()
            if not self.data.created_at:
                self.data.created_at = self.data.updated_at
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(asdict(self.data), ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # ── setters ──────────────────────────────────────────────────────────────

    def set_value(self, key: str, value: float) -> None:
        if key not in self.NUMERIC_FIELDS:
            raise KeyError(f"Unknown calibration field: {key}")
        with self._lock:
            setattr(self.data, key, float(value))
            if key in {"room_width_m", "room_depth_m"}:
                self._invalidate_locked()
            self.save()

    def set_aim(self, x: int, y: int) -> None:
        with self._lock:
            self.data.aim_px = int(x)
            self.data.aim_py = int(y)
            self.save()

    def set_cal_mode(self, mode: str) -> None:
        if mode not in CAL_MODES:
            return
        with self._lock:
            self.data.cal_mode = mode
            self._invalidate_locked()
            self.save()

    # ── calibration points management ─────────────────────────────────────────

    def add_cal_point(
        self,
        px: int, py: int,
        x_m: float = 0.0, y_m: float = 0.0,
        dist_m: float = 0.0, angle_deg: float = 0.0,
    ) -> int:
        with self._lock:
            pts = self.data.cal_points or []
            pts.append({
                "px": int(px), "py": int(py),
                "x_m": float(x_m), "y_m": float(y_m),
                "dist_m": float(dist_m), "angle_deg": float(angle_deg),
            })
            self.data.cal_points = pts
            self._invalidate_locked()
            self.save()
            return len(pts)

    def remove_last_cal_point(self) -> None:
        with self._lock:
            pts = self.data.cal_points or []
            if pts:
                pts.pop()
                self.data.cal_points = pts
                self._invalidate_locked()
                self.save()

    def clear_cal_points(self) -> None:
        with self._lock:
            self.data.cal_points = []
            self._invalidate_locked()
            self.save()

    # Legacy floor-points API (kept for fallback)

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
            self._invalidate_locked()
            self.save()
            return len(self.data.floor_points)

    def set_room_size(self, width_m: float, depth_m: float) -> None:
        with self._lock:
            self.data.room_width_m = float(width_m)
            self.data.room_depth_m = float(depth_m)
            self._invalidate_locked()
            self.save()

    # ── coordinate mapping ────────────────────────────────────────────────────

    def pixel_to_floor(self, px: float, py: float) -> Optional[GeoPoint]:
        with self._lock:
            H = self._homography_locked()
            pts_cal = list(self.data.cal_points or [])
            mode = self.data.cal_mode
            cam_to_aim = float(self.data.cam_to_aim_m)
            cx = float(self.data.camera_floor_x_m)
            cy = float(self.data.camera_floor_y_m)
            cam_h = float(self.data.camera_height_m)
            width = float(self.data.room_width_m)
            depth = float(self.data.room_depth_m)
            fp = list(self.data.floor_points or [])

        if H is None:
            return None

        p = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, H)[0][0]
        x_m, y_m = float(out[0]), float(out[1])

        # Floor distance from AIM
        dist_floor = math.sqrt(x_m * x_m + y_m * y_m)

        # Camera 3D position relative to AIM:
        # cam_to_aim_m is the straight-line laser distance AIM → lens.
        # camera_floor_x/y_m is where the camera projects onto the floor.
        if cam_to_aim > 0:
            r2 = cx * cx + cy * cy
            cz = math.sqrt(max(0.0, cam_to_aim * cam_to_aim - r2))
        else:
            cz = cam_h
            cx = cy = 0.0

        dist_cam = math.sqrt((x_m - cx) ** 2 + (y_m - cy) ** 2 + cz * cz)

        # inside_calibration_zone: pixel polygon test
        if pts_cal:
            pix = np.array([[pt["px"], pt["py"]] for pt in pts_cal], dtype=np.int32)
            hull = cv2.convexHull(pix)
            inside_cal = cv2.pointPolygonTest(hull, (float(px), float(py)), False) >= 0
            world_xys = [_point_to_xy(pt, mode) for pt in pts_cal]
            min_x = min(w[0] for w in world_xys)
            max_x = max(w[0] for w in world_xys)
            min_y = min(w[1] for w in world_xys)
            max_y = max(w[1] for w in world_xys)
            inside_room = min_x <= x_m <= max_x and min_y <= y_m <= max_y
        else:
            inside_room = 0 <= x_m <= width and 0 <= y_m <= depth
            pts_arr = np.array(fp, dtype=np.int32)
            inside_cal = (
                cv2.pointPolygonTest(pts_arr, (float(px), float(py)), False) >= 0
                if len(fp) >= 3 else False
            )

        return GeoPoint(
            x_m=x_m,
            y_m=y_m,
            distance_m=dist_floor,
            inside_room=inside_room,
            inside_calibration_zone=inside_cal,
            distance_cam_m=dist_cam,
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _homography_locked(self) -> Optional[np.ndarray]:
        if self._H is not None:
            return self._H

        pts_cal = self.data.cal_points or []
        mode = self.data.cal_mode

        if len(pts_cal) >= 4:
            src = np.array([[float(pt["px"]), float(pt["py"])] for pt in pts_cal], dtype=np.float32)
            dst = np.array([list(_point_to_xy(pt, mode)) for pt in pts_cal], dtype=np.float32)
            method = cv2.RANSAC if len(pts_cal) > 4 else 0
            H, _ = cv2.findHomography(src, dst, method)
            self._H = H
            return H

        # Legacy: 4 floor pixels → room rectangle
        fp = self.data.floor_points or []
        if len(fp) != 4:
            return None
        src = np.array(fp, dtype=np.float32)
        dst = np.array(self._legacy_world_points(), dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        self._H = H
        return H

    def _legacy_world_points(self) -> list[list[float]]:
        w, d = float(self.data.room_width_m), float(self.data.room_depth_m)
        return [[0.0, 0.0], [w, 0.0], [w, d], [0.0, d]]

    def _invalidate_locked(self) -> None:
        self._H = None
        self._version += 1

    def reload(self) -> None:
        with self._lock:
            self.data = self._load()
            self._invalidate_locked()

    # hybrid mode: validate inter-point distances (returns list of (idx_a, idx_b, measured, computed, error_m))
    def validate_hybrid_distances(self) -> list[tuple[int, int, float, float, float]]:
        pts = self.data.cal_points or []
        mode = self.data.cal_mode
        if mode != CAL_MODE_HYBRID or len(pts) < 2:
            return []
        results = []
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            ax, ay = _point_to_xy(a, mode)
            bx, by = _point_to_xy(b, mode)
            computed = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            measured = float(b.get("dist_m", 0.0))
            if measured > 0:
                results.append((i, i + 1, measured, computed, abs(measured - computed)))
        return results


def _point_to_xy(pt: dict, mode: str) -> tuple[float, float]:
    """Convert a cal_point dict to world (x_m, y_m) based on calibration mode."""
    if mode == CAL_MODE_LASER:
        rad = math.radians(float(pt.get("angle_deg", 0.0)))
        dist = float(pt.get("dist_m", 0.0))
        return dist * math.sin(rad), dist * math.cos(rad)
    # xy and hybrid: use direct x_m, y_m
    return float(pt.get("x_m", 0.0)), float(pt.get("y_m", 0.0))
