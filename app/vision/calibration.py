from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import ConfigManager
from app.types import GeoPoint

log = logging.getLogger("calibration")


@dataclass(slots=True)
class CalibrationData:
    """Floor calibration: 4 floor pixels -> room rectangle homography.

    The model is intentionally simple — everything is computed from:
      * floor_points: 4 pixel corners clicked clockwise on the floor
      * room_width_m / room_depth_m: the real size of that rectangle
      * aim_px / aim_py: the floor point the camera sits above
      * camera_height_m: lens height above the floor
    Distances are then derived automatically.
    """

    room_width_m: float = 2.5
    room_depth_m: float = 2.5

    aim_px: int = 320
    aim_py: int = 240
    floor_points: list | None = None    # 4 pixel corners clockwise
    world_points: list | None = None    # 4 world corners (derived from room size)

    camera_height_m: float = 2.5
    cam_to_aim_m: float = 0.0           # measured laser distance lens -> AIM (0 = use height)
    camera_pitch_deg: float = 45.0
    camera_yaw_deg: float = 0.0
    camera_roll_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 55.0
    rotation_deg: float = 0.0
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0

    # Trapezoid calibration (preferred): 4 clicked image corners P1..P4,
    # plus measured edge lengths and interior angles. P1 is the world origin.
    quad_px: list | None = None         # [[x,y], ...] 4 image corners
    trap_edges_m: list | None = None    # [AB, BC, CD, DA] metres
    trap_angles_deg: list | None = None  # [A1, A2, A3, A4] interior angles

    zones: list | None = None           # list of dicts {name, polygon_px, color}

    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        if self.floor_points is None:
            self.floor_points = []
        if self.world_points is None:
            self.world_points = []
        if self.quad_px is None:
            self.quad_px = []
        if self.trap_edges_m is None:
            self.trap_edges_m = [0.0, 0.0, 0.0, 0.0]
        if self.trap_angles_deg is None:
            self.trap_angles_deg = [90.0, 90.0, 90.0, 90.0]
        if self.zones is None:
            self.zones = []


class CalibrationManager:
    """Homography-based floor calibration plus editable camera geometry."""

    NUMERIC_FIELDS = {
        "room_width_m", "room_depth_m",
        "camera_height_m", "cam_to_aim_m",
        "camera_pitch_deg", "camera_yaw_deg", "camera_roll_deg",
        "hfov_deg", "vfov_deg", "rotation_deg",
        "lens_distortion_k1", "lens_distortion_k2",
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

            data.quad_px = [list(map(float, p)) for p in raw.get("quad_px", [])]
            if "trap_edges_m" in raw:
                data.trap_edges_m = [float(v) for v in raw["trap_edges_m"]]
            if "trap_angles_deg" in raw:
                data.trap_angles_deg = [float(v) for v in raw["trap_angles_deg"]]
            data.zones = list(raw.get("zones", []))
            data.created_at = float(raw.get("created_at", data.created_at))
            data.updated_at = float(raw.get("updated_at", raw.get("created_at", data.updated_at)))

        return data

    def snapshot(self) -> CalibrationData:
        with self._lock:
            return CalibrationData(**asdict(self.data))

    def save(self) -> None:
        with self._lock:
            self.data.world_points = self._world_points_locked()
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

    def set_room_size(self, width_m: float, depth_m: float) -> None:
        with self._lock:
            self.data.room_width_m = float(width_m)
            self.data.room_depth_m = float(depth_m)
            self._invalidate_locked()
            self.save()

    # ── floor points (4-corner calibration) ───────────────────────────────────

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

    # ── trapezoid calibration ───────────────────────────────────────────────────

    def set_quad_point(self, index: int, x: int, y: int) -> int:
        """Set one of the 4 image corners P1..P4 (index 0..3). Cycles after 4."""
        with self._lock:
            q = list(self.data.quad_px or [])
            if index == 0 and len(q) >= 4:
                q = []
            while len(q) <= index:
                q.append([0.0, 0.0])
            q[index] = [float(x), float(y)]
            self.data.quad_px = q[:4]
            self._invalidate_locked()
            self.save()
            return len(self.data.quad_px)

    def clear_quad(self) -> None:
        with self._lock:
            self.data.quad_px = []
            self._invalidate_locked()
            self.save()

    def set_trap_edge(self, index: int, value: float) -> None:
        with self._lock:
            e = list(self.data.trap_edges_m or [0.0, 0.0, 0.0, 0.0])
            while len(e) < 4:
                e.append(0.0)
            if 0 <= index < 4:
                e[index] = float(value)
            self.data.trap_edges_m = e
            self._invalidate_locked()
            self.save()

    def set_trap_angle(self, index: int, value: float) -> None:
        with self._lock:
            a = list(self.data.trap_angles_deg or [90.0, 90.0, 90.0, 90.0])
            while len(a) < 4:
                a.append(90.0)
            if 0 <= index < 4:
                a[index] = float(value)
            self.data.trap_angles_deg = a
            self._invalidate_locked()
            self.save()

    def zones_world(self) -> list[dict]:
        """Zones with polygons in metres and normalised (0..1 of room) — for a
        floorplan/SVG card. polygon_m/polygon_norm present only when calibrated."""
        with self._lock:
            H = self._homography_locked()
            zones = list(self.data.zones or [])
            w = float(self.data.room_width_m)
            d = float(self.data.room_depth_m)
        out: list[dict] = []
        for z in zones:
            poly_px = z.get("polygon_px", [])
            item: dict = {"name": z.get("name", ""), "color": z.get("color"),
                          "polygon_px": poly_px}
            if H is not None and len(poly_px) >= 3:
                pm, pn = [], []
                for p in poly_px:
                    xm, ym = self._project(H, float(p[0]), float(p[1]))
                    pm.append([round(xm, 3), round(ym, 3)])
                    pn.append([round(xm / w, 4) if w > 0 else 0.0,
                               round(ym / d, 4) if d > 0 else 0.0])
                item["polygon_m"] = pm
                item["polygon_norm"] = pn
            out.append(item)
        return out

    def trapezoid_closure_error(self) -> Optional[float]:
        """Metres mismatch between the walked P4->P1 edge and the entered DA."""
        with self._lock:
            edges = list(self.data.trap_edges_m or [])
            angles = list(self.data.trap_angles_deg or [90.0, 90.0, 90.0, 90.0])
        if len(edges) != 4 or not all(e > 0 for e in edges[:3]):
            return None
        _pts, err = trapezoid_world_points(edges, angles)
        return err

    # ── zone management ────────────────────────────────────────────────────────

    def add_zone(self, name: str, polygon_px: list, color: list | None = None) -> None:
        _color = color or [0, 200, 200]
        with self._lock:
            zones = list(self.data.zones or [])
            zones.append({
                "name": name,
                "polygon_px": [[int(p[0]), int(p[1])] for p in polygon_px],
                "color": _color,
            })
            self.data.zones = zones
            self.save()

    def delete_zone(self, index: int) -> None:
        with self._lock:
            zones = list(self.data.zones or [])
            if 0 <= index < len(zones):
                zones.pop(index)
                self.data.zones = zones
                self.save()

    def clear_zones(self) -> None:
        with self._lock:
            self.data.zones = []
            self.save()

    # ── coordinate mapping ─────────────────────────────────────────────────────

    def pixel_to_floor(self, px: float, py: float) -> Optional[GeoPoint]:
        with self._lock:
            H = self._homography_locked()
            width = float(self.data.room_width_m)
            depth = float(self.data.room_depth_m)
            cam_h = float(self.data.camera_height_m)
            cam_to_aim = float(self.data.cam_to_aim_m)
            aim_px = float(self.data.aim_px)
            aim_py = float(self.data.aim_py)
            fp = list(self.data.floor_points or [])
            quad = list(self.data.quad_px or [])
            zones_data = list(self.data.zones or [])

        if H is None:
            log.debug(
                "pixel_to_floor px=(%.0f,%.0f) -> no homography (need 4 floor points)",
                px, py,
            )
            return None

        # Person foot -> floor metres
        x_m, y_m = self._project(H, px, py)
        # AIM (camera ground spot) -> floor metres
        ax_m, ay_m = self._project(H, aim_px, aim_py)

        # Floor distance from AIM, and 3D distance from the camera lens.
        # Camera is modelled directly above AIM; its elevation is the measured
        # laser distance to AIM when provided, otherwise the estimated height.
        dist_floor = math.hypot(x_m - ax_m, y_m - ay_m)
        elev = cam_to_aim if cam_to_aim > 0 else cam_h
        dist_cam = math.sqrt(dist_floor * dist_floor + elev * elev)

        # Inside-the-calibrated-area test uses the active image polygon (the
        # trapezoid quad if defined, else the legacy rectangle). Points outside
        # are still projected — the planar homography extrapolates exactly.
        poly = quad if len(quad) == 4 else fp
        inside_cal = (
            cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (float(px), float(py)), False) >= 0
            if len(poly) == 4 else False
        )
        inside_room = inside_cal if len(quad) == 4 else (0.0 <= x_m <= width and 0.0 <= y_m <= depth)

        zone_name = ""
        for zone in zones_data:
            poly_px = zone.get("polygon_px", [])
            if len(poly_px) >= 3:
                poly_arr = np.array(poly_px, dtype=np.int32)
                if cv2.pointPolygonTest(poly_arr, (float(px), float(py)), False) >= 0:
                    zone_name = zone.get("name", "")
                    break

        log.debug(
            "pixel_to_floor px=(%.0f,%.0f) -> x_m=%.3f y_m=%.3f dist_floor=%.3f "
            "dist_cam=%.3f inside_room=%s zone=%r",
            px, py, x_m, y_m, dist_floor, dist_cam, inside_room, zone_name,
        )
        return GeoPoint(
            x_m=x_m,
            y_m=y_m,
            distance_m=dist_floor,
            inside_room=inside_room,
            inside_calibration_zone=inside_cal,
            distance_cam_m=dist_cam,
            zone=zone_name,
        )

    # ── internal ───────────────────────────────────────────────────────────────

    @staticmethod
    def _project(H: np.ndarray, px: float, py: float) -> tuple[float, float]:
        p = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, H)[0][0]
        return float(out[0]), float(out[1])

    def _world_points_locked(self) -> list[list[float]]:
        w, d = float(self.data.room_width_m), float(self.data.room_depth_m)
        return [[0.0, 0.0], [w, 0.0], [w, d], [0.0, d]]

    def _homography_locked(self) -> Optional[np.ndarray]:
        if self._H is not None:
            return self._H

        # Preferred: trapezoid (4 clicked corners + measured edges/angles).
        quad = self.data.quad_px or []
        edges = self.data.trap_edges_m or []
        if len(quad) == 4 and len(edges) == 4 and all(e > 0 for e in edges[:3]):
            world, _err = trapezoid_world_points(edges, self.data.trap_angles_deg or [90.0] * 4)
            src = np.array(quad, dtype=np.float32)
            dst = np.array(world, dtype=np.float32)
            H, _ = cv2.findHomography(src, dst)
            self._H = H
            return H

        # Fallback: legacy 4 floor points -> room rectangle.
        fp = self.data.floor_points or []
        if len(fp) != 4:
            return None
        src = np.array(fp, dtype=np.float32)
        dst = np.array(self._world_points_locked(), dtype=np.float32)
        H, _ = cv2.findHomography(src, dst)
        self._H = H
        return H

    def _invalidate_locked(self) -> None:
        self._H = None
        self._version += 1

    def reload(self) -> None:
        with self._lock:
            self.data = self._load()
            self._invalidate_locked()


def trapezoid_world_points(
    edges_m: list, angles_deg: list,
) -> tuple[list[tuple[float, float]], float]:
    """Build the 4 floor-plane world coordinates of a trapezoid.

    P1 is the origin (0, 0) and the first edge P1->P2 runs along +X. Then the
    polygon is walked using the interior angles: at each vertex the heading turns
    by (180 - interior_angle). Returns the 4 points [P1, P2, P3, P4] and the
    closure error (how far the walked P4->P1 distance differs from the entered
    DA length) — a quality check on the measurements.

    edges_m  = [AB, BC, CD, DA]   (metres)
    angles_deg = [A1, A2, A3, A4] (interior degrees; A2, A3 drive the walk)
    """
    ab, bc, cd, da = (float(e) for e in edges_m[:4])
    _a1, a2, a3, _a4 = (float(a) for a in angles_deg[:4])

    pts: list[tuple[float, float]] = [(0.0, 0.0)]
    heading = 0.0
    p = (0.0, 0.0)
    for length, turn_at in ((ab, None), (bc, a2), (cd, a3)):
        if turn_at is not None:
            heading += 180.0 - turn_at
        rad = math.radians(heading)
        p = (p[0] + length * math.cos(rad), p[1] + length * math.sin(rad))
        pts.append(p)

    closure = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
    return pts, abs(closure - da)
