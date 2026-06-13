from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.types import TrackSnapshot, VisionPacket
from app.vision.calibration import CalibrationData

POSE_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Editable numeric fields shown in the panel. Camera height drives the
# camera-distance calculation, so it comes first.
FIELD_ORDER = [
    ("camera_height_m",   "Cam height, m"),
    ("room_width_m",      "Room W, m"),
    ("room_depth_m",      "Room D, m"),
    ("camera_pitch_deg",  "Pitch, deg"),
    ("camera_yaw_deg",    "Yaw, deg"),
    ("camera_roll_deg",   "Roll, deg"),
    ("hfov_deg",          "HFOV, deg"),
    ("vfov_deg",          "VFOV, deg"),
    ("rotation_deg",      "Rot, deg"),
    ("lens_distortion_k1", "Lens k1"),
    ("lens_distortion_k2", "Lens k2"),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_AIM_COLOR = (60, 90, 255)      # warm red (BGR)
_FLOOR_COLOR = (255, 190, 60)   # cyan-blue (BGR)


def _t(img: np.ndarray, text: str, org: tuple[int, int],
       scale: float = 0.48, color: tuple[int, int, int] = (235, 235, 235),
       thickness: int = 1) -> None:
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


# ── calibration overlay ─────────────────────────────────────────────────────────

def draw_calibration(
    frame: np.ndarray,
    cal: CalibrationData,
    scale: float = 1.0,
    zone_polygon_px: Optional[list] = None,
) -> None:
    # Saved zones (behind everything), semi-transparent.
    for zone in (cal.zones or []):
        poly_px = zone.get("polygon_px", [])
        if len(poly_px) < 3:
            continue
        name = zone.get("name", "zone")
        color = tuple(int(c) for c in zone.get("color", [0, 200, 200]))
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in poly_px], dtype=np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, 0.16, frame, 0.84, 0, frame)
        cv2.polylines(frame, [pts], True, color, 1, cv2.LINE_AA)
        cx_z, cy_z = int(pts[:, 0].mean()), int(pts[:, 1].mean())
        (tw, _th), _ = cv2.getTextSize(name, _FONT, 0.48, 1)
        cv2.rectangle(frame, (cx_z - tw // 2 - 3, cy_z - 14), (cx_z + tw // 2 + 3, cy_z + 4), (0, 0, 0), -1)
        _t(frame, name, (cx_z - tw // 2, cy_z), 0.48, (255, 255, 255), 1)

    # In-progress zone polygon.
    if zone_polygon_px:
        z_pts = [(int(p[0] * scale), int(p[1] * scale)) for p in zone_polygon_px]
        for i in range(1, len(z_pts)):
            cv2.line(frame, z_pts[i - 1], z_pts[i], (0, 255, 128), 1, cv2.LINE_AA)
        for zp in z_pts:
            cv2.circle(frame, zp, 4, (0, 255, 128), -1, cv2.LINE_AA)
        if len(z_pts) >= 3:
            cv2.line(frame, z_pts[-1], z_pts[0], (0, 255, 128), 1, cv2.LINE_AA)

    # Floor calibration quad: thin lines, neat points, translucent fill.
    pts = [(int(p[0] * scale), int(p[1] * scale)) for p in (cal.floor_points or [])]
    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], _FLOOR_COLOR, 1, cv2.LINE_AA)
    if len(pts) == 4:
        cv2.line(frame, pts[3], pts[0], _FLOOR_COLOR, 1, cv2.LINE_AA)
        ov = frame.copy()
        cv2.fillPoly(ov, [np.array(pts, dtype=np.int32)], _FLOOR_COLOR)
        cv2.addWeighted(ov, 0.14, frame, 0.86, 0, frame)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 4, _FLOOR_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 5, (255, 255, 255), 1, cv2.LINE_AA)
        _t(frame, str(i + 1), (p[0] + 8, p[1] - 8), 0.45, (255, 255, 255), 1)

    # AIM marker: thin cross + ring + center dot.
    ax, ay = int(cal.aim_px * scale), int(cal.aim_py * scale)
    cv2.drawMarker(frame, (ax, ay), _AIM_COLOR, cv2.MARKER_CROSS, 22, 1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 8, _AIM_COLOR, 1, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 2, _AIM_COLOR, -1, cv2.LINE_AA)
    _t(frame, "AIM", (ax + 12, ay - 8), 0.48, _AIM_COLOR, 1)


# ── track overlay ───────────────────────────────────────────────────────────────

def draw_pose(frame: np.ndarray, keypoints: Optional[np.ndarray], scale: float = 1.0) -> None:
    if keypoints is None:
        return
    for a, b in POSE_EDGES:
        if len(keypoints) > max(a, b):
            ax, ay = int(keypoints[a][0] * scale), int(keypoints[a][1] * scale)
            bx, by = int(keypoints[b][0] * scale), int(keypoints[b][1] * scale)
            if ax > 1 and ay > 1 and bx > 1 and by > 1:
                cv2.line(frame, (ax, ay), (bx, by), (0, 220, 255), 2, cv2.LINE_AA)
    for p in keypoints:
        x, y = int(p[0] * scale), int(p[1] * scale)
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1, cv2.LINE_AA)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackSnapshot],
    scale: float = 1.0,
    show_pose: bool = True,
    show_tracks: bool = True,
) -> None:
    for tr in tracks:
        if show_pose:
            draw_pose(frame, tr.keypoints, scale)

        x1, y1 = int(tr.box[0] * scale), int(tr.box[1] * scale)
        x2, y2 = int(tr.box[2] * scale), int(tr.box[3] * scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 190, 60), 2, cv2.LINE_AA)

        fx, fy = int(tr.foot[0] * scale), int(tr.foot[1] * scale)
        cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (fx, fy), 6, (255, 255, 255), 1, cv2.LINE_AA)

        if show_tracks:
            hist = [(int(p[0] * scale), int(p[1] * scale)) for p in tr.history]
            for i in range(1, len(hist)):
                cv2.line(frame, hist[i - 1], hist[i], (0, 230, 255), 2, cv2.LINE_AA)

        parts = [f"ID{tr.track_id}", tr.state]
        if tr.motion and tr.motion != "stationary":
            parts.append(tr.motion)
        if tr.activity:
            parts.append(tr.activity)
        parts.append(f"{tr.conf:.2f}")
        if tr.geo:
            parts.append(f"D:{tr.geo.distance_cam_m:.2f}m")
            parts.append(f"({tr.geo.x_m:.2f},{tr.geo.y_m:.2f})")
            if tr.geo.zone:
                parts.append(f"[{tr.geo.zone}]")

        txt = "  ".join(parts)
        fh, fw = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.46, 1)
        tx = max(2, min(fx + 10, fw - tw - 6))
        ty = max(th + 8, min(fy - 8, fh - 6))
        cv2.rectangle(frame, (tx - 3, ty - th - 6), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        _t(frame, txt, (tx, ty), 0.46, (255, 255, 255), 1)


# ── side panel ────────────────────────────────────────────────────────────────

def draw_panel(
    canvas: np.ndarray,
    panel_x: int,
    packet: Optional[VisionPacket],
    ui_state: dict,
    buttons: dict,
    fields: dict,
    field_values: dict[str, str],
    editing_key: Optional[str],
) -> None:
    h, w = canvas.shape[:2]
    cv2.rectangle(canvas, (panel_x, 0), (w, h), (24, 24, 28), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), (80, 80, 90), 1)

    pad = 12
    x = panel_x + pad
    right = w - pad
    y = 20

    LINE = 19
    FIELD_H = 26
    BTN_H = 30

    calib_mode = ui_state.get("calib_mode")
    floor_count = ui_state.get("floor_count", 0)
    zones = ui_state.get("zones", [])
    zone_draw = ui_state.get("zone_draw_active", False)
    zone_nm_mode = ui_state.get("zone_name_mode", False)
    zone_polygon = ui_state.get("zone_polygon_px", [])
    zone_nm_buf = ui_state.get("zone_name_buf", "")
    activity_enabled = ui_state.get("activity_enabled", False)
    activity_available = ui_state.get("activity_available", False)

    def _check_y(extra: int = 0) -> bool:
        return y + extra < h - 30

    def section(title: str, color: tuple = (210, 210, 220)) -> None:
        nonlocal y
        if not _check_y(20):
            return
        y += 6
        _t(canvas, title, (x, y), 0.46, color, 1)
        y += 4
        cv2.line(canvas, (x, y), (right, y), (62, 62, 70), 1)
        y += 14

    def row(label: str, value: str, vcol: tuple = (235, 235, 235)) -> None:
        nonlocal y
        if not _check_y():
            return
        _t(canvas, label, (x, y), 0.40, (155, 155, 162), 1)
        _t(canvas, value, (x + 122, y), 0.40, vcol, 1)
        y += LINE

    def button(name: str, label: str, active: bool = False, col: Optional[tuple] = None) -> None:
        nonlocal y
        if not _check_y(BTN_H):
            return
        x1, y1, x2, y2 = x, y, right, y + BTN_H
        buttons[name] = (x1, y1, x2, y2)
        if col:
            bg = col if active else tuple(max(0, c - 30) for c in col)
            brd = col
        else:
            bg = (70, 70, 90) if active else (44, 44, 54)
            brd = (190, 190, 210) if active else (95, 95, 110)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 1)
        _t(canvas, label, (x1 + 8, y1 + 20), 0.43, (245, 245, 250), 1)
        y += BTN_H + 4

    def field(key: str, label: str) -> None:
        nonlocal y
        if not _check_y(FIELD_H + 2):
            return
        active = (editing_key == key)
        _t(canvas, label, (x, y + 17), 0.38, (160, 160, 168), 1)
        bx1, by1, bx2, by2 = x + 120, y, right, y + FIELD_H
        fields[key] = (bx1, by1, bx2, by2)
        bg = (65, 58, 110) if active else (38, 38, 46)
        brd = (175, 148, 255) if active else (78, 78, 92)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), brd, 1)
        val = field_values.get(key, "")
        disp = (val + "_") if active else val
        _t(canvas, disp, (bx1 + 5, by1 + 18), 0.42, (255, 255, 255), 1)
        y += FIELD_H + 4

    # ── STATUS ──────────────────────────────────────────────────────────────
    section("STATUS")
    p = packet
    row("FPS in/inf", f"{(p.reader_fps if p else 0):.1f} / {(p.inference_fps if p else 0):.1f}")
    row("Infer ms", f"{(p.infer_ms if p else 0):.0f}")
    row("CPU/RAM", f"{(p.cpu_percent if p else 0):.0f}% / {(p.ram_percent if p else 0):.0f}%")
    row("People", str(len(p.tracks) if p else 0))
    fc_col = (90, 220, 90) if floor_count == 4 else (220, 190, 60) if floor_count else (160, 160, 170)
    mode_txt = f"  [{calib_mode}]" if calib_mode else ""
    row("Floor pts", f"{floor_count}/4{mode_txt}", fc_col)

    # ── ACTIONS ─────────────────────────────────────────────────────────────
    section("ACTIONS")
    button("aim", "Set AIM  [A]", active=(calib_mode == "aim"))
    button("floor4", "Set 4 floor points  [F]", active=(calib_mode == "floor4"))
    button("save", "Save  [S]")
    if activity_available:
        act_label = "Activity: ON  [toggle]" if activity_enabled else "Activity: OFF [toggle]"
        act_col = (40, 130, 40) if activity_enabled else (80, 80, 90)
        button("toggle_activity", act_label, active=activity_enabled, col=act_col)

    # ── ZONES ───────────────────────────────────────────────────────────────
    section("ZONES")
    if zone_nm_mode and _check_y(FIELD_H + LINE):
        bx1, by1, bx2, by2 = x, y, right, y + FIELD_H
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (28, 52, 52), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 200), 1)
        _t(canvas, f"Name: {zone_nm_buf}_", (bx1 + 5, by1 + 18), 0.42, (0, 230, 230), 1)
        y += FIELD_H + 3
        if _check_y():
            _t(canvas, "Enter=save  Esc=cancel", (x, y), 0.34, (90, 170, 170), 1)
            y += LINE
    elif zone_draw and _check_y(34):
        n_z = len(zone_polygon)
        bx1, by1, bx2, by2 = x, y, right, y + 32
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 48, 22), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 100), 1)
        _t(canvas, f">> CLICK IN VIDEO ({n_z} pts)", (bx1 + 5, by1 + 13), 0.38, (0, 240, 120), 1)
        _t(canvas, "Enter=finish  Esc=cancel", (bx1 + 5, by1 + 27), 0.32, (0, 180, 90), 1)
        y += 36
        if n_z >= 3:
            button("zone_finish", f"Finish zone ({n_z} pts)", col=(0, 140, 60))
    else:
        button("zone_draw", "+ Draw zone")
        if zones and _check_y(BTN_H):
            button("zone_delete_last", "- Delete last zone")
        for zone in zones[-4:]:
            if not _check_y():
                break
            zname = zone.get("name", "zone")
            zcolor = tuple(int(c) for c in zone.get("color", [0, 200, 200]))
            n_zpts = len(zone.get("polygon_px", []))
            cv2.circle(canvas, (x + 6, y - LINE // 2), 5, zcolor, -1)
            _t(canvas, f"  {zname} ({n_zpts} pts)", (x, y), 0.36, (130, 195, 155), 1)
            y += LINE

    # ── CALIBRATION FIELDS ──────────────────────────────────────────────────
    if _check_y(20):
        section("CALIBRATION")
        for key, label in FIELD_ORDER:
            if not _check_y(FIELD_H + 2):
                break
            field(key, label)

    # ── HINTS ───────────────────────────────────────────────────────────────
    _t(canvas, "A:AIM  F:floor  S:save  Esc:quit", (x, h - 14), 0.32, (120, 120, 130), 1)
