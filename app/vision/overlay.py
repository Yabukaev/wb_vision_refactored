from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.types import TrackSnapshot, VisionPacket
from app.vision.calibration import (
    CAL_MODE_HYBRID, CAL_MODE_LASER, CAL_MODE_XY, _point_to_xy,
    CalibrationData,
)

POSE_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

FIELD_ORDER = [
    ("cam_to_aim_m",      "Laser AIM->cam, m"),
    ("camera_floor_x_m",  "Cam X offset, m"),
    ("camera_floor_y_m",  "Cam Y offset, m"),
    ("camera_height_m",   "Cam H (legacy), m"),
    ("room_width_m",      "Room W, m"),
    ("room_depth_m",      "Room D, m"),
    ("camera_pitch_deg",  "Pitch, deg"),
    ("camera_yaw_deg",    "Yaw, deg"),
    ("camera_roll_deg",   "Roll, deg"),
    ("hfov_deg",          "HFOV, deg"),
    ("vfov_deg",          "VFOV, deg"),
    ("rotation_deg",      "Rot, deg"),
    ("lens_distortion_k1","Lens k1"),
    ("lens_distortion_k2","Lens k2"),
]

# Only the params relevant to each calibration mode
_MODE_PARAMS: dict[str, list[tuple[str, str]]] = {
    CAL_MODE_XY: [
        ("cam_to_aim_m",     "AIM->lens, m"),
        ("camera_floor_x_m", "Cam offset X, m"),
        ("camera_floor_y_m", "Cam offset Y, m"),
        ("room_width_m",     "Room W, m"),
        ("room_depth_m",     "Room D, m"),
    ],
    CAL_MODE_LASER: [
        ("cam_to_aim_m",     "AIM->lens, m"),
        ("camera_floor_x_m", "Cam offset X, m"),
        ("camera_floor_y_m", "Cam offset Y, m"),
        ("room_width_m",     "Room W, m"),
        ("room_depth_m",     "Room D, m"),
    ],
    CAL_MODE_HYBRID: [
        ("cam_to_aim_m",     "AIM->lens, m"),
        ("camera_floor_x_m", "Cam offset X, m"),
        ("camera_floor_y_m", "Cam offset Y, m"),
        ("room_width_m",     "Room W, m"),
        ("room_depth_m",     "Room D, m"),
    ],
}

_MODE_COLORS = {
    CAL_MODE_XY:     (60, 180, 60),
    CAL_MODE_LASER:  (60, 130, 220),
    CAL_MODE_HYBRID: (200, 130, 40),
}

_MODE_BTN_LABELS = {
    CAL_MODE_XY:     "XY  (tape measure)",
    CAL_MODE_LASER:  "Laser + angle",
    CAL_MODE_HYBRID: "Hybrid",
}

_MODE_ENTRY_LABELS: dict[str, list[tuple[str, str]]] = {
    CAL_MODE_XY:     [("x_m", "X from AIM, m"), ("y_m", "Y from AIM, m")],
    CAL_MODE_LASER:  [("dist_m", "Dist from AIM, m"), ("angle_deg", "Angle, deg")],
    CAL_MODE_HYBRID: [("x_m", "X from AIM, m"), ("y_m", "Y from AIM, m"), ("dist_m", "Dist (check), m")],
}

_MODE_DESC: dict[str, str] = {
    CAL_MODE_XY:     "Per point: tape X, Y from AIM",
    CAL_MODE_LASER:  "Per point: laser dist + angle",
    CAL_MODE_HYBRID: "Per point: XY tape + dist check",
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _t(img: np.ndarray, text: str, org: tuple[int, int],
       scale: float = 0.48, color: tuple[int, int, int] = (235, 235, 235),
       thickness: int = 1) -> None:
    cv2.putText(img, text, org, _FONT, scale, color, thickness, cv2.LINE_AA)


# ── calibration overlay ───────────────────────────────────────────────────────

def draw_calibration(
    frame: np.ndarray,
    cal: CalibrationData,
    scale: float = 1.0,
    pending_px: Optional[int] = None,
    pending_py: Optional[int] = None,
    zone_polygon_px: Optional[list] = None,
) -> None:
    # Draw saved zones first (behind everything)
    for zone in (cal.zones or []):
        poly_px = zone.get("polygon_px", [])
        if len(poly_px) < 3:
            continue
        name = zone.get("name", "zone")
        color = tuple(int(c) for c in zone.get("color", [0, 200, 200]))
        pts = np.array([[int(p[0] * scale), int(p[1] * scale)] for p in poly_px], dtype=np.int32)
        ov = frame.copy()
        cv2.fillPoly(ov, [pts], color)
        cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [pts], True, color, 2, cv2.LINE_AA)
        cx_z = int(pts[:, 0].mean())
        cy_z = int(pts[:, 1].mean())
        (tw, _th), _ = cv2.getTextSize(name, _FONT, 0.50, 1)
        cv2.rectangle(frame, (cx_z - tw // 2 - 3, cy_z - 14), (cx_z + tw // 2 + 3, cy_z + 4), (0, 0, 0), -1)
        _t(frame, name, (cx_z - tw // 2, cy_z), 0.50, (255, 255, 255), 1)

    # Draw in-progress zone polygon
    if zone_polygon_px:
        z_pts = [(int(p[0] * scale), int(p[1] * scale)) for p in zone_polygon_px]
        for i in range(1, len(z_pts)):
            cv2.line(frame, z_pts[i - 1], z_pts[i], (0, 255, 128), 2, cv2.LINE_AA)
        for zp in z_pts:
            cv2.circle(frame, zp, 5, (0, 255, 128), -1, cv2.LINE_AA)
        if len(z_pts) >= 3:
            cv2.line(frame, z_pts[-1], z_pts[0], (0, 255, 128), 1, cv2.LINE_AA)

    mode = cal.cal_mode
    col = _MODE_COLORS.get(mode, (0, 0, 255))

    ax, ay = int(cal.aim_px * scale), int(cal.aim_py * scale)
    cv2.drawMarker(frame, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 7, (0, 0, 255), 2, cv2.LINE_AA)
    _t(frame, "AIM", (ax + 12, ay - 6), 0.52, (0, 80, 255), 2)

    pts_cal = cal.cal_points or []
    if pts_cal:
        scaled = [(int(pt["px"] * scale), int(pt["py"] * scale)) for pt in pts_cal]

        for i in range(1, len(scaled)):
            cv2.line(frame, scaled[i - 1], scaled[i], col, 1, cv2.LINE_AA)
        if len(scaled) >= 4:
            cv2.line(frame, scaled[-1], scaled[0], col, 1, cv2.LINE_AA)
            ov = frame.copy()
            cv2.fillPoly(ov, [np.array(scaled, dtype=np.int32)], col)
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)

        for sp in scaled:
            cv2.line(frame, (ax, ay), sp, (80, 80, 200), 1, cv2.LINE_AA)

        for i, (sp, pt) in enumerate(zip(scaled, pts_cal)):
            cv2.circle(frame, sp, 7, col, -1, cv2.LINE_AA)
            cv2.circle(frame, sp, 9, (255, 255, 255), 1, cv2.LINE_AA)
            wx, wy = _point_to_xy(pt, mode)
            _t(frame, f"P{i+1}({wx:.1f},{wy:.1f})", (sp[0] + 10, sp[1] + 5), 0.40, (255, 255, 255), 1)

    if pending_px is not None and pending_py is not None:
        spx, spy = int(pending_px * scale), int(pending_py * scale)
        cv2.drawMarker(frame, (spx, spy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(frame, (spx, spy), 10, (0, 255, 255), 2, cv2.LINE_AA)
        _t(frame, "Enter values ->", (spx + 14, spy + 5), 0.44, (0, 255, 255), 1)


# ── track overlay ─────────────────────────────────────────────────────────────

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
        cv2.circle(frame, (fx, fy), 6, (0, 0, 255), -1, cv2.LINE_AA)

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
            d = tr.geo.distance_cam_m if tr.geo.distance_cam_m > 0 else tr.geo.distance_m
            parts.append(f"D:{d:.2f}m")
            if tr.geo.zone:
                parts.append(f"[{tr.geo.zone}]")

        txt = "  ".join(parts)
        tx, ty = fx + 10, max(18, fy - 8)
        (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.46, 1)
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

    pad = 10
    x = panel_x + pad
    right = w - pad
    y = 16

    LINE   = 18   # compact row height
    FIELD_H = 24  # input field total height
    BTN_H   = 22  # button height

    cal_mode          = ui_state.get("cal_mode", CAL_MODE_XY)
    pick_active       = ui_state.get("point_pick_active", False)
    pending_px        = ui_state.get("pending_px")
    pending_py        = ui_state.get("pending_py")
    entry_fields      = ui_state.get("point_entry_fields", [])
    entry_cursor      = ui_state.get("point_entry_cursor", 0)
    entry_buf         = ui_state.get("point_entry_buf", "")
    entry_vals        = ui_state.get("point_entry_values", {})
    cal_points        = ui_state.get("cal_points", [])
    hybrid_errors     = ui_state.get("hybrid_errors", [])
    activity_enabled  = ui_state.get("activity_enabled", False)
    activity_available = ui_state.get("activity_available", False)

    in_entry = pick_active or (pending_px is not None)

    def _check_y(extra: int = 0) -> bool:
        return y + extra < h - 36

    def section(title: str, color: tuple = (200, 200, 210)) -> None:
        nonlocal y
        if not _check_y(18):
            return
        y += 5
        cv2.putText(canvas, title, (x, y), _FONT, 0.45, color, 1, cv2.LINE_AA)
        y += 3
        cv2.line(canvas, (x, y), (right, y), (60, 60, 68), 1)
        y += 12

    def row(label: str, value: str, vcol: tuple = (230, 230, 230)) -> None:
        nonlocal y
        if not _check_y():
            return
        cv2.putText(canvas, label, (x, y), _FONT, 0.38, (155, 155, 162), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (x + 120, y), _FONT, 0.38, vcol, 1, cv2.LINE_AA)
        y += LINE

    def button(name: str, label: str, active: bool = False, col: Optional[tuple] = None) -> None:
        nonlocal y
        if not _check_y(BTN_H):
            return
        x1, y1, x2, y2 = x, y, right, y + BTN_H
        buttons[name] = (x1, y1, x2, y2)
        if col:
            bg  = tuple(max(0, c - 30) for c in col) if not active else col
            brd = col
        else:
            bg  = (70, 70, 90) if active else (40, 40, 50)
            brd = (190, 190, 210) if active else (90, 90, 105)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 1)
        cv2.putText(canvas, label, (x1 + 6, y1 + 15), _FONT, 0.40, (245, 245, 250), 1, cv2.LINE_AA)
        y += BTN_H + 3

    def mode_btn(name: str, mode_key: str) -> None:
        nonlocal y
        if not _check_y(BTN_H):
            return
        is_active = cal_mode == mode_key
        col = _MODE_COLORS.get(mode_key, (60, 60, 60))
        bg  = col if is_active else tuple(int(c * 0.4) for c in col)
        brd = col
        x1, y1, x2, y2 = x, y, right, y + BTN_H
        buttons[name] = (x1, y1, x2, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 2 if is_active else 1)
        prefix = "* " if is_active else "  "
        cv2.putText(canvas, prefix + _MODE_BTN_LABELS[mode_key],
                    (x1 + 6, y1 + 15), _FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        y += BTN_H + 3

    def param_field(key: str, label: str) -> None:
        """Editable numeric parameter field (label left, value box right)."""
        nonlocal y
        if not _check_y(FIELD_H + 2):
            return
        active = (editing_key == key)
        lw = 118
        # label
        cv2.putText(canvas, label, (x, y + 15), _FONT, 0.36, (160, 160, 168), 1, cv2.LINE_AA)
        # value box
        bx1, by1, bx2, by2 = x + lw, y, right, y + FIELD_H
        fields[key] = (bx1, by1, bx2, by2)
        bg  = (65, 58, 110) if active else (35, 35, 44)
        brd = (175, 148, 255) if active else (75, 75, 88)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), brd, 1)
        val = field_values.get(key, "")
        disp = (val + "_") if active else val
        cv2.putText(canvas, disp, (bx1 + 4, by1 + 16), _FONT, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        y += FIELD_H + 3

    def entry_field(key: str, label: str, value: str, active: bool, done: bool) -> None:
        """Point entry field — keyboard-only, shows status via color."""
        nonlocal y
        if not _check_y(FIELD_H + 2):
            return
        fields[f"pe_{key}"] = (x, y, right, y + FIELD_H)  # hit box (click returns early)
        if done:
            label_col = (100, 200, 100)
            bg = (28, 52, 28)
            brd = (60, 160, 60)
            tick = "OK "
        elif active:
            label_col = (255, 220, 80)
            bg = (70, 58, 10)
            brd = (200, 170, 40)
            tick = ">> "
        else:
            label_col = (140, 140, 148)
            bg = (32, 32, 40)
            brd = (65, 65, 78)
            tick = "   "
        # label
        cv2.putText(canvas, tick + label, (x, y + 15), _FONT, 0.37, label_col, 1, cv2.LINE_AA)
        # value box
        bx1, by1, bx2, by2 = x + 118, y, right, y + FIELD_H
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg, -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), brd, 1)
        disp = (value + "_") if active else value
        cv2.putText(canvas, disp, (bx1 + 4, by1 + 16), _FONT, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        y += FIELD_H + 3

    # ── STATUS ────────────────────────────────────────────────────────────────
    section("STATUS")
    p = packet
    row("FPS in/inf",
        f"{(p.reader_fps if p else 0):.1f} / {(p.inference_fps if p else 0):.1f}")
    row("Infer ms", f"{(p.infer_ms if p else 0):.0f}")
    row("CPU/RAM",
        f"{(p.cpu_percent if p else 0):.0f}% / {(p.ram_percent if p else 0):.0f}%")
    row("People", str(len(p.tracks) if p else 0))
    n_pts = len(cal_points)
    pts_col = (80, 220, 80) if n_pts >= 4 else (220, 190, 60) if n_pts > 0 else (160, 160, 170)
    mode_names = {CAL_MODE_XY: "XY", CAL_MODE_LASER: "Laser+A", CAL_MODE_HYBRID: "Hybrid"}
    row("Mode / pts",
        f"{mode_names.get(cal_mode, cal_mode)}  {n_pts} pt{'s' if n_pts != 1 else ''}",
        pts_col)

    # ── CAL MODE ──────────────────────────────────────────────────────────────
    if not in_entry:
        section("CAL MODE")
        mode_btn("mode_xy",     CAL_MODE_XY)
        mode_btn("mode_laser",  CAL_MODE_LASER)
        mode_btn("mode_hybrid", CAL_MODE_HYBRID)
        if _check_y(LINE):
            desc = _MODE_DESC.get(cal_mode, "")
            cv2.putText(canvas, desc, (x, y), _FONT, 0.33,
                        _MODE_COLORS.get(cal_mode, (160, 160, 170)), 1, cv2.LINE_AA)
            y += LINE + 2

    # ── POINT PICK ACTIVE: big banner ─────────────────────────────────────────
    if pick_active and _check_y(36):
        y += 4
        bx1, by1, bx2, by2 = x, y, right, y + 34
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 80, 100), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
        cv2.putText(canvas, ">> CLICK FLOOR IN VIDEO <<",
                    (bx1 + 4, by1 + 22), _FONT, 0.44, (0, 240, 255), 1, cv2.LINE_AA)
        y += 38
        if _check_y():
            cv2.putText(canvas, "Esc to cancel",
                        (x, y), _FONT, 0.34, (120, 180, 190), 1, cv2.LINE_AA)
            y += LINE

    # ── POINT ENTRY FORM ──────────────────────────────────────────────────────
    elif pending_px is not None and _check_y(20):
        n = n_pts + 1
        mode_col = _MODE_COLORS.get(cal_mode, (120, 120, 120))
        section(f"ENTERING POINT #{n}   px({pending_px},{pending_py})", mode_col)

        mode_fields = _MODE_ENTRY_LABELS.get(cal_mode, [])
        for i, (fkey, flabel) in enumerate(mode_fields):
            is_cur  = (i == entry_cursor)
            is_done = (i < entry_cursor)
            val_str = str(entry_vals.get(fkey, "")) if is_done else (entry_buf if is_cur else "")
            entry_field(fkey, flabel, val_str, is_cur, is_done)

        if _check_y():
            cv2.putText(canvas, "Enter=confirm field   Esc=cancel",
                        (x, y), _FONT, 0.33, (140, 140, 150), 1, cv2.LINE_AA)
            y += LINE + 2

    # ── ACTIONS (only in normal mode) ─────────────────────────────────────────
    if not in_entry:
        section("ACTIONS")
        button("aim", "Set AIM  [A]")
        button("add_point", "+ Add point  [F]")
        if cal_points:
            button("remove_point", "- Remove last")
            button("clear_points", "Clear all")
        button("save", "Save  [S]")
        if activity_available:
            act_label = "Activity: ON  [toggle]" if activity_enabled else "Activity: OFF [toggle]"
            act_col = (40, 130, 40) if activity_enabled else (80, 80, 90)
            button("toggle_activity", act_label, active=activity_enabled, col=act_col)

    # ── ZONES ─────────────────────────────────────────────────────────────────
    zones        = ui_state.get("zones", [])
    zone_draw    = ui_state.get("zone_draw_active", False)
    zone_nm_mode = ui_state.get("zone_name_mode", False)
    zone_polygon = ui_state.get("zone_polygon_px", [])
    zone_nm_buf  = ui_state.get("zone_name_buf", "")

    if _check_y(18):
        section("ZONES")

    if zone_nm_mode and _check_y(FIELD_H + LINE):
        bx1, by1, bx2, by2 = x, y, right, y + FIELD_H
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (28, 52, 52), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 200), 1)
        cv2.putText(canvas, f"Name: {zone_nm_buf}_",
                    (bx1 + 4, by1 + 16), _FONT, 0.40, (0, 230, 230), 1, cv2.LINE_AA)
        y += FIELD_H + 3
        if _check_y():
            cv2.putText(canvas, "Enter=done  Esc=cancel",
                        (x, y), _FONT, 0.32, (80, 160, 160), 1, cv2.LINE_AA)
            y += LINE
    elif zone_draw and _check_y(34):
        n_z = len(zone_polygon)
        bx1, by1, bx2, by2 = x, y, right, y + 34
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 48, 22), -1)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 200, 100), 2)
        cv2.putText(canvas, f">> CLICK IN VIDEO ({n_z} pts)",
                    (bx1 + 4, by1 + 14), _FONT, 0.38, (0, 240, 120), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Enter=finish  Esc=cancel",
                    (bx1 + 4, by1 + 28), _FONT, 0.31, (0, 180, 90), 1, cv2.LINE_AA)
        y += 38
        if n_z >= 3:
            button("zone_finish", f"Finish zone ({n_z} pts)", col=(0, 140, 60))
    else:
        button("zone_draw", "+ Draw zone")
        if zones and _check_y(BTN_H):
            button("zone_delete_last", "- Delete last zone")
        for zone in zones[-4:]:
            if not _check_y():
                break
            zname  = zone.get("name", "zone")
            zcolor = tuple(int(c) for c in zone.get("color", [0, 200, 200]))
            n_zpts = len(zone.get("polygon_px", []))
            cv2.circle(canvas, (x + 6, y - 4), 5, zcolor, -1)
            cv2.putText(canvas, f"  {zname} ({n_zpts} pts)",
                        (x, y), _FONT, 0.36, (130, 195, 155), 1, cv2.LINE_AA)
            y += LINE

    # ── POINTS LIST ───────────────────────────────────────────────────────────
    if cal_points and _check_y(16):
        ok = n_pts >= 4
        list_col = (100, 220, 100) if ok else (200, 180, 60)
        section(f"POINTS ({n_pts}){' - ready' if ok else ' - need 4+'}", list_col)
        show = cal_points[-4:] if not in_entry else cal_points[-2:]
        offset = len(cal_points) - len(show)
        for i, pt in enumerate(show):
            if not _check_y():
                break
            wx, wy = _point_to_xy(pt, cal_mode)
            cv2.putText(canvas,
                        f"P{offset+i+1}  ({wx:.2f}, {wy:.2f}) m",
                        (x, y), _FONT, 0.34, (140, 190, 140), 1, cv2.LINE_AA)
            y += 15
        for ia, ib, meas, comp, err in hybrid_errors[:2]:
            if not _check_y():
                break
            ec = (80, 220, 80) if err < 0.05 else (220, 160, 60) if err < 0.15 else (80, 80, 220)
            cv2.putText(canvas,
                        f"P{ia+1}->P{ib+1} err={err:.3f}m",
                        (x, y), _FONT, 0.32, ec, 1, cv2.LINE_AA)
            y += 14

    # ── PARAMS (mode-specific, only in normal mode) ───────────────────────────
    if not in_entry and _check_y(36):
        section("PARAMS")
        mode_param_list = _MODE_PARAMS.get(cal_mode, list(_MODE_PARAMS.values())[0])
        for key, label in mode_param_list:
            if not _check_y(FIELD_H + 2):
                break
            param_field(key, label)

    # ── HINTS ─────────────────────────────────────────────────────────────────
    yy = h - 20
    cv2.putText(canvas, "A:AIM  F:+pt  S:save  Esc:quit  P/T/C:toggle",
                (x, yy), _FONT, 0.30, (120, 120, 130), 1, cv2.LINE_AA)
