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
) -> None:
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

    elif cal.floor_points:
        pts = [(int(p[0] * scale), int(p[1] * scale)) for p in cal.floor_points]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (120, 60, 60), 1)
        if len(pts) == 4:
            cv2.line(frame, pts[3], pts[0], (120, 60, 60), 1)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 5, (180, 80, 80), -1)
            _t(frame, f"L{i+1}", (p[0] + 8, p[1] + 5), 0.40, (200, 120, 120), 1)

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

    pad = 12
    x = panel_x + pad
    right = w - pad
    y = 18

    LINE = 20    # row height
    FIELD_H = 26 # input field height
    BTN_H = 26   # button height

    cal_mode      = ui_state.get("cal_mode", CAL_MODE_XY)
    pick_active   = ui_state.get("point_pick_active", False)
    pending_px    = ui_state.get("pending_px")
    pending_py    = ui_state.get("pending_py")
    entry_fields  = ui_state.get("point_entry_fields", [])
    entry_cursor  = ui_state.get("point_entry_cursor", 0)
    entry_buf     = ui_state.get("point_entry_buf", "")
    entry_vals    = ui_state.get("point_entry_values", {})
    cal_points    = ui_state.get("cal_points", [])
    hybrid_errors = ui_state.get("hybrid_errors", [])

    def _check_y(extra: int = 0) -> bool:
        return y + extra < h - 40

    def section(title: str) -> None:
        nonlocal y
        if not _check_y(20):
            return
        y += 6
        cv2.putText(canvas, title, (x, y), _FONT, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        y += 4
        cv2.line(canvas, (x, y), (right, y), (70, 70, 76), 1)
        y += 14

    def row(label: str, value: str, vcol: tuple = (245, 245, 245)) -> None:
        nonlocal y
        if not _check_y():
            return
        cv2.putText(canvas, label, (x, y), _FONT, 0.40, (170, 170, 175), 1, cv2.LINE_AA)
        cv2.putText(canvas, value, (x + 128, y), _FONT, 0.40, vcol, 1, cv2.LINE_AA)
        y += LINE

    def button(name: str, label: str, active: bool = False) -> None:
        nonlocal y
        if not _check_y(BTN_H):
            return
        x1, y1, x2, y2 = x, y, right, y + BTN_H
        buttons[name] = (x1, y1, x2, y2)
        bg = (70, 70, 90) if active else (44, 44, 52)
        brd = (190, 190, 210) if active else (100, 100, 112)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 1)
        cv2.putText(canvas, label, (x1 + 7, y1 + 17), _FONT, 0.42, (240, 240, 245), 1, cv2.LINE_AA)
        y += BTN_H + 4

    def mode_btn(name: str, mode_key: str) -> None:
        nonlocal y
        if not _check_y(BTN_H):
            return
        is_active = cal_mode == mode_key
        col = _MODE_COLORS.get(mode_key, (60, 60, 60))
        bg = col if is_active else tuple(max(0, c - 40) for c in col)
        brd = col if is_active else (80, 80, 90)
        x1, y1, x2, y2 = x, y, right, y + BTN_H
        buttons[name] = (x1, y1, x2, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 1 if not is_active else 2)
        prefix = "* " if is_active else "  "
        cv2.putText(canvas, prefix + _MODE_BTN_LABELS[mode_key],
                    (x1 + 6, y1 + 17), _FONT, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        y += BTN_H + 4

    def input_field(key: str, label: str, value: str, active: bool) -> None:
        nonlocal y
        if not _check_y(FIELD_H + 4):
            return
        lw = 120
        x1, y1, x2, y2 = x + lw, y - 18, right, y + FIELD_H - 18
        fields[key] = (x1, y1, x2, y2)
        cv2.putText(canvas, label, (x, y), _FONT, 0.38, (185, 185, 190), 1, cv2.LINE_AA)
        bg = (70, 64, 110) if active else (38, 38, 46)
        brd = (180, 155, 255) if active else (85, 85, 95)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), brd, 1)
        disp = (value + "_") if active else value
        cv2.putText(canvas, disp, (x1 + 5, y1 + 17), _FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28

    # ── Status ────────────────────────────────────────────────────────────────
    section("STATUS")
    p = packet
    row("RTSP / Infer FPS",
        f"{(p.reader_fps if p else 0):.1f} / {(p.inference_fps if p else 0):.1f}")
    row("Infer", f"{(p.infer_ms if p else 0):.0f} ms")
    row("CPU / RAM",
        f"{(p.cpu_percent if p else 0):.0f}% / {(p.ram_percent if p else 0):.0f}%")
    row("People", str(len(p.tracks) if p else 0))
    mode_names = {CAL_MODE_XY: "XY", CAL_MODE_LASER: "Laser+A", CAL_MODE_HYBRID: "Hybrid"}
    row("Cal mode", f"{mode_names.get(cal_mode, cal_mode)}  [{len(cal_points)} pts]",
        (160, 200, 255))

    # ── Cal mode buttons ──────────────────────────────────────────────────────
    section("CAL MODE")
    mode_btn("mode_xy",     CAL_MODE_XY)
    mode_btn("mode_laser",  CAL_MODE_LASER)
    mode_btn("mode_hybrid", CAL_MODE_HYBRID)

    # ── Actions ───────────────────────────────────────────────────────────────
    section("ACTIONS")
    button("aim", "Set AIM  (A)")
    if not pick_active and pending_px is None:
        button("add_point", "+ Add point  (F)")
    if cal_points:
        button("remove_point", "- Remove last")
        button("clear_points", "Clear all points")
    button("save", "Save  (S)")

    # ── Point pick hint ───────────────────────────────────────────────────────
    if pick_active and _check_y(20):
        y += 4
        cv2.putText(canvas, ">> CLICK FLOOR IN VIDEO <<",
                    (x, y), _FONT, 0.44, (0, 255, 255), 1, cv2.LINE_AA)
        y += LINE

    # ── Point entry form ──────────────────────────────────────────────────────
    elif pending_px is not None:
        n = len(cal_points) + 1
        section(f"POINT #{n}  px({pending_px},{pending_py})")
        labels = _MODE_ENTRY_LABELS.get(cal_mode, [])
        for i, (fkey, flabel) in enumerate(labels):
            done = entry_vals.get(fkey, "")
            is_cur = i == entry_cursor
            is_done = i < entry_cursor
            input_field(
                f"pe_{fkey}", flabel,
                done if is_done else (entry_buf if is_cur else "--"),
                is_cur,
            )
        if _check_y():
            cv2.putText(canvas, "Enter=next  Esc=cancel",
                        (x, y), _FONT, 0.36, (150, 150, 158), 1, cv2.LINE_AA)
            y += LINE

    # ── Points list ───────────────────────────────────────────────────────────
    if cal_points and _check_y(16):
        section(f"POINTS  ({len(cal_points)})")
        for i, pt in enumerate(cal_points[-5:]):
            if not _check_y():
                break
            idx = len(cal_points) - min(5, len(cal_points)) + i
            wx, wy = _point_to_xy(pt, cal_mode)
            cv2.putText(canvas,
                        f"P{idx+1} px({pt['px']},{pt['py']}) -> ({wx:.2f},{wy:.2f})m",
                        (x, y), _FONT, 0.36, (150, 195, 150), 1, cv2.LINE_AA)
            y += 16
        for ia, ib, meas, comp, err in hybrid_errors[:3]:
            if not _check_y():
                break
            ec = (80, 220, 80) if err < 0.05 else (80, 140, 255) if err < 0.15 else (60, 60, 220)
            cv2.putText(canvas,
                        f"P{ia+1}->P{ib+1} meas={meas:.2f} calc={comp:.2f} err={err:.3f}m",
                        (x, y), _FONT, 0.34, ec, 1, cv2.LINE_AA)
            y += 15

    # ── Numeric params ────────────────────────────────────────────────────────
    if _check_y(40):
        section("PARAMS")
        for key, label in FIELD_ORDER:
            if not _check_y(FIELD_H + 2):
                break
            input_field(key, label, field_values.get(key, ""), editing_key == key)

    # ── Hints ─────────────────────────────────────────────────────────────────
    yy = h - 32
    cv2.putText(canvas, "A:AIM  F:+pt  S:save  Esc:quit  P/T/C:toggle",
                (x, yy), _FONT, 0.35, (140, 140, 148), 1, cv2.LINE_AA)
