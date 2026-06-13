from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.types import TrackSnapshot, VisionPacket
from app.vision.calibration import (
    CAL_MODE_HYBRID, CAL_MODE_LASER, CAL_MODE_XY,
    CalibrationData, mode_entry_fields, mode_label,
)

POSE_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Numeric fields shown in the panel (key, label)
FIELD_ORDER = [
    ("cam_to_aim_m",    "Лазер AIM→кам, м"),
    ("camera_floor_x_m", "Cam X смещ, м"),
    ("camera_floor_y_m", "Cam Y смещ, м"),
    ("camera_height_m", "Высота кам (уст), м"),
    ("room_width_m",    "Ширина зоны, м"),
    ("room_depth_m",    "Глубина зоны, м"),
    ("camera_pitch_deg","Pitch, °"),
    ("camera_yaw_deg",  "Yaw, °"),
    ("camera_roll_deg", "Roll, °"),
    ("hfov_deg",        "HFOV, °"),
    ("vfov_deg",        "VFOV, °"),
    ("rotation_deg",    "Rot, °"),
    ("lens_distortion_k1", "Lens k1"),
    ("lens_distortion_k2", "Lens k2"),
]

_MODE_COLORS = {
    CAL_MODE_XY:     (60, 180, 60),
    CAL_MODE_LASER:  (60, 130, 220),
    CAL_MODE_HYBRID: (200, 130, 40),
}

_MODE_KEYS = (CAL_MODE_XY, CAL_MODE_LASER, CAL_MODE_HYBRID)
_MODE_BTN_LABELS = {
    CAL_MODE_XY:     "XY (рулетка)",
    CAL_MODE_LASER:  "Лазер + угол",
    CAL_MODE_HYBRID: "Гибрид",
}


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


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


def draw_calibration(
    frame: np.ndarray,
    cal: CalibrationData,
    scale: float = 1.0,
    pending_px: Optional[int] = None,
    pending_py: Optional[int] = None,
) -> None:
    mode = cal.cal_mode
    col = _MODE_COLORS.get(mode, (0, 0, 255))

    # AIM point
    ax, ay = int(cal.aim_px * scale), int(cal.aim_py * scale)
    cv2.drawMarker(frame, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 7, (0, 0, 255), 2, cv2.LINE_AA)
    _text(frame, "AIM (0,0)", (ax + 12, ay - 10), 0.52, (0, 80, 255), 2)

    # Calibration points (new system)
    pts_cal = cal.cal_points or []
    if pts_cal:
        scaled = [(int(pt["px"] * scale), int(pt["py"] * scale)) for pt in pts_cal]

        # Lines connecting consecutive points
        for i in range(1, len(scaled)):
            cv2.line(frame, scaled[i - 1], scaled[i], col, 1, cv2.LINE_AA)
        if len(scaled) >= 4:
            cv2.line(frame, scaled[-1], scaled[0], col, 1, cv2.LINE_AA)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [np.array(scaled, dtype=np.int32)], col)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # Line from AIM to each point
        for sp in scaled:
            cv2.line(frame, (ax, ay), sp, (80, 80, 200), 1, cv2.LINE_AA)

        # Point markers + labels
        from app.vision.calibration import _point_to_xy
        for i, (sp, pt) in enumerate(zip(scaled, pts_cal)):
            cv2.circle(frame, sp, 7, col, -1, cv2.LINE_AA)
            cv2.circle(frame, sp, 9, (255, 255, 255), 1, cv2.LINE_AA)
            wx, wy = _point_to_xy(pt, mode)
            label = f"P{i + 1} ({wx:.1f},{wy:.1f})"
            _text(frame, label, (sp[0] + 10, sp[1] - 6), 0.44, (255, 255, 255), 1)

    # Legacy floor points (4-corner fallback, shown dimmer)
    elif cal.floor_points:
        pts = [(int(p[0] * scale), int(p[1] * scale)) for p in cal.floor_points]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (120, 60, 60), 1)
        if len(pts) == 4:
            cv2.line(frame, pts[3], pts[0], (120, 60, 60), 1)
        for i, p in enumerate(pts):
            cv2.circle(frame, p, 5, (180, 80, 80), -1)
            _text(frame, f"L{i + 1}", (p[0] + 8, p[1] - 6), 0.42, (200, 120, 120), 1)

    # Pending pixel (user clicked, waiting for value entry)
    if pending_px is not None and pending_py is not None:
        spx, spy = int(pending_px * scale), int(pending_py * scale)
        cv2.drawMarker(frame, (spx, spy), (0, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.circle(frame, (spx, spy), 10, (0, 255, 255), 2, cv2.LINE_AA)
        _text(frame, "Введите значения →", (spx + 14, spy - 8), 0.48, (0, 255, 255), 1)


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
        tx, ty = fx + 10, max(24, fy - 10)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (tx - 4, ty - th - 8), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        _text(frame, txt, (tx, ty), 0.5, (255, 255, 255), 1)


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
    panel_w = max(1, w - panel_x)

    cv2.rectangle(canvas, (panel_x, 0), (w, h), (24, 24, 28), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), (80, 80, 90), 1)

    pad = 14
    x = panel_x + pad
    right = w - pad
    y = 22

    title_scale = 0.58
    normal_scale = 0.46
    small_scale = 0.42
    line_h = 22
    field_h = 28

    cal_mode = ui_state.get("cal_mode", CAL_MODE_XY)
    point_pick_active = ui_state.get("point_pick_active", False)
    pending_px = ui_state.get("pending_px")
    pending_py = ui_state.get("pending_py")
    entry_fields = ui_state.get("point_entry_fields", [])
    entry_cursor = ui_state.get("point_entry_cursor", 0)
    entry_buf = ui_state.get("point_entry_buf", "")
    entry_vals = ui_state.get("point_entry_values", {})
    cal_points = ui_state.get("cal_points", [])
    hybrid_errors = ui_state.get("hybrid_errors", [])

    def section(title: str) -> None:
        nonlocal y
        y += 6
        _text(canvas, title.upper(), (x, y), title_scale, (255, 255, 255), 1)
        y += 6
        cv2.line(canvas, (x, y), (right, y), (70, 70, 76), 1)
        y += 16

    def row(label: str, value: str, col: tuple = (185, 185, 190)) -> None:
        nonlocal y
        _text(canvas, label, (x, y), normal_scale, col, 1)
        _text(canvas, value, (x + 130, y), normal_scale, (245, 245, 245), 1)
        y += line_h

    def button(name: str, text: str, active: bool = False, color: tuple = (52, 52, 60)) -> None:
        nonlocal y
        if y + 36 > h - 10:
            return
        x1, y1, x2, y2 = x, y, right, y + 30
        buttons[name] = (x1, y1, x2, y2)
        bg = color if not active else tuple(min(255, c + 60) for c in color)
        border = (200, 200, 210) if active else (110, 110, 120)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        _text(canvas, text, (x1 + 8, y1 + 20), 0.46, (255, 255, 255) if active else (220, 220, 225), 1)
        y += 36

    def mode_button(name: str, mode_key: str) -> None:
        nonlocal y
        if y + 30 > h - 10:
            return
        label = _MODE_BTN_LABELS[mode_key]
        is_active = cal_mode == mode_key
        col = _MODE_COLORS.get(mode_key, (60, 60, 60))
        btn_col = tuple(max(0, c - 30) for c in col) if not is_active else col
        x1, y1, x2, y2 = x, y, right, y + 28
        buttons[name] = (x1, y1, x2, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), btn_col, -1)
        border_col = col if is_active else (90, 90, 100)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1 if not is_active else 2)
        _text(canvas, ("✓ " if is_active else "  ") + label, (x1 + 6, y1 + 19), 0.44, (255, 255, 255), 1)
        y += 32

    def inline_field(key: str, label: str, value: str, active: bool) -> None:
        nonlocal y
        if y + field_h + 4 > h - 60:
            return
        label_w = min(130, int(panel_w * 0.40))
        x1, y1, x2, y2 = x + label_w, y - 18, right, y + field_h - 18
        fields[key] = (x1, y1, x2, y2)
        _text(canvas, label, (x, y), small_scale, (195, 195, 200), 1)
        bg = (78, 72, 110) if active else (42, 42, 48)
        border = (190, 170, 255) if active else (90, 90, 100)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
        val = (value + "_") if active else value
        _text(canvas, val, (x1 + 6, y1 + 19), 0.46, (255, 255, 255), 1)
        y += 30

    # ── Status ────────────────────────────────────────────────────────────────
    section("Статус")
    people = len(packet.tracks) if packet else 0
    row("FPS rtsp/infer", f"{(packet.reader_fps if packet else 0):.1f} / {(packet.inference_fps if packet else 0):.1f}")
    row("Инференс", f"{(packet.infer_ms if packet else 0):.0f} мс")
    row("CPU / RAM", f"{(packet.cpu_percent if packet else 0):.0f}% / {(packet.ram_percent if packet else 0):.0f}%")
    row("Людей", str(people))
    mode_str = mode_label(cal_mode)
    n_pts = len(cal_points)
    row("Режим кал.", f"{mode_str} [{n_pts} тч]", (180, 200, 255))

    # ── Mode selection ────────────────────────────────────────────────────────
    section("Режим позиционирования")
    mode_button("mode_xy", CAL_MODE_XY)
    mode_button("mode_laser", CAL_MODE_LASER)
    mode_button("mode_hybrid", CAL_MODE_HYBRID)

    # ── Actions ───────────────────────────────────────────────────────────────
    section("Действия")
    button("aim", "  Установить AIM (A)")
    if not point_pick_active and pending_px is None:
        button("add_point", "  + Добавить точку (F)")
    if cal_points:
        button("remove_point", "  − Удалить последнюю")
        button("clear_points", "  Очистить все точки")
    button("save", "  Сохранить (S)")

    # ── Point entry form ──────────────────────────────────────────────────────
    if point_pick_active:
        y += 6
        _text(canvas, "КЛИКНИТЕ ТОЧКУ НА ВИДЕО", (x, y), 0.50, (0, 255, 255), 1)
        y += 22
        _text(canvas, f"Точка #{len(cal_points) + 1}", (x, y), small_scale, (180, 180, 185), 1)
        y += 18

    elif pending_px is not None:
        section(f"ТОЧКА #{len(cal_points) + 1} — введите значения")
        for i, (fkey, flabel) in enumerate(entry_fields):
            done_val = entry_vals.get(fkey, "")
            is_active = i == entry_cursor
            if i < entry_cursor:
                inline_field(f"pe_{fkey}", flabel, done_val, False)
            elif is_active:
                inline_field(f"pe_{fkey}", flabel, entry_buf, True)
            else:
                inline_field(f"pe_{fkey}", flabel, "—", False)
        y += 4
        _text(canvas, "Enter — далее   Esc — отмена", (x, y), small_scale, (160, 160, 165), 1)
        y += 20

    # ── Cal points list ───────────────────────────────────────────────────────
    if cal_points:
        section(f"Точки ({len(cal_points)})")
        from app.vision.calibration import _point_to_xy
        for i, pt in enumerate(cal_points[-6:]):  # show last 6
            idx = len(cal_points) - min(6, len(cal_points)) + i
            wx, wy = _point_to_xy(pt, cal_mode)
            if y + line_h > h - 60:
                break
            lbl = f"P{idx + 1}  px({pt['px']},{pt['py']})  →  ({wx:.2f}, {wy:.2f})м"
            _text(canvas, lbl, (x, y), 0.38, (170, 200, 170), 1)
            y += 18

        # Hybrid validation errors
        if hybrid_errors:
            y += 4
            for ia, ib, meas, comp, err in hybrid_errors[:3]:
                col = (100, 255, 100) if err < 0.05 else (80, 120, 255) if err < 0.15 else (80, 80, 255)
                _text(canvas, f"P{ia+1}→P{ib+1}: изм={meas:.2f} расч={comp:.2f} Δ={err:.3f}м",
                      (x, y), 0.38, col, 1)
                y += 16

    # ── Numeric fields ────────────────────────────────────────────────────────
    if y + 60 < h:
        section("Параметры")
        for key, label in FIELD_ORDER:
            if y + field_h + 4 > h - 55:
                break
            val = field_values.get(key, "")
            inline_field(key, label, val, editing_key == key)

    # ── Hints ─────────────────────────────────────────────────────────────────
    hints = ["A: AIM  F: +точка  S: сохр  Esc: выйти", "P: поза  T: треки  C: калибр"]
    yy = h - 36
    for line in hints:
        _text(canvas, line, (x, yy), 0.38, (155, 155, 162), 1)
        yy += 18
