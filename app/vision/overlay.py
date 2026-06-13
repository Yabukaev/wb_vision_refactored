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

FIELD_ORDER = [
    ("room_width_m", "Room W, m"),
    ("room_depth_m", "Room D, m"),
    ("camera_height_m", "Cam H, m"),
    ("camera_pitch_deg", "Pitch, deg"),
    ("camera_yaw_deg", "Yaw, deg"),
    ("camera_roll_deg", "Roll, deg"),
    ("hfov_deg", "HFOV, deg"),
    ("vfov_deg", "VFOV, deg"),
    ("rotation_deg", "Rot, deg"),
    ("lens_distortion_k1", "Lens k1"),
    ("lens_distortion_k2", "Lens k2"),
]


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
    # P-06: scale coordinates when drawing on a resized frame
    if keypoints is None:
        return

    for a, b in POSE_EDGES:
        if len(keypoints) > max(a, b):
            ax = int(keypoints[a][0] * scale)
            ay = int(keypoints[a][1] * scale)
            bx = int(keypoints[b][0] * scale)
            by = int(keypoints[b][1] * scale)
            if ax > 1 and ay > 1 and bx > 1 and by > 1:
                cv2.line(frame, (ax, ay), (bx, by), (0, 220, 255), 2, cv2.LINE_AA)

    for p in keypoints:
        x, y = int(p[0] * scale), int(p[1] * scale)
        if x > 1 and y > 1:
            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1, cv2.LINE_AA)


def draw_calibration(frame: np.ndarray, cal: CalibrationData, scale: float = 1.0) -> None:
    # P-06: scale calibration coordinates to match resized frame
    ax, ay = int(cal.aim_px * scale), int(cal.aim_py * scale)

    cv2.drawMarker(frame, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
    cv2.circle(frame, (ax, ay), 7, (0, 0, 255), 2, cv2.LINE_AA)
    _text(frame, "AIM", (ax + 12, ay - 10), 0.55, (0, 0, 255), 2)

    pts = [(int(p[0] * scale), int(p[1] * scale)) for p in (cal.floor_points or [])]

    if len(pts) >= 2:
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 2, cv2.LINE_AA)

    if len(pts) == 4:
        cv2.line(frame, pts[3], pts[0], (0, 0, 255), 2, cv2.LINE_AA)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 0, 140))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for i, p in enumerate(pts):
        cv2.circle(frame, p, 7, (0, 0, 255), -1, cv2.LINE_AA)
        _text(frame, str(i + 1), (p[0] + 10, p[1] - 8), 0.55, (255, 255, 255), 2)


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackSnapshot],
    scale: float = 1.0,
    show_pose: bool = True,
    show_tracks: bool = True,
) -> None:
    # P-06: scale all track coordinates to match resized frame
    for tr in tracks:
        if show_pose:
            draw_pose(frame, tr.keypoints, scale)

        x1 = int(tr.box[0] * scale)
        y1 = int(tr.box[1] * scale)
        x2 = int(tr.box[2] * scale)
        y2 = int(tr.box[3] * scale)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 190, 60), 2, cv2.LINE_AA)

        fx = int(tr.foot[0] * scale)
        fy = int(tr.foot[1] * scale)
        cv2.circle(frame, (fx, fy), 6, (0, 0, 255), -1, cv2.LINE_AA)

        if show_tracks:
            scaled_history = [(int(p[0] * scale), int(p[1] * scale)) for p in tr.history]
            for i in range(1, len(scaled_history)):
                cv2.line(frame, scaled_history[i - 1], scaled_history[i], (0, 230, 255), 2, cv2.LINE_AA)

        txt = f"ID {tr.track_id} {tr.state} {tr.conf:.2f}"
        if tr.geo:
            txt += f"  X:{tr.geo.x_m:.2f} Y:{tr.geo.y_m:.2f} D:{tr.geo.distance_m:.2f}"

        tx, ty = fx + 10, max(24, fy - 10)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (tx - 4, ty - th - 8), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        _text(frame, txt, (tx, ty), 0.5, (255, 255, 255), 1)


def draw_panel(
    canvas: np.ndarray,
    panel_x: int,
    packet: Optional[VisionPacket],
    calib_mode: Optional[str],
    buttons: dict,
    fields: dict,
    field_values: dict[str, str],
    editing_key: Optional[str],
) -> None:
    h, w = canvas.shape[:2]
    panel_w = max(1, w - panel_x)

    cv2.rectangle(canvas, (panel_x, 0), (w, h), (24, 24, 28), -1)
    cv2.line(canvas, (panel_x, 0), (panel_x, h), (80, 80, 90), 1)

    pad = 16
    x = panel_x + pad
    right = w - pad
    y = 30

    title_scale = 0.62
    normal_scale = 0.47
    small_scale = 0.42
    line_h = 24
    field_h = 30

    def section(title: str) -> None:
        nonlocal y
        y += 8
        _text(canvas, title.upper(), (x, y), title_scale, (255, 255, 255), 1)
        y += 8
        cv2.line(canvas, (x, y), (right, y), (70, 70, 76), 1)
        y += 18

    def row(label: str, value: str) -> None:
        nonlocal y
        _text(canvas, label, (x, y), normal_scale, (185, 185, 190), 1)
        _text(canvas, value, (x + 128, y), normal_scale, (245, 245, 245), 1)
        y += line_h

    def button(name: str, text: str) -> None:
        nonlocal y
        x1, y1, x2, y2 = x, y, right, y + 34
        buttons[name] = (x1, y1, x2, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (52, 52, 60), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 120, 130), 1)
        _text(canvas, text, (x1 + 10, y1 + 23), 0.48, (245, 245, 245), 1)
        y += 42

    def field(key: str, label: str) -> None:
        nonlocal y
        if y + field_h + 6 > h - 70:
            return

        label_w = min(140, int(panel_w * 0.42))
        x1 = x + label_w
        y1 = y - 20
        x2 = right
        y2 = y1 + field_h

        fields[key] = (x1, y1, x2, y2)

        _text(canvas, label, (x, y), small_scale, (195, 195, 200), 1)

        active = editing_key == key
        bg = (78, 72, 110) if active else (42, 42, 48)
        border = (190, 170, 255) if active else (95, 95, 105)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)

        val = field_values.get(key, "")
        if active:
            val = val + "_"

        _text(canvas, val, (x1 + 8, y1 + 21), 0.48, (255, 255, 255), 1)
        y += 34

    people = len(packet.tracks) if packet else 0

    section("Status")
    row("RTSP FPS", f"{(packet.reader_fps if packet else 0):.1f}")
    row("Infer FPS", f"{(packet.inference_fps if packet else 0):.1f}")
    row("Infer", f"{(packet.infer_ms if packet else 0):.0f} ms")
    row("CPU/RAM", f"{(packet.cpu_percent if packet else 0):.0f}% / {(packet.ram_percent if packet else 0):.0f}%")
    row("People", str(people))
    row("Mode", calib_mode or "none")

    section("Actions")
    button("aim", "Set AIM point")
    button("floor4", "Set 4 floor points")
    button("save", "Save calibration")

    section("Calibration")
    for key, label in FIELD_ORDER:
        field(key, label)

    hint_lines = [
        "Click field -> type -> Enter",
        "A: aim   F: floor   Esc: quit",
        "4 floor points clockwise",
    ]

    yy = h - 58
    for line in hint_lines:
        _text(canvas, line, (x, yy), 0.40, (170, 170, 176), 1)
        yy += 18
