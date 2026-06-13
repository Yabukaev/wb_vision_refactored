from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from app.config import UISection
from app.core.latest_value import LatestValue
from app.types import FramePacket, VisionPacket
from app.vision.activity_classifier import ActivityClassifier
from app.vision.calibration import (
    CAL_MODE_HYBRID, CAL_MODE_LASER, CAL_MODE_XY, CAL_MODES,
    CalibrationData, CalibrationManager, mode_entry_fields,
)
from app.vision.overlay import FIELD_ORDER, draw_calibration, draw_panel, draw_tracks

_ZONE_COLORS = [
    [0, 200, 200],
    [200, 120, 0],
    [180, 0, 180],
    [200, 200, 0],
    [0, 180, 100],
    [0, 100, 200],
]


class UIWorker:
    """OpenCV UI loop. On Windows this must run in the main thread."""

    def __init__(
        self,
        ui_cfg: UISection,
        frames: LatestValue[FramePacket],
        results: LatestValue[VisionPacket],
        calibration: CalibrationManager,
        stop_event: threading.Event,
        activity: Optional[ActivityClassifier] = None,
    ) -> None:
        self.ui_cfg = ui_cfg
        self.frames = frames
        self.results = results
        self.calibration = calibration
        self.stop_event = stop_event
        self.activity = activity

        self.window_name = "VISION STABLE"

        # Generic field editing (numeric params)
        self.editing_key: Optional[str] = None
        self.edit_buffer = ""

        # AIM pick mode
        self.aim_mode = False

        # Calibration point pick / entry state machine
        self.point_pick_active = False
        self.pending_px: Optional[int] = None
        self.pending_py: Optional[int] = None
        self.point_entry_fields: list[tuple[str, str]] = []
        self.point_entry_cursor: int = 0
        self.point_entry_buf: str = ""
        self.point_entry_values: dict[str, float] = {}

        # Zone drawing state
        self.zone_draw_active = False
        self.zone_polygon_px: list[list[int]] = []
        self.zone_name_mode = False
        self.zone_name_buf = ""

        # Panel hit-testing dicts (rebuilt each frame)
        self.buttons: dict[str, tuple[int, int, int, int]] = {}
        self.fields: dict[str, tuple[int, int, int, int]] = {}

        # Video geometry (updated each frame)
        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.draw_w = 0
        self.draw_h = 0
        self.src_w = 0
        self.src_h = 0

        self.fullscreen = False

        self._waiting_canvas: Optional[np.ndarray] = None
        self._waiting_canvas_size: tuple[int, int] = (0, 0)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, int(self.ui_cfg.window_width), int(self.ui_cfg.window_height))
        cv2.setMouseCallback(self.window_name, self._mouse_cb)

        last_seq = 0

        while not self.stop_event.is_set():
            frame_packet, seq = self.frames.wait_next(last_seq=last_seq, timeout=0.03)
            if seq > last_seq:
                last_seq = seq

            if frame_packet is None:
                self._show_waiting()
                if self._handle_key(cv2.waitKeyEx(30)):
                    break
                continue

            cal = self.calibration.snapshot()
            result, _ = self.results.get()

            canvas = self._compose(frame_packet.image, result, cal)
            cv2.imshow(self.window_name, canvas)

            if self._handle_key(cv2.waitKeyEx(1)):
                break

        self.stop_event.set()
        cv2.destroyAllWindows()

    # ── rendering ─────────────────────────────────────────────────────────────

    def _show_waiting(self) -> None:
        ww, wh = self._current_window_size()
        if (ww, wh) != self._waiting_canvas_size:
            canvas = np.zeros((wh, ww, 3), dtype=np.uint8)
            lines = [
                "Waiting for RTSP stream...",
                "ESC - quit",
                "A - set AIM point",
                "F - add cal point",
                "S - save",
            ]
            y = 54
            for line in lines:
                cv2.putText(canvas, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.68, (235, 235, 235), 1, cv2.LINE_AA)
                y += 32
            self._waiting_canvas = canvas
            self._waiting_canvas_size = (ww, wh)
        cv2.imshow(self.window_name, self._waiting_canvas)

    def _compose(self, frame: np.ndarray, packet: Optional[VisionPacket], cal: CalibrationData) -> np.ndarray:
        ww, wh = self._current_window_size()
        panel_w = int(max(340, min(460, ww * 0.27)))
        video_w = max(1, ww - panel_w)
        fh, fw = frame.shape[:2]
        canvas = np.zeros((wh, ww, 3), dtype=np.uint8)

        self.scale = min(video_w / fw, wh / fh)
        self.draw_w = max(1, int(fw * self.scale))
        self.draw_h = max(1, int(fh * self.scale))
        self.off_x = int((video_w - self.draw_w) / 2)
        self.off_y = int((wh - self.draw_h) / 2)
        self.src_w, self.src_h = fw, fh

        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (self.draw_w, self.draw_h), interpolation=interp)

        if packet is not None:
            draw_tracks(
                resized,
                packet.tracks,
                scale=self.scale,
                show_pose=self.ui_cfg.show_pose,
                show_tracks=self.ui_cfg.show_tracks,
            )

        if self.ui_cfg.show_calibration:
            draw_calibration(
                resized, cal, scale=self.scale,
                pending_px=self.pending_px,
                pending_py=self.pending_py,
                zone_polygon_px=self.zone_polygon_px if self.zone_draw_active else None,
            )

        canvas[self.off_y:self.off_y + self.draw_h, self.off_x:self.off_x + self.draw_w] = resized

        # Video area border — bright when in pick/aim mode to signal "click here"
        if self.point_pick_active or self.aim_mode:
            border_col = (0, 210, 255) if self.point_pick_active else (0, 220, 100)
            cv2.rectangle(
                canvas,
                (self.off_x, self.off_y),
                (self.off_x + self.draw_w - 1, self.off_y + self.draw_h - 1),
                border_col, 3, cv2.LINE_AA,
            )
            hint = "CLICK IN VIDEO TO PLACE POINT" if self.point_pick_active else "CLICK IN VIDEO TO SET AIM"
            (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            hx = self.off_x + (self.draw_w - hw) // 2
            hy = self.off_y + self.draw_h - 14
            cv2.rectangle(canvas, (hx - 6, hy - hh - 6), (hx + hw + 6, hy + 4), (0, 0, 0), -1)
            cv2.putText(canvas, hint, (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_col, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(
                canvas,
                (self.off_x, self.off_y),
                (self.off_x + self.draw_w - 1, self.off_y + self.draw_h - 1),
                (70, 70, 75), 1, cv2.LINE_AA,
            )

        self.buttons = {}
        self.fields = {}

        field_values: dict[str, str] = {}
        for key, _ in FIELD_ORDER:
            if self.editing_key == key:
                field_values[key] = self.edit_buffer
            else:
                value = getattr(cal, key, "")
                field_values[key] = f"{value:g}" if isinstance(value, float) else str(value)

        hybrid_errors = (
            self.calibration.validate_hybrid_distances()
            if cal.cal_mode == CAL_MODE_HYBRID else []
        )

        ui_state = {
            "cal_mode": cal.cal_mode,
            "point_pick_active": self.point_pick_active,
            "pending_px": self.pending_px,
            "pending_py": self.pending_py,
            "point_entry_fields": self.point_entry_fields,
            "point_entry_cursor": self.point_entry_cursor,
            "point_entry_buf": self.point_entry_buf,
            "point_entry_values": {k: str(v) for k, v in self.point_entry_values.items()},
            "cal_points": cal.cal_points or [],
            "hybrid_errors": hybrid_errors,
            "activity_enabled": self.activity.is_enabled if self.activity else False,
            "activity_available": self.activity is not None,
            "zones": cal.zones or [],
            "zone_draw_active": self.zone_draw_active,
            "zone_polygon_px": self.zone_polygon_px,
            "zone_name_mode": self.zone_name_mode,
            "zone_name_buf": self.zone_name_buf,
        }

        draw_panel(
            canvas,
            video_w,
            packet,
            ui_state,
            self.buttons,
            self.fields,
            field_values,
            self.editing_key,
        )

        return canvas

    # ── keyboard ──────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        if key == -1:
            return False

        low = key & 0xFF

        # Zone name entry (highest priority)
        if self.zone_name_mode:
            if low in (13, 10):
                name = self.zone_name_buf.strip() or "zone"
                cal = self.calibration.snapshot()
                n = len(cal.zones or [])
                color = _ZONE_COLORS[n % len(_ZONE_COLORS)]
                self.calibration.add_zone(name, [[p[0], p[1]] for p in self.zone_polygon_px], color)
                self.zone_polygon_px = []
                self.zone_name_mode = False
                self.zone_name_buf = ""
                self.zone_draw_active = False
                print(f"Zone '{name}' saved")
            elif low == 27:
                self.zone_name_mode = False
                self.zone_name_buf = ""
                self.zone_polygon_px = []
                self.zone_draw_active = False
                print("Zone cancelled")
            elif low in (8, 127):
                self.zone_name_buf = self.zone_name_buf[:-1]
            elif 32 <= low <= 126:
                if len(self.zone_name_buf) < 20:
                    self.zone_name_buf += chr(low)
            return False

        # Cal point entry mode
        if self.pending_px is not None:
            if low in (13, 10):
                self._point_entry_advance()
            elif low == 27:
                self._point_entry_cancel()
            elif low in (8, 127):
                self.point_entry_buf = self.point_entry_buf[:-1]
            elif 0 <= low <= 255:
                ch = chr(low)
                if ch in "0123456789.-":
                    self.point_entry_buf += ch
            return False

        # Numeric field editing mode
        if self.editing_key:
            if low in (13, 10):
                self._commit_edit()
            elif low == 27:
                self.editing_key = None
                self.edit_buffer = ""
            elif low in (8, 127):
                self.edit_buffer = self.edit_buffer[:-1]
            elif 0 <= low <= 255:
                ch = chr(low)
                if ch in "0123456789.-":
                    self.edit_buffer += ch
            return False

        # Zone draw mode (intercepts Enter/Esc)
        if self.zone_draw_active:
            if low == 27:
                self.zone_draw_active = False
                self.zone_polygon_px = []
                print("Zone cancelled")
                return False
            if low in (13, 10):
                if len(self.zone_polygon_px) >= 3:
                    self.zone_draw_active = False
                    self.zone_name_mode = True
                    print("Enter zone name on keyboard")
                return False
            return False

        # Normal commands
        if low == 27:
            if self.point_pick_active or self.aim_mode:
                self.point_pick_active = False
                self.aim_mode = False
                print("Cancelled")
                return False
            return True
        if low in (ord("a"), ord("A")):
            self.aim_mode = True
            self.point_pick_active = False
            print("Click AIM point on floor in video")
        elif low in (ord("f"), ord("F")):
            self._start_point_pick()
        elif low in (ord("s"), ord("S")):
            self.calibration.save()
            print("Calibration saved")
        elif low in (ord("p"), ord("P")):
            self.ui_cfg.show_pose = not self.ui_cfg.show_pose
        elif low in (ord("t"), ord("T")):
            self.ui_cfg.show_tracks = not self.ui_cfg.show_tracks
        elif low in (ord("c"), ord("C")):
            self.ui_cfg.show_calibration = not self.ui_cfg.show_calibration
        elif low in (ord("m"), ord("M")):
            self.fullscreen = not self.fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

        return False

    # ── mouse ─────────────────────────────────────────────────────────────────

    def _mouse_cb(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Panel buttons
        for name, rect in self.buttons.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._handle_button(name)
                return

        # Panel numeric fields
        for key, rect in self.fields.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if key.startswith("pe_"):
                    return
                cal = self.calibration.snapshot()
                self.editing_key = key
                self.edit_buffer = f"{getattr(cal, key, 0.0):g}"
                return

        # Video area clicks
        p = self._screen_to_frame(x, y)
        if p is None:
            return
        fx, fy = p

        if self.aim_mode:
            self.calibration.set_aim(fx, fy)
            self.aim_mode = False
            print(f"AIM set: ({fx}, {fy})")

        elif self.point_pick_active:  # cal pick has priority over zone draw
            self.pending_px = fx
            self.pending_py = fy
            self.point_pick_active = False
            cal = self.calibration.snapshot()
            self.point_entry_fields = mode_entry_fields(cal.cal_mode)
            self.point_entry_cursor = 0
            self.point_entry_buf = ""
            self.point_entry_values = {}
            n = len(cal.cal_points or []) + 1
            print(f"Point P{n} selected: pixel ({fx}, {fy}). Enter values.")

        elif self.zone_draw_active and not self.zone_name_mode:
            self.zone_polygon_px.append([fx, fy])
            print(f"Zone point {len(self.zone_polygon_px)}: ({fx}, {fy})")

    def _handle_button(self, name: str) -> None:
        if name == "aim":
            self.aim_mode = True
            self.point_pick_active = False
            print("Click AIM point on floor in video")

        elif name == "add_point":
            self._start_point_pick()

        elif name == "remove_point":
            self.calibration.remove_last_cal_point()
            print("Last point removed")

        elif name == "clear_points":
            self.calibration.clear_cal_points()
            self._point_entry_cancel()
            print("All points cleared")

        elif name == "save":
            self.calibration.save()
            print("Calibration saved")

        elif name == "mode_xy":
            self.calibration.set_cal_mode(CAL_MODE_XY)
            self._point_entry_cancel()
            print("Mode: XY coords")

        elif name == "mode_laser":
            self.calibration.set_cal_mode(CAL_MODE_LASER)
            self._point_entry_cancel()
            print("Mode: Laser+angle")

        elif name == "mode_hybrid":
            self.calibration.set_cal_mode(CAL_MODE_HYBRID)
            self._point_entry_cancel()
            print("Mode: Hybrid")

        elif name == "toggle_activity":
            if self.activity is not None:
                enabled = self.activity.toggle_enabled()
                print(f"Activity: {'ON' if enabled else 'OFF'}")

        elif name == "zone_draw":
            self.zone_draw_active = True
            self.zone_polygon_px = []
            self.zone_name_mode = False
            self.zone_name_buf = ""
            print("Zone draw: click points in video. Enter to finish (need 3+).")

        elif name == "zone_finish":
            if len(self.zone_polygon_px) >= 3:
                self.zone_draw_active = False
                self.zone_name_mode = True
                self.zone_name_buf = ""
                print("Type zone name, then Enter")

        elif name == "zone_delete_last":
            cal = self.calibration.snapshot()
            n = len(cal.zones or [])
            if n > 0:
                self.calibration.delete_zone(n - 1)
                print("Last zone deleted")

    # ── point entry helpers ────────────────────────────────────────────────────

    def _start_point_pick(self) -> None:
        # Reset any half-finished entry FIRST — _point_entry_cancel() clears
        # point_pick_active, so it must run before we arm pick mode, or the
        # pick is silently disabled and clicks never place a point.
        self._point_entry_cancel()
        self.point_pick_active = True
        self.aim_mode = False
        self.editing_key = None
        print("Click floor in video to place point")

    def _point_entry_advance(self) -> None:
        if not self.point_entry_fields or self.pending_px is None:
            return

        fkey, _ = self.point_entry_fields[self.point_entry_cursor]
        try:
            self.point_entry_values[fkey] = float(self.point_entry_buf.replace(",", ".") or "0")
        except ValueError:
            self.point_entry_values[fkey] = 0.0

        self.point_entry_cursor += 1
        self.point_entry_buf = ""

        if self.point_entry_cursor >= len(self.point_entry_fields):
            vals = self.point_entry_values
            n = self.calibration.add_cal_point(
                px=self.pending_px,
                py=self.pending_py,
                x_m=vals.get("x_m", 0.0),
                y_m=vals.get("y_m", 0.0),
                dist_m=vals.get("dist_m", 0.0),
                angle_deg=vals.get("angle_deg", 0.0),
            )
            print(f"Point P{n} added: px=({self.pending_px},{self.pending_py}) vals={vals}")
            self.pending_px = None
            self.pending_py = None
            self.point_entry_fields = []
            self.point_entry_cursor = 0
            self.point_entry_values = {}

    def _point_entry_cancel(self) -> None:
        self.pending_px = None
        self.pending_py = None
        self.point_pick_active = False
        self.point_entry_fields = []
        self.point_entry_cursor = 0
        self.point_entry_buf = ""
        self.point_entry_values = {}

    # ── utilities ─────────────────────────────────────────────────────────────

    def _screen_to_frame(self, x: int, y: int) -> Optional[tuple[int, int]]:
        if (
            x < self.off_x or x > self.off_x + self.draw_w
            or y < self.off_y or y > self.off_y + self.draw_h
            or self.scale <= 0
        ):
            return None
        fx = int((x - self.off_x) / self.scale)
        fy = int((y - self.off_y) / self.scale)
        return max(0, min(self.src_w - 1, fx)), max(0, min(self.src_h - 1, fy))

    def _current_window_size(self) -> tuple[int, int]:
        try:
            _x, _y, ww, wh = cv2.getWindowImageRect(self.window_name)
            if ww > 100 and wh > 100:
                return int(ww), int(wh)
        except Exception:
            pass
        return int(self.ui_cfg.window_width), int(self.ui_cfg.window_height)

    def _commit_edit(self) -> None:
        try:
            if not self.editing_key:
                return
            value = float(self.edit_buffer.replace(",", "."))
            self.calibration.set_value(self.editing_key, value)
            print(f"Saved {self.editing_key} = {value}")
        except Exception as exc:
            print(f"Invalid value for {self.editing_key}: {exc}")
        finally:
            self.editing_key = None
            self.edit_buffer = ""
