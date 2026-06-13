from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from app.config import UISection
from app.core.latest_value import LatestValue
from app.types import FramePacket, VisionPacket
from app.vision.activity_classifier import ActivityClassifier
from app.vision.calibration import CalibrationData, CalibrationManager
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

        # Numeric field editing
        self.editing_key: Optional[str] = None
        self.edit_buffer = ""

        # Calibration interaction: None / "aim" / "floor4"
        self.calib_mode: Optional[str] = None

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
                "F - set 4 floor points",
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
                resized, packet.tracks, scale=self.scale,
                show_pose=self.ui_cfg.show_pose,
                show_tracks=self.ui_cfg.show_tracks,
            )

        if self.ui_cfg.show_calibration:
            draw_calibration(
                resized, cal, scale=self.scale,
                zone_polygon_px=self.zone_polygon_px if self.zone_draw_active else None,
            )

        canvas[self.off_y:self.off_y + self.draw_h, self.off_x:self.off_x + self.draw_w] = resized
        self._draw_video_border(canvas)

        self.buttons = {}
        self.fields = {}

        field_values: dict[str, str] = {}
        for key, _ in FIELD_ORDER:
            if self.editing_key == key:
                field_values[key] = self.edit_buffer
            else:
                value = getattr(cal, key, "")
                field_values[key] = f"{value:g}" if isinstance(value, float) else str(value)

        ui_state = {
            "calib_mode": self.calib_mode,
            "floor_count": len(cal.floor_points or []),
            "zones": cal.zones or [],
            "zone_draw_active": self.zone_draw_active,
            "zone_polygon_px": self.zone_polygon_px,
            "zone_name_mode": self.zone_name_mode,
            "zone_name_buf": self.zone_name_buf,
            "activity_enabled": self.activity.is_enabled if self.activity else False,
            "activity_available": self.activity is not None,
        }

        draw_panel(canvas, video_w, packet, ui_state, self.buttons, self.fields,
                   field_values, self.editing_key)
        return canvas

    def _draw_video_border(self, canvas: np.ndarray) -> None:
        x1, y1 = self.off_x, self.off_y
        x2, y2 = self.off_x + self.draw_w - 1, self.off_y + self.draw_h - 1
        if self.calib_mode in ("aim", "floor4"):
            col = (60, 90, 255) if self.calib_mode == "aim" else (255, 190, 60)
            hint = "CLICK AIM POINT IN VIDEO" if self.calib_mode == "aim" else "CLICK 4 FLOOR CORNERS (clockwise)"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), col, 2, cv2.LINE_AA)
            (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.66, 2)
            hx = x1 + (self.draw_w - hw) // 2
            hy = y2 - 14
            cv2.rectangle(canvas, (hx - 6, hy - hh - 6), (hx + hw + 6, hy + 4), (0, 0, 0), -1)
            cv2.putText(canvas, hint, (hx, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.66, col, 2, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (70, 70, 75), 1, cv2.LINE_AA)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> bool:
        if key == -1:
            return False
        low = key & 0xFF

        # Numeric field editing
        if self.editing_key:
            if low in (13, 10):
                self._commit_edit()
            elif low == 27:
                self.editing_key = None
                self.edit_buffer = ""
            elif low in (8, 127):
                self.edit_buffer = self.edit_buffer[:-1]
            elif 0 <= low <= 255 and chr(low) in "0123456789.-":
                self.edit_buffer += chr(low)
            return False

        # Zone name entry
        if self.zone_name_mode:
            if low in (13, 10):
                self._save_zone()
            elif low == 27:
                self._cancel_zone("Zone cancelled")
            elif low in (8, 127):
                self.zone_name_buf = self.zone_name_buf[:-1]
            elif 32 <= low <= 126 and len(self.zone_name_buf) < 20:
                self.zone_name_buf += chr(low)
            return False

        # Zone drawing (intercepts Enter/Esc)
        if self.zone_draw_active:
            if low == 27:
                self._cancel_zone("Zone cancelled")
            elif low in (13, 10) and len(self.zone_polygon_px) >= 3:
                self.zone_draw_active = False
                self.zone_name_mode = True
                print("Type zone name, then Enter")
            return False

        # Normal commands
        if low == 27:
            if self.calib_mode:
                self.calib_mode = None
                print("Cancelled")
                return False
            return True
        if low in (ord("a"), ord("A")):
            self.calib_mode = "aim"
            print("Click AIM point in video")
        elif low in (ord("f"), ord("F")):
            self.calib_mode = "floor4"
            self.calibration.clear_floor_points()
            print("Click 4 floor points clockwise")
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

    # ── mouse ───────────────────────────────────────────────────────────────────

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
                cal = self.calibration.snapshot()
                self.editing_key = key
                self.edit_buffer = f"{getattr(cal, key, 0.0):g}"
                return

        # Video area clicks
        p = self._screen_to_frame(x, y)
        if p is None:
            return
        fx, fy = p

        if self.calib_mode == "aim":
            self.calibration.set_aim(fx, fy)
            self.calib_mode = None
            print(f"AIM set: ({fx}, {fy})")

        elif self.calib_mode == "floor4":
            n = self.calibration.add_floor_point(fx, fy)
            print(f"Floor point {n}/4: ({fx}, {fy})")
            if n >= 4:
                self.calib_mode = None
                print("Floor calibration complete")

        elif self.zone_draw_active and not self.zone_name_mode:
            self.zone_polygon_px.append([fx, fy])
            print(f"Zone point {len(self.zone_polygon_px)}: ({fx}, {fy})")

    def _handle_button(self, name: str) -> None:
        if name == "aim":
            self.calib_mode = "aim"
            print("Click AIM point in video")
        elif name == "floor4":
            self.calib_mode = "floor4"
            self.calibration.clear_floor_points()
            print("Click 4 floor points clockwise")
        elif name == "save":
            self.calibration.save()
            print("Calibration saved")
        elif name == "toggle_activity":
            if self.activity is not None:
                enabled = self.activity.toggle_enabled()
                print(f"Activity: {'ON' if enabled else 'OFF'}")
        elif name == "zone_draw":
            self.zone_draw_active = True
            self.zone_polygon_px = []
            self.zone_name_mode = False
            self.zone_name_buf = ""
            self.calib_mode = None
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

    # ── zone helpers ────────────────────────────────────────────────────────────

    def _save_zone(self) -> None:
        name = self.zone_name_buf.strip() or "zone"
        cal = self.calibration.snapshot()
        color = _ZONE_COLORS[len(cal.zones or []) % len(_ZONE_COLORS)]
        self.calibration.add_zone(name, [[p[0], p[1]] for p in self.zone_polygon_px], color)
        self._cancel_zone(f"Zone '{name}' saved")

    def _cancel_zone(self, msg: str) -> None:
        self.zone_polygon_px = []
        self.zone_name_mode = False
        self.zone_name_buf = ""
        self.zone_draw_active = False
        print(msg)

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
