from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from app.config import UISection
from app.core.latest_value import LatestValue
from app.types import FramePacket, VisionPacket
from app.vision.calibration import CalibrationData, CalibrationManager
from app.vision.overlay import FIELD_ORDER, draw_calibration, draw_panel, draw_tracks


class UIWorker:
    """OpenCV UI loop. On Windows this should run in the main thread."""

    def __init__(
        self,
        ui_cfg: UISection,
        frames: LatestValue[FramePacket],
        results: LatestValue[VisionPacket],
        calibration: CalibrationManager,
        stop_event: threading.Event,
    ) -> None:
        self.ui_cfg = ui_cfg
        self.frames = frames
        self.results = results
        self.calibration = calibration
        self.stop_event = stop_event

        self.window_name = "VISION STABLE"
        self.calib_mode: Optional[str] = None
        self.editing_key: Optional[str] = None
        self.edit_buffer = ""

        self.buttons: dict[str, tuple[int, int, int, int]] = {}
        self.fields: dict[str, tuple[int, int, int, int]] = {}

        self.scale = 1.0
        self.off_x = 0
        self.off_y = 0
        self.draw_w = 0
        self.draw_h = 0
        self.src_w = 0
        self.src_h = 0

        self.fullscreen = False

        # B-18: pre-allocated waiting canvas; rebuilt only on window resize
        self._waiting_canvas: Optional[np.ndarray] = None
        self._waiting_canvas_size: tuple[int, int] = (0, 0)

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

            # B-17: single calibration snapshot per frame, passed to both draw calls
            cal = self.calibration.snapshot()
            result, _ = self.results.get()

            # P-06: draw_tracks and draw_calibration happen inside _compose on resized frame
            canvas = self._compose(frame_packet.image, result, cal)
            cv2.imshow(self.window_name, canvas)

            if self._handle_key(cv2.waitKeyEx(1)):
                break

        self.stop_event.set()
        cv2.destroyAllWindows()

    def _show_waiting(self) -> None:
        # B-18: reuse pre-allocated canvas; only reallocate when window is resized
        window_w, window_h = self._current_window_size()
        if (window_w, window_h) != self._waiting_canvas_size:
            canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
            lines = [
                "Waiting for RTSP frame...",
                "ESC - exit",
                "A - set aim point",
                "F - set 4 floor points",
                "Click field -> type number -> Enter",
            ]
            y = 54
            for line in lines:
                cv2.putText(canvas, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 1, cv2.LINE_AA)
                y += 34
            self._waiting_canvas = canvas
            self._waiting_canvas_size = (window_w, window_h)

        cv2.imshow(self.window_name, self._waiting_canvas)

    def _current_window_size(self) -> tuple[int, int]:
        try:
            _x, _y, w, h = cv2.getWindowImageRect(self.window_name)
            if w > 100 and h > 100:
                return int(w), int(h)
        except Exception:
            pass

        return int(self.ui_cfg.window_width), int(self.ui_cfg.window_height)

    def _compose(self, frame: np.ndarray, packet: Optional[VisionPacket], cal: CalibrationData) -> np.ndarray:
        # B-17: accepts cal snapshot from run() — no second snapshot() call here
        window_w, window_h = self._current_window_size()

        panel_w = int(max(320, min(430, window_w * 0.25)))
        video_w = max(1, window_w - panel_w)
        video_h = window_h

        fh, fw = frame.shape[:2]
        canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)

        self.scale = min(video_w / fw, video_h / fh)
        self.draw_w = max(1, int(fw * self.scale))
        self.draw_h = max(1, int(fh * self.scale))
        self.off_x = int((video_w - self.draw_w) / 2)
        self.off_y = int((video_h - self.draw_h) / 2)
        self.src_w, self.src_h = fw, fh

        # P-06: resize FIRST, draw overlay on the smaller surface (avoids full-res copy)
        interp = cv2.INTER_AREA if self.scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (self.draw_w, self.draw_h), interpolation=interp)

        if result is not None:
            draw_tracks(
                resized,
                result.tracks,
                scale=self.scale,
                show_pose=self.ui_cfg.show_pose,
                show_tracks=self.ui_cfg.show_tracks,
            )

        if self.ui_cfg.show_calibration:
            draw_calibration(resized, cal, scale=self.scale)

        canvas[self.off_y:self.off_y + self.draw_h, self.off_x:self.off_x + self.draw_w] = resized

        cv2.rectangle(
            canvas,
            (self.off_x, self.off_y),
            (self.off_x + self.draw_w - 1, self.off_y + self.draw_h - 1),
            (70, 70, 75),
            1,
            cv2.LINE_AA,
        )

        self.buttons = {}
        self.fields = {}

        values: dict[str, str] = {}
        for key, _label in FIELD_ORDER:
            if self.editing_key == key:
                values[key] = self.edit_buffer
            else:
                value = getattr(cal, key, "")
                if isinstance(value, float):
                    values[key] = f"{value:g}"
                else:
                    values[key] = str(value)

        draw_panel(
            canvas,
            video_w,
            packet,
            self.calib_mode,
            self.buttons,
            self.fields,
            values,
            self.editing_key,
        )

        return canvas

    def _screen_to_frame(self, x: int, y: int) -> Optional[tuple[int, int]]:
        if (
            x < self.off_x
            or x > self.off_x + self.draw_w
            or y < self.off_y
            or y > self.off_y + self.draw_h
            or self.scale <= 0
        ):
            return None

        fx = int((x - self.off_x) / self.scale)
        fy = int((y - self.off_y) / self.scale)

        return max(0, min(self.src_w - 1, fx)), max(0, min(self.src_h - 1, fy))

    def _mouse_cb(self, event, x, y, flags, param) -> None:  # noqa: ANN001
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for name, rect in self.buttons.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "aim":
                    self.calib_mode = "aim"
                    print("Click AIM point")
                elif name == "floor4":
                    self.calib_mode = "floor4"
                    self.calibration.clear_floor_points()
                    print("Click 4 floor points clockwise")
                elif name == "save":
                    self.calibration.save()
                    print("Calibration saved")
                return

        for key, rect in self.fields.items():
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                cal = self.calibration.snapshot()
                self.editing_key = key
                self.edit_buffer = str(getattr(cal, key, ""))
                print(f"Editing {key}. Type value and press Enter.")
                return

        p = self._screen_to_frame(x, y)
        if p is None:
            return

        fx, fy = p

        if self.calib_mode == "aim":
            self.calibration.set_aim(fx, fy)
            self.calib_mode = None
            print(f"AIM saved: {fx}, {fy}")

        elif self.calib_mode == "floor4":
            n = self.calibration.add_floor_point(fx, fy)
            print(f"Floor point {n}: {fx}, {fy}")
            if n >= 4:
                self.calib_mode = None
                print("Floor calibration saved")

    def _handle_key(self, key: int) -> bool:
        if key == -1:
            return False

        low = key & 0xFF

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

        if low == 27:
            return True

        if low in (ord("a"), ord("A")):
            self.calib_mode = "aim"
            print("Click AIM point")
        elif low in (ord("f"), ord("F")):
            self.calib_mode = "floor4"
            self.calibration.clear_floor_points()
            print("Click 4 floor points clockwise")
        elif low in (ord("s"), ord("S")):
            self.calibration.save()
            print("Calibration saved")
        elif low in (ord("p"), ord("P")):
            self.ui_cfg.show_pose = not self.ui_cfg.show_pose
            print(f"show_pose={self.ui_cfg.show_pose}")
        elif low in (ord("t"), ord("T")):
            self.ui_cfg.show_tracks = not self.ui_cfg.show_tracks
            print(f"show_tracks={self.ui_cfg.show_tracks}")
        elif low in (ord("c"), ord("C")):
            self.ui_cfg.show_calibration = not self.ui_cfg.show_calibration
            print(f"show_calibration={self.ui_cfg.show_calibration}")
        elif low in (ord("m"), ord("M")):
            self.fullscreen = not self.fullscreen
            prop = cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, prop)

        return False

    def _commit_edit(self) -> None:
        try:
            if not self.editing_key:
                return

            value = float(self.edit_buffer.replace(",", "."))
            self.calibration.set_value(self.editing_key, value)
            print(f"Saved {self.editing_key} = {value}")

        except Exception as exc:
            print(f"Bad value for {self.editing_key}: {exc}")

        finally:
            self.editing_key = None
            self.edit_buffer = ""
