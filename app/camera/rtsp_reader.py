from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from app.config import CameraSection
from app.core.latest_value import LatestValue
from app.types import FramePacket


@dataclass(slots=True)
class RtspStats:
    connected: bool = False
    frames: int = 0
    reconnects: int = 0
    last_error: str = ""
    fps: float = 0.0
    last_frame_ts: float = 0.0


class RtspReader(threading.Thread):
    """Continuously reads RTSP into a LatestValue slot."""

    def __init__(self, camera: CameraSection, out: LatestValue[FramePacket], stop_event: threading.Event) -> None:
        super().__init__(name="rtsp-reader", daemon=True)
        self.camera = camera
        self.out = out
        self.stop_event = stop_event
        self.stats = RtspStats()
        self._frame_id = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_fps_t = time.monotonic()
        self._fps_counter = 0

    def run(self) -> None:
        if self.camera.ffmpeg_capture_options:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", self.camera.ffmpeg_capture_options)

        bad_reads = 0
        while not self.stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._connect()
                bad_reads = 0
                if self._cap is None or not self._cap.isOpened():
                    self.stop_event.wait(self.camera.reconnect_delay_sec)
                    continue

            ok, frame = self._cap.read()
            now = time.time()
            if not ok or frame is None:
                bad_reads += 1
                self.stats.last_error = f"no frame ({bad_reads})"
                if bad_reads >= self.camera.read_fail_limit:
                    self._release()
                    self.stats.reconnects += 1
                    self.stats.connected = False
                    self.stop_event.wait(self.camera.reconnect_delay_sec)
                continue

            bad_reads = 0
            self._frame_id += 1
            self.stats.frames += 1
            self.stats.last_frame_ts = now
            self.stats.connected = True
            self._update_fps()
            self.out.set(FramePacket(frame_id=self._frame_id, ts=now, image=frame))

        self._release()
        self.out.close()

    def _connect(self) -> None:
        self._release()
        backend = cv2.CAP_FFMPEG if self.camera.backend.lower() == "ffmpeg" else 0
        cap = cv2.VideoCapture(self.camera.rtsp_url, backend)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(self.camera.buffer_size))
        except Exception:
            pass
        self._cap = cap
        self.stats.connected = cap.isOpened()
        if not self.stats.connected:
            self.stats.last_error = "cannot open RTSP"

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _update_fps(self) -> None:
        self._fps_counter += 1
        now = time.monotonic()
        dt = now - self._last_fps_t
        if dt >= 1.0:
            inst = self._fps_counter / max(dt, 1e-6)
            self.stats.fps = inst if self.stats.fps <= 0 else 0.8 * self.stats.fps + 0.2 * inst
            self._fps_counter = 0
            self._last_fps_t = now

