from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from app.config import WebSection
from app.core.latest_value import LatestValue
from app.types import FramePacket, VisionPacket
from app.vision.calibration import CalibrationManager
from app.vision.overlay import draw_calibration, draw_tracks
from app.web.page import INDEX_HTML


def build_app(
    frames: LatestValue[FramePacket],
    results: LatestValue[VisionPacket],
    calibration: CalibrationManager,
    web_cfg: WebSection,
) -> FastAPI:
    app = FastAPI(title="WB Vision")

    # ── pages ────────────────────────────────────────────────────────────────
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    # ── live state ───────────────────────────────────────────────────────────
    @app.get("/api/state")
    def state() -> JSONResponse:
        cal = calibration.snapshot()
        packet, _ = results.get()
        fp, _ = frames.get()
        src_w = fp.width if fp else 0
        src_h = fp.height if fp else 0
        tracks = []
        if packet is not None:
            for tr in packet.tracks:
                t = {
                    "id": tr.track_id, "state": tr.state, "motion": tr.motion,
                    "activity": tr.activity, "conf": round(tr.conf, 2),
                    "foot": [tr.foot[0], tr.foot[1]],
                }
                if tr.geo:
                    t["geo"] = {
                        "x_m": round(tr.geo.x_m, 2), "y_m": round(tr.geo.y_m, 2),
                        "dist_cam_m": round(tr.geo.distance_cam_m, 2),
                        "zone": tr.geo.zone,
                    }
                tracks.append(t)
        return JSONResponse({
            "source": {"width": src_w, "height": src_h},
            "calibration": {
                "aim_px": cal.aim_px, "aim_py": cal.aim_py,
                "quad_px": cal.quad_px or [],
                "trap_edges_m": cal.trap_edges_m or [0, 0, 0, 0],
                "trap_angles_deg": cal.trap_angles_deg or [90, 90, 90, 90],
                "camera_height_m": cal.camera_height_m,
                "cam_to_aim_m": cal.cam_to_aim_m,
                "room_width_m": cal.room_width_m,
                "room_depth_m": cal.room_depth_m,
                "zones": cal.zones or [],
                "closure_error_m": calibration.trapezoid_closure_error(),
            },
            "status": {
                "reader_fps": round(packet.reader_fps, 1) if packet else 0.0,
                "inference_fps": round(packet.inference_fps, 1) if packet else 0.0,
                "infer_ms": round(packet.infer_ms, 0) if packet else 0.0,
                "people": len(packet.tracks) if packet else 0,
                "cpu": round(packet.cpu_percent, 0) if packet else 0.0,
                "ram": round(packet.ram_percent, 0) if packet else 0.0,
            },
            "tracks": tracks,
        })

    # ── calibration mutations ────────────────────────────────────────────────
    @app.post("/api/aim")
    def set_aim(body: dict) -> dict:
        calibration.set_aim(int(body["x"]), int(body["y"]))
        return {"ok": True}

    @app.post("/api/quad_point")
    def set_quad_point(body: dict) -> dict:
        n = calibration.set_quad_point(int(body["index"]), int(body["x"]), int(body["y"]))
        return {"ok": True, "count": n}

    @app.post("/api/quad/clear")
    def clear_quad() -> dict:
        calibration.clear_quad()
        return {"ok": True}

    @app.post("/api/edge")
    def set_edge(body: dict) -> dict:
        calibration.set_trap_edge(int(body["index"]), float(body["value"]))
        return {"ok": True}

    @app.post("/api/angle")
    def set_angle(body: dict) -> dict:
        calibration.set_trap_angle(int(body["index"]), float(body["value"]))
        return {"ok": True}

    @app.post("/api/value")
    def set_value(body: dict) -> JSONResponse:
        try:
            calibration.set_value(str(body["key"]), float(body["value"]))
        except KeyError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True})

    @app.post("/api/zone/add")
    def zone_add(body: dict) -> dict:
        poly = [[int(p[0]), int(p[1])] for p in body.get("polygon_px", [])]
        if len(poly) >= 3:
            calibration.add_zone(str(body.get("name", "zone")), poly, body.get("color"))
        return {"ok": True}

    @app.post("/api/zone/delete")
    def zone_delete(body: dict) -> dict:
        calibration.delete_zone(int(body["index"]))
        return {"ok": True}

    @app.post("/api/save")
    def save() -> dict:
        calibration.save()
        return {"ok": True}

    # ── MJPEG video ──────────────────────────────────────────────────────────
    @app.get("/video")
    def video() -> StreamingResponse:
        return StreamingResponse(
            _mjpeg(frames, results, calibration, web_cfg),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def _mjpeg(frames, results, calibration, web_cfg: WebSection):
    boundary = b"--frame\r\n"
    min_interval = 1.0 / max(1.0, float(web_cfg.stream_fps))
    max_w = int(web_cfg.stream_max_width)
    last_seq = 0
    last_emit = 0.0
    while True:
        fp, seq = frames.wait_next(last_seq=last_seq, timeout=1.0)
        if fp is None:
            time.sleep(0.05)
            continue
        last_seq = seq
        now = time.monotonic()
        if now - last_emit < min_interval:
            continue
        last_emit = now

        frame = fp.image
        fh, fw = frame.shape[:2]
        scale = min(1.0, max_w / fw) if fw > 0 else 1.0
        if scale < 1.0:
            frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame = frame.copy()

        packet, _ = results.get()
        if packet is not None:
            draw_tracks(frame, packet.tracks, scale=scale)
        draw_calibration(frame, calibration.snapshot(), scale=scale)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ok:
            continue
        yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


class WebServer(threading.Thread):
    """Runs the FastAPI control UI via uvicorn in a background thread."""

    def __init__(
        self,
        web_cfg: WebSection,
        frames: LatestValue[FramePacket],
        results: LatestValue[VisionPacket],
        calibration: CalibrationManager,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="web-server", daemon=True)
        self.cfg = web_cfg
        self.frames = frames
        self.results = results
        self.calibration = calibration
        self.stop_event = stop_event
        self._server: Optional[object] = None

    def run(self) -> None:
        import logging

        import uvicorn

        log = logging.getLogger("web")
        try:
            app = build_app(self.frames, self.results, self.calibration, self.cfg)
            config = uvicorn.Config(
                app, host=self.cfg.host, port=int(self.cfg.port),
                log_level="warning", access_log=False,
            )
            server = uvicorn.Server(config)
            # uvicorn installs OS signal handlers in serve(); that only works in
            # the main thread, so disable it when running as a worker thread.
            server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
            self._server = server
            threading.Thread(target=self._watch_stop, daemon=True).start()
            log.info("serving on http://%s:%s", self.cfg.host, self.cfg.port)
            server.run()
        except Exception:
            log.exception("web server crashed")

    def _watch_stop(self) -> None:
        self.stop_event.wait()
        if self._server is not None:
            self._server.should_exit = True
