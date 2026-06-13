from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.camera.rtsp_reader import RtspReader
from app.config import ConfigManager
from app.core.latest_value import LatestValue
from app.mqtt.mqtt_worker import MqttWorker
from app.types import FramePacket, VisionPacket
from app.ui.ui_worker import UIWorker
from app.vision.calibration import CalibrationManager
from app.vision.inference_worker import InferenceWorker
from app.vision.tracker import StableTracker


log = logging.getLogger("main")


def setup_logging() -> None:
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(log_dir / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WB Vision refactored RTSP/YOLO/MQTT app")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--debug", action="store_true", help="Enable UI window and verbose logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        config = ConfigManager(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Config error: %s", exc)
        return 2
    settings = config.get()

    stop_event = threading.Event()

    def request_stop(signum=None, frame=None) -> None:  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    # B-10: on Windows, SIGTERM from outside (NSSM, Task Scheduler) is TerminateProcess
    # and never reaches Python. Register a console ctrl handler for CTRL_CLOSE_EVENT
    # so services that send WM_CLOSE / GenerateConsoleCtrlEvent get graceful shutdown.
    if sys.platform == "win32":
        import ctypes
        _HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_uint)

        def _win_ctrl_handler(ctrl_type: int) -> bool:
            if ctrl_type in (0, 2, 5, 6):  # C, CLOSE, LOGOFF, SHUTDOWN
                request_stop()
                return True
            return False

        # Keep a module-level reference so the callable isn't GC'd
        main._win_handler = _HandlerRoutine(_win_ctrl_handler)  # type: ignore[attr-defined]
        ctypes.windll.kernel32.SetConsoleCtrlHandler(main._win_handler, True)  # type: ignore[attr-defined]

    frames: LatestValue[FramePacket] = LatestValue()
    results: LatestValue[VisionPacket] = LatestValue()
    calibration = CalibrationManager(config)
    tracker = StableTracker(settings.tracker)

    mqtt_worker = MqttWorker(settings.mqtt, stop_event)
    rtsp_reader = RtspReader(settings.camera, frames, stop_event)
    inference_worker = InferenceWorker(
        vision_cfg=settings.vision,
        camera_cfg=settings.camera,
        mqtt_cfg=settings.mqtt,
        frames=frames,
        results=results,
        calibration=calibration,
        tracker=tracker,
        mqtt_worker=mqtt_worker,
        stop_event=stop_event,
        reader_fps_getter=lambda: rtsp_reader.stats.fps,
        activity_cfg=settings.activity,
    )

    log.info("START")
    log.info("CONFIG: %s", Path(args.config).resolve())
    log.info("CAMERA: %s", settings.camera.id)
    log.info("MODEL: %s", settings.vision.model_path)
    log.info("UI: %s", "on" if args.debug and settings.ui.enabled else "off")

    web_server = None
    if settings.web.enabled:
        from app.web.server import WebServer
        web_server = WebServer(
            settings.web, frames, results, calibration, stop_event,
            vision_cfg=settings.vision, tracker_cfg=settings.tracker,
        )
        log.info("WEB: http://%s:%s", settings.web.host, settings.web.port)

    mqtt_worker.start()
    rtsp_reader.start()
    inference_worker.start()
    if web_server is not None:
        web_server.start()

    try:
        if args.debug and settings.ui.enabled:
            UIWorker(
                settings.ui, frames, results, calibration, stop_event,
                activity=inference_worker.activity_classifier,
            ).run()
        else:
            while not stop_event.is_set():
                time.sleep(0.5)
    finally:
        stop_event.set()
        # B-03: workers own their LatestValue lifecycle (RtspReader closes frames,
        # InferenceWorker closes results). No redundant close() calls here.
        workers = [rtsp_reader, inference_worker, mqtt_worker]
        if web_server is not None:
            workers.append(web_server)
        for worker in workers:
            worker.join(timeout=3.0)
        log.info("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
