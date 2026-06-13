from __future__ import annotations

import argparse
import logging
import signal
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
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV UI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    try:
        config = ConfigManager(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Config error: %s", exc)
        return 2
    settings = config.get()

    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

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
    )

    log.info("START")
    log.info("CONFIG: %s", Path(args.config).resolve())
    log.info("CAMERA: %s", settings.camera.id)
    log.info("MODEL: %s", settings.vision.model_path)
    log.info("UI: %s", "off" if args.headless or not settings.ui.enabled else "on")

    mqtt_worker.start()
    rtsp_reader.start()
    inference_worker.start()

    try:
        if not args.headless and settings.ui.enabled:
            UIWorker(settings.ui, frames, results, calibration, stop_event).run()
        else:
            while not stop_event.is_set():
                time.sleep(0.5)
    finally:
        stop_event.set()
        frames.close()
        results.close()
        for worker in (rtsp_reader, inference_worker, mqtt_worker):
            worker.join(timeout=3.0)
        log.info("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

