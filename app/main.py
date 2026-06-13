from __future__ import annotations

import argparse
import signal
import threading
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WB Vision refactored RTSP/YOLO/MQTT app")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--headless", action="store_true", help="Run without OpenCV UI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ConfigManager(args.config)
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

    print("START")
    print(f"CONFIG: {Path(args.config).resolve()}")
    print(f"CAMERA: {settings.camera.id}")
    print(f"MODEL: {settings.vision.model_path}")
    print(f"UI: {'off' if args.headless or not settings.ui.enabled else 'on'}")

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
        print("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

