from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import psutil

from dataclasses import replace

from app.config import ActivitySection, CameraSection, MqttSection, VisionSection
from app.core.latest_value import LatestValue
from app.mqtt.mqtt_worker import MqttWorker
from app.types import FramePacket, TrackSnapshot, VisionPacket
from app.vision.activity_classifier import ActivityClassifier
from app.vision.calibration import CalibrationManager
from app.vision.detector import YoloPoseDetector, suppress_duplicates
from app.vision.tracker import StableTracker


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        vision_cfg: VisionSection,
        camera_cfg: CameraSection,
        mqtt_cfg: MqttSection,
        frames: LatestValue[FramePacket],
        results: LatestValue[VisionPacket],
        calibration: CalibrationManager,
        tracker: StableTracker,
        mqtt_worker: Optional[MqttWorker],
        stop_event: threading.Event,
        reader_fps_getter=None,
        activity_cfg: Optional[ActivitySection] = None,
    ) -> None:
        super().__init__(name="inference-worker", daemon=True)
        self.vision_cfg = vision_cfg
        self.camera_cfg = camera_cfg
        self.mqtt_cfg = mqtt_cfg
        self.frames = frames
        self.results = results
        self.calibration = calibration
        self.tracker = tracker
        self.mqtt = mqtt_worker
        self.stop_event = stop_event
        self.reader_fps_getter = reader_fps_getter
        self.activity_classifier: Optional[ActivityClassifier] = (
            ActivityClassifier(activity_cfg) if activity_cfg is not None else None
        )
        self.detector: Optional[YoloPoseDetector] = None
        self._inference_fps = 0.0
        self._last_infer_ts = 0.0
        self._last_health_ts = 0.0
        self._last_tracks_pub_ts = 0.0
        self._last_psutil_ts = 0.0  # P-04: slow psutil timer
        self._cpu = 0.0
        self._ram = 0.0
        self._prev_done_ts = time.time()
        self._published_track_ids: set[int] = set()
        self._model_lock = threading.Lock()
        self._pending_pose_model: Optional[str] = None
        self._pending_object_model: Optional[str] = None
        psutil.cpu_percent(interval=None)  # B-02: primer — first real call returns 0 otherwise

    # ── model hot-swap (web thread requests, inference thread applies) ──────────

    def request_pose_model(self, path: str) -> None:
        with self._model_lock:
            self._pending_pose_model = str(path)

    def request_object_model(self, path: str) -> None:
        with self._model_lock:
            self._pending_object_model = str(path)

    def current_models(self) -> dict:
        with self._model_lock:
            pose = self._pending_pose_model or self.vision_cfg.model_path
            obj = self._pending_object_model
        if obj is None and self.activity_classifier is not None:
            obj = self.activity_classifier.object_model_path
        return {"pose": pose, "object": obj}

    def _apply_pending_models(self) -> None:
        with self._model_lock:
            pose = self._pending_pose_model
            obj = self._pending_object_model
            self._pending_pose_model = None
            self._pending_object_model = None
        if pose and pose != self.vision_cfg.model_path:
            try:
                self.vision_cfg.model_path = pose
                self.detector = YoloPoseDetector(self.vision_cfg)
                logging.getLogger("inference").info("pose model -> %s", pose)
            except Exception:
                logging.getLogger("inference").exception("pose model swap failed")
        if obj and self.activity_classifier is not None:
            self.activity_classifier.set_object_model(obj)

    def run(self) -> None:
        self.detector = YoloPoseDetector(self.vision_cfg)
        if self.activity_classifier is not None:
            self.activity_classifier.start()
        self._mqtt(f"{self.camera_cfg.id}/status", "online", retain=True)
        last_seq = 0

        while not self.stop_event.is_set():
            packet, seq = self.frames.wait_next(last_seq=last_seq, timeout=0.5)
            if packet is None:
                continue
            last_seq = seq

            now = time.time()
            # read inference_fps live so the web UI can retune it without restart
            min_interval = 1.0 / max(0.1, float(self.vision_cfg.inference_fps))
            if now - self._last_infer_ts < min_interval:
                continue

            self._last_infer_ts = now
            self._apply_pending_models()  # hot-swap requested via web, if any
            t0 = time.perf_counter()
            detections = self.detector.predict(packet.image)
            detections = suppress_duplicates(
                detections,
                foot_dist_px=float(self.vision_cfg.duplicate_foot_dist_px),
                iou_threshold=float(self.vision_cfg.duplicate_iou),
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0

            dt = now - self._prev_done_ts
            self._prev_done_ts = now
            inst_fps = 1.0 / max(dt, 1e-6)
            self._inference_fps = inst_fps if self._inference_fps <= 0 else 0.85 * self._inference_fps + 0.15 * inst_fps

            tracks = self.tracker.update(detections, now=now, geo_fn=self.calibration.pixel_to_floor)
            if self.activity_classifier is not None:
                activities = self.activity_classifier.classify(packet.image, tracks, now)
                tracks = [replace(tr, activity=activities.get(tr.track_id, "")) for tr in tracks]
            self._update_system_stats(now)  # P-04: only calls psutil every 2s
            reader_fps = float(self.reader_fps_getter() or 0.0) if self.reader_fps_getter else 0.0

            result = VisionPacket(
                frame_id=packet.frame_id,
                ts=now,
                infer_ms=infer_ms,
                inference_fps=self._inference_fps,
                source_width=packet.width,
                source_height=packet.height,
                tracks=tracks,
                detections_count=len(detections),
                cpu_percent=self._cpu,
                ram_percent=self._ram,
                reader_fps=reader_fps,
            )
            self.results.set(result)
            self._publish(now, result)

        self._mqtt(f"{self.camera_cfg.id}/status", "offline", retain=True)
        self.results.close()

    def _update_system_stats(self, now: float) -> None:
        # P-04: psutil polls OS every call; limit to once per 2s
        if now - self._last_psutil_ts >= 2.0:
            self._cpu = psutil.cpu_percent(interval=None)
            self._ram = psutil.virtual_memory().percent
            self._last_psutil_ts = now

    def _publish(self, now: float, packet: VisionPacket) -> None:
        tracks_hz = max(0.2, float(self.mqtt_cfg.publish_tracks_hz))
        publish_tracks = (now - self._last_tracks_pub_ts) >= (1.0 / tracks_hz)
        if publish_tracks:
            self._last_tracks_pub_ts = now
            current_ids = {tr.track_id for tr in packet.tracks}
            for gone_id in self._published_track_ids - current_ids:
                self._publish_gone(gone_id, now)
            self._published_track_ids = current_ids
            for tr in packet.tracks:
                self._publish_track(tr, packet.source_width, packet.source_height, now)

        if now - self._last_health_ts >= 1.0:
            self._last_health_ts = now
            cid = self.camera_cfg.id
            self._mqtt(f"{cid}/health/reader_fps", round(packet.reader_fps, 2))
            self._mqtt(f"{cid}/health/inference_fps", round(packet.inference_fps, 2))
            self._mqtt(f"{cid}/health/infer_ms", round(packet.infer_ms, 1))
            self._mqtt(f"{cid}/health/cpu", round(packet.cpu_percent, 1))
            self._mqtt(f"{cid}/health/ram", round(packet.ram_percent, 1))
            self._mqtt(f"{cid}/health/people_count", len(packet.tracks))
            self._mqtt(f"{cid}/presence", "ON" if packet.tracks else "OFF")
            self._mqtt(f"{cid}/status", "online", retain=True)

    def _publish_gone(self, track_id: int, now: float) -> None:
        base = f"{self.camera_cfg.id}/person/{track_id}"
        self._mqtt(f"{base}/state", "gone")
        self._mqtt(f"{base}/json", json.dumps({"id": track_id, "state": "gone", "ts": now}))

    def _publish_track(self, tr: TrackSnapshot, fw: int, fh: int, now: float) -> None:
        # B-04: use the passed `now` timestamp instead of calling time.time() again
        cid = self.camera_cfg.id
        fx, fy = tr.foot
        # Stable schema: id, coordinates, posture (sit/stand/lie), motion
        # (walk/sleep/...), and classification activity (phone/book/eating/...).
        zone = tr.geo.zone if tr.geo else ""
        payload: dict = {
            "id": tr.track_id,
            "pose": tr.state,            # sitting / standing / lying
            "motion": tr.motion,         # walking / sleeping / fallen / stationary
            "activity": tr.activity,     # on phone / reading / eating / ... ("" if none)
            "zone": zone,
            "confidence": round(tr.conf, 3),
            "foot_px": {"x": fx, "y": fy},
            "frame": {"width": fw, "height": fh},
            "hits": tr.hits,
            "age_sec": round(tr.age_sec, 2),
            "ts": now,
        }
        if tr.geo:
            payload["x_m"] = round(tr.geo.x_m, 3)
            payload["y_m"] = round(tr.geo.y_m, 3)
            payload["distance_cam_m"] = round(tr.geo.distance_cam_m, 3)
            payload["distance_floor_m"] = round(tr.geo.distance_m, 3)
            payload["inside_room"] = tr.geo.inside_room
            payload["inside_calibration_zone"] = tr.geo.inside_calibration_zone

        # P-07: serialize JSON once and reuse
        payload_json = json.dumps(payload, ensure_ascii=False)
        base = f"{cid}/person/{tr.track_id}"
        self._mqtt(f"{base}/json", payload_json)
        self._mqtt(f"{base}/pose", tr.state)
        self._mqtt(f"{base}/state", tr.state)   # kept for backward compatibility
        self._mqtt(f"{base}/motion", tr.motion)
        self._mqtt(f"{base}/activity", tr.activity or "none")
        self._mqtt(f"{base}/zone", zone or "none")

        # P-03: most-used flat geo fields; everything else is in /json
        if tr.geo:
            self._mqtt(f"{base}/x_m", round(tr.geo.x_m, 3))
            self._mqtt(f"{base}/y_m", round(tr.geo.y_m, 3))
            self._mqtt(f"{base}/inside_room", "ON" if tr.geo.inside_room else "OFF")

    def _mqtt(self, topic: str, value, retain: bool = False) -> None:
        if self.mqtt is not None:
            self.mqtt.enqueue(topic, value, retain=retain)
