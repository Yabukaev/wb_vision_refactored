from __future__ import annotations

import threading

from app.config import CameraSection, MqttSection, VisionSection
from app.types import TrackSnapshot, VisionPacket
from app.vision.inference_worker import InferenceWorker


class FakeMqtt:
    def __init__(self) -> None:
        self.messages: list[tuple[str, object]] = []

    def enqueue(self, topic: str, value, retain: bool = False) -> None:
        self.messages.append((topic, value))


def _snapshot(track_id: int) -> TrackSnapshot:
    return TrackSnapshot(
        track_id=track_id,
        box=(0, 0, 50, 100),
        conf=0.9,
        foot=(25, 100),
        center=(25, 50),
        state="standing",
        last_seen=1.0,
        hits=3,
        age_sec=1.0,
    )


def _packet(track_ids: list[int]) -> VisionPacket:
    tracks = [_snapshot(tid) for tid in track_ids]
    return VisionPacket(
        frame_id=1,
        ts=1.0,
        infer_ms=10.0,
        inference_fps=4.0,
        source_width=640,
        source_height=480,
        tracks=tracks,
        detections_count=len(tracks),
    )


def _worker(fake: FakeMqtt) -> InferenceWorker:
    return InferenceWorker(
        vision_cfg=VisionSection(),
        camera_cfg=CameraSection(id="cam1"),
        mqtt_cfg=MqttSection(publish_tracks_hz=1000.0),
        frames=None,
        results=None,
        calibration=None,
        tracker=None,
        mqtt_worker=fake,
        stop_event=threading.Event(),
    )


def test_disappeared_track_publishes_gone_state():
    fake = FakeMqtt()
    worker = _worker(fake)

    worker._publish(now=1.0, packet=_packet([7]))
    fake.messages.clear()
    worker._publish(now=2.0, packet=_packet([]))

    assert ("cam1/person/7/state", "gone") in fake.messages


def test_alive_track_is_not_marked_gone():
    fake = FakeMqtt()
    worker = _worker(fake)

    worker._publish(now=1.0, packet=_packet([7]))
    worker._publish(now=2.0, packet=_packet([7]))

    assert ("cam1/person/7/state", "gone") not in fake.messages
