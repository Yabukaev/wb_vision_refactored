from __future__ import annotations

import threading

from app.config import MqttSection
from app.mqtt.mqtt_worker import MqttWorker


def _worker(**overrides) -> MqttWorker:
    cfg = MqttSection(enabled=True, host="127.0.0.1", port=1, client_id="test-client", queue_size=3)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return MqttWorker(cfg, threading.Event())


def test_connect_failure_keeps_worker_enabled_for_retry():
    worker = _worker()
    assert worker._connect() is False
    # P0.2: a failed connect must not permanently disable MQTT.
    assert worker.enabled is True
    assert worker.connected is False


def test_client_construction_works_with_paho_2x():
    worker = _worker()
    worker._connect()
    # P0.1: under paho-mqtt 2.x the old Client(client_id=...) call raises
    # "Unsupported callback API version" before any network I/O.
    assert "callback api" not in worker.last_error.lower()


def test_enqueue_drops_oldest_when_full():
    worker = _worker()
    for i in range(5):
        worker.enqueue("t", i)
    values = [worker.queue.get_nowait().value for _ in range(worker.queue.qsize())]
    assert values == [2, 3, 4]


def test_enqueue_noop_when_disabled():
    worker = _worker(enabled=False)
    worker.enabled = False
    worker.enqueue("t", 1)
    assert worker.queue.qsize() == 0
