from __future__ import annotations

import json
import queue
import threading
from typing import Any, Optional

import paho.mqtt.client as mqtt

from app.config import MqttSection
from app.types import MqttMessage


class MqttWorker(threading.Thread):
    """MQTT publisher in its own thread with drop-oldest queue."""

    def __init__(self, cfg: MqttSection, stop_event: threading.Event) -> None:
        super().__init__(name="mqtt-worker", daemon=True)
        self.cfg = cfg
        self.stop_event = stop_event
        self.queue: queue.Queue[MqttMessage] = queue.Queue(maxsize=int(cfg.queue_size))
        self.enabled = bool(cfg.enabled)
        self.connected = False
        self.last_error = ""
        self._client: Optional[mqtt.Client] = None

    def enqueue(self, topic: str, value: Any, retain: bool = False) -> None:
        if not self.enabled:
            return
        msg = MqttMessage(topic=topic, value=value, retain=retain)
        try:
            self.queue.put_nowait(msg)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(msg)
            except queue.Full:
                pass

    def run(self) -> None:
        if not self.enabled:
            return
        self._connect()
        while not self.stop_event.is_set():
            try:
                msg = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._publish(msg)
        self._disconnect()

    def _connect(self) -> None:
        try:
            self._client = mqtt.Client(client_id=self.cfg.client_id)
            if self.cfg.username:
                self._client.username_pw_set(self.cfg.username, self.cfg.password or None)
            self._client.connect(self.cfg.host, int(self.cfg.port), 5)
            self._client.loop_start()
            self.connected = True
        except Exception as exc:
            self.enabled = False
            self.connected = False
            self.last_error = str(exc)
            print(f"MQTT disabled: {exc}")

    def _publish(self, msg: MqttMessage) -> None:
        if not self.enabled or self._client is None:
            return
        topic = f"{self.cfg.prefix}/{msg.topic}".replace("//", "/")
        value = msg.value
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        try:
            self._client.publish(topic, value, retain=msg.retain)
        except Exception as exc:
            self.last_error = str(exc)

    def _disconnect(self) -> None:
        if self._client is None:
            return
        try:
            self._client.loop_stop()
            self._client.disconnect()
        except Exception:
            pass

