from __future__ import annotations

import json
import logging
import queue
import threading
from typing import Any, Optional

import paho.mqtt.client as mqtt

from app.config import MqttSection
from app.types import MqttMessage

log = logging.getLogger("mqtt")


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

    def enqueue(self, topic: str, value: Any, retain: bool = False, absolute: bool = False) -> None:
        if not self.enabled:
            return
        msg = MqttMessage(topic=topic, value=value, retain=retain, absolute=absolute)
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
        while not self.stop_event.is_set():
            if self._client is None:
                if not self._connect():
                    self.stop_event.wait(float(self.cfg.reconnect_delay_sec))
                    continue
            try:
                msg = self.queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._publish(msg)

        # B-01: drain remaining queue so retain=True messages (e.g. status=offline)
        # enqueued by InferenceWorker just before stop are actually sent.
        self._drain()
        self._disconnect()

    def _drain(self) -> None:
        while True:
            try:
                msg = self.queue.get_nowait()
                self._publish(msg)
            except queue.Empty:
                break

    def _connect(self) -> bool:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.cfg.client_id)
            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            if self.cfg.username:
                client.username_pw_set(self.cfg.username, self.cfg.password or None)
            client.connect(self.cfg.host, int(self.cfg.port), 5)
            client.loop_start()
            self._client = client
            self.last_error = ""
            return True
        except Exception as exc:
            # Keep the worker enabled: it retries until the broker is reachable.
            self.connected = False
            if str(exc) != self.last_error:
                log.warning("MQTT connect to %s:%s failed: %s", self.cfg.host, self.cfg.port, exc)
            self.last_error = str(exc)
            return False

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:  # noqa: ANN001
        self.connected = not reason_code.is_failure
        if self.connected:
            log.info("MQTT connected to %s:%s", self.cfg.host, self.cfg.port)
        else:
            log.warning("MQTT connection refused: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:  # noqa: ANN001
        self.connected = False
        log.warning("MQTT disconnected: %s", reason_code)

    def _publish(self, msg: MqttMessage) -> None:
        if not self.enabled or self._client is None:
            return
        topic = msg.topic if msg.absolute else f"{self.cfg.prefix}/{msg.topic}".replace("//", "/")
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
