"""Home Assistant MQTT discovery config builder.

Publishes retained `<discovery_prefix>/<component>/<node>/<object>/config`
messages so HA auto-creates entities (grouped under one device) for camera
status, presence, people count, health metrics, and a fixed set of person
slots. Person slots carry x_norm/y_norm in their attributes so a picture-
elements / floorplan card can place the little person icons by coordinate.
"""
from __future__ import annotations


def build_discovery_configs(
    camera_id: str,
    prefix: str,
    discovery_prefix: str = "homeassistant",
    person_slots: int = 4,
) -> list[tuple[str, dict]]:
    base = f"{prefix}/{camera_id}"
    node = f"wbvision_{camera_id}"
    device = {
        "identifiers": [node],
        "name": f"WB Vision {camera_id}",
        "manufacturer": "WB Vision",
        "model": "Beta 0.1",
    }
    avail = {
        "availability_topic": f"{base}/status",
        "payload_available": "online",
        "payload_not_available": "offline",
    }

    def cfg(component: str, obj: str, extra: dict) -> tuple[str, dict]:
        payload = {
            "name": extra.pop("name", obj.replace("_", " ").title()),
            "unique_id": f"{node}_{obj}",
            "device": device,
            **avail,
            **extra,
        }
        return f"{discovery_prefix}/{component}/{node}/{obj}/config", payload

    out: list[tuple[str, dict]] = []
    out.append(cfg("binary_sensor", "presence", {
        "name": "Presence", "state_topic": f"{base}/presence",
        "payload_on": "ON", "payload_off": "OFF",
        "device_class": "occupancy", "icon": "mdi:motion-sensor",
    }))
    out.append(cfg("sensor", "people_count", {
        "name": "People", "state_topic": f"{base}/health/people_count",
        "icon": "mdi:account-group", "state_class": "measurement",
    }))
    for obj, name, unit, icon in (
        ("inference_fps", "Inference FPS", "fps", "mdi:speedometer"),
        ("reader_fps", "Reader FPS", "fps", "mdi:speedometer"),
        ("infer_ms", "Infer time", "ms", "mdi:timer-outline"),
        ("cpu", "CPU", "%", "mdi:cpu-64-bit"),
        ("ram", "RAM", "%", "mdi:memory"),
    ):
        out.append(cfg("sensor", obj, {
            "name": name, "state_topic": f"{base}/health/{obj}",
            "unit_of_measurement": unit, "icon": icon,
            "state_class": "measurement", "entity_category": "diagnostic",
        }))

    for k in range(1, int(person_slots) + 1):
        out.append(cfg("sensor", f"person_{k}", {
            "name": f"Person {k}",
            "state_topic": f"{base}/person_slot/{k}/state",
            "json_attributes_topic": f"{base}/person_slot/{k}/json",
            "icon": "mdi:human",
        }))
    return out
