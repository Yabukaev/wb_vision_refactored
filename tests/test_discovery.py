"""Home Assistant MQTT discovery config builder."""
from __future__ import annotations

from app.mqtt.discovery import build_discovery_configs


def test_builds_camera_and_person_entities():
    cfgs = build_discovery_configs("cam1", "frigate", "homeassistant", 2)
    topics = [t for t, _ in cfgs]
    assert "homeassistant/binary_sensor/wbvision_cam1/presence/config" in topics
    assert any(t.endswith("person_1/config") for t in topics)
    assert any(t.endswith("person_2/config") for t in topics)


def test_presence_payload_is_ha_ready():
    cfgs = build_discovery_configs("cam1", "frigate", "homeassistant", 1)
    pres = next(p for t, p in cfgs if t.endswith("presence/config"))
    assert pres["state_topic"] == "frigate/cam1/presence"
    assert pres["device_class"] == "occupancy"
    assert pres["availability_topic"] == "frigate/cam1/status"
    assert pres["device"]["identifiers"] == ["wbvision_cam1"]
    assert pres["unique_id"] == "wbvision_cam1_presence"


def test_person_slot_uses_attributes_topic():
    cfgs = build_discovery_configs("cam1", "frigate", "homeassistant", 1)
    p1 = next(p for t, p in cfgs if t.endswith("person_1/config"))
    assert p1["json_attributes_topic"] == "frigate/cam1/person_slot/1/json"
    assert p1["state_topic"] == "frigate/cam1/person_slot/1/state"
    assert p1["icon"] == "mdi:human"
