from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import MISSING, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
log = logging.getLogger("config")


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.getenv(name, default)

    # Substituted values stay strings: coercing here corrupts credentials
    # like "007" or all-digit passwords. Consumers cast explicitly.
    return _ENV_PATTERN.sub(repl, value)


def _cast(value: Any, field_default: Any) -> Any:
    """B-11: cast env-substituted strings to the expected primitive type."""
    if not isinstance(value, str):
        return value
    if field_default is MISSING or not isinstance(field_default, (bool, int, float)):
        return value
    if isinstance(field_default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(field_default, int):
        try:
            return int(value)
        except ValueError:
            return value
    if isinstance(field_default, float):
        try:
            return float(value)
        except ValueError:
            return value
    return value


@dataclass(slots=True)
class AppSection:
    name: str = "VISION STABLE"
    data_dir: str = "data"


@dataclass(slots=True)
class CameraSection:
    id: str = "hikvision_01"
    rtsp_url: str = ""
    backend: str = "ffmpeg"
    buffer_size: int = 1
    reconnect_delay_sec: float = 1.5
    read_fail_limit: int = 20
    ffmpeg_capture_options: str = "rtsp_transport;tcp|max_delay;500000|stimeout;5000000"


@dataclass(slots=True)
class VisionSection:
    model_path: str = "yolo11n-pose.pt"
    imgsz: int = 640
    conf: float = 0.40
    iou: float = 0.55
    classes: list[int] = field(default_factory=lambda: [0])
    inference_fps: float = 5.0
    min_box_area_ratio: float = 0.0015
    duplicate_foot_dist_px: float = 35.0
    duplicate_iou: float = 0.65


@dataclass(slots=True)
class TrackerSection:
    keep_sec: float = 3.5
    match_distance_px: float = 100.0
    iou_match: float = 0.10
    smoothing: float = 0.65
    min_hits: int = 1
    max_history: int = 80
    walking_px_s: float = 15.0
    still_px_s: float = 8.0
    fallen_window_sec: float = 2.0
    fallen_persist_sec: float = 10.0
    sleep_still_sec: float = 30.0


@dataclass(slots=True)
class MqttSection:
    enabled: bool = True
    host: str = ""
    port: int = 1883
    username: str = ""
    password: str = ""
    prefix: str = "frigate"
    client_id: str = "vision_stable_pose"
    queue_size: int = 500
    publish_tracks_hz: float = 5.0
    reconnect_delay_sec: float = 5.0
    discovery: bool = True              # publish Home Assistant MQTT discovery configs
    discovery_prefix: str = "homeassistant"
    person_slots: int = 4              # fixed HA person entities for the floorplan


@dataclass(slots=True)
class CalibrationSection:
    file: str = "data/calibration.json"

    room_width_m: float = 2.5
    room_depth_m: float = 2.5

    aim_px: int = 320
    aim_py: int = 240
    floor_points: list[list[float]] = field(default_factory=list)

    camera_height_m: float = 2.5
    camera_pitch_deg: float = 45.0
    camera_yaw_deg: float = 0.0
    camera_roll_deg: float = 0.0
    hfov_deg: float = 90.0
    vfov_deg: float = 55.0
    rotation_deg: float = 0.0
    lens_distortion_k1: float = 0.0
    lens_distortion_k2: float = 0.0


@dataclass(slots=True)
class ActivitySection:
    enabled: bool = False
    det_model_path: str = "yolo11n.pt"
    det_conf: float = 0.30
    det_fps: float = 2.0
    det_imgsz: int = 1280            # object-detector input size (small items need >640)
    assoc_margin_ratio: float = 0.6  # object-to-person margin as a fraction of bbox size
    clip_enabled: bool = False
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_fps: float = 0.5


@dataclass(slots=True)
class UISection:
    enabled: bool = True
    window_width: int = 1600
    window_height: int = 900
    panel_width: int = 380
    show_pose: bool = True
    show_tracks: bool = True
    show_calibration: bool = True


@dataclass(slots=True)
class WebSection:
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    stream_max_width: int = 1280   # MJPEG downscale cap for bandwidth
    stream_fps: float = 15.0
    open_browser: bool = True      # open the control UI in the default browser on start


@dataclass(slots=True)
class Settings:
    root_dir: Path
    config_path: Path
    app: AppSection = field(default_factory=AppSection)
    camera: CameraSection = field(default_factory=CameraSection)
    vision: VisionSection = field(default_factory=VisionSection)
    tracker: TrackerSection = field(default_factory=TrackerSection)
    mqtt: MqttSection = field(default_factory=MqttSection)
    calibration: CalibrationSection = field(default_factory=CalibrationSection)
    activity: ActivitySection = field(default_factory=ActivitySection)
    ui: UISection = field(default_factory=UISection)
    web: WebSection = field(default_factory=WebSection)


def _section(cls: type, data: dict[str, Any]) -> Any:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"Config section {cls.__name__} must be a mapping/dict, got {type(data).__name__}: {data!r}")
    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    # B-12: warn about unknown keys so typos are caught immediately
    unknown = [k for k in data if k not in fields]
    if unknown:
        log.warning("Unknown config keys in %s (ignored): %s", cls.__name__, unknown)
    # B-11: cast env-substituted strings to the declared primitive type
    return cls(**{k: _cast(v, fields[k].default) for k, v in data.items() if k in fields})


class ConfigManager:
    """Single source of runtime configuration."""

    def __init__(self, config_path: str | Path = "configs/config.yaml") -> None:
        self._lock = threading.RLock()
        self.root_dir = Path.cwd()
        self.config_path = Path(config_path)
        if not self.config_path.is_absolute():
            self.config_path = self.root_dir / self.config_path
        self.settings = self._load()

    def _load(self) -> Settings:
        _load_dotenv(self.root_dir / ".env")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8-sig")) or {}
        raw = _expand_env(raw)
        settings = Settings(
            root_dir=self.root_dir,
            config_path=self.config_path,
            app=_section(AppSection, raw.get("app", {})),
            camera=_section(CameraSection, raw.get("camera", {})),
            vision=_section(VisionSection, raw.get("vision", {})),
            tracker=_section(TrackerSection, raw.get("tracker", {})),
            mqtt=_section(MqttSection, raw.get("mqtt", {})),
            calibration=_section(CalibrationSection, raw.get("calibration", {})),
            activity=_section(ActivitySection, raw.get("activity", {})),
            ui=_section(UISection, raw.get("ui", {})),
            web=_section(WebSection, raw.get("web", {})),
        )
        if not str(settings.camera.rtsp_url).strip():
            raise ValueError(
                "camera.rtsp_url is empty. Set RTSP_URL in .env "
                "(see .env.example) or in configs/config.yaml."
            )
        return settings

    def reload(self) -> Settings:
        with self._lock:
            self.settings = self._load()
            return self.settings

    def get(self) -> Settings:
        with self._lock:
            return self.settings

    def resolve_path(self, path: str | Path) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        return self.root_dir / p
