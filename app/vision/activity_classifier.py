from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from app.config import ActivitySection
from app.types import TrackSnapshot

log = logging.getLogger("activity")

_OBJECT_TO_ACTIVITY: dict[str, str] = {
    "cell phone": "с телефоном",
    "book": "читает",
    "laptop": "у компьютера",
    "tv": "у компьютера",
    "mouse": "у компьютера",
    "keyboard": "у компьютера",
    "monitor": "у компьютера",
    "bowl": "ест",
    "fork": "ест",
    "spoon": "ест",
    "sandwich": "ест",
    "banana": "ест",
    "apple": "ест",
    "orange": "ест",
    "pizza": "ест",
    "donut": "ест",
    "cake": "ест",
    "hot dog": "ест",
    "cup": "пьёт",
    "bottle": "пьёт",
    "wine glass": "пьёт",
    "knife": "готовит",
    "oven": "готовит",
    "microwave": "готовит",
    "refrigerator": "готовит",
    "toaster": "готовит",
    "sink": "у раковины",
    "toilet": "в туалете",
    "toothbrush": "чистит зубы",
    "hair drier": "сушит волосы",
}

_ACTIVITY_PRIORITY: list[str] = [
    "в туалете",
    "чистит зубы",
    "сушит волосы",
    "готовит",
    "ест",
    "пьёт",
    "читает",
    "с телефоном",
    "у компьютера",
    "у раковины",
]

_CLIP_PROMPTS: dict[str, str] = {
    "в душе": "a person standing in a shower",
    "красится": "a person applying makeup in front of mirror",
    "у компьютера": "a person sitting at a desk with a computer monitor",
}


class ObjectDetector:
    def __init__(self, cfg: ActivitySection) -> None:
        self._cfg = cfg
        self._model = None
        self._last_run_ts: float = 0.0
        self._cached: list[dict] = []

    def load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
            self._model = YOLO(self._cfg.det_model_path)
            log.info("ObjectDetector loaded: %s", self._cfg.det_model_path)
        except Exception as exc:
            log.warning("ObjectDetector not available: %s", exc)

    def detect(self, image: np.ndarray, now: float) -> list[dict]:
        if self._model is None:
            return []
        min_interval = 1.0 / max(0.1, float(self._cfg.det_fps))
        if now - self._last_run_ts < min_interval:
            return self._cached
        self._last_run_ts = now
        try:
            results = self._model(image, conf=float(self._cfg.det_conf), verbose=False)
            detections: list[dict] = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = r.names[cls_id]
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2
                    cy = (xyxy[1] + xyxy[3]) / 2
                    detections.append({"label": label, "cx": cx, "cy": cy, "conf": float(box.conf[0])})
            self._cached = detections
        except Exception as exc:
            log.debug("ObjectDetector error: %s", exc)
        return self._cached


class ActivityRules:
    def classify(self, track: TrackSnapshot, objects: list[dict]) -> str:
        fx, fy = track.foot
        cx, cy = track.center
        nearby: list[str] = []
        for obj in objects:
            dist = ((obj["cx"] - cx) ** 2 + (obj["cy"] - fy) ** 2) ** 0.5
            if dist < 200:
                label = obj["label"]
                if label in _OBJECT_TO_ACTIVITY:
                    nearby.append(_OBJECT_TO_ACTIVITY[label])
        if not nearby:
            return ""
        for priority_act in _ACTIVITY_PRIORITY:
            if priority_act in nearby:
                return priority_act
        return nearby[0]


class ClipClassifier:
    def __init__(self, cfg: ActivitySection) -> None:
        self._cfg = cfg
        self._model = None
        self._processor = None
        self._last_run_ts: float = 0.0
        self._cached: str = ""

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-untyped]
            self._model = CLIPModel.from_pretrained(self._cfg.clip_model)
            self._processor = CLIPProcessor.from_pretrained(self._cfg.clip_model)
            log.info("ClipClassifier loaded: %s", self._cfg.clip_model)
        except Exception as exc:
            log.warning("ClipClassifier not available: %s", exc)

    def classify(self, image: np.ndarray, now: float) -> str:
        if self._model is None or self._processor is None:
            return ""
        min_interval = 1.0 / max(0.1, float(self._cfg.clip_fps))
        if now - self._last_run_ts < min_interval:
            return self._cached
        self._last_run_ts = now
        try:
            from PIL import Image as PILImage  # type: ignore[import-untyped]
            pil_img = PILImage.fromarray(image[..., ::-1])
            labels = list(_CLIP_PROMPTS.keys())
            prompts = list(_CLIP_PROMPTS.values())
            inputs = self._processor(text=prompts, images=pil_img, return_tensors="pt", padding=True)
            import torch  # type: ignore[import-untyped]
            with torch.no_grad():
                outputs = self._model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_idx = int(probs.argmax())
            best_prob = float(probs[best_idx])
            self._cached = labels[best_idx] if best_prob > 0.4 else ""
        except Exception as exc:
            log.debug("ClipClassifier error: %s", exc)
        return self._cached


class ActivityClassifier:
    def __init__(self, cfg: ActivitySection) -> None:
        self._cfg = cfg
        self._detector = ObjectDetector(cfg)
        self._rules = ActivityRules()
        self._clip: Optional[ClipClassifier] = ClipClassifier(cfg) if cfg.clip_enabled else None

    def start(self) -> None:
        if not self._cfg.enabled:
            return
        self._detector.load()
        if self._clip is not None:
            self._clip.load()

    def classify(self, image: np.ndarray, tracks: list[TrackSnapshot], now: float) -> dict[int, str]:
        if not self._cfg.enabled:
            return {}
        objects = self._detector.detect(image, now)
        clip_label = self._clip.classify(image, now) if self._clip is not None else ""
        result: dict[int, str] = {}
        for tr in tracks:
            activity = self._rules.classify(tr, objects)
            if not activity and clip_label:
                activity = clip_label
            result[tr.track_id] = activity
        return result
