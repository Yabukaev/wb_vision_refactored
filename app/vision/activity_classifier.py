from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import numpy as np

from app.config import ActivitySection
from app.types import TrackSnapshot

log = logging.getLogger("activity")

_OBJECT_TO_ACTIVITY: dict[str, str] = {
    "cell phone": "on phone",
    "book": "reading",
    "laptop": "at computer",
    "tv": "at computer",
    "mouse": "at computer",
    "keyboard": "at computer",
    "monitor": "at computer",
    "bowl": "eating",
    "fork": "eating",
    "spoon": "eating",
    "sandwich": "eating",
    "banana": "eating",
    "apple": "eating",
    "orange": "eating",
    "pizza": "eating",
    "donut": "eating",
    "cake": "eating",
    "hot dog": "eating",
    "cup": "drinking",
    "bottle": "drinking",
    "wine glass": "drinking",
    "knife": "cooking",
    "oven": "cooking",
    "microwave": "cooking",
    "refrigerator": "cooking",
    "toaster": "cooking",
    "sink": "at sink",
    "toilet": "in bathroom",
    "toothbrush": "brushing teeth",
    "hair drier": "drying hair",
}

_ACTIVITY_PRIORITY: list[str] = [
    "in bathroom",
    "brushing teeth",
    "drying hair",
    "cooking",
    "eating",
    "drinking",
    "reading",
    "on phone",
    "at computer",
    "at sink",
]

_CLIP_PROMPTS: dict[str, str] = {
    "showering": "a person standing in a shower",
    "applying makeup": "a person applying makeup in front of mirror",
    "at computer": "a person sitting at a desk with a computer monitor",
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
            results = self._model(
                image, conf=float(self._cfg.det_conf),
                imgsz=int(self._cfg.det_imgsz), verbose=False,
            )
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
    def classify(self, track: TrackSnapshot, objects: list[dict], margin_ratio: float = 0.6) -> str:
        # Associate objects with a person by their bounding box plus a margin
        # scaled to the person's size — resolution-independent, unlike a fixed
        # pixel radius.
        bx1, by1, bx2, by2 = track.box
        bw = max(1.0, float(bx2 - bx1))
        bh = max(1.0, float(by2 - by1))
        margin = float(margin_ratio) * max(bw, bh)
        x1, y1, x2, y2 = bx1 - margin, by1 - margin, bx2 + margin, by2 + margin

        nearby: list[str] = []
        for obj in objects:
            ocx, ocy = obj["cx"], obj["cy"]
            if x1 <= ocx <= x2 and y1 <= ocy <= y2:
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
    """Async CLIP classifier — runs in a daemon thread so it never blocks the inference loop."""

    def __init__(self, cfg: ActivitySection) -> None:
        self._cfg = cfg
        self._model = None
        self._processor = None
        self._last_submit_ts: float = 0.0
        self._cached: str = ""
        self._cache_lock = threading.Lock()
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor  # type: ignore[import-untyped]
            self._model = CLIPModel.from_pretrained(self._cfg.clip_model)
            self._processor = CLIPProcessor.from_pretrained(self._cfg.clip_model)
            log.info("ClipClassifier loaded: %s", self._cfg.clip_model)
            self._thread = threading.Thread(target=self._worker, name="clip-worker", daemon=True)
            self._thread.start()
        except Exception as exc:
            log.warning("ClipClassifier not available: %s", exc)

    def _worker(self) -> None:
        while True:
            try:
                image = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            result = self._run(image)
            with self._cache_lock:
                self._cached = result

    def _run(self, image: np.ndarray) -> str:
        if self._model is None or self._processor is None:
            return ""
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
            return labels[best_idx] if best_prob > 0.4 else ""
        except Exception as exc:
            log.debug("ClipClassifier error: %s", exc)
            return ""

    def classify(self, image: np.ndarray, now: float) -> str:
        """Submit frame asynchronously; return last cached result immediately."""
        if self._model is None:
            return ""
        min_interval = 1.0 / max(0.1, float(self._cfg.clip_fps))
        if now - self._last_submit_ts >= min_interval:
            self._last_submit_ts = now
            img_copy = image.copy()
            # Drop stale queued frame; always keep latest
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(img_copy)
            except queue.Full:
                pass
        with self._cache_lock:
            return self._cached


class ActivityClassifier:
    """Facade: object-detection rules + optional async CLIP, with runtime enable/disable."""

    def __init__(self, cfg: ActivitySection) -> None:
        self._cfg = cfg
        self._enabled: bool = cfg.enabled
        self._load_needed: bool = False   # set True when toggled on before models load
        self._detector = ObjectDetector(cfg)
        self._rules = ActivityRules()
        self._clip: Optional[ClipClassifier] = ClipClassifier(cfg) if cfg.clip_enabled else None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Called by InferenceWorker at startup. Loads models if enabled at launch."""
        if not self._enabled:
            return
        self._load_models()

    def _load_models(self) -> None:
        self._detector.load()
        if self._clip is not None:
            self._clip.load()
        self._load_needed = False

    # ── runtime toggle (UI thread calls this) ─────────────────────────────────

    def toggle_enabled(self) -> bool:
        self._enabled = not self._enabled
        if self._enabled and self._detector._model is None:
            # Models haven't loaded yet — load on next classify() call
            # (happens in inference thread, safe to block briefly there)
            self._load_needed = True
        log.info("Activity classifier %s", "enabled" if self._enabled else "disabled")
        return self._enabled

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def object_model_path(self) -> str:
        return self._cfg.det_model_path

    def set_object_model(self, path: str) -> None:
        """Swap the object-detection model (called from the inference thread)."""
        self._cfg.det_model_path = str(path)
        self._detector = ObjectDetector(self._cfg)
        if self._enabled:
            self._detector.load()
        else:
            self._load_needed = True
        log.info("object model -> %s", path)

    # ── inference ─────────────────────────────────────────────────────────────

    def classify(self, image: np.ndarray, tracks: list[TrackSnapshot], now: float) -> dict[int, str]:
        if not self._enabled:
            return {}
        if self._load_needed:
            self._load_models()
        objects = self._detector.detect(image, now)
        clip_label = self._clip.classify(image, now) if self._clip is not None else ""
        result: dict[int, str] = {}
        for tr in tracks:
            activity = self._rules.classify(tr, objects, self._cfg.assoc_margin_ratio)
            if not activity and clip_label:
                activity = clip_label
            result[tr.track_id] = activity
        return result
