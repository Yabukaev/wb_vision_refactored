"""Persist runtime overrides (live tuning + selected models) to a JSON file.

Config (config.yaml/.env) stays the immutable baseline; this file records what
the operator changes from the web UI so it survives a restart.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class RuntimeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8-sig"))
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")

    def tuning(self) -> dict:
        with self._lock:
            return dict(self._data.get("tuning", {}))

    def models(self) -> dict:
        with self._lock:
            return dict(self._data.get("models", {}))

    def set_tuning(self, key: str, value) -> None:
        with self._lock:
            self._data.setdefault("tuning", {})[key] = value
            self._save()

    def set_model(self, kind: str, path: str) -> None:
        with self._lock:
            self._data.setdefault("models", {})[kind] = path
            self._save()
