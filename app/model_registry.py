"""Discover available YOLO model files for hot-swapping.

A file is classified as a *pose* model when "pose" appears in its name,
otherwise as an *object* model. Search dirs that don't exist are skipped.
"""
from __future__ import annotations

from pathlib import Path


def discover_models(dirs) -> dict[str, list[str]]:
    pose: set[str] = set()
    obj: set[str] = set()
    for d in dirs:
        path = Path(d)
        if not path.is_dir():
            continue
        for f in path.glob("*.pt"):
            target = pose if "pose" in f.name.lower() else obj
            target.add(str(f.resolve()))
    return {"pose": sorted(pose), "object": sorted(obj)}


def with_current(models: dict[str, list[str]], pose_current: str | None,
                 object_current: str | None) -> dict[str, list[str]]:
    """Ensure the currently-configured paths appear in the lists."""
    out = {"pose": list(models.get("pose", [])), "object": list(models.get("object", []))}
    for key, cur in (("pose", pose_current), ("object", object_current)):
        if cur:
            resolved = str(Path(cur).resolve()) if Path(cur).exists() else cur
            if resolved not in out[key]:
                out[key].append(resolved)
                out[key].sort()
    return out
