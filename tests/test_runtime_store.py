"""Runtime overrides persistence (tuning + selected models)."""
from __future__ import annotations

from app.runtime_store import RuntimeStore


def test_persists_tuning_and_models(tmp_path):
    p = tmp_path / "runtime.json"
    s = RuntimeStore(p)
    s.set_tuning("inference_fps", 8.0)
    s.set_model("pose", "models/yolo11s-pose.pt")
    assert p.exists()

    reloaded = RuntimeStore(p)
    assert reloaded.tuning()["inference_fps"] == 8.0
    assert reloaded.models()["pose"] == "models/yolo11s-pose.pt"


def test_empty_when_missing(tmp_path):
    s = RuntimeStore(tmp_path / "nope.json")
    assert s.tuning() == {} and s.models() == {}


def test_corrupt_file_is_ignored(tmp_path):
    p = tmp_path / "runtime.json"
    p.write_text("{ not json", encoding="utf-8")
    s = RuntimeStore(p)
    assert s.tuning() == {}
