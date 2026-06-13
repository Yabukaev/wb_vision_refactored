"""Phase 4: model discovery for hot-swap."""
from __future__ import annotations

from app.model_registry import discover_models, with_current


def test_discover_classifies_pose_vs_object(tmp_path):
    (tmp_path / "yolo11n-pose.pt").write_bytes(b"x")
    (tmp_path / "yolo11s-pose.pt").write_bytes(b"x")
    (tmp_path / "yolo11n.pt").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    models = discover_models([tmp_path])
    assert len(models["pose"]) == 2
    assert len(models["object"]) == 1
    assert all(p.endswith(".pt") for p in models["pose"])


def test_discover_skips_missing_dir(tmp_path):
    models = discover_models([tmp_path / "nope"])
    assert models == {"pose": [], "object": []}


def test_with_current_adds_configured_paths():
    out = with_current({"pose": [], "object": []}, "custom-pose.pt", "custom.pt")
    assert "custom-pose.pt" in out["pose"]
    assert "custom.pt" in out["object"]
