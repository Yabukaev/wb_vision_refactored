# Changelog

## Beta 0.1 — 2026-06-13

First beta. Local CV presence system: RTSP → YOLO11 pose detection → stable
tracking → floor calibration → distance/zone computation → MQTT, with a browser
control UI.

### Highlights
- **Browser control UI (FastAPI, port 8000)** — live MJPEG video with overlays,
  click-to-calibrate, zone editor, status, runtime tuning and model switching.
  Opens automatically on start.
- **Trapezoid floor calibration** — click 4 floor corners, enter edge lengths
  (AB/BC/CD/DA) and interior angles; P1 is the origin. Homography maps pixels to
  metres and extrapolates outside the quad; a closure error flags bad measures.
- **Distance from camera** — derived from the floor distance to AIM and the
  camera elevation (measured laser distance to AIM, or estimated height).
- **Runtime tuning (no restart)** — inference FPS, detect conf/IoU, and tracker
  params (match distance, IoU, smoothing, min hits, keep sec, walking/still
  thresholds) editable live from the web UI.
- **Model hot-swap** — pick pose (yolo11 n/s/m-pose) and object (yolo11 n/s/m)
  models from dropdowns; applied on the next frame without restarting.
- **Zones** — draw polygon zones in the browser; track membership published per
  person.
- **MQTT** — presence, health, and per-person state/geo topics with reconnect
  and clean `gone` on track loss.

### Run
- `scripts\serve.ps1` — start in the background (writes `.service.pid`, opens
  the web UI at http://localhost:8000).
- `scripts\stop.ps1` — stop the background service.

### Notes
- OpenCV dev window is off by default (`ui.enabled: false`); control via the web.
- Place extra weights in `models/` to expand the hot-swap dropdowns.
- Tests: `python -m pytest -q` (99 passing).

### Known limitations
- Camera assumed roughly above AIM for the distance model (angled wall mounts
  are approximate).
- Single camera; MQTT topic prefix targets Home Assistant (`frigate`).
- `imgsz` is fixed; high-resolution streams are downscaled for inference.
