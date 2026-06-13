# WB Vision

Local computer-vision presence system. Reads an RTSP camera, detects people with
YOLO11 pose, tracks them with stable IDs, maps foot position to floor metres via
homography calibration, classifies posture/motion/activity, and publishes
everything to MQTT with **Home Assistant auto-discovery**. Calibration and live
tuning are done from a browser UI.

Architecture (independent worker threads): `RtspReader` → `LatestValue` (drop-old
frame buffer) → `InferenceWorker` (YOLO + tracking + geo + MQTT) → `MqttWorker`;
plus a FastAPI web control UI. See `CHANGELOG.md` for the feature list.

## Quick start (one command)

**Windows (PowerShell):**

```powershell
irm https://raw.githubusercontent.com/Yabukaev/wb_vision_refactored/main/scripts/bootstrap.ps1 | iex
```

**Linux / macOS:**

```bash
curl -fsSL https://raw.githubusercontent.com/Yabukaev/wb_vision_refactored/main/scripts/bootstrap.sh | bash
```

The bootstrap clones the repo (if needed), creates a virtualenv, **detects an
NVIDIA GPU** and installs CUDA or CPU PyTorch accordingly, installs the app,
creates `.env` from the template, and runs a smoke test.

Already have the repo checked out? Just run the script from inside it:
`./scripts/bootstrap.ps1` or `bash scripts/bootstrap.sh`.

## Configure

Edit `.env` (created from `.env.example`):

| Variable | Meaning |
|---|---|
| `RTSP_URL` | Camera stream, e.g. `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101` |
| `CAMERA_ID` | Camera id used in MQTT topics |
| `MQTT_HOST` / `MQTT_PORT` | Broker address |
| `MQTT_USER` / `MQTT_PASSWORD` | Broker credentials |
| `MQTT_PREFIX` | MQTT topic prefix (default `frigate`) |
| `MODEL_PATH` | Pose model file (default `yolo11n-pose.pt`) |

Runtime options (FPS, tracker, web port, person slots, activity) live in
`configs/config.yaml` and most are also tunable live from the web UI.

## Run

```powershell
.\scripts\serve.ps1     # Windows: start in background (writes .service.pid)
.\scripts\stop.ps1      # stop
```

```bash
# Linux/macOS
. .venv/bin/activate && python -m app.main --config configs/config.yaml
```

Then open the control UI at **http://localhost:8000** (opens automatically on
Windows): live video, click-to-calibrate the floor trapezoid, draw zones, tune
FPS/tracker, and hot-swap pose/object models.

## Smoke test

```bash
python scripts/smoke_test.py     # imports + config parse, no camera needed
python -m pytest -q              # full unit test suite
```

## CPU / GPU

The bootstrap picks PyTorch automatically:

- `nvidia-smi` present and working → installs CUDA wheels (`requirements-gpu.txt`).
- No GPU → CPU wheels (`requirements-cpu.txt`).
- GPU install fails → automatic fallback to CPU. If CUDA is unavailable at
  runtime, PyTorch silently runs on CPU.

`requirements.txt` holds the framework-agnostic deps; PyTorch is installed first
from the CPU/GPU file so ultralytics reuses it.

## Models

Weights are **not** committed (`*.pt` is git-ignored). `yolo11n-pose.pt` and
`yolo11n.pt` download on first use. For hot-swap, drop extra weights into
`models/` (e.g. `yolo11s/m-pose.pt`, `yolo11s/m.pt`) — they appear in the web
dropdowns. To pre-fetch:

```bash
python -c "from ultralytics.utils.downloads import attempt_download_asset as g; [g('models/'+n) for n in ['yolo11s-pose.pt','yolo11m-pose.pt','yolo11s.pt','yolo11m.pt']]"
```

## Home Assistant

With MQTT configured, the service publishes discovery configs so a device
**“WB Vision <camera_id>”** appears automatically with: presence, people count,
FPS/CPU/RAM, and `Person 1..N` entities whose attributes carry pose, motion,
activity, zone and `x_norm/y_norm` (0..1) for a floorplan card. Zone polygons
(pixels + metres + normalised) are published retained at `<prefix>/<cam>/zones`.

## Known limitations

- Single camera; MQTT prefix targets Home Assistant.
- Distance model assumes the camera is roughly above the AIM point (angled wall
  mounts are approximate).
- Activity classification is limited to the object model's classes (COCO by
  default) near a person; small items need a larger `Class imgsz` / model.
- CPU inference is the bottleneck; `inference_fps` is capped accordingly.

## Project layout

```text
app/            core service (camera, vision, mqtt, web, config)
configs/        config.yaml
data/           calibration.json (runtime state)
scripts/        bootstrap / serve / stop / smoke_test
tests/          pytest suite
```
