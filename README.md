# WB Vision Refactored

Новый вариант проекта разделён на независимые воркеры:

- `RtspReader` — отдельный поток чтения RTSP, постоянный reconnect, буфер камеры = 1.
- `LatestValue` — latest-frame хранилище: кадры заменяются, очередь не растёт, задержка не накапливается.
- `InferenceWorker` — отдельный поток YOLO/pose inference и трекинга.
- `UIWorker` — отдельный UI-loop OpenCV: видео, оверлей, кнопки калибровки.
- `MqttWorker` — отдельный MQTT-поток и очередь публикаций, чтобы сеть не блокировала inference.
- `ConfigManager` — единая точка загрузки `.env`, YAML-конфига и JSON-калибровки.
- `CalibrationManager` — homography-калибровка пола, сохранение 4 точек, координаты в метрах.
- `StableTracker` — стабильные ID с greedy matching по foot-distance + IoU, сглаживание, история.

## Быстрый старт на Windows PowerShell

```powershell
cd путь\к\wb_vision_refactored
copy .env.example .env
notepad .env
.\scripts\install.ps1
.\scripts\run.ps1
```

Открыть в VS Code:

```powershell
.\scripts\open_vscode.ps1
```

Запуск без GUI:

```powershell
.\scripts\run_headless.ps1
```

## Настройка

1. В `.env` укажите RTSP и MQTT credentials.
2. В `configs/config.yaml` настройте FPS inference, модель, размеры окна, MQTT prefix.
3. Для pose лучше использовать `yolo11n-pose.pt`. Если указать detection-only модель, проект будет работать по bbox, но без keypoints.

## Калибровка в UI

- `CALIBRATE AIM` — кликните точку направления/центра.
- `CALIBRATE 4 FLOOR POINTS` — кликните 4 точки пола по часовой стрелке:
  1. ближний левый угол,
  2. ближний правый,
  3. дальний правый,
  4. дальний левый.
- `Room W` и `Room D` — реальные размеры зоны в метрах.
- `SAVE CONFIG` — сохранить `data/calibration.json`.

Горячие клавиши:

- `a` — режим AIM.
- `f` — режим 4 точек пола.
- `Esc` — выход.

## Структура

```text
app/
  main.py
  config.py
  types.py
  core/latest_value.py
  camera/rtsp_reader.py
  vision/calibration.py
  vision/detector.py
  vision/tracker.py
  vision/inference_worker.py
  vision/overlay.py
  mqtt/mqtt_worker.py
  ui/ui_worker.py
configs/config.yaml
data/calibration.json
scripts/*.ps1
.vscode/tasks.json
.vscode/launch.json
```

