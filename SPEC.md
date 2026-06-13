# Spec: WB Vision Refactored — путь к стабильному MVP

Дата анализа: 2026-06-13. Анализ выполнен по коду без запуска приложения и без изменений исходников.

## Objective (Что делает продукт)

**Локальная CV-система присутствия людей для умного дома.**

- Читает RTSP-поток с IP-камеры (Hikvision), детектирует людей с позой через YOLO11n-pose.
- Трекает людей со стабильными ID (`StableTracker`: greedy matching по foot-distance + IoU, сглаживание).
- Через homography-калибровку пола (4 точки + размеры комнаты) переводит позицию ног в метры.
- Публикует presence/координаты/health в MQTT (префикс `frigate` — интеграция с Home Assistant).
- OpenCV-GUI для калибровки и мониторинга; headless-режим для работы 24/7.

Пользователь — владелец системы (single-tenant, одна камера, локальная сеть).

**Главный сценарий end-to-end:**

1. `install.ps1` → venv + зависимости; заполнить `.env` (RTSP, MQTT).
2. `run.ps1` → камера подключается, YOLO детектит людей.
3. В UI: 4 клика по углам пола + размеры комнаты в метрах → homography → `SAVE`.
4. `run_headless.ps1`: координаты в метрах, presence и health публикуются в MQTT → автоматизации Home Assistant.

**Сценарий сейчас сломан на шаге 4** — см. P0.1 ниже.

## Tech Stack

- Python 3.12, многопоточность (threading)
- ultralytics 8.4 (YOLO11n-pose), torch 2.12 (CPU-build)
- opencv-python 4.13 (RTSP-чтение через FFmpeg backend + GUI)
- paho-mqtt 2.1 (установлен; код написан под API 1.x — см. P0.1)
- PyYAML, numpy, psutil; pydantic в requirements, но **не используется**

## Commands

```powershell
.\scripts\install.ps1        # venv + pip install -r requirements.txt + .env из примера
.\scripts\run.ps1            # запуск с GUI
.\scripts\run_headless.ps1   # запуск без GUI
.\scripts\check.ps1          # python -m compileall app (единственная "проверка")
```

Тестов и линтера нет (см. P1.4).

## Project Structure

```
app/
  main.py                  → точка входа, сборка воркеров, graceful shutdown
  config.py                → ConfigManager: .env → ${VAR:default} в YAML → dataclass-секции
  types.py                 → FramePacket, Detection, TrackSnapshot, VisionPacket, GeoPoint
  core/latest_value.py     → LatestValue: thread-safe latest-only слот (drop-old)
  camera/rtsp_reader.py    → RtspReader: поток чтения RTSP, reconnect, FPS-статистика
  vision/detector.py       → YoloPoseDetector + suppress_duplicates + foot/state из позы
  vision/tracker.py        → StableTracker: greedy matching, сглаживание, история
  vision/calibration.py    → CalibrationManager: homography пола, pixel→метры
  vision/inference_worker.py → InferenceWorker: инференс-цикл + публикация в MQTT
  vision/overlay.py        → отрисовка треков/позы/калибровки/панели
  mqtt/mqtt_worker.py      → MqttWorker: очередь публикаций (drop-oldest)
  ui/ui_worker.py          → UIWorker: OpenCV-loop, кнопки, редактируемые поля
configs/config.yaml        → конфигурация (camera/vision/tracker/mqtt/calibration/ui)
data/calibration.json      → единственное персистентное состояние
scripts/*.ps1              → Windows-only установка и запуск
.env / .env.example        → секреты (RTSP/MQTT credentials)
backups/                   → ручные копии файлов (заменить на git — P0.3)
wheelhouse/                → офлайн-кэш wheel-пакетов
yolo11n-pose.pt            → веса модели в корне
```

Архитектура в целом **хорошая**: `LatestValue` правильно решает накопление лага RTSP; воркеры развязаны; shutdown через общий `stop_event`.

**Внешний интерфейс** — только MQTT-топики (HTTP API нет):

- `{prefix}/{camera_id}/status` (online/offline, retain)
- `{prefix}/{camera_id}/presence` (ON/OFF)
- `{prefix}/{camera_id}/health/{reader_fps,inference_fps,infer_ms,cpu,ram,people_count}`
- `{prefix}/{camera_id}/person/{id}/{json,state,foot_x,foot_y,confidence,x_m,y_m,distance_axis_m,inside_room,inside_calibration_zone}`

## Найденные проблемы

### P0 — блокируют MVP

1. **MQTT не работает с установленным paho-mqtt 2.1.0.** `app/mqtt/mqtt_worker.py:57` вызывает
   `mqtt.Client(client_id=...)` — в paho 2.x первым аргументом обязателен `CallbackAPIVersion`,
   конструктор кидает `ValueError`. Исключение ловится, печатается «MQTT disabled» и воркер молча
   выключается навсегда. Основной выход продукта мёртв.
   Фикс: `mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id=...)` или пин `paho-mqtt<2`.
2. **MQTT не переподключается.** Любая ошибка первого коннекта → `self.enabled = False` навсегда
   (`mqtt_worker.py:64`). Для headless-сервиса 24/7 нужен retry-loop, как у `RtspReader`.
3. **Нет git и нет `.gitignore`.** Версионирование — папка `backups/` с ручными копиями.
   В корне лежит `.env` с реальными кредами RTSP/MQTT: первый же `git init && git add .`
   закоммитит секреты. Инженерный и security-риск одновременно.

### P1 — нужны для стабильного MVP

4. **Тестов нет.** `StableTracker`, `suppress_duplicates`, `_expand_env`, `pixel_to_floor` —
   чистая логика, тестируемая без камеры.
5. **Тихие деградации.** Пустой `RTSP_URL` → `_expand_env` подставит пустую строку → бесконечный
   молчаливый reconnect. Нужна fail-fast валидация конфига на старте.
6. **`_expand_env` агрессивно приводит типы** (`app/config.py:42-48`): пароль `"123456"` станет
   `int`, `"007"` → `7`. Подстановку из env оставлять строкой, типизацию делать в секциях.
7. **Топики `person/{id}` не очищаются**: при смерти трека последние значения остаются в брокере
   как «живые». Нужен финальный `state=gone`/пустой payload при удалении трека.
8. **Per-track topic-fanout** — 10+ топиков на человека на тик; достаточно JSON-топика
   (уже есть) + presence/count.
9. **`.env` рассинхронизирован с конфигом**: в `.env` есть `CAMERA_ID`, но `config.yaml`
   хардкодит `camera.id: hikvision_01`. `pydantic` в requirements не используется.
10. **Деплой как сервиса отсутствует**: нет автозапуска (Task Scheduler/NSSM/Docker),
    нет лог-файлов (всё через `print`), нет ротации логов.
11. **README битый**: кириллица в mojibake (двойная перекодировка cp1251/utf-8).

### P2 — качество и полировка

12. **`state_by_pose` не использует позу** — только аспект bbox (`app/vision/detector.py:30`);
    keypoints дали бы точнее sitting/lying.
13. **Калибровка пишется на диск на каждый клик** (`add_floor_point` → `save()`);
    4 точки лучше сохранять одним save.
14. **Поля камеры (height/pitch/fov/distortion) редактируются в UI, но нигде не используются** —
    мёртвый UI, вводит в заблуждение.
15. **`RtspStats` читается из другого потока без синхронизации** — на CPython практически
    безопасно, но стоит зафиксировать осознанно.
16. **`min_hits: 1`** — каждый одиночный ложный детект сразу становится треком и улетает в MQTT;
    `min_hits: 2-3` сильно чистит ложняки.
17. **Текущий `data/calibration.json`** — тестовые значения (комната 0.4×0.4 м).

## Производительность

Узкое место — **CPU-инференс** (torch CPU-build): yolo11n-pose @ 640 ≈ 80–250 мс/кадр,
отсюда лимит `inference_fps: 4`. Осознанный и нормальный для MVP компромисс. Улучшения:

- **Декод RTSP** — скрытый расход: `RtspReader` декодирует все ~15–25 FPS камеры, используется 4.
  URL указывает на substream (`Channels/102`) — правильно; убедиться в низком разрешении/FPS.
- **Экспорт модели в ONNX/OpenVINO** (ultralytics умеет из коробки) — типично 2–3× на CPU.
- `imgsz: 480` вместо 640 — ещё ~40% при достаточной точности для комнаты.
- Убрать MQTT topic-fanout (P1.8) — меньше JSON/publish на тик.

Алгоритмически всё дёшево: трекер O(D×T) на единицах объектов, homography — один
`perspectiveTransform` на трек.

## Code Style

Существующий стиль (сохранять): `from __future__ import annotations`, type hints везде,
`@dataclass(slots=True)` для данных, потоки наследуют `threading.Thread` с `name=` и `daemon=True`,
остановка через общий `stop_event: threading.Event`.

```python
@dataclass(slots=True)
class FramePacket:
    frame_id: int
    ts: float
    image: np.ndarray
```

## Testing Strategy

Сейчас: только `compileall`. Цель для MVP:

- Framework: pytest, тесты в `tests/`.
- Unit (без камеры/GPU): `StableTracker.update` (матчинг, expire, smoothing),
  `suppress_duplicates`, `box_iou`, `_expand_env`/`_load_dotenv`,
  `CalibrationManager.pixel_to_floor` (синтетическая homography), `MqttWorker.enqueue` (drop-oldest).
- Интеграционный smoke: запуск `main` с фейковым источником кадров, headless, MQTT off.
- Запуск: `python -m pytest tests/` — добавить в `check.ps1`.

## Boundaries

- **Always:** запускать `check.ps1`/pytest перед коммитом; держать секреты только в `.env`;
  сохранять существующий стиль (dataclasses, type hints, воркеры с `stop_event`).
- **Ask first:** смена формата MQTT-топиков (ломает подписчиков HA); новые зависимости;
  смена модели/весов; изменение формата `calibration.json`.
- **Never:** коммитить `.env`, `.venv/`, `wheelhouse/`, `*.pt`; удалять `backups/` до первого
  git-коммита; менять поведение `LatestValue` (ядро анти-лаг механики).

## MVP Checklist

- [ ] Починить MQTT под paho 2.x (P0.1) — без этого продукта нет
- [ ] MQTT reconnect-loop вместо одноразового disable (P0.2)
- [ ] `git init` + `.gitignore` (`.env`, `.venv/`, `wheelhouse/`, `__pycache__/`, `*.pt`, `backups/`), первый коммит (P0.3)
- [ ] Fail-fast валидация: пустой `RTSP_URL` → понятная ошибка при старте (P1.5)
- [ ] `state=gone` при удалении трека + упростить топики (P1.7–8)
- [ ] Тесты: tracker, suppress_duplicates, `_expand_env`, pixel_to_floor (P1.4)
- [ ] Логирование через `logging` в файл вместо `print` (P1.10)
- [ ] Автозапуск headless как Windows-сервиса/задачи (P1.10)
- [ ] Починить README-кодировку, синхронизировать `.env.example` ↔ `config.yaml` (P1.9, P1.11)
- [ ] `min_hits: 2`, перекалибровать реальную комнату (P2.16–17)

## Success Criteria

- `run_headless.ps1` работает ≥24 ч: переживает рестарт MQTT-брокера и обрывы RTSP без ручного вмешательства.
- `mosquitto_sub -t 'frigate/hikvision_01/#'` показывает presence/health/person-топики в реальном времени.
- Уход человека из кадра приводит к `presence=OFF` и финальному состоянию его person-топика (нет «вечно живых» треков в брокере).
- Пустой/неверный `RTSP_URL` → процесс завершается с понятной ошибкой, а не молча крутится.
- `python -m pytest tests/` зелёный; тесты покрывают tracker, dedup, config-expansion, калибровку.
- Проект в git, секреты не в истории.

## Первый шаг (самый безопасный)

**`git init` + `.gitignore` + первый коммит** (исключив `.env`, `.venv`, `wheelhouse`, `*.pt`).
Нулевой риск: код не меняется, появляется страховка для всех правок и закрывается риск утечки кредов.
Сразу после — однострочный фикс конструктора MQTT-клиента (P0.1), проверка через
`run_headless.ps1` + `mosquitto_sub`.

## Open Questions

- Подтвердить целевого потребителя MQTT (Home Assistant через префикс `frigate`?) — влияет на формат топиков (P1.8) и discovery.
- Нужен ли GUI в проде, или он только для калибровки? (влияет на приоритет UI-полировки)
- Одна камера навсегда или планируется несколько? (влияет на конфиг и топики)

---

# REDESIGN (2026-06-13): trapezoid calibration + browser control UI

Подтверждено с пользователем:
- **UI**: новый браузерный веб-UI (FastAPI + HTML). Видео через MJPEG,
  калибровка кликами в браузере, выпадающие списки моделей, слайдеры FPS/трекинга.
- **Калибровка**: трапеция на полу. P1 = (0,0). Оператор кликает 4 угла пола в
  кадре и вводит длины сторон (AB, BC, CD, DA) и внутренние углы; алгоритм строит
  4 мировые координаты и гомографию. Точки вне четырёхугольника обрабатывает сама
  планарная гомография (точная экстраполяция для плоскости).
- **Модели**: горячая смена pose (yolo11n/s/m-pose) и объектов/активности
  (yolo11n/s/m). Файлы скачиваются заранее; сервис берёт из папки `models/`.
- **Порядок**: поэтапно.

## Фазы
1. **Математика калибровки (headless + тесты).**
   - Чистый builder: длины сторон + внутренние углы → 4 мировые точки (P1=0,0)
     + ошибка замыкания (P4→P1 vs DA) для валидации.
   - `CalibrationData`: `quad_px` (4 угла в пикселях), `trap_edges_m`
     [ab,bc,cd,da], `trap_angles_deg` [a1,a2,a3,a4].
   - Гомография предпочитает трапецию, когда задана; иначе fallback на старый
     прямоугольник floor-points (текущий OpenCV-UI продолжает работать).
   - `pixel_to_floor` снаружи без изменений (x_m,y_m,dist_floor,dist_cam,zone).
2. **Веб-UI (FastAPI)** — MJPEG, клики AIM/4 угла, формы сторон/углов, зоны, статус.
3. **Рантайм-тюнинг** — inference FPS + параметры трекера вживую.
4. **Горячая смена моделей** — дропдауны pose/объекты из `models/`, reload без рестарта.

## Success criteria — Фаза 1
По 4 углам в кадре + длинам сторон + углам `pixel_to_floor` возвращает корректные
(x_m,y_m) с P1 в начале координат; прямоугольник — частный случай; считается ошибка
замыкания; весь тест-сьют зелёный.

## Boundaries (redesign)
- Always: полный pytest перед коммитом; обратная совместимость; один логический
  юнит на коммит; ветка `fix/positioning-mvp`.
- Ask first: отключение OpenCV-UI по умолчанию; тяжёлые зависимости.
- Never: коммитить секреты/.env, RTSP/MQTT креды; логировать креды.
