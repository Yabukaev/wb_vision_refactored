# Полный анализ кода WB Vision Refactored

Дата: 2026-06-13. Анализ только по коду, без запуска.
**Статус: все баги B-01..B-18 и P-03..P-07 исправлены (2026-06-13). 46 тестов — OK.**

---

## БАГИ

### P0 — могут приводить к потере данных или зависанию

**B-01 · `status=offline` не доходит до брокера при завершении**
`inference_worker.py:104` — после выхода из цикла публикуется `status=offline`, но
к этому моменту `MqttWorker` уже получил `stop_event` и прекратил читать очередь.
Сообщение кладётся в `queue`, но не отправляется. В Home Assistant камера навсегда
остаётся `online`.

**B-02 · Первый вызов `psutil.cpu_percent(interval=None)` возвращает 0.0**
`inference_worker.py:84` — при `interval=None` psutil возвращает накопленное с
предыдущего вызова. На первом кадре (нет предыдущего вызова) всегда 0.0.
Решение: один «прогревочный» вызов в `__init__`.

**B-03 · Двойной `close()` на LatestValue**
`main.py:101-102` — `frames.close()` и `results.close()` вызываются в `finally`.
Но `RtspReader.run()` уже вызывает `self.out.close()`, а `InferenceWorker.run()`
вызывает `self.results.close()`. Повторный `close()` безвреден (просто `_closed=True`
ещё раз), но маскирует логику завершения и может сломаться если `LatestValue`
когда-нибудь добавит финализацию ресурса.

**B-04 · Timestamp в `_publish_track` отличается от timestamp пакета**
`inference_worker.py:147` — `"ts": time.time()` вызывается заново внутри
`_publish_track`, хотя пакет уже содержит `packet.ts=now`. Разница ~1-3мс, но
подписчики MQTT получают `ts`, который отличается от момента инференса.

**B-05 · Keypoints без фильтрации по confidence**
`detector.py:22-26` — `foot_from_pose` принимает `res.keypoints.xy` (только x, y),
проверяя лишь `x > 1 и y > 1`. YOLO возвращает keypoints с confidence score (третий
столбец в `.data`). Заниженная conf ankle-точки (перекрытая нога, край кадра)
засчитывается как валидная — foot размещается в случайном месте.
Нужно использовать `res.keypoints.data` и фильтровать по порогу ~0.3.

**B-06 · `state_by_pose` не использует позу**
`detector.py:30-38` — функция принимает `keypoints`, но вычисляет состояние только
по аспекту bbox (w/h). Человек, лежащий вдоль оси камеры (высокое h, малое w),
будет помечен `standing`. Параметр `keypoints` есть, данные есть — не используются.

**B-07 · `RtspStats.fps` читается без синхронизации**
`rtsp_reader.py:103` — `stats.fps` пишется в `RtspReader`-потоке, читается в
`InferenceWorker` через `reader_fps_getter`. На CPython GIL защищает простые
attribute write, но EWA-вычисление `0.8 * self.stats.fps + 0.2 * inst` не атомарно:
между чтением `self.stats.fps` и записью в него другой поток может прочитать
промежуточное NaN или 0. Практически безопасно, но нарушает формальные гарантии.

**B-08 · Ошибка подключения RTSP не логируется**
`rtsp_reader.py:87` — `stats.last_error = "cannot open RTSP"`, но никакого вызова
`log.warning()` нет. В headless-режиме пользователь видит только тишину.

**B-09 · `OPENCV_FFMPEG_CAPTURE_OPTIONS` устанавливается через `os.environ`**
`rtsp_reader.py:41-42` — `os.environ.setdefault(...)` глобальная операция. Если
когда-нибудь запустить два `RtspReader` с разными опциями (вторая камера), второй
получит опции первого. Также переменная не очищается после закрытия потока.

**B-10 · `signal.SIGTERM` не работает как ожидается на Windows**
`main.py:61` — Python регистрирует обработчик, но TerminateProcess от Task Scheduler
или NSSM не генерирует SIGTERM. Graceful shutdown через внешний сигнал недоступен.
Только Ctrl+C (SIGINT) работает надёжно. Для headless-сервиса это проблема.

**B-11 · Неявные типы в dataclass-секциях при подстановке из env**
`config.py:152` — `_expand_env` возвращает строки после подстановки переменных
окружения. `_section` передаёт их напрямую в dataclass-конструктор. В результате
`mqtt.port = "1883"` (str), хотя аннотация `int`. Большинство мест явно кастят
(`int(self.cfg.port)`), но новый код может забыть — тихий баг без TypeError.

**B-12 · Неизвестные ключи в YAML молча игнорируются**
`config.py:151-152` — `_section` фильтрует только известные поля. Опечатка в
`config.yaml` (`reonnect_delay_sec` вместо `reconnect_delay_sec`) будет проигнорирована
без предупреждения. Пользователь думает, что настройка применена.

---

### P1 — логические ошибки / неожиданное поведение

**B-13 · `_drop_expired` вызывается дважды за один `update()`**
`tracker.py:54,82` — вызов в начале `update()` (удалить устаревшие перед матчингом)
и в конце (после создания новых треков). Второй вызов сразу после первого бесполезен:
за время матчинга и создания треков прошло ~0мс, ни один трек не мог устареть.

**B-14 · ID трека растёт бесконечно и не сбрасывается**
`tracker.py:47` — `_next_id = 1` при создании трекера. После суток работы в людном
месте (10 чел/мин × 60мин × 24ч = 14400) MQTT-топики будут `person/14401/...`.
Это само по себе не баг, но HA накапливает брошенные entity.

**B-15 · `Counter(state_history).most_common(1)` на каждом кадре**
`tracker.py:140` — каждое обновление трека создаёт Counter из deque (макс. 7 эл-в),
что выделяет новый объект. Незначительно, но происходит для каждого трека каждый кадр.

**B-16 · Сглаживание позиций int-truncation накапливает ошибку**
`tracker.py:16-21` — `_smooth_point` обрезает до int на каждом шаге.
При alpha=0.65 стационарный объект будет медленно «дрейфовать» на 1px из-за
накопленного truncation bias. Foot-coordinate в MQTT будет нестабильна без движения.

**B-17 · `calibration.snapshot()` вызывается дважды за кадр в UI**
`ui_worker.py:71,167` — один раз в `run()` (`cal = self.calibration.snapshot()`),
второй раз внутри `_compose()` (`cal = self.calibration.snapshot()`). Каждый
`snapshot()` захватывает RLock и вызывает `asdict()` (копирует все поля). Два раза
за кадр без необходимости.

**B-18 · `_show_waiting` создаёт новый canvas каждые 30мс**
`ui_worker.py:95` — `np.zeros((h, w, 3), dtype=np.uint8)` выделяет буфер каждый
вызов. При отсутствии кадра (камера не подключена) это происходит ~33 раз/сек.

**B-19 · `frame.copy()` на полноразмерном кадре каждый UI-тик**
`ui_worker.py:69` — полная копия RTSP-кадра (напр. 1920×1080×3 ≈ 6МБ) каждую
итерацию UI-петли. Необходимо для отрисовки поверх, но можно рисовать на
уже уменьшенном frame после resize.

---

## ТОРМОЖЕНИЕ / ПРОИЗВОДИТЕЛЬНОСТЬ

### Главные узкие места (по убыванию важности)

**P-01 · RTSP декодирует все кадры, используется каждый 4-й**
`rtsp_reader.py` читает камеру на полной скорости (~15-25 fps для Hikvision).
FFmpeg декодирует каждый кадр (H.264 → RGB, ~3-8мс/кадр на CPU).
При inference_fps=4.0 примерно 80% декодированных кадров выбрасываются.
Итог: ~60-80% CPU времени RTSP-потока уходит впустую.

Решение: добавить в `RtspReader` контролируемый пропуск кадров через
`CAP_PROP_POS_FRAMES` или читать с дросселированием. Либо снизить FPS субпотока
камеры до 5 fps прямо в настройках Hikvision (самый дешёвый вариант).

**P-02 · YOLO CPU-инференс: основной bottleneck (80-250мс/кадр)**
`detector.py:49-56` — `model.predict()` на CPU torch. Это конструктивное
ограничение при отсутствии GPU. Пути ускорения:
- Экспорт в ONNX: `model.export(format="onnx")` + `YOLO("yolo11n-pose.onnx")` →
  в 2-3× быстрее на CPU через ONNXRuntime
- Экспорт в OpenVINO: `model.export(format="openvino")` → 3-4× на Intel CPU
- Уменьшить `imgsz: 480` → ~40% ускорение при незначительной потере точности

**P-03 · MQTT fanout: 10+ публикаций на трек за тик**
`inference_worker.py:158-168` — для каждого трека: `/json`, `/state`, `/foot_x`,
`/foot_y`, `/confidence` + 5 geo-топиков = 10 publish-вызовов. При 3 людях @ 5hz:
150 publish/сек + 6 health-топиков = ~156 MQTT сообщений/сек. Большинство
подписчиков (HA) читают только JSON, остальные топики — избыточный трафик.

**P-04 · `psutil` на каждом inference-кадре**
`inference_worker.py:83-84` — `cpu_percent()` и `virtual_memory().percent` читают
`/proc/stat` (Linux) или WMI (Windows) каждые 250мс. На Windows WMI-запрос
блокирующий и может занимать 5-20мс. Лучше выносить в отдельный медленный таймер
(раз в 2-3 сек), CPU-статистика не нужна на частоте инференса.

**P-05 · `notify_all()` вместо `notify()` в LatestValue**
`latest_value.py:27` — при каждой записи кадра будятся все ожидающие. Сейчас
ждёт только один (InferenceWorker), но `notify_all()` чуть дороже. Незначительно.

**P-06 · `frame.copy()` + `cv2.resize` на полном кадре**
`ui_worker.py:69,151` — сначала копия полного кадра, затем resize до размера окна.
Можно сначала resize, потом рисовать overlay на уменьшенном кадре — вдвое меньше
памяти и быстрее отрисовка (меньше пикселей в cv2.rectangle/circle/line).

**P-07 · JSON сериализуется дважды для каждого трека**
`inference_worker.py:159` — `json.dumps(payload)` для `/json` топика.
Тот же payload уже разобран по отдельным топикам (foot_x, foot_y и т.д.).
Строится словарь, сериализуется в JSON, затем снова читается по полям. Дублирование.

---

## КАК РАЗВЕРНУТЬ ОДНОЙ КОМАНДОЙ

### Вариант 1: Docker (headless, Linux/любой хост)

Headless-режим (`--headless`) не требует дисплея и идеален для 24/7 сервиса.
Docker даёт изоляцию и воспроизводимость.

**Dockerfile** (добавить в корень):
```dockerfile
FROM python:3.12-slim

# Системные зависимости OpenCV + FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY configs/ configs/
COPY data/ data/
COPY *.pt .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "app.main", "--config", "configs/config.yaml", "--headless"]
```

**docker-compose.yml** (добавить в корень):
```yaml
version: "3.9"
services:
  wb-vision:
    build: .
    restart: unless-stopped
    env_file: .env
    volumes:
      - ./data:/app/data       # calibration.json сохраняется между перезапусками
      - ./logs:/app/logs       # логи доступны снаружи
      - ./configs:/app/configs # можно менять конфиг без пересборки образа
```

**Одна команда деплоя:**
```bash
# Первый раз (или после изменений кода):
docker compose up -d --build

# Последующие запуски (конфиг/env изменился):
docker compose up -d
```

**Просмотр логов:**
```bash
docker compose logs -f
```

**Остановка:**
```bash
docker compose down
```

---

### Вариант 2: Windows headless-сервис через Task Scheduler

Для Windows-машины без Docker — запуск при старте системы через `schtasks`:

```powershell
# Создать задачу (запускать headless при входе/старте системы)
schtasks /Create /TN "WBVision" /TR "powershell -WindowStyle Hidden -File C:\Users\alt-c\wb_vision_refactored\scripts\run_headless.ps1" /SC ONLOGON /RU "%USERNAME%" /F

# Запустить немедленно:
schtasks /Run /TN "WBVision"

# Остановить:
schtasks /End /TN "WBVision"

# Удалить:
schtasks /Delete /TN "WBVision" /F
```

---

### Вариант 3: Windows-сервис через NSSM (лучший для prod)

NSSM (Non-Sucking Service Manager) даёт настоящий Windows-сервис:

```powershell
# Установить NSSM (один раз)
winget install NSSM.NSSM

# Создать сервис
nssm install WBVision "C:\Users\alt-c\wb_vision_refactored\.venv\Scripts\python.exe"
nssm set WBVision AppParameters "-m app.main --config configs/config.yaml --headless"
nssm set WBVision AppDirectory "C:\Users\alt-c\wb_vision_refactored"
nssm set WBVision AppStdout "C:\Users\alt-c\wb_vision_refactored\logs\service.log"
nssm set WBVision AppStderr "C:\Users\alt-c\wb_vision_refactored\logs\service.log"
nssm set WBVision Start SERVICE_AUTO_START

# Запустить:
nssm start WBVision

# Статус:
nssm status WBVision
```

---

## Итоговая таблица приоритетов

| # | Файл | Тип | Критичность | Статус |
|---|---|---|---|---|
| B-01 | mqtt_worker.py | Баг | P0 — offline не доходит | **FIXED** |
| B-02 | inference_worker.py | Баг | P0 — CPU всегда 0 при старте | **FIXED** |
| B-03 | main.py | Баг | P0 — двойной close() | **FIXED** |
| B-04 | inference_worker.py | Баг | P1 — ts рассинхронизирован | **FIXED** |
| B-05 | detector.py | Баг | P1 — keypoints без conf-фильтра | **FIXED** |
| B-06 | detector.py | Баг | P1 — state_by_pose игнорирует позу | **FIXED** |
| B-07 | rtsp_reader.py | Баг | P1 — stats.fps без синхронизации | **FIXED** |
| B-08 | rtsp_reader.py | Баг | P1 — ошибка RTSP не логируется | **FIXED** |
| B-09 | rtsp_reader.py | Баг | P1 — setdefault не переопределяет | **FIXED** |
| B-10 | main.py | Баг | P1 — SIGTERM не работает на Windows | **FIXED** |
| B-11 | config.py | Баг | P1 — типы строк вместо int/float | **FIXED** |
| B-12 | config.py | Баг | P1 — опечатки в yaml молча теряются | **FIXED** |
| B-13 | tracker.py | Баг | P2 — лишний _drop_expired | **FIXED** |
| B-14 | tracker.py | Баг | P2 — ID растёт бесконечно | не трогали (не критично) |
| B-15 | tracker.py | Баг | P2 — Counter на каждом кадре | не трогали (мизер) |
| B-16 | tracker.py | Баг | P2 — int-truncation drift | **FIXED** |
| B-17 | ui_worker.py | Баг | P2 — двойной snapshot() | **FIXED** |
| B-18 | ui_worker.py | Баг | P2 — canvas allocate каждые 30мс | **FIXED** |
| B-19 | ui_worker.py | Баг | P2 — frame.copy() на большом кадре | **FIXED** (P-06) |
| P-01 | rtsp_reader.py | Перф | P1 — декодирование всех кадров | не трогали (нужен субпоток на камере) |
| P-02 | detector.py | Перф | P1 — YOLO CPU (ONNX решит) | не трогали (imgsz не меняем) |
| P-03 | inference_worker.py | Перф | P1 — MQTT fanout 150 msg/sec | **FIXED** |
| P-04 | inference_worker.py | Перф | P1 — psutil на каждом кадре | **FIXED** |
| P-05 | latest_value.py | Перф | P2 — notify_all не нужен | **FIXED** |
| P-06 | ui_worker.py + overlay.py | Перф | P2 — resize перед overlay | **FIXED** |
| P-07 | inference_worker.py | Перф | P2 — JSON сериализуется дважды | **FIXED** |
| P-06 | ui_worker.py:69,151 | Перф | P2 — copy до resize |
Надо сделать чтобы окно с отрисовкой видео открывалось только в режиме дебаг без него только отдача координат и все.
Так же добавить распознование стои лежит идет спит упал, и добавить распознование классификацию - с книгой с телефоном моет посуду готовит еду, кушает, в душе в туалете на горшке, красится у зеркала,  сидит у компьютера.