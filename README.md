# WB Vision Refactored

РќРѕРІС‹Р№ РІР°СЂРёР°РЅС‚ РїСЂРѕРµРєС‚Р° СЂР°Р·РґРµР»С‘РЅ РЅР° РЅРµР·Р°РІРёСЃРёРјС‹Рµ РІРѕСЂРєРµСЂС‹:

- `RtspReader` вЂ” РѕС‚РґРµР»СЊРЅС‹Р№ РїРѕС‚РѕРє С‡С‚РµРЅРёСЏ RTSP, РїРѕСЃС‚РѕСЏРЅРЅС‹Р№ reconnect, Р±СѓС„РµСЂ РєР°РјРµСЂС‹ = 1.
- `LatestValue` вЂ” latest-frame С…СЂР°РЅРёР»РёС‰Рµ: РєР°РґСЂС‹ Р·Р°РјРµРЅСЏСЋС‚СЃСЏ, РѕС‡РµСЂРµРґСЊ РЅРµ СЂР°СЃС‚С‘С‚, Р·Р°РґРµСЂР¶РєР° РЅРµ РЅР°РєР°РїР»РёРІР°РµС‚СЃСЏ.
- `InferenceWorker` вЂ” РѕС‚РґРµР»СЊРЅС‹Р№ РїРѕС‚РѕРє YOLO/pose inference Рё С‚СЂРµРєРёРЅРіР°.
- `UIWorker` вЂ” РѕС‚РґРµР»СЊРЅС‹Р№ UI-loop OpenCV: РІРёРґРµРѕ, РѕРІРµСЂР»РµР№, РєРЅРѕРїРєРё РєР°Р»РёР±СЂРѕРІРєРё.
- `MqttWorker` вЂ” РѕС‚РґРµР»СЊРЅС‹Р№ MQTT-РїРѕС‚РѕРє Рё РѕС‡РµСЂРµРґСЊ РїСѓР±Р»РёРєР°С†РёР№, С‡С‚РѕР±С‹ СЃРµС‚СЊ РЅРµ Р±Р»РѕРєРёСЂРѕРІР°Р»Р° inference.
- `ConfigManager` вЂ” РµРґРёРЅР°СЏ С‚РѕС‡РєР° Р·Р°РіСЂСѓР·РєРё `.env`, YAML-РєРѕРЅС„РёРіР° Рё JSON-РєР°Р»РёР±СЂРѕРІРєРё.
- `CalibrationManager` вЂ” homography-РєР°Р»РёР±СЂРѕРІРєР° РїРѕР»Р°, СЃРѕС…СЂР°РЅРµРЅРёРµ 4 С‚РѕС‡РµРє, РєРѕРѕСЂРґРёРЅР°С‚С‹ РІ РјРµС‚СЂР°С….
- `StableTracker` вЂ” СЃС‚Р°Р±РёР»СЊРЅС‹Рµ ID СЃ greedy matching РїРѕ foot-distance + IoU, СЃРіР»Р°Р¶РёРІР°РЅРёРµ, РёСЃС‚РѕСЂРёСЏ.

## Р‘С‹СЃС‚СЂС‹Р№ СЃС‚Р°СЂС‚ РЅР° Windows PowerShell

```powershell
cd РїСѓС‚СЊ\Рє\wb_vision_refactored
copy .env.example .env
notepad .env
.\scripts\install.ps1
.\scripts\run.ps1
```

РћС‚РєСЂС‹С‚СЊ РІ VS Code:

```powershell
.\scripts\open_vscode.ps1
```

Р—Р°РїСѓСЃРє Р±РµР· GUI:

```powershell
.\scripts\run_headless.ps1
```

## РќР°СЃС‚СЂРѕР№РєР°

1. Р’ `.env` СѓРєР°Р¶РёС‚Рµ RTSP Рё MQTT credentials.
2. Р’ `configs/config.yaml` РЅР°СЃС‚СЂРѕР№С‚Рµ FPS inference, РјРѕРґРµР»СЊ, СЂР°Р·РјРµСЂС‹ РѕРєРЅР°, MQTT prefix.
3. Р”Р»СЏ pose Р»СѓС‡С€Рµ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ `yolo11n-pose.pt`. Р•СЃР»Рё СѓРєР°Р·Р°С‚СЊ detection-only РјРѕРґРµР»СЊ, РїСЂРѕРµРєС‚ Р±СѓРґРµС‚ СЂР°Р±РѕС‚Р°С‚СЊ РїРѕ bbox, РЅРѕ Р±РµР· keypoints.

## РљР°Р»РёР±СЂРѕРІРєР° РІ UI

- `CALIBRATE AIM` вЂ” РєР»РёРєРЅРёС‚Рµ С‚РѕС‡РєСѓ РЅР°РїСЂР°РІР»РµРЅРёСЏ/С†РµРЅС‚СЂР°.
- `CALIBRATE 4 FLOOR POINTS` вЂ” РєР»РёРєРЅРёС‚Рµ 4 С‚РѕС‡РєРё РїРѕР»Р° РїРѕ С‡Р°СЃРѕРІРѕР№ СЃС‚СЂРµР»РєРµ:
  1. Р±Р»РёР¶РЅРёР№ Р»РµРІС‹Р№ СѓРіРѕР»,
  2. Р±Р»РёР¶РЅРёР№ РїСЂР°РІС‹Р№,
  3. РґР°Р»СЊРЅРёР№ РїСЂР°РІС‹Р№,
  4. РґР°Р»СЊРЅРёР№ Р»РµРІС‹Р№.
- `Room W` Рё `Room D` вЂ” СЂРµР°Р»СЊРЅС‹Рµ СЂР°Р·РјРµСЂС‹ Р·РѕРЅС‹ РІ РјРµС‚СЂР°С….
- `SAVE CONFIG` вЂ” СЃРѕС…СЂР°РЅРёС‚СЊ `data/calibration.json`.

Р“РѕСЂСЏС‡РёРµ РєР»Р°РІРёС€Рё:

- `a` вЂ” СЂРµР¶РёРј AIM.
- `f` вЂ” СЂРµР¶РёРј 4 С‚РѕС‡РµРє РїРѕР»Р°.
- `Esc` вЂ” РІС‹С…РѕРґ.

## РЎС‚СЂСѓРєС‚СѓСЂР°

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

