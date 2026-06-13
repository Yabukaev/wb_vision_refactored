from __future__ import annotations

import math
from typing import Optional

import numpy as np

from app.config import VisionSection
from app.types import Detection

_KP_CONF_THRESHOLD = 0.25  # B-05: minimum keypoint confidence to treat as valid


def foot_from_pose(keypoints: Optional[np.ndarray], box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, _y1, x2, y2 = box
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    if keypoints is None:
        return foot_x, foot_y
    ankle_pts: list[tuple[float, float]] = []
    for idx in (15, 16):
        if len(keypoints) > idx:
            kp = keypoints[idx]
            x, y = float(kp[0]), float(kp[1])
            # B-05: filter by confidence when available (keypoints.data has 3 cols: x,y,conf)
            conf = float(kp[2]) if kp.shape[0] > 2 else 1.0
            if x > 1 and y > 1 and conf >= _KP_CONF_THRESHOLD:
                ankle_pts.append((x, y))
    if ankle_pts:
        foot_x = int(sum(p[0] for p in ankle_pts) / len(ankle_pts))
        foot_y = int(max(p[1] for p in ankle_pts))
    return foot_x, foot_y


def _kp_y(keypoints: np.ndarray, idx: int) -> float | None:
    """Return y coordinate of keypoint if confidence passes threshold."""
    if len(keypoints) <= idx:
        return None
    kp = keypoints[idx]
    x, y = float(kp[0]), float(kp[1])
    conf = float(kp[2]) if kp.shape[0] > 2 else 1.0
    return y if (x > 1 and y > 1 and conf >= _KP_CONF_THRESHOLD) else None


def state_by_pose(keypoints: Optional[np.ndarray], box: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    # B-06: use body keypoints when confident enough
    if keypoints is not None and len(keypoints) >= 17:
        shoulder_ys = [y for i in (5, 6) if (y := _kp_y(keypoints, i)) is not None]
        hip_ys = [y for i in (11, 12) if (y := _kp_y(keypoints, i)) is not None]
        ankle_ys = [y for i in (15, 16) if (y := _kp_y(keypoints, i)) is not None]

        if shoulder_ys and ankle_ys:
            sh_y = sum(shoulder_ys) / len(shoulder_ys)
            an_y = sum(ankle_ys) / len(ankle_ys)
            body_h = abs(an_y - sh_y)

            # Body compressed vertically relative to bbox height → lying
            if body_h < h * 0.4:
                return "lying"

            # Hip close to ankle in proportion to full body height → sitting
            if hip_ys:
                hip_y = sum(hip_ys) / len(hip_ys)
                if body_h > 0 and abs(an_y - hip_y) / body_h < 0.3:
                    return "sitting"

            return "standing"

    # Fallback: bbox aspect ratio
    ratio = w / h
    if ratio > 1.25:
        return "lying"
    if ratio > 0.65:
        return "sitting"
    return "standing"


class YoloPoseDetector:
    def __init__(self, cfg: VisionSection) -> None:
        from ultralytics import YOLO
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)

    def predict(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame,
            imgsz=int(self.cfg.imgsz),
            conf=float(self.cfg.conf),
            iou=float(self.cfg.iou),
            classes=list(self.cfg.classes) if self.cfg.classes else None,
            verbose=False,
        )[0]
        if res.boxes is None:
            return []
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        keypoints_arr = None
        if getattr(res, "keypoints", None) is not None and res.keypoints is not None:
            try:
                # B-05: use .data (x, y, conf per keypoint) instead of .xy (x, y only)
                keypoints_arr = res.keypoints.data.cpu().numpy()
            except Exception:
                keypoints_arr = None
        fh, fw = frame.shape[:2]
        min_area = fw * fh * float(self.cfg.min_box_area_ratio)
        detections: list[Detection] = []
        for i, box_raw in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box_raw[:4])
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            kp = keypoints_arr[i] if keypoints_arr is not None and i < len(keypoints_arr) else None
            box = (x1, y1, x2, y2)
            foot = foot_from_pose(kp, box)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            detections.append(Detection(box=box, conf=float(confs[i]), foot=foot, center=center, state=state_by_pose(kp, box), keypoints=kp))
        return detections


def box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / max(1, aa + bb - inter)


def _box_area(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def suppress_duplicates(dets: list[Detection], foot_dist_px: float, iou_threshold: float) -> list[Detection]:
    kept: list[Detection] = []
    for det in sorted(dets, key=lambda d: d.conf, reverse=True):
        duplicate = False
        det_area = _box_area(det.box)
        for old in kept:
            foot_close = math.hypot(det.foot[0] - old.foot[0], det.foot[1] - old.foot[1]) < foot_dist_px
            overlap = box_iou(det.box, old.box) > iou_threshold
            if foot_close or overlap:
                # Guard: if boxes differ in area by >3x, they are different people
                # (one crouching, one standing, or partial vs full view).
                old_area = _box_area(old.box)
                size_ratio = max(det_area, old_area) / max(min(det_area, old_area), 1.0)
                if size_ratio > 3.0:
                    continue
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept
