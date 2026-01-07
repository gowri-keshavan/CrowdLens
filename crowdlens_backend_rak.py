# crowdlens_backend_rak.py
import cv2
import time
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

VIDEO_PATH   = r"D:\Crowd_lens\datasets\overcrowded market.mp4"
MODEL_PATH   = "yolov8n.pt"
IMG_SIZE     = 640
CONF_THR     = 0.3
IOU_THR      = 0.4
MAX_AGE      = 25
INFER_EVERY  = 2

FLOW_STEP    = 2
MOTION_WIN   = 50
MOTION_MAX   = 4.0
SPEED_MAX    = 25.0

# Use the per-track-speed based aggression logic (moved here)

yolo = YOLO(MODEL_PATH)
deepsort = DeepSort(max_age=MAX_AGE)

def run_analysis():
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_ids = set()
    last_tracks = []
    frame_idx = 0

    # per-track last positions for speed-based aggression
    last_positions = {}
    speed_history = deque(maxlen=MOTION_WIN)

    # lists for frontend
    steps, densities, aggr_scores, counts = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % INFER_EVERY == 0:
            det = yolo(frame, imgsz=IMG_SIZE, conf=CONF_THR, iou=IOU_THR)[0]
            det_boxes   = det.boxes.xyxy.cpu().numpy()
            det_confs   = det.boxes.conf.cpu().numpy()
            det_classes = det.boxes.cls.cpu().numpy()

            dets = []
            for (x1, y1, x2, y2), conf, cls in zip(det_boxes, det_confs, det_classes):
                if int(cls) != 0:
                    continue
                dets.append(([x1, y1, x2, y2], float(conf), "person"))
            tracks = deepsort.update_tracks(dets, frame=frame)
            last_tracks = tracks
        else:
            tracks = last_tracks

        current_ids = set()
        xs, ys, xe, ye = [], [], [], []
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            all_ids.add(t.track_id)
            current_ids.add(t.track_id)
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            xs.append(x1); ys.append(y1); xe.append(x2); ye.append(y2)

        frame_area = frame.shape[0] * frame.shape[1]
        density = len(current_ids) / max(frame_area, 1)

        if xs:
            x_min = max(min(xs) - 10, 0)
            y_min = max(min(ys) - 10, 0)
            x_max = min(max(xe) + 10, frame.shape[1])
            y_max = min(max(ye) + 10, frame.shape[0])
            crowd_roi = frame[y_min:y_max, x_min:x_max]
        else:
            crowd_roi = frame

        # --- compute per-track speeds (pixels/frame) and use them for aggression ---
        speeds = []
        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            tid = t.track_id
            if tid in last_positions:
                lx, ly, lf = last_positions[tid]
                df = frame_idx - lf
                if df > 0:
                    speed = ((cx - lx)**2 + (cy - ly)**2)**0.5 / df
                    speeds.append(float(speed))
            last_positions[tid] = (cx, cy, frame_idx)

        # use a robust percentile (75th) to reduce outlier influence
        if speeds:
            speed_75 = float(np.percentile(speeds, 75))
        else:
            speed_75 = 0.0

        speed_history.append(speed_75)
        if speed_history:
            speed_avg = sum(speed_history) / len(speed_history)
        else:
            speed_avg = 0.0

        norm_aggr = min(speed_avg / SPEED_MAX, 1.0)
        aggr_percent = norm_aggr * 100.0

        # store stats
        steps.append(frame_idx)
        densities.append(density)
        aggr_scores.append(aggr_percent)
        counts.append(len(current_ids))

    cap.release()
    return {
        "steps": steps,
        "density": densities,
        "aggression": aggr_scores,
        "counts": counts,
        "total_people": len(all_ids),
    }
