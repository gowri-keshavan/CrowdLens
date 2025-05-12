# VERSION 1
# from deep_sort_realtime.deepsort_tracker import DeepSort
# import cv2
# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
# cap = cv2.VideoCapture(r"D:\Crowd_lens\datasets\ShanghaiTechCampus\training\videos\10_005.avi")  # Replace with your .avi file path

# deepsort = DeepSort(max_age=30)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)
#     boxes = results[0].boxes.xyxy.cpu().numpy()
#     confidences = results[0].boxes.conf.cpu().numpy()
#     classes = results[0].boxes.cls.cpu().numpy()

#     detections = []
#     for box, conf, cls in zip(boxes, confidences, classes):
#         if int(cls) == 0:  # Only detect persons
#             x1, y1, x2, y2 = map(float, box)
#             detections.append(([x1, y1, x2, y2], conf, 'person'))

#     tracks = deepsort.update_tracks(detections, frame=frame)

#     for track in tracks:
#         if not track.is_confirmed():
#             continue
#         track_id = track.track_id
#         ltrb = track.to_ltrb()
#         x1, y1, x2, y2 = map(int, ltrb)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#     cv2.imshow("YOLOv8 + DeepSort Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# VERSION 2
# import cv2
# import time
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort

# # ─── CONFIG ────────────────────────────────────────────────────────────────────
# VIDEO_PATH     = r"datasets\people1.avi"
# MODEL_PATH     = "yolov8n.pt"
# IMG_SIZE       = 640
# CONF_THR       = 0.3
# IOU_THR        = 0.4
# MAX_AGE        = 0            # immediate drop of lost tracks
# SHRINK_FACTOR  = 0.1
# INFER_EVERY    = 3            # run detection+tracking every 3rd frame
# EMA_ALPHA      = 0.1          # smoothing factor for FPS
# # ───────────────────────────────────────────────────────────────────────────────

# model    = YOLO(MODEL_PATH)
# deepsort = DeepSort(max_age=MAX_AGE)

# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     raise RuntimeError(f"Cannot open {VIDEO_PATH}")
# last_tracks = []

# prev_time = time.time()
# smoothed_fps = None
# frame_idx = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_idx += 1

#     # Only run heavy inference every Nth frame
#     if frame_idx % INFER_EVERY == 0:
#         # Letterbox + inference
#         h0, w0 = frame.shape[:2]
#         scale = IMG_SIZE / max(h0, w0)
#         nw, nh = int(w0*scale), int(h0*scale)
#         resized = cv2.resize(frame, (nw, nh))
#         dw, dh = (IMG_SIZE-nw)//2, (IMG_SIZE-nh)//2
#         padded = cv2.copyMakeBorder(resized, dh, dh, dw, dw,
#                                     cv2.BORDER_CONSTANT, value=(114,114,114))

#         det = model(padded, imgsz=IMG_SIZE, conf=CONF_THR, iou=IOU_THR)[0]
#         boxes = det.boxes.xyxy.cpu().numpy()
#         confs = det.boxes.conf.cpu().numpy()
#         clss  = det.boxes.cls.cpu().numpy()

#         dets = []
#         for (x1,y1,x2,y2), c, cls in zip(boxes, confs, clss):
#             if int(cls) != 0: continue  # only person
#             # unpad+unscale
#             x1 = (x1 - dw)/scale; y1 = (y1 - dh)/scale
#             x2 = (x2 - dw)/scale; y2 = (y2 - dh)/scale
#             # shrink box
#             w,h = x2-x1, y2-y1
#             x1 += w*SHRINK_FACTOR/2; y1 += h*SHRINK_FACTOR/2
#             x2 -= w*SHRINK_FACTOR/2; y2 -= h*SHRINK_FACTOR/2
#             dets.append(([x1,y1,x2,y2], float(c), "person"))

#         tracks = deepsort.update_tracks(dets, frame=frame)
#         last_tracks = tracks
#     else:
#         tracks = last_tracks

#     # Draw & count
#     count = 0
#     for t in tracks:
#         if not t.is_confirmed() or t.time_since_update>0:
#             continue
#         count += 1
#         x1,y1,x2,y2 = map(int, t.to_ltrb())
#         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
#         cv2.putText(frame, f"ID:{t.track_id}", (x1,y1-6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

#     # === FPS calculation with EMA smoothing ===
#     now = time.time()
#     dt = now - prev_time
#     prev_time = now
#     current_fps = 1/dt if dt>0 else 0
#     if smoothed_fps is None:
#         smoothed_fps = current_fps
#     else:
#         smoothed_fps = EMA_ALPHA * current_fps + (1-EMA_ALPHA) * smoothed_fps

#     # Overlay counter + FPS
#     cv2.rectangle(frame, (0,0), (200,35), (0,0,0), -1)
#     cv2.putText(frame, f"People: {count}", (5,20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
#     cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (150,20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

#     cv2.imshow("Crowd Detection + Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# VERSION 3

import cv2
import time
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH    = r"D:\Crowd_lens\datasets\overcrowded market.mp4"
MODEL_PATH    = "yolov8n.pt"     # generic COCO person detector
IMG_SIZE      = 640              # inference resolution
CONF_THR      = 0.3              # min detection confidence
IOU_THR       = 0.4              # NMS IoU threshold
MAX_AGE       = 0                # immediate drop of lost tracks
SHRINK_FACTOR = 0.1              # shrink each box by 10%
INFER_EVERY   = 3                # run detection+tracking every Nth frame
FPS_WINDOW    = 30               # number of displayed frames to avg FPS over
# ───────────────────────────────────────────────────────────────────────────────

# 1. Load detector & tracker
model    = YOLO(MODEL_PATH)
deepsort = DeepSort(max_age=MAX_AGE)

# 2. Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

# 3. Initialize display window
win_name = "Crowd Detection + Tracking"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, 800, 600)

# 4. Prepare for frame skipping & FPS calc
last_tracks = []
frame_idx   = 0
timestamps  = deque(maxlen=FPS_WINDOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # ─── RUN DETECTION + TRACKING every INFER_EVERY frames ─────────────────────
    if frame_idx % INFER_EVERY == 0:
        # letterbox resize
        h0, w0 = frame.shape[:2]
        scale  = IMG_SIZE / max(h0, w0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh  = (IMG_SIZE - nw)//2, (IMG_SIZE - nh)//2
        padded = cv2.copyMakeBorder(resized, dh, dh, dw, dw,
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=(114,114,114))

        # inference
        det = model(padded, imgsz=IMG_SIZE, conf=CONF_THR, iou=IOU_THR)[0]
        boxes = det.boxes.xyxy.cpu().numpy()
        confs = det.boxes.conf.cpu().numpy()
        clss  = det.boxes.cls.cpu().numpy()

        # unpad, unscale, shrink & filter “person” class
        dets = []
        for (x1,y1,x2,y2), c, cls in zip(boxes, confs, clss):
            if int(cls) != 0:  # only COCO “person”
                continue
            x1 = (x1 - dw) / scale
            y1 = (y1 - dh) / scale
            x2 = (x2 - dw) / scale
            y2 = (y2 - dh) / scale
            w, h = x2 - x1, y2 - y1
            x1 += w * SHRINK_FACTOR/2
            y1 += h * SHRINK_FACTOR/2
            x2 -= w * SHRINK_FACTOR/2
            y2 -= h * SHRINK_FACTOR/2
            dets.append(([x1, y1, x2, y2], float(c), "person"))

        # update tracks
        tracks = deepsort.update_tracks(dets, frame=frame)
        last_tracks = tracks
    else:
        # reuse last known tracks
        tracks = last_tracks

    # ─── DRAW BOXES & COUNT PEOPLE ──────────────────────────────────────────────
    count = 0
    for t in tracks:
        # only confirmed & updated tracks
        if not t.is_confirmed() or t.time_since_update > 0:
            continue
        count += 1
        x1, y1, x2, y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box
        cv2.putText(frame, f"ID:{t.track_id}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ─── CALCULATE MOVING-WINDOW FPS ────────────────────────────────────────────
    now = time.time()
    timestamps.append(now)
    if len(timestamps) >= 2:
        fps = len(timestamps) / (timestamps[-1] - timestamps[0])
    else:
        fps = 0.0

    # ─── OVERLAY COUNTER & FPS ──────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (300, 35), (0, 0, 0), -1)
    cv2.putText(frame, f"People: {count}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (150, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ─── SHOW FRAME ─────────────────────────────────────────────────────────────
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
