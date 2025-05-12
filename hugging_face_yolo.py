import cv2
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_PATH    = r"D:\Crowd_lens\datasets\ShanghaiTechCampus\training\videos\10_005.avi" # 0 for live camera else put file path
MODEL_PATH    = r"D:\Crowd_lens\YOLOv8-Face-Detection\model.pt"
IMG_SIZE      = 640       # inference resolution
CONF_THR      = 0.3       # min detection confidence
IOU_THR       = 0.4       # NMS IoU threshold
MAX_AGE       = 30        # DeepSort max lost frames
FRAME_SKIP    = 1         # process every (FRAME_SKIP+1)th frame
MAX_BOX_AREA  = 0.5       # drop detections covering >50% of frame area
# ───────────────────────────────────────────────────────────────────────────────

# 1. load detector & tracker
model   = YOLO(MODEL_PATH)
deepsort = DeepSort(max_age=MAX_AGE)

# 2. open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_PATH}")

frame_idx = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # skip frames to speed up
    if frame_idx % (FRAME_SKIP+1) != 0:
        continue

    # ─── PREPROCESS & INFERENCE ────────────────────────────────────────────────
    h0, w0 = frame.shape[:2]
    scale = IMG_SIZE / max(h0, w0)
    nw, nh = int(w0*scale), int(h0*scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (IMG_SIZE - nw)//2, (IMG_SIZE - nh)//2
    padded = cv2.copyMakeBorder(resized, dh, dh, dw, dw,
                                cv2.BORDER_CONSTANT, value=(114,114,114))

    results = model(padded, imgsz=IMG_SIZE, conf=CONF_THR, iou=IOU_THR)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    # ─── SCALE BACK & FILTER ───────────────────────────────────────────────────
    dets = []
    frame_area = h0 * w0
    for (x1,y1,x2,y2), c in zip(boxes, confs):
        # unpad & unscale
        x1 = max(0, (x1 - dw) / scale)
        y1 = max(0, (y1 - dh) / scale)
        x2 = min(w0, (x2 - dw) / scale)
        y2 = min(h0, (y2 - dh) / scale)


        # drop absurdly large boxes (likely false positives)
        box_area = (x2-x1)*(y2-y1)
        if box_area / frame_area > MAX_BOX_AREA:
            continue

        dets.append(([x1, y1, x2, y2], float(c), "face"))

    # ─── TRACKING ───────────────────────────────────────────────────────────────
    tracks = deepsort.update_tracks(dets, frame=frame)

    # ─── DRAW & COUNT ──────────────────────────────────────────────────────────
    face_count = 0
    for t in tracks:
        if not t.is_confirmed():
            continue
        face_count += 1
        x1,y1,x2,y2 = map(int, t.to_ltrb())
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,255), 2)
        cv2.putText(frame, f"ID:{t.track_id}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 1)

    # FPS computation
    now = time.time()
    fps = 1/(now - prev_time)
    prev_time = now

    # overlay counter + FPS
    cv2.rectangle(frame, (0,0), (200,35), (0,0,0), -1)
    cv2.putText(frame, f"Faces: {face_count}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (110,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imshow("Face Detection + Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
