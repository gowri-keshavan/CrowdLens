import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
cap = cv2.VideoCapture(r"D:\Crowd_lens\datasets\ShanghaiTechCampus\training\videos\10_005.avi")
deepsort = DeepSort(max_age=30)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    detections = []
    for (x, y, w, h) in faces:
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        conf = 1.0
        detections.append((bbox, conf, 0))  # Class 0 is for faces

    tracks = deepsort.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            f'ID: {track.track_id}',
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow("Haar Cascade Face Detection + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
