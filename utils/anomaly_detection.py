import cv2
import numpy as np

def detect_anomalies(frame, models):
    results = models['yolo'](frame)
    people_count = 0

    for *box, conf, cls in results.xyxy[0]:
        if int(cls) == 0:  # class 0 = person
            people_count += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Simulated fight/aggression prediction
    # In real app, extract features and use models['fight'], models['aggression']
    fight_detected = people_count > 4  # placeholder logic
    aggression_detected = people_count > 6  # placeholder logic

    return {
        "crowd_count": people_count,
        "fight": fight_detected,
        "aggression": aggression_detected
    }
