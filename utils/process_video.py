import numpy as np
import cv2
from utils.models_loader import load_models
from utils.anomaly_detection import detect_anomalies

models = load_models()

def process_frame(frame_file):
    # Convert file to OpenCV frame
    np_img = np.frombuffer(frame_file.read(), np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Analyze frame
    result = detect_anomalies(frame, models)
    return result
