import torch
import pickle
from tensorflow.keras.models import load_model

def load_models():
    models = {}

    # Load YOLOv8 for person detection
    models['yolo'] = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov8 via Ultralytics

    # Load fight detection model
    with open('models/fight_model.pkl', 'rb') as f:
        models['fight'] = pickle.load(f)

    # Load aggression detection model
    models['aggression'] = load_model('models/aggression_model.h5')

    return models
