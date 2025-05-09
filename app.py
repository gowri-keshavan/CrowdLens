from flask import Flask, render_template, request, jsonify
import csv
import os
from datetime import datetime

import cv2
import numpy as np

# Dummy implementation of process_frame
def process_frame(frame):
    # Load the image from the file-like object (using PIL or OpenCV)
    img = frame.read()

    # You can convert the image into an OpenCV format for analysis
    # Example: Converting the image to an array using numpy
    nparr = np.frombuffer(img, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Placeholder logic for crowd analysis (replace with real logic)
    crowd_count = np.random.randint(1, 100)  # Random number for demonstration
    fight_detected = False  # Placeholder (you would use an actual model)
    aggression_detected = False  # Placeholder

    # Return the results as a dictionary
    result = {
        'crowd_count': crowd_count,
        'fight': fight_detected,
        'aggression': aggression_detected
    }
    
    return result
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('test_upload.html')

log_data = []

@app.route('/data_endpoint', methods=['GET'])
def data_endpoint():
    timestamps = []
    counts = []

    with open('data/logs.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(row['timestamp'])
            counts.append(int(row['crowd_count']))

    return jsonify({'timestamps': timestamps, 'counts': counts})


@app.route('/analyze', methods=['POST'])
def analyze():
    frame = request.files['frame']
    result = process_frame(frame)

    timestamp = datetime.now().strftime("%H:%M:%S")

    # Log the results to CSV
    log_file = 'data/logs.csv'
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'crowd_count', 'fight', 'aggression'])
        writer.writerow([timestamp, result['crowd_count'], result['fight'], result['aggression']])

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)