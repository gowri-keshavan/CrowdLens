# app.py
from flask import Flask, request, jsonify
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files['video']
    path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(path)

    # Run detection as subprocess
    subprocess.Popen(["python", "test_yolo.py", path])

    return jsonify({"status": "Processing started", "filename": video.filename})

if __name__ == "__main__":
    app.run(debug=True)
