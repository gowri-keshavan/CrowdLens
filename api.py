# backend/api.py
# Flask REST API wrapping crowdlens_backend_rak.run_analysis()
# Run with: python api.py
# React frontend proxies /api/* to http://localhost:5000

from flask import Flask, jsonify, Response
from flask_cors import CORS
import sys
import os

# Allow importing from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crowdlens_backend_rak import run_analysis

import csv
import io

app = Flask(__name__)
CORS(app)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Runs the full YOLO + DeepSort analysis on the configured video.
    Returns JSON matching the shape run_analysis() produces:
      {
        "steps":       [int, ...],
        "density":     [float, ...],
        "aggression":  [float, ...],
        "counts":      [int, ...],
        "total_people": int
      }
    """
    try:
        result = run_analysis()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-csv", methods=["POST"])
def download_csv():
    """
    Runs analysis and returns results as a downloadable CSV file.
    """
    try:
        result = run_analysis()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Frame", "Current People", "Density", "Aggression (%)"])
        for i, frame in enumerate(result["steps"]):
            writer.writerow([
                frame,
                result["counts"][i],
                result["density"][i],
                result["aggression"][i],
            ])
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=crowd_stats.csv"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
