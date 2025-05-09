
# CrowdLens – AI-Based Crowd Behavior Analyzer

CrowdLens is a backend system designed to analyze crowd behavior from video input using artificial intelligence. It detects anomalies such as sudden increases in crowd density, aggression, or physical altercations that may lead to stampede-prone situations. This project aims to support public safety monitoring systems in real-time environments.

## Features

- Detects crowd count from video frames
- Identifies signs of aggression or fighting (planned)
- Logs analysis data to a CSV file with timestamps
- Structured backend using Flask
- API-based design for integration with frontend dashboards

## Project Structure
crowdlens-backend/
├── app.py # Main Flask application
├── models/ # Directory for ML models (to be added)
├── utils/ # Utility functions and helpers
├── data/ # Storage for logs (e.g., logs.csv)
├── static/ # Static assets (CSS, JS, etc.)
├── templates/ # HTML templates
├── requirements.txt # Python dependencies
└── README.md # Project documentation

