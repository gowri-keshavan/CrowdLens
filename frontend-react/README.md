# CrowdLens – React Dashboard

A React 18 replication of the Streamlit dashboard (`crowdlens_dashboard_rak.py`), consuming the Flask REST API wrapper around `crowdlens_backend_rak.run_analysis()`.

## What it replicates from the Streamlit dashboard

| Streamlit feature | React equivalent |
|---|---|
| `st.sidebar.slider` – rolling window (1–200, default 10) | Range input, live-updates charts |
| `st.sidebar.slider` – aggression threshold (0–100, default 60) | Range input, live-updates charts + KPIs |
| `st.sidebar.selectbox` – sensitivity preset | `<select>` dropdown |
| `st.expander` – "How to interpret the results" | Collapsible section |
| `st.button("Run Analysis")` + spinner | Button → POST `/api/analyze` |
| `st.metric` × 3 — Total Unique People, Peak Aggression %, Status | KPI cards with 🟢/🟡/🔴 |
| Plotly aggression chart — raw (gray) + rolling mean (firebrick) + threshold line + red shaded regions | Recharts `ComposedChart` with `Line` × 2, `ReferenceLine`, `ReferenceArea` |
| Plotly density line chart | Recharts `LineChart` |
| `st.download_button` – CSV export | Button → POST `/api/download-csv` |

## Architecture

```
React (port 3000)
  POST /api/analyze      →  Flask api.py  →  run_analysis()  →  returns JSON
  POST /api/download-csv →  Flask api.py  →  run_analysis()  →  returns CSV
```

## Getting started

### 1. Backend
```bash
pip install flask flask-cors
python backend/api.py        # runs on http://localhost:5000
```

### 2. Frontend
```bash
cd frontend
npm install
npm start                    # runs on http://localhost:3000
```

The `"proxy": "http://localhost:5000"` in `package.json` forwards all `/api/*` calls to Flask during development.

## Tech stack

| Layer | Technology |
|---|---|
| UI | React 18, CSS Modules |
| Charts | Recharts (ComposedChart, LineChart, ReferenceArea) |
| API | Fetch API |
| Backend | Flask, Flask-CORS |
| Analysis | YOLOv8 + DeepSort (`crowdlens_backend_rak.py`) |
