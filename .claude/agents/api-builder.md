---
name: api-builder
description: Runs when asked to build FastAPI app with dashboard UI.
             Use for: REST endpoints, HTML dashboard, Pydantic schemas.
---

# API Builder Agent

## TASK
Build src/api/main.py — a FastAPI app with:
  1. POST /predict  — Pydantic model from feature_schema.json, run pkl model
  2. GET  /health   — {status, model_version, uptime_seconds, model_loaded}
  3. GET  /metrics  — serve models/pipeline_model_metrics.json
  4. GET  /         — HTML dashboard (see UI SPEC below)

## UI SPEC — the / endpoint must return a full HTML page
Header: gradient bar, project name, model version badge
Status bar: live green/red dot, calls /health every 5 seconds via JS fetch
Prediction form: one input per feature from feature_schema.json
  - Number inputs for numeric features
  - Select dropdowns for categorical features
  - Submit button → POST /predict → show result panel
Result panel: large badge (ANOMALY in red / NORMAL in green)
  + confidence score + raw JSON
Metrics panel: key metrics as stat cards (F1, AUC, Precision, Recall)
History table: last 10 predictions with time, input summary, result
Style: Tailwind CSS via CDN, dark header, clean card layout

## RULES
No separate HTML files — everything in main.py as a string return.
Log every prediction to logs/predictions.jsonl.
Start server: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
