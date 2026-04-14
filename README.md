# Transaction Anomaly Detection ML Pipeline

An end-to-end ML pipeline for real-time transaction fraud/anomaly detection, built with XGBoost, FastAPI, and automated testing.

## Pipeline Stages

| Stage | Description | Output |
|-------|-------------|--------|
| 0 | Data Discovery | 10,000 transactions, 11 cols |
| 1 | Ingest & Validate | `data/processed/clean.parquet` · 10/10 checks |
| 2 | Feature Engineering + EDA | 19 features · 5 charts · 12 validation checks |
| 3 | Model Training | XGBoost · ROC-AUC: 0.9999 · F1: 0.9831 |
| 4 | Unit Tests | 8/8 passed |
| 5 | FastAPI + Dashboard | `http://localhost:8000` |
| 6 | Playwright E2E Tests | 6/6 passed · 6 screenshots |
| 7 | Git + GitHub | This repo |
| 8 | JIRA | Project + Epic + Sprint |
| 9 | Confluence | 11-section documentation |
| 10 | Scheduler | Nightly retrain + drift detection |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data
python3 data/raw/generate_data.py

# Run full pipeline
python3 src/data/ingest.py
python3 src/features/engineer.py
python3 src/train.py

# Start API
uvicorn src.api.main:app --port 8000

# Run tests
pytest tests/
```

## Model Performance

- **Algorithm**: XGBoost (binary classification)
- **ROC-AUC**: 0.9999
- **PR-AUC**: 0.9970
- **F1 Score**: 0.9831
- **Precision**: 0.9831
- **Recall**: 0.9831

## API Endpoints

- `GET /` — Interactive dashboard
- `POST /predict` — Score a transaction
- `GET /health` — Liveness check
- `GET /metrics` — Model performance metrics
- `GET /docs` — Swagger UI

## Project Structure

```
data/raw/           → Raw transaction CSV
data/processed/     → Cleaned + feature-engineered parquet
src/                → Pipeline source code
tests/unit/         → 8 pytest unit tests
tests/e2e/          → 6 Playwright E2E tests
models/             → Trained model + metrics
logs/               → Audit logs + quality reports
reports/figures/    → 5 EDA charts
reports/screenshots/→ 6 Playwright screenshots
```
