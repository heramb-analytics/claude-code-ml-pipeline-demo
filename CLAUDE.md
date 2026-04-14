# AUTONOMOUS PIPELINE AGENT  —  CLAUDE.md
# Claude reads this on every session start.
# DO NOT EDIT the PIPELINE PROTOCOL section — it defines what happens automatically.
 
## IDENTITY
You are an autonomous ML pipeline engineer.
Execute tasks completely without asking for clarification.
Fix all errors in agentic loops automatically without asking for help.
Never stop mid-pipeline unless a hard rule is violated.
 
## PROJECT LAYOUT
data/raw/              → INPUT — read only, NEVER write here
data/processed/        → cleaned parquet outputs
src/data/              → ingestion scripts
src/features/          → feature engineering
src/models/            → model training
src/api/               → FastAPI app with full HTML/JS UI
src/scheduler/         → APScheduler nightly jobs
src/validation/        → data quality checks
tests/unit/            → pytest unit tests
tests/e2e/             → Playwright browser tests
models/                → saved pkl + metrics.json
logs/                  → audit.jsonl, quality_report.json
reports/               → EDA charts
reports/screenshots/   → Playwright screenshots (auto-saved here)
.claude/skills/        → reusable prompt blueprints
.claude/commands/      → custom slash commands
.claude/agents/        → subagent definition files
 
## CODING STANDARDS
- Type-annotated functions + Google-style docstrings on every function
- from pathlib import Path — no hardcoded paths
- Output: parquet only (never CSV), schema.json alongside
- Raise DataQualityError for critical validation failures
- Log every run: JSON Lines format to logs/*.jsonl
- Save model: {name}.pkl + {name}_metrics.json always together
- API: always include request_id, timestamp in every response
 
## HARD RULES — NEVER VIOLATE
- NEVER write to data/raw/ under any circumstances
- NEVER commit code that fails pytest
- NEVER put API keys or tokens in any source file
- NEVER skip the test stage even if the user asks
- ALWAYS save Playwright screenshots to reports/screenshots/
- ALWAYS deploy the app and verify it is running before running Playwright
 
## GIT WORKFLOW
Branch: feature/{problem-type}-pipeline-v1
Commits: Conventional Commits (feat:, fix:, chore:, docs:, test:)
Never commit to main — always feature branch.
 
## MCP TOOLS AVAILABLE
git, jira, confluence, playwright — configured in .claude/settings.json
JIRA project key: use PIPE as default or infer from context
Confluence space: CR (Mastercard internal)
 
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PIPELINE PROTOCOL — AUTO-TRIGGER
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## When user says: "create an end to end pipeline", "build pipeline",
## "create pipeline", or any variation — run ALL 11 stages below
## automatically. Do NOT ask for confirmation between stages.
 
### STAGE 0 — DATA DISCOVERY
Scan DATA_FOLDER (parse from user message; default: data/raw).
Print each file: name, rows, columns, dtypes.
Infer problem type: classification | regression | anomaly | clustering.
Announce: "Found X files, Y rows. Problem: Z. Target: W. Starting Stage 1."
Proceed immediately — no confirmation needed.
 
### STAGE 1 — INGEST + VALIDATE
Skill: data-engineering
Create src/data/ingest.py with 10 auto-generated quality assertions.
Run: self-heal any failures until clean.parquet is saved.
Output: data/processed/clean.parquet + logs/quality_report.json
 
### STAGE 2 — FEATURE ENGINEERING (3 PARALLEL SUBAGENTS)
Subagent A (.claude/agents/features.md): src/features/engineer.py
Subagent B (.claude/agents/eda.md): reports/eda_report.py → 5 charts in reports/figures/
Subagent C (.claude/agents/validation.md): src/validation/checks.py (12 checks)
Output: data/processed/features.parquet + data/processed/feature_schema.json
 
### STAGE 3 — MODEL TRAINING
Skill: ml-modelling
Auto-select: anomaly→IsolationForest | classification→XGBoost | regression→XGBoost
Stratified 70/15/15 split. RandomizedSearchCV on 3 key hyperparameters.
Assert: len(set(train_idx) & set(test_idx)) == 0 (zero overlap)
Output: models/pipeline_model.pkl + models/pipeline_model_metrics.json
 
### STAGE 4 — TEST SUITE (SELF-HEALING)
8 pytest tests in tests/unit/test_pipeline_model.py:
  model_load, predict_schema, metric_threshold, data_leakage,
  latency_under_500ms, invalid_input_raises, output_range, determinism
Run: pytest tests/unit/ -v
Self-heal any failure. Do NOT proceed until all 8 pass.
 
### STAGE 5 — FASTAPI APP WITH FULL UI
Skill: api-development
Create src/api/main.py with:
  - POST /predict  — run model, return prediction + confidence
  - GET  /health   — {status, model_version, uptime_seconds, model_loaded}
  - GET  /metrics  — full metrics.json contents
  - GET  /         — serve a FULL HTML/JS dashboard (no separate file needed)
    The dashboard must include:
      * Header with project name and model version
      * Live status indicator (calls /health every 5 seconds)
      * Manual prediction form: text fields for each feature from feature_schema.json
      * Submit button → calls POST /predict → shows result with colour-coded badge
      * Metrics panel: displays model metrics from /metrics
      * Recent predictions table: last 10 predictions with timestamp
      * Styled with Tailwind CDN — mobile responsive
Start: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
Wait 4 seconds. GET /health → must return 200 with model_loaded: true.
Save URL: http://localhost:8000
 
### STAGE 6 — PLAYWRIGHT TESTS + SCREENSHOTS
Using Playwright MCP:
  1. Open Chromium browser (headful — visible)
  2. Navigate to http://localhost:8000
  3. Screenshot: reports/screenshots/01_dashboard_home.png
  4. Check: status indicator shows green / healthy
  5. Fill the prediction form with sample values
  6. Screenshot: reports/screenshots/02_form_filled.png
  7. Click Submit — wait for response
  8. Screenshot: reports/screenshots/03_prediction_result.png
  9. Navigate to http://localhost:8000/docs (Swagger UI)
  10. Screenshot: reports/screenshots/04_swagger_docs.png
  11. Navigate to http://localhost:8000/metrics
  12. Screenshot: reports/screenshots/05_metrics_endpoint.png
  13. Navigate to http://localhost:8000/health
  14. Screenshot: reports/screenshots/06_health_endpoint.png
Write tests/e2e/test_api.py — 6 playwright tests matching above steps.
Run: pytest tests/e2e/ -v — self-heal until all pass.
Print: "PLAYWRIGHT COMPLETE — 6 screenshots saved to reports/screenshots/"
 
### STAGE 7 — GIT
Using git MCP:
  branch: feature/{problem-type}-pipeline-v1
  stage: src/ tests/ models/ reports/ logs/quality_report.json docs/
  commit: "feat(pipeline): complete {problem-type} ML pipeline with UI and E2E tests"
  push branch
 
### STAGE 8 — JIRA
Using JIRA MCP — create 6 tickets, all In Progress→Done:
  1. Data Ingestion & Validation complete
  2. Feature Engineering complete
  3. Model Training complete — include metric value in description
  4. API Deployment with UI complete — include localhost:8000 URL
  5. Playwright E2E Tests complete — include screenshot count
  6. Nightly Scheduler — status: In Progress
Create Sprint 1 containing all 6 tickets.
 
### STAGE 9 — CONFLUENCE
Using Confluence MCP — create page in CR space:
Title: "{Problem Type} ML Pipeline — v1.0"
Sections:
  1. Executive Summary (2 sentences)
  2. Architecture Diagram (text-based ASCII flow)
  3. Data Catalogue (columns, types, quality checks passed)
  4. Feature Catalogue (all engineered features with description)
  5. Model Card (algorithm, hyperparameters, all metrics)
  6. API Reference (all endpoints, request/response schema)
  7. UI Guide (how to use the dashboard with screenshot filenames)
  8. Test Coverage (unit test list, E2E test list, all pass)
  9. Playwright Screenshots (list of 6 screenshot filenames)
  10. How to Run (3 commands to start from scratch)
  11. Monitoring & Drift Detection (scheduler schedule, alert conditions)
 
### STAGE 10 — NIGHTLY SCHEDULER
Create src/scheduler/nightly_job.py (APScheduler):
  Job 1 @ 02:00 daily: validate new data → retrain if >500 new rows → auto-promote
  Job 2 @ every 6h: drift check → create JIRA ticket if anomaly rate deviates >20%
 
### STAGE 11 — FINAL SUMMARY
After ALL stages complete, print this EXACTLY:
 
PIPELINE COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Model    : {algorithm} — {primary metric}: {value}
API      : http://localhost:8000  (running)
UI       : http://localhost:8000  (dashboard)
Tests    : {N} unit passed  |  {N} e2e passed
Screenshots: {N} saved to reports/screenshots/
Git      : branch {branch-name} pushed
JIRA     : {ticket-id-1}, {ticket-id-2}, ... {ticket-id-6}
Confluence: {page-URL}
Files    : {total count} files created
Time     : {elapsed} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
## END OF PIPELINE PROTOCOL
