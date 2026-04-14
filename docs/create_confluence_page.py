"""Create the pipeline documentation page in Confluence via REST API.

Usage:
    export CONFLUENCE_URL=https://your-instance.atlassian.net
    export CONFLUENCE_USER=your-email@example.com
    export CONFLUENCE_TOKEN=your-api-token
    python docs/create_confluence_page.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("Install requests: pip install requests")

CONFLUENCE_URL = os.environ.get("CONFLUENCE_URL", "")
CONFLUENCE_USER = os.environ.get("CONFLUENCE_USER", "")
CONFLUENCE_TOKEN = os.environ.get("CONFLUENCE_TOKEN", "")
SPACE_KEY = "CR"

PAGE_TITLE = "Anomaly Detection ML Pipeline — v1.0"

PAGE_BODY = """
<h1>Executive Summary</h1>
<p>This pipeline detects fraudulent transactions using an IsolationForest model trained on 10,000
synthetic transactions with 23 engineered features. It achieves <strong>F1=0.8333</strong> and
<strong>ROC-AUC=0.9982</strong> on the hold-out test set, with a live FastAPI dashboard at
<a href="http://localhost:8000">http://localhost:8000</a>.</p>

<h1>Architecture Diagram</h1>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">text</ac:parameter><ac:plain-text-body><![CDATA[
data/raw/transactions.csv
        ↓
[Stage 1] src/data/ingest.py  →  data/processed/clean.parquet
        ↓
[Stage 2] src/features/engineer.py   →  data/processed/features.parquet
          src/validation/checks.py   →  logs/validation_report.json
          reports/eda_report.py      →  reports/figures/ (5 charts)
        ↓
[Stage 3] src/models/train.py  →  models/pipeline_model.pkl
        ↓
[Stage 4] pytest tests/unit/   →  8/8 passed
        ↓
[Stage 5] src/api/main.py  →  http://localhost:8000
        ↓
[Stage 6] pytest tests/e2e/  →  6/6 passed + 6 screenshots
        ↓
[Stage 7] git commit  →  branch feature/anomaly-pipeline-v1
        ↓
[Stage 8] JIRA KAN-1 … KAN-6  →  5× Done, 1× In Progress
]]></ac:plain-text-body></ac:structured-macro>

<h1>Data Catalogue</h1>
<table>
  <tr><th>Column</th><th>Type</th><th>Description</th></tr>
  <tr><td>transaction_id</td><td>string</td><td>Unique TXN identifier (TXNxxxxxxxx)</td></tr>
  <tr><td>timestamp</td><td>datetime</td><td>Transaction timestamp (5-min intervals, 2024)</td></tr>
  <tr><td>merchant_id</td><td>string</td><td>Merchant ID (MERxxxxx, 500 unique merchants)</td></tr>
  <tr><td>merchant_category</td><td>string</td><td>grocery / electronics / travel / dining / retail / online</td></tr>
  <tr><td>card_type</td><td>string</td><td>credit / debit / prepaid</td></tr>
  <tr><td>amount</td><td>float</td><td>Transaction amount $1–$50,000; anomalies $8,000–$50,000</td></tr>
  <tr><td>num_prev_transactions</td><td>int</td><td>Prior transactions for this card (0–199)</td></tr>
  <tr><td>customer_age_years</td><td>int</td><td>Card-holder age 18–80</td></tr>
  <tr><td>transaction_hour</td><td>int</td><td>Hour of day 0–23</td></tr>
  <tr><td>is_international</td><td>0/1</td><td>1 = cross-border transaction (15% of rows)</td></tr>
  <tr><td>is_anomaly</td><td>0/1</td><td>Ground-truth label — 1 = fraud (2.0% rate)</td></tr>
</table>

<h1>Feature Catalogue</h1>
<table>
  <tr><th>Feature</th><th>Type</th><th>Description</th></tr>
  <tr><td>amount</td><td>float (scaled)</td><td>StandardScaler on train split</td></tr>
  <tr><td>num_prev_transactions</td><td>float (scaled)</td><td>StandardScaler on train split</td></tr>
  <tr><td>customer_age_years</td><td>float (scaled)</td><td>StandardScaler on train split</td></tr>
  <tr><td>is_international</td><td>binary</td><td>Pass-through</td></tr>
  <tr><td>is_weekend</td><td>binary</td><td>1 if Saturday or Sunday</td></tr>
  <tr><td>hour_sin / hour_cos</td><td>float</td><td>Cyclic 24-h encoding</td></tr>
  <tr><td>is_night_hour</td><td>binary</td><td>1 if hour in 22:00–05:59</td></tr>
  <tr><td>amount_log1p</td><td>float (scaled)</td><td>log1p(amount) — compresses right-skewed dist</td></tr>
  <tr><td>amount_zscore</td><td>float (scaled)</td><td>Z-score of amount on train mean/std</td></tr>
  <tr><td>amount_iqr_flag</td><td>binary</td><td>1 if amount > Q3 + 1.5×IQR on train</td></tr>
  <tr><td>amount_to_mean_ratio</td><td>float (scaled)</td><td>amount / train mean</td></tr>
  <tr><td>amount_x_international</td><td>float (scaled)</td><td>Interaction: amount × is_international</td></tr>
  <tr><td>amount_x_night</td><td>float (scaled)</td><td>Interaction: amount × is_night_hour</td></tr>
  <tr><td>merchant_cat_* (×6)</td><td>binary OHE</td><td>dining, electronics, grocery, online, retail, travel</td></tr>
  <tr><td>card_type_* (×3)</td><td>binary OHE</td><td>credit, debit, prepaid</td></tr>
</table>

<h1>Model Card</h1>
<table>
  <tr><th>Property</th><th>Value</th></tr>
  <tr><td>Algorithm</td><td>IsolationForest (sklearn)</td></tr>
  <tr><td>n_estimators</td><td>100</td></tr>
  <tr><td>max_samples</td><td>auto</td></tr>
  <tr><td>max_features</td><td>1.0</td></tr>
  <tr><td>contamination</td><td>0.02 (estimated from train)</td></tr>
  <tr><td>random_state</td><td>42</td></tr>
  <tr><td>Train F1 / ROC-AUC</td><td>0.8500 / 0.9988</td></tr>
  <tr><td>Val F1 / ROC-AUC</td><td>0.8814 / 0.9995</td></tr>
  <tr><td><strong>Test F1 / ROC-AUC</strong></td><td><strong>0.8333 / 0.9982</strong></td></tr>
  <tr><td>Test Precision / Recall</td><td>0.8333 / 0.8333</td></tr>
  <tr><td>Test Avg Precision</td><td>0.9269</td></tr>
</table>

<h1>API Reference</h1>
<table>
  <tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
  <tr><td>POST</td><td>/predict</td><td>Body: 23 feature floats. Returns: {prediction, is_anomaly, confidence, anomaly_score, request_id, timestamp, model_version}</td></tr>
  <tr><td>GET</td><td>/health</td><td>Returns: {status, model_version, model_loaded, uptime_seconds, timestamp}</td></tr>
  <tr><td>GET</td><td>/metrics</td><td>Returns full pipeline_model_metrics.json</td></tr>
  <tr><td>GET</td><td>/</td><td>HTML/JS Tailwind dashboard</td></tr>
  <tr><td>GET</td><td>/recent_predictions</td><td>Last 10 predictions ring buffer</td></tr>
  <tr><td>GET</td><td>/docs</td><td>Swagger UI (auto-generated)</td></tr>
</table>

<h1>UI Guide</h1>
<ol>
  <li>Navigate to <a href="http://localhost:8000">http://localhost:8000</a></li>
  <li>Check green status dot — confirms model loaded</li>
  <li>Fill the prediction form: enter raw dollar Amount, select Merchant Category and Card Type, set Transaction Hour (0–23) and Customer Age, tick International if applicable</li>
  <li>Click <strong>Detect Anomaly</strong></li>
  <li>Result panel shows ANOMALY (red) or NORMAL (green) badge with confidence % and raw anomaly score</li>
  <li>Recent Predictions table updates automatically</li>
  <li>Screenshots: 01_dashboard_home.png, 02_form_filled.png, 03_prediction_result.png</li>
</ol>

<h1>Test Coverage</h1>
<h2>Unit Tests (8/8 PASS)</h2>
<ul>
  <li>test_model_load — model pkl loads, correct type</li>
  <li>test_predict_schema — output shape and value set {-1, 1}</li>
  <li>test_metric_threshold — F1 ≥ 0.60, ROC-AUC ≥ 0.85</li>
  <li>test_data_leakage — zero index overlap train/val/test</li>
  <li>test_latency_under_500ms — single inference &lt; 500ms</li>
  <li>test_invalid_input_raises — wrong feature count raises</li>
  <li>test_output_range — scores finite, preds in {-1, 1}</li>
  <li>test_determinism — identical input → identical output</li>
</ul>
<h2>E2E Tests (6/6 PASS)</h2>
<ul>
  <li>test_dashboard_loads</li>
  <li>test_form_fields_present</li>
  <li>test_prediction_result (ANOMALY at 98% confidence)</li>
  <li>test_swagger_docs</li>
  <li>test_metrics_endpoint</li>
  <li>test_health_endpoint</li>
</ul>

<h1>Playwright Screenshots</h1>
<ul>
  <li>01_dashboard_home.png — full dashboard, green healthy indicator</li>
  <li>02_form_filled.png — $15,000 Electronics/Prepaid/International/3am transaction</li>
  <li>03_prediction_result.png — ANOMALY badge, 98% confidence, red bar</li>
  <li>04_swagger_docs.png — Swagger UI all 5 endpoints visible</li>
  <li>05_metrics_endpoint.png — /metrics JSON response</li>
  <li>06_health_endpoint.png — /health JSON response, model_loaded: true</li>
</ul>
<p>All 6 saved to <code>reports/screenshots/</code></p>

<h1>How to Run</h1>
<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">bash</ac:parameter><ac:plain-text-body><![CDATA[
# 1. Build the pipeline (ingest → features → train)
python src/data/ingest.py && python src/features/engineer.py && python src/models/train.py

# 2. Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 3. Run all tests
pytest tests/
]]></ac:plain-text-body></ac:structured-macro>

<h1>Monitoring &amp; Drift Detection</h1>
<p>Managed by <code>src/scheduler/nightly_job.py</code> (APScheduler):</p>
<ul>
  <li><strong>Job 1 @ 02:00 UTC daily:</strong> Re-ingest → re-engineer → retrain if &gt;500 new rows accumulated in data/raw/. Auto-promotes new model on success.</li>
  <li><strong>Job 2 @ every 6 hours:</strong> Reads last 500 predictions from logs/predictions.jsonl. If current anomaly rate deviates &gt;20% from baseline (2.0%), creates a JIRA High-priority Bug ticket in the KAN project.</li>
</ul>
<p>Drift logs: <code>logs/drift_checks.jsonl</code> | Retrain logs: <code>logs/retrain_history.jsonl</code></p>
"""


def main() -> None:
    """Create or update the Confluence page via REST API."""
    if not all([CONFLUENCE_URL, CONFLUENCE_USER, CONFLUENCE_TOKEN]):
        print("ERROR: Set CONFLUENCE_URL, CONFLUENCE_USER, CONFLUENCE_TOKEN env vars")
        sys.exit(1)

    api_url = f"{CONFLUENCE_URL}/wiki/rest/api/content"
    auth = (CONFLUENCE_USER, CONFLUENCE_TOKEN)
    headers = {"Content-Type": "application/json"}

    payload = {
        "type": "page",
        "title": PAGE_TITLE,
        "space": {"key": SPACE_KEY},
        "body": {
            "storage": {
                "value": PAGE_BODY,
                "representation": "storage",
            }
        },
    }

    resp = requests.post(api_url, json=payload, auth=auth, headers=headers, timeout=30)

    if resp.status_code in (200, 201):
        page = resp.json()
        page_id = page["id"]
        page_url = f"{CONFLUENCE_URL}/wiki/spaces/{SPACE_KEY}/pages/{page_id}"
        print(f"SUCCESS: Page created — {page_url}")
        # Save URL to docs/
        Path("docs/confluence_page_url.txt").write_text(page_url + "\n")
    else:
        print(f"ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
