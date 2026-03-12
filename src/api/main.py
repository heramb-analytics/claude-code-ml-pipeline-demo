"""FastAPI application for transaction anomaly detection.

Endpoints:
    POST /predict  — run model inference
    GET  /health   — service health + model status
    GET  /metrics  — model performance metrics
    GET  /         — full HTML/JS dashboard
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pipeline_model.pkl"
METRICS_PATH = ROOT / "models" / "pipeline_model_metrics.json"
SCHEMA_PATH = ROOT / "data" / "processed" / "feature_schema.json"
LOGS_DIR = ROOT / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions.jsonl"

# ---------------------------------------------------------------------------
# App + startup state
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Real-time fraud/anomaly detection powered by IsolationForest",
    version="1.0.0",
)

_start_time = time.time()
_model_bundle: dict[str, Any] = {}
_feature_schema: list[dict[str, Any]] = []
_model_metrics: dict[str, Any] = {}
_recent_predictions: deque = deque(maxlen=10)


@app.on_event("startup")
def load_model() -> None:
    """Load model, schema, and metrics on startup."""
    global _model_bundle, _feature_schema, _model_metrics

    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)

    if SCHEMA_PATH.exists():
        _feature_schema = json.loads(SCHEMA_PATH.read_text())

    if METRICS_PATH.exists():
        _model_metrics = json.loads(METRICS_PATH.read_text())

    LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Prediction request — accepts all 23 engineered features."""

    amount: float = Field(default=50.0, description="Transaction amount (scaled)")
    num_prev_transactions: float = Field(default=0.0, description="Num prior transactions (scaled)")
    customer_age_years: float = Field(default=0.0, description="Customer age (scaled)")
    is_international: float = Field(default=0.0, description="1 if international")
    is_weekend: float = Field(default=0.0, description="1 if weekend")
    hour_sin: float = Field(default=0.0, description="Sine of transaction hour")
    hour_cos: float = Field(default=1.0, description="Cosine of transaction hour")
    is_night_hour: float = Field(default=0.0, description="1 if 22:00-05:59")
    amount_log1p: float = Field(default=0.0, description="log1p(amount) scaled")
    amount_zscore: float = Field(default=0.0, description="Amount z-score scaled")
    amount_iqr_flag: float = Field(default=0.0, description="1 if IQR outlier")
    amount_to_mean_ratio: float = Field(default=0.0, description="Amount / train mean scaled")
    amount_x_international: float = Field(default=0.0, description="amount * is_international scaled")
    amount_x_night: float = Field(default=0.0, description="amount * is_night_hour scaled")
    merchant_cat_dining: float = Field(default=0.0, description="Merchant: dining")
    merchant_cat_electronics: float = Field(default=0.0, description="Merchant: electronics")
    merchant_cat_grocery: float = Field(default=0.0, description="Merchant: grocery")
    merchant_cat_online: float = Field(default=0.0, description="Merchant: online")
    merchant_cat_retail: float = Field(default=0.0, description="Merchant: retail")
    merchant_cat_travel: float = Field(default=0.0, description="Merchant: travel")
    card_type_credit: float = Field(default=1.0, description="Card: credit")
    card_type_debit: float = Field(default=0.0, description="Card: debit")
    card_type_prepaid: float = Field(default=0.0, description="Card: prepaid")


class PredictResponse(BaseModel):
    """Prediction response with full metadata."""

    request_id: str
    timestamp: str
    prediction: str  # "ANOMALY" or "NORMAL"
    is_anomaly: int  # 1 or 0
    confidence: float
    anomaly_score: float
    model_version: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_feature_names() -> list[str]:
    return _model_bundle.get("feature_names", [f.get("name") for f in _feature_schema])


def _log_prediction(request_id: str, inputs: dict, result: dict) -> None:
    record = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "inputs": inputs,
        **result,
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    """Return service health and model status."""
    return JSONResponse({
        "status": "healthy",
        "model_version": _model_metrics.get("version", "1.0.0"),
        "model_loaded": bool(_model_bundle),
        "uptime_seconds": round(time.time() - _start_time, 1),
        "timestamp": datetime.now().isoformat(),
    })


@app.get("/metrics")
def metrics() -> JSONResponse:
    """Return full model metrics."""
    if not _model_metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return JSONResponse(_model_metrics)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run anomaly detection on a single transaction."""
    if not _model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = _get_feature_names()
    input_dict = request.model_dump()
    X = np.array([[input_dict[f] for f in feature_names]], dtype=float)

    model = _model_bundle["model"]
    raw_pred = model.predict(X)[0]       # -1 anomaly, 1 normal
    score = float(model.score_samples(X)[0])

    is_anomaly = 1 if raw_pred == -1 else 0
    label = "ANOMALY" if is_anomaly else "NORMAL"
    # Normalise score to [0,1] confidence: more negative score = higher anomaly confidence
    # score_samples returns negative path lengths; more negative = more anomalous
    confidence = float(np.clip(1.0 / (1.0 + np.exp(score * 5)), 0.0, 1.0))

    request_id = str(uuid.uuid4())
    result: dict[str, Any] = {
        "prediction": label,
        "is_anomaly": is_anomaly,
        "confidence": round(confidence, 4),
        "anomaly_score": round(score, 6),
        "model_version": _model_metrics.get("version", "1.0.0"),
    }
    _log_prediction(request_id, input_dict, result)

    # Store in recent predictions ring buffer
    _recent_predictions.appendleft({
        "request_id": request_id[:8],
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "amount": input_dict.get("amount", 0),
        "prediction": label,
        "confidence": round(confidence, 4),
    })

    return PredictResponse(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        **result,
    )


@app.get("/recent_predictions")
def recent_predictions() -> JSONResponse:
    """Return last 10 predictions for dashboard table."""
    return JSONResponse(list(_recent_predictions))


@app.get("/", response_class=HTMLResponse)
def dashboard() -> HTMLResponse:
    """Serve the full HTML/JS monitoring dashboard."""
    test_metrics = _model_metrics.get("test", {})
    model_version = _model_metrics.get("version", "1.0.0")
    algorithm = _model_metrics.get("algorithm", "IsolationForest")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Transaction Anomaly Detection — Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .badge-anomaly {{ background:#ef4444;color:#fff;padding:4px 12px;border-radius:9999px;font-weight:700;font-size:1.1rem; }}
    .badge-normal  {{ background:#22c55e;color:#fff;padding:4px 12px;border-radius:9999px;font-weight:700;font-size:1.1rem; }}
    .pulse-green {{ animation: pulse 2s infinite; }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.5}} }}
  </style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen font-sans">

  <!-- Header -->
  <header class="bg-gradient-to-r from-blue-900 via-indigo-900 to-purple-900 shadow-lg">
    <div class="max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
      <div>
        <h1 class="text-2xl font-bold tracking-tight text-white">Transaction Anomaly Detection</h1>
        <p class="text-blue-200 text-sm mt-1">Real-time fraud detection pipeline · {algorithm}</p>
      </div>
      <div class="flex items-center gap-3">
        <span class="bg-blue-800 text-blue-100 text-xs px-3 py-1 rounded-full font-mono">v{model_version}</span>
        <div id="status-dot" class="w-3 h-3 rounded-full bg-gray-500"></div>
        <span id="status-text" class="text-sm text-gray-300">Checking...</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-3 gap-6">

    <!-- Prediction Form -->
    <section class="lg:col-span-1 bg-gray-900 rounded-2xl p-6 shadow-xl border border-gray-800">
      <h2 class="text-lg font-semibold text-white mb-4">Run Prediction</h2>
      <form id="predict-form" class="space-y-3">
        <div>
          <label class="text-xs text-gray-400 uppercase tracking-wide">Amount (raw $)</label>
          <input id="raw_amount" type="number" step="0.01" value="150.00"
                 class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" />
        </div>
        <div>
          <label class="text-xs text-gray-400 uppercase tracking-wide">Merchant Category</label>
          <select id="merchant_category"
                  class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            <option value="grocery">Grocery</option>
            <option value="dining">Dining</option>
            <option value="electronics">Electronics</option>
            <option value="online">Online</option>
            <option value="retail">Retail</option>
            <option value="travel">Travel</option>
          </select>
        </div>
        <div>
          <label class="text-xs text-gray-400 uppercase tracking-wide">Card Type</label>
          <select id="card_type"
                  class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none">
            <option value="credit">Credit</option>
            <option value="debit">Debit</option>
            <option value="prepaid">Prepaid</option>
          </select>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs text-gray-400 uppercase tracking-wide">Hour (0–23)</label>
            <input id="txn_hour" type="number" min="0" max="23" value="14"
                   class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" />
          </div>
          <div>
            <label class="text-xs text-gray-400 uppercase tracking-wide">Customer Age</label>
            <input id="cust_age" type="number" min="18" max="90" value="35"
                   class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" />
          </div>
        </div>
        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs text-gray-400 uppercase tracking-wide">Prev Transactions</label>
            <input id="prev_txns" type="number" min="0" value="12"
                   class="w-full mt-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm focus:ring-2 focus:ring-blue-500 focus:outline-none" />
          </div>
          <div class="flex items-end pb-2">
            <label class="flex items-center gap-2 cursor-pointer">
              <input id="is_intl" type="checkbox" class="w-4 h-4 accent-blue-500" />
              <span class="text-sm text-gray-300">International</span>
            </label>
          </div>
        </div>
        <button type="submit"
                class="w-full mt-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2.5 rounded-xl transition-colors text-sm">
          Detect Anomaly
        </button>
      </form>

      <!-- Result Panel -->
      <div id="result-panel" class="hidden mt-5 p-4 bg-gray-800 rounded-xl border border-gray-700">
        <div class="flex items-center justify-between mb-3">
          <span id="result-badge" class="text-lg font-bold"></span>
          <span class="text-xs text-gray-400">Confidence</span>
        </div>
        <div class="flex items-center justify-between">
          <div>
            <div id="confidence-bar-wrap" class="w-48 h-2 bg-gray-700 rounded-full overflow-hidden">
              <div id="confidence-bar" class="h-2 rounded-full bg-blue-500" style="width:0%"></div>
            </div>
            <span id="confidence-text" class="text-xs text-gray-400 mt-1 block"></span>
          </div>
          <span id="score-text" class="text-xs text-gray-500 font-mono"></span>
        </div>
        <details class="mt-3">
          <summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-300">Raw JSON</summary>
          <pre id="raw-json" class="mt-2 text-xs text-green-400 font-mono overflow-x-auto bg-gray-900 p-2 rounded"></pre>
        </details>
      </div>
    </section>

    <!-- Right column: metrics + history -->
    <section class="lg:col-span-2 flex flex-col gap-6">

      <!-- Metrics Cards -->
      <div class="bg-gray-900 rounded-2xl p-6 shadow-xl border border-gray-800">
        <h2 class="text-lg font-semibold text-white mb-4">Model Performance</h2>
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div class="bg-gray-800 rounded-xl p-4 text-center">
            <div class="text-2xl font-bold text-blue-400">{round(float(str(test_metrics.get('f1', 0))), 3)}</div>
            <div class="text-xs text-gray-400 mt-1 uppercase tracking-wide">F1 Score</div>
          </div>
          <div class="bg-gray-800 rounded-xl p-4 text-center">
            <div class="text-2xl font-bold text-green-400">{round(float(str(test_metrics.get('roc_auc', 0))), 3)}</div>
            <div class="text-xs text-gray-400 mt-1 uppercase tracking-wide">ROC-AUC</div>
          </div>
          <div class="bg-gray-800 rounded-xl p-4 text-center">
            <div class="text-2xl font-bold text-yellow-400">{round(float(str(test_metrics.get('precision', 0))), 3)}</div>
            <div class="text-xs text-gray-400 mt-1 uppercase tracking-wide">Precision</div>
          </div>
          <div class="bg-gray-800 rounded-xl p-4 text-center">
            <div class="text-2xl font-bold text-purple-400">{round(float(str(test_metrics.get('recall', 0))), 3)}</div>
            <div class="text-xs text-gray-400 mt-1 uppercase tracking-wide">Recall</div>
          </div>
        </div>
        <div class="mt-4 grid grid-cols-2 gap-4 text-sm text-gray-400">
          <div><span class="text-gray-500">Algorithm:</span> <span class="text-white">{algorithm}</span></div>
          <div><span class="text-gray-500">Features:</span> <span class="text-white">{_model_metrics.get('feature_count', 23)}</span></div>
          <div><span class="text-gray-500">Trained:</span> <span class="text-white font-mono text-xs">{_model_metrics.get('trained_at', 'N/A')[:19]}</span></div>
          <div><span class="text-gray-500">Version:</span> <span class="text-white">{model_version}</span></div>
        </div>
      </div>

      <!-- Recent Predictions Table -->
      <div class="bg-gray-900 rounded-2xl p-6 shadow-xl border border-gray-800 flex-1">
        <div class="flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-white">Recent Predictions</h2>
          <button onclick="refreshHistory()" class="text-xs text-blue-400 hover:text-blue-300">Refresh</button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="text-gray-500 text-xs uppercase tracking-wide border-b border-gray-800">
                <th class="pb-2 text-left">Time</th>
                <th class="pb-2 text-left">ID</th>
                <th class="pb-2 text-right">Amount</th>
                <th class="pb-2 text-center">Result</th>
                <th class="pb-2 text-right">Confidence</th>
              </tr>
            </thead>
            <tbody id="history-body" class="divide-y divide-gray-800">
              <tr><td colspan="5" class="py-4 text-center text-gray-600 text-xs">No predictions yet</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

<script>
// Constants (from server - known mean and std for de-scaling display)
const AMOUNT_MEAN = 60.47;

// Health polling
async function checkHealth() {{
  try {{
    const r = await fetch('/health');
    const d = await r.json();
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (d.model_loaded) {{
      dot.className = 'w-3 h-3 rounded-full bg-green-400 pulse-green';
      txt.textContent = `Healthy · uptime ${{Math.round(d.uptime_seconds)}}s`;
      txt.className = 'text-sm text-green-300';
    }} else {{
      dot.className = 'w-3 h-3 rounded-full bg-red-500';
      txt.textContent = 'Model not loaded';
      txt.className = 'text-sm text-red-400';
    }}
  }} catch(e) {{
    document.getElementById('status-dot').className = 'w-3 h-3 rounded-full bg-red-500';
    document.getElementById('status-text').textContent = 'Unreachable';
  }}
}}
setInterval(checkHealth, 5000);
checkHealth();

// Build feature vector from form inputs
function buildPayload() {{
  const amount  = parseFloat(document.getElementById('raw_amount').value) || 50;
  const hour    = parseInt(document.getElementById('txn_hour').value) || 12;
  const age     = parseInt(document.getElementById('cust_age').value) || 35;
  const prevTxn = parseInt(document.getElementById('prev_txns').value) || 10;
  const isIntl  = document.getElementById('is_intl').checked ? 1 : 0;
  const cat     = document.getElementById('merchant_category').value;
  const ctype   = document.getElementById('card_type').value;

  // Approximate the same transformations used in feature engineering
  // (using train-split statistics baked into the feature schema)
  const amtMean = 60.47, amtStd = 251.3;
  const prevMean = 99.65, prevStd = 57.85;
  const ageMean  = 48.94, ageStd  = 18.02;

  const amtScaled   = (amount - amtMean) / amtStd;
  const logScaled   = (Math.log1p(amount) - Math.log1p(amtMean)) / 1.2;
  const zScore      = amtScaled;
  const iqrFlag     = amount > 130 ? 1 : 0;
  const ratioScaled = (amount / amtMean - 1) / 1.2;
  const prevScaled  = (prevTxn - prevMean) / prevStd;
  const ageScaled   = (age - ageMean) / ageStd;
  const hourSin     = Math.sin(2 * Math.PI * hour / 24);
  const hourCos     = Math.cos(2 * Math.PI * hour / 24);
  const isNight     = (hour >= 22 || hour <= 5) ? 1 : 0;
  const isWeekend   = 0;
  const axI_raw     = amount * isIntl;
  const axN_raw     = amount * isNight;
  const axIScaled   = (axI_raw - 9.07) / 77.5;
  const axNScaled   = (axN_raw - 11.96) / 100.2;

  return {{
    amount: amtScaled,
    num_prev_transactions: prevScaled,
    customer_age_years: ageScaled,
    is_international: isIntl,
    is_weekend: isWeekend,
    hour_sin: hourSin,
    hour_cos: hourCos,
    is_night_hour: isNight,
    amount_log1p: logScaled,
    amount_zscore: zScore,
    amount_iqr_flag: iqrFlag,
    amount_to_mean_ratio: ratioScaled,
    amount_x_international: axIScaled,
    amount_x_night: axNScaled,
    merchant_cat_dining:       cat === 'dining'       ? 1 : 0,
    merchant_cat_electronics:  cat === 'electronics'  ? 1 : 0,
    merchant_cat_grocery:      cat === 'grocery'      ? 1 : 0,
    merchant_cat_online:       cat === 'online'       ? 1 : 0,
    merchant_cat_retail:       cat === 'retail'       ? 1 : 0,
    merchant_cat_travel:       cat === 'travel'       ? 1 : 0,
    card_type_credit:  ctype === 'credit'  ? 1 : 0,
    card_type_debit:   ctype === 'debit'   ? 1 : 0,
    card_type_prepaid: ctype === 'prepaid' ? 1 : 0,
  }};
}}

// Prediction form submit
document.getElementById('predict-form').addEventListener('submit', async (e) => {{
  e.preventDefault();
  const btn = e.target.querySelector('button[type=submit]');
  btn.textContent = 'Running...';
  btn.disabled = true;
  try {{
    const payload = buildPayload();
    const resp = await fetch('/predict', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify(payload),
    }});
    const data = await resp.json();

    const panel  = document.getElementById('result-panel');
    const badge  = document.getElementById('result-badge');
    panel.classList.remove('hidden');

    if (data.prediction === 'ANOMALY') {{
      badge.innerHTML = '<span class="badge-anomaly">⚠ ANOMALY</span>';
    }} else {{
      badge.innerHTML = '<span class="badge-normal">✓ NORMAL</span>';
    }}

    const pct = Math.round(data.confidence * 100);
    document.getElementById('confidence-bar').style.width = pct + '%';
    document.getElementById('confidence-bar').className =
      'h-2 rounded-full ' + (data.prediction === 'ANOMALY' ? 'bg-red-500' : 'bg-green-500');
    document.getElementById('confidence-text').textContent = `Confidence: ${{pct}}%`;
    document.getElementById('score-text').textContent = `score: ${{data.anomaly_score.toFixed(4)}}`;
    document.getElementById('raw-json').textContent = JSON.stringify(data, null, 2);

    refreshHistory();
  }} catch(err) {{
    alert('Prediction failed: ' + err.message);
  }} finally {{
    btn.textContent = 'Detect Anomaly';
    btn.disabled = false;
  }}
}});

// History table
async function refreshHistory() {{
  try {{
    const r = await fetch('/recent_predictions');
    const rows = await r.json();
    const tbody = document.getElementById('history-body');
    if (!rows.length) return;
    tbody.innerHTML = rows.map(p => `
      <tr class="hover:bg-gray-800 transition-colors">
        <td class="py-2 text-gray-400 text-xs">${{p.timestamp}}</td>
        <td class="py-2 text-gray-500 font-mono text-xs">${{p.request_id}}</td>
        <td class="py-2 text-right text-gray-300 font-mono text-xs">${{parseFloat(p.amount).toFixed(3)}}</td>
        <td class="py-2 text-center">
          ${{p.prediction === 'ANOMALY'
            ? '<span class="text-xs bg-red-900 text-red-300 px-2 py-0.5 rounded-full">ANOMALY</span>'
            : '<span class="text-xs bg-green-900 text-green-300 px-2 py-0.5 rounded-full">NORMAL</span>'}}
        </td>
        <td class="py-2 text-right text-gray-400 text-xs">${{Math.round(p.confidence*100)}}%</td>
      </tr>
    `).join('');
  }} catch(e) {{}}
}}
setInterval(refreshHistory, 10000);
</script>
</body>
</html>"""
    return HTMLResponse(content=html)
