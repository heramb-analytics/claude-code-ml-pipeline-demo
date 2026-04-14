"""FastAPI application for transaction anomaly detection.

Endpoints:
  POST /predict   — score a transaction
  GET  /health    — liveness check
  GET  /metrics   — model performance metrics
  GET  /          — HTML dashboard
"""
import json
import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
AUDIT_LOG = Path("logs/audit.jsonl")
Path("logs").mkdir(exist_ok=True)

app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="Real-time transaction scoring with XGBoost anomaly detection",
    version="1.0.0",
)

# Load model at startup
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)
_model = _bundle["model"]
_feature_cols = _bundle["feature_cols"]
_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

_predictions_history: list[dict[str, Any]] = []


class TransactionInput(BaseModel):
    """Input schema for a single transaction."""
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Category: grocery/electronics/travel/dining/retail/online")
    card_type: str = Field(..., description="Card type: credit/debit/prepaid")
    num_prev_transactions: int = Field(0, ge=0)
    customer_age_years: int = Field(35, ge=18, le=100)
    transaction_hour: int = Field(12, ge=0, le=23)
    is_international: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    """Response schema for anomaly prediction."""
    request_id: str
    timestamp: str
    is_anomaly: int
    anomaly_probability: float
    risk_level: str
    amount: float
    merchant_category: str


def _build_features(txn: TransactionInput) -> pd.DataFrame:
    """Build feature DataFrame from transaction input.

    Args:
        txn: Transaction input data.

    Returns:
        Single-row feature DataFrame.
    """
    import numpy as np
    cat_map = {"grocery": 0, "dining": 1, "retail": 2, "online": 3, "electronics": 4, "travel": 5}
    card_map = {"debit": 0, "credit": 1, "prepaid": 2}
    log_amount = np.log1p(txn.amount)
    # zscore relative to training mean/std (approximate from dataset stats)
    amount_zscore = (txn.amount - 120.0) / 450.0
    composite_risk = (
        (2.0 if txn.amount > 5000 else 0.0) +
        txn.is_international * 1.5 +
        (1.0 if txn.transaction_hour >= 22 or txn.transaction_hour < 6 else 0.0) +
        (0.5 if txn.customer_age_years < 25 or txn.customer_age_years > 70 else 0.0)
    )
    row = {
        "amount": txn.amount,
        "log_amount": log_amount,
        "amount_zscore": amount_zscore,
        "num_prev_transactions": txn.num_prev_transactions,
        "customer_age_years": txn.customer_age_years,
        "transaction_hour": txn.transaction_hour,
        "is_international": txn.is_international,
        "day_of_week": datetime.now(timezone.utc).weekday(),
        "is_weekend": int(datetime.now(timezone.utc).weekday() >= 5),
        "is_night": int(txn.transaction_hour >= 22 or txn.transaction_hour < 6),
        "category_code": cat_map.get(txn.merchant_category.lower(), -1),
        "card_type_code": card_map.get(txn.card_type.lower(), 0),
        "high_frequency_merchant": int(txn.num_prev_transactions > 50),
        "amount_risk": int(txn.amount > 5000),
        "age_risk": int(txn.customer_age_years < 25 or txn.customer_age_years > 70),
        "composite_risk": composite_risk,
    }
    return pd.DataFrame([row])[_feature_cols]


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: TransactionInput) -> PredictionResponse:
    """Score a transaction for anomaly probability.

    Args:
        txn: Transaction data to score.

    Returns:
        Prediction with risk level and probability.
    """
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    try:
        X = _build_features(txn)
        pred = int(_model.predict(X)[0])
        prob = float(_model.predict_proba(X)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
    result = {
        "request_id": request_id,
        "timestamp": ts,
        "is_anomaly": pred,
        "anomaly_probability": round(prob, 4),
        "risk_level": risk_level,
        "amount": txn.amount,
        "merchant_category": txn.merchant_category,
    }
    _predictions_history.append(result)
    if len(_predictions_history) > 100:
        _predictions_history.pop(0)

    audit = {**result, "input": txn.model_dump()}
    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(audit) + "\n")

    return PredictionResponse(**result)


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness check endpoint.

    Returns:
        Health status dict.
    """
    return {
        "status": "healthy",
        "model": "XGBoost",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Return model performance metrics.

    Returns:
        Metrics dict from training run.
    """
    return {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_metrics,
        "total_predictions": len(_predictions_history),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    """HTML dashboard with live prediction form and metrics.

    Args:
        request: FastAPI request object.

    Returns:
        HTML dashboard page.
    """
    last_10 = _predictions_history[-10:][::-1]
    rows_html = ""
    for p in last_10:
        color = "#ef4444" if p["is_anomaly"] else "#22c55e"
        badge = "ANOMALY" if p["is_anomaly"] else "NORMAL"
        rows_html += f"""
        <tr class="border-b border-gray-700">
          <td class="py-2 px-4 text-xs text-gray-400">{p['request_id'][:8]}...</td>
          <td class="py-2 px-4">${p['amount']:,.2f}</td>
          <td class="py-2 px-4">{p['merchant_category']}</td>
          <td class="py-2 px-4">{p['anomaly_probability']:.2%}</td>
          <td class="py-2 px-4"><span style="color:{color};font-weight:bold">{badge}</span></td>
        </tr>"""

    roc = _metrics.get("roc_auc", "N/A")
    pr = _metrics.get("pr_auc", "N/A")
    f1 = _metrics.get("f1_score", "N/A")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Transaction Anomaly Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
  <div class="max-w-5xl mx-auto p-8">
    <div class="flex items-center gap-4 mb-8">
      <div class="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
      <h1 class="text-3xl font-bold text-white">Transaction Anomaly Detection</h1>
      <span class="ml-auto text-xs text-gray-400 bg-gray-800 px-3 py-1 rounded-full">v1.0.0 · XGBoost</span>
    </div>

    <!-- Metrics Cards -->
    <div class="grid grid-cols-3 gap-4 mb-8">
      <div class="bg-gray-800 rounded-xl p-4 text-center">
        <div class="text-2xl font-bold text-blue-400">{roc}</div>
        <div class="text-xs text-gray-400 mt-1">ROC-AUC</div>
      </div>
      <div class="bg-gray-800 rounded-xl p-4 text-center">
        <div class="text-2xl font-bold text-purple-400">{pr}</div>
        <div class="text-xs text-gray-400 mt-1">PR-AUC</div>
      </div>
      <div class="bg-gray-800 rounded-xl p-4 text-center">
        <div class="text-2xl font-bold text-green-400">{f1}</div>
        <div class="text-xs text-gray-400 mt-1">F1 Score</div>
      </div>
    </div>

    <!-- Prediction Form -->
    <div class="bg-gray-800 rounded-xl p-6 mb-8">
      <h2 class="text-lg font-semibold mb-4 text-gray-200">Score a Transaction</h2>
      <form id="predict-form" class="grid grid-cols-2 gap-4">
        <div>
          <label class="text-xs text-gray-400">Amount ($)</label>
          <input name="amount" type="number" value="150.00" step="0.01"
            class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white"/>
        </div>
        <div>
          <label class="text-xs text-gray-400">Merchant Category</label>
          <select name="merchant_category" class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white">
            <option>grocery</option><option>electronics</option><option>travel</option>
            <option>dining</option><option>retail</option><option>online</option>
          </select>
        </div>
        <div>
          <label class="text-xs text-gray-400">Card Type</label>
          <select name="card_type" class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white">
            <option>credit</option><option>debit</option><option>prepaid</option>
          </select>
        </div>
        <div>
          <label class="text-xs text-gray-400">Transaction Hour (0-23)</label>
          <input name="transaction_hour" type="number" value="14" min="0" max="23"
            class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white"/>
        </div>
        <div>
          <label class="text-xs text-gray-400">Customer Age</label>
          <input name="customer_age_years" type="number" value="35" min="18" max="100"
            class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white"/>
        </div>
        <div>
          <label class="text-xs text-gray-400">International (0/1)</label>
          <input name="is_international" type="number" value="0" min="0" max="1"
            class="w-full bg-gray-700 rounded px-3 py-2 mt-1 text-white"/>
        </div>
        <div class="col-span-2">
          <button type="submit"
            class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded-lg transition">
            Predict Anomaly
          </button>
        </div>
      </form>
      <div id="result" class="mt-4 hidden">
        <div id="result-badge" class="text-center py-4 rounded-lg text-xl font-bold"></div>
        <div id="result-detail" class="text-center text-sm text-gray-400 mt-2"></div>
      </div>
    </div>

    <!-- Recent Predictions Table -->
    <div class="bg-gray-800 rounded-xl p-6">
      <h2 class="text-lg font-semibold mb-4 text-gray-200">Last 10 Predictions</h2>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-gray-400 text-xs border-b border-gray-700">
            <th class="py-2 px-4 text-left">Request ID</th>
            <th class="py-2 px-4 text-left">Amount</th>
            <th class="py-2 px-4 text-left">Category</th>
            <th class="py-2 px-4 text-left">Probability</th>
            <th class="py-2 px-4 text-left">Result</th>
          </tr>
        </thead>
        <tbody id="predictions-table">{rows_html}</tbody>
      </table>
    </div>
  </div>

  <script>
    document.getElementById('predict-form').addEventListener('submit', async (e) => {{
      e.preventDefault();
      const fd = new FormData(e.target);
      const payload = {{
        amount: parseFloat(fd.get('amount')),
        merchant_category: fd.get('merchant_category'),
        card_type: fd.get('card_type'),
        num_prev_transactions: 10,
        customer_age_years: parseInt(fd.get('customer_age_years')),
        transaction_hour: parseInt(fd.get('transaction_hour')),
        is_international: parseInt(fd.get('is_international')),
      }};
      const res = await fetch('/predict', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify(payload),
      }});
      const data = await res.json();
      const badge = document.getElementById('result-badge');
      const detail = document.getElementById('result-detail');
      badge.className = 'text-center py-4 rounded-lg text-xl font-bold ' +
        (data.is_anomaly ? 'bg-red-900 text-red-300' : 'bg-green-900 text-green-300');
      badge.textContent = data.is_anomaly ? '⚠ ANOMALY DETECTED' : '✓ NORMAL TRANSACTION';
      detail.textContent = `Probability: ${{(data.anomaly_probability * 100).toFixed(2)}}%  ·  Risk: ${{data.risk_level}}  ·  ID: ${{data.request_id.substring(0,8)}}...`;
      document.getElementById('result').classList.remove('hidden');
      setTimeout(() => location.reload(), 2000);
    }});
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)
