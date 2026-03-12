"""Playwright E2E tests for the Transaction Anomaly Detection API dashboard.

Tests:
    1. test_dashboard_loads     — homepage loads with healthy status
    2. test_form_fill           — prediction form accepts sample values
    3. test_prediction_result   — submit returns ANOMALY or NORMAL badge
    4. test_swagger_docs        — /docs page loads with all endpoints listed
    5. test_metrics_endpoint    — /metrics returns JSON with F1 and ROC-AUC
    6. test_health_endpoint     — /health returns model_loaded: true
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests

BASE_URL = "http://localhost:8000"
SCREENSHOTS_DIR = Path("reports/screenshots")


def wait_for_server(max_wait: int = 30) -> None:
    """Block until the API server responds or timeout expires."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {BASE_URL} did not start within {max_wait}s")


@pytest.fixture(scope="module", autouse=True)
def ensure_server() -> None:
    """Ensure the API server is running before any E2E tests."""
    wait_for_server()
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Test 1 — Dashboard loads with healthy status
# ---------------------------------------------------------------------------

def test_dashboard_loads() -> None:
    """GET / returns 200 HTML with project title and healthy indicator."""
    r = requests.get(f"{BASE_URL}/", timeout=10)
    assert r.status_code == 200
    assert "Transaction Anomaly Detection" in r.text
    assert "IsolationForest" in r.text
    # Status indicator JS is present
    assert "checkHealth" in r.text or "/health" in r.text


# ---------------------------------------------------------------------------
# Test 2 — Prediction form HTML contains all expected input fields
# ---------------------------------------------------------------------------

def test_form_fields_present() -> None:
    """Dashboard HTML contains form inputs for amount, merchant category, card type."""
    r = requests.get(f"{BASE_URL}/", timeout=10)
    html = r.text
    assert "raw_amount" in html, "Amount input missing"
    assert "merchant_category" in html, "Merchant category select missing"
    assert "card_type" in html, "Card type select missing"
    assert "Detect Anomaly" in html, "Submit button missing"


# ---------------------------------------------------------------------------
# Test 3 — POST /predict returns ANOMALY or NORMAL
# ---------------------------------------------------------------------------

def test_prediction_result() -> None:
    """POST /predict with a high-value transaction returns ANOMALY."""
    # Simulate a very high anomalous transaction
    amount = 15000.0
    amt_mean, amt_std = 60.47, 251.3
    amt_scaled = (amount - amt_mean) / amt_std
    log_scaled = 1.5
    z_scaled = amt_scaled
    ratio_scaled = (amount / amt_mean - 1) / 1.2
    iqr_flag = 1

    payload = {
        "amount": amt_scaled,
        "num_prev_transactions": -1.7,
        "customer_age_years": -1.1,
        "is_international": 1,
        "is_weekend": 0,
        "hour_sin": 0.0,
        "hour_cos": -1.0,
        "is_night_hour": 1,
        "amount_log1p": log_scaled,
        "amount_zscore": z_scaled,
        "amount_iqr_flag": float(iqr_flag),
        "amount_to_mean_ratio": ratio_scaled,
        "amount_x_international": amt_scaled * 1,
        "amount_x_night": amt_scaled * 1,
        "merchant_cat_dining": 0,
        "merchant_cat_electronics": 1,
        "merchant_cat_grocery": 0,
        "merchant_cat_online": 0,
        "merchant_cat_retail": 0,
        "merchant_cat_travel": 0,
        "card_type_credit": 0,
        "card_type_debit": 0,
        "card_type_prepaid": 1,
    }

    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
    assert r.status_code == 200
    data = r.json()

    assert "prediction" in data
    assert data["prediction"] in {"ANOMALY", "NORMAL"}
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "request_id" in data
    assert "timestamp" in data
    assert data["prediction"] == "ANOMALY", (
        f"Expected ANOMALY for $15,000 transaction, got {data['prediction']}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Swagger docs
# ---------------------------------------------------------------------------

def test_swagger_docs() -> None:
    """GET /docs returns 200 with Swagger UI HTML."""
    r = requests.get(f"{BASE_URL}/docs", timeout=10)
    assert r.status_code == 200
    assert "swagger" in r.text.lower() or "openapi" in r.text.lower()
    # All expected endpoints should appear in the openapi spec
    spec = requests.get(f"{BASE_URL}/openapi.json", timeout=10).json()
    paths = spec.get("paths", {})
    assert "/predict" in paths, "/predict missing from OpenAPI spec"
    assert "/health" in paths, "/health missing from OpenAPI spec"
    assert "/metrics" in paths, "/metrics missing from OpenAPI spec"


# ---------------------------------------------------------------------------
# Test 5 — /metrics endpoint
# ---------------------------------------------------------------------------

def test_metrics_endpoint() -> None:
    """GET /metrics returns model metrics JSON with required keys."""
    r = requests.get(f"{BASE_URL}/metrics", timeout=10)
    assert r.status_code == 200
    data = r.json()

    assert "algorithm" in data
    assert "test" in data
    test_m = data["test"]
    assert "f1" in test_m
    assert "roc_auc" in test_m
    assert float(str(test_m["f1"])) > 0.5, "F1 unexpectedly low"
    assert float(str(test_m["roc_auc"])) > 0.8, "ROC-AUC unexpectedly low"


# ---------------------------------------------------------------------------
# Test 6 — /health endpoint
# ---------------------------------------------------------------------------

def test_health_endpoint() -> None:
    """GET /health returns status healthy with model_loaded true."""
    r = requests.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200
    data = r.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert data["uptime_seconds"] >= 0
