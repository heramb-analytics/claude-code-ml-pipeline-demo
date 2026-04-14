"""Playwright E2E tests for the transaction anomaly detection API."""
import json
import pytest
import requests
from playwright.sync_api import Page, expect

BASE_URL = "http://localhost:8000"


def test_health_endpoint():
    """Test /health returns 200 with status:healthy."""
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_metrics_endpoint():
    """Test /metrics returns model performance fields."""
    r = requests.get(f"{BASE_URL}/metrics", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "roc_auc" in data
    assert "f1_score" in data
    assert "request_id" in data


def test_predict_normal_transaction():
    """Test POST /predict returns low anomaly probability for normal txn."""
    payload = {
        "amount": 50.0,
        "merchant_category": "grocery",
        "card_type": "debit",
        "num_prev_transactions": 30,
        "customer_age_years": 40,
        "transaction_hour": 14,
        "is_international": 0,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] == 0
    assert data["anomaly_probability"] < 0.5
    assert "request_id" in data
    assert "timestamp" in data


def test_predict_anomalous_transaction():
    """Test POST /predict flags high-amount international transaction."""
    payload = {
        "amount": 45000.0,
        "merchant_category": "electronics",
        "card_type": "prepaid",
        "num_prev_transactions": 2,
        "customer_age_years": 19,
        "transaction_hour": 3,
        "is_international": 1,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] == 1
    assert data["anomaly_probability"] > 0.5


def test_dashboard_loads(page: Page):
    """Test that dashboard HTML page loads with key elements."""
    page.goto(BASE_URL + "/", wait_until="networkidle")
    expect(page.locator("h1")).to_contain_text("Transaction Anomaly Detection")
    expect(page.locator('button[type="submit"]')).to_be_visible()
    expect(page.locator('input[name="amount"]')).to_be_visible()


def test_swagger_docs_load(page: Page):
    """Test that Swagger docs page loads."""
    page.goto(BASE_URL + "/docs", wait_until="networkidle")
    page.wait_for_timeout(1000)
    assert "swagger" in page.title().lower() or page.locator(".swagger-ui").count() > 0
