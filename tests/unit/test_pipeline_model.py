"""Unit tests for transaction anomaly detection pipeline."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
FEATURES_PATH = Path("data/processed/features.parquet")
CLEAN_PATH = Path("data/processed/clean.parquet")


@pytest.fixture(scope="module")
def model_bundle():
    """Load model bundle once for all tests."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def sample_features():
    """Load a sample of feature data."""
    return pd.read_parquet(FEATURES_PATH).head(500)


def test_model_load(model_bundle):
    """Test that model bundle loads correctly and contains expected keys."""
    assert "model" in model_bundle
    assert "feature_cols" in model_bundle
    assert model_bundle["model"] is not None


def test_predict_schema(model_bundle, sample_features):
    """Test that model predictions have correct shape and binary values."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = sample_features[feature_cols].fillna(0)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_predict_proba_range(model_bundle, sample_features):
    """Test that probability outputs are in [0, 1]."""
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = sample_features[feature_cols].fillna(0)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_metrics_file_exists():
    """Test that metrics JSON file exists and has required fields."""
    assert METRICS_PATH.exists()
    metrics = json.loads(METRICS_PATH.read_text())
    for field in ["roc_auc", "pr_auc", "f1_score", "precision", "recall", "algorithm"]:
        assert field in metrics, f"Missing field: {field}"


def test_roc_auc_threshold(model_bundle, sample_features):
    """Test that ROC-AUC on sample data exceeds minimum threshold."""
    from sklearn.metrics import roc_auc_score
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    X = sample_features[feature_cols].fillna(0)
    y = sample_features["is_anomaly"]
    if y.nunique() < 2:
        pytest.skip("Sample has only one class")
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    assert auc >= 0.70, f"ROC-AUC {auc:.4f} below threshold 0.70"


def test_clean_parquet_exists():
    """Test that clean.parquet was created by ingestion pipeline."""
    assert CLEAN_PATH.exists()
    df = pd.read_parquet(CLEAN_PATH)
    assert len(df) > 0
    assert "is_anomaly" in df.columns


def test_features_parquet_exists():
    """Test that features.parquet exists with engineered columns."""
    assert FEATURES_PATH.exists()
    df = pd.read_parquet(FEATURES_PATH)
    engineered = ["log_amount", "amount_zscore", "composite_risk", "is_weekend"]
    for col in engineered:
        assert col in df.columns, f"Missing engineered column: {col}"


def test_no_data_leakage(model_bundle):
    """Test feature list contains no target or ID columns."""
    feature_cols = model_bundle["feature_cols"]
    leakage_cols = {"is_anomaly", "transaction_id"}
    overlap = set(feature_cols) & leakage_cols
    assert len(overlap) == 0, f"Data leakage detected: {overlap}"
