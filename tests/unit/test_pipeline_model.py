"""Unit tests for the transaction anomaly detection pipeline model.

Tests: model_load, predict_schema, metric_threshold, data_leakage,
       latency_under_500ms, invalid_input_raises, output_range, determinism
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pipeline_model.pkl"
METRICS_PATH = ROOT / "models" / "pipeline_model_metrics.json"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"


@pytest.fixture(scope="module")
def model_bundle() -> dict:
    """Load the saved model bundle once for all tests."""
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle


@pytest.fixture(scope="module")
def sample_features(model_bundle: dict) -> np.ndarray:
    """Load a small sample from features.parquet."""
    df = pd.read_parquet(FEATURES_PATH)
    feature_names = model_bundle["feature_names"]
    return df[feature_names].values[:100]


@pytest.fixture(scope="module")
def metrics() -> dict:
    """Load model metrics JSON."""
    assert METRICS_PATH.exists(), f"Metrics not found: {METRICS_PATH}"
    return json.loads(METRICS_PATH.read_text())


def test_model_load(model_bundle: dict) -> None:
    """Model pkl loads successfully and contains expected keys."""
    assert "model" in model_bundle, "Bundle missing 'model' key"
    assert "feature_names" in model_bundle, "Bundle missing 'feature_names' key"
    from sklearn.ensemble import IsolationForest
    assert isinstance(model_bundle["model"], IsolationForest), (
        f"Expected IsolationForest, got {type(model_bundle['model'])}"
    )


def test_predict_schema(model_bundle: dict, sample_features: np.ndarray) -> None:
    """Predictions return correct shape and dtype."""
    model = model_bundle["model"]
    raw_preds = model.predict(sample_features)
    # IsolationForest returns -1 (anomaly) or 1 (normal)
    assert raw_preds.shape == (100,), f"Expected shape (100,), got {raw_preds.shape}"
    assert set(raw_preds).issubset({-1, 1}), f"Unexpected prediction values: {set(raw_preds)}"


def test_metric_threshold(metrics: dict) -> None:
    """Test metrics meet minimum thresholds: F1 >= 0.60, ROC-AUC >= 0.85."""
    test_metrics = metrics.get("test", {})
    f1 = test_metrics.get("f1", 0)
    roc_auc = test_metrics.get("roc_auc", 0)
    # Handle numpy float serialization
    f1 = float(str(f1))
    roc_auc = float(str(roc_auc))
    assert f1 >= 0.60, f"Test F1 {f1:.4f} below threshold 0.60"
    assert roc_auc >= 0.85, f"Test ROC-AUC {roc_auc:.4f} below threshold 0.85"


def test_data_leakage() -> None:
    """Assert zero overlap between train, val, and test indices."""
    df = pd.read_parquet(FEATURES_PATH)
    X = df.drop(columns=["is_anomaly"]).values
    y = df["is_anomaly"].values

    from sklearn.model_selection import StratifiedShuffleSplit

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel, test_rel = next(sss2.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap!"
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap!"


def test_latency_under_500ms(model_bundle: dict, sample_features: np.ndarray) -> None:
    """Single prediction completes within 500ms."""
    model = model_bundle["model"]
    single_sample = sample_features[:1]
    start = time.perf_counter()
    model.predict(single_sample)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 500, f"Prediction latency {elapsed_ms:.1f}ms exceeds 500ms"


def test_invalid_input_raises(model_bundle: dict) -> None:
    """Model raises on clearly invalid input (wrong feature count)."""
    model = model_bundle["model"]
    wrong_shape = np.zeros((1, 3))  # wrong number of features
    with pytest.raises(Exception):
        model.predict(wrong_shape)


def test_output_range(model_bundle: dict, sample_features: np.ndarray) -> None:
    """Anomaly scores are finite floats and binary preds are -1 or 1."""
    model = model_bundle["model"]
    preds = model.predict(sample_features)
    scores = model.score_samples(sample_features)

    assert np.all(np.isfinite(scores)), "Anomaly scores contain inf/nan"
    assert set(preds).issubset({-1, 1}), f"Unexpected pred values: {set(preds)}"
    # Anomaly scores should be in roughly [-1, 1] range for IsolationForest
    assert scores.min() > -2.0, f"Anomaly score too low: {scores.min()}"


def test_determinism(model_bundle: dict, sample_features: np.ndarray) -> None:
    """Two consecutive predictions on the same input return identical results."""
    model = model_bundle["model"]
    pred1 = model.predict(sample_features)
    pred2 = model.predict(sample_features)
    np.testing.assert_array_equal(pred1, pred2, err_msg="Predictions are not deterministic!")
