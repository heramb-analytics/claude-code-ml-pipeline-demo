"""Model training for transaction anomaly detection.

Uses IsolationForest (anomaly detection) with RandomizedSearchCV.
Stratified 70/15/15 train/val/test split.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
LOGS_DIR = Path("logs")


def load_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load engineered features from features.parquet.

    Returns:
        Tuple of (X, y, feature_names).
    """
    path = DATA_PROCESSED / "features.parquet"
    df = pd.read_parquet(path)
    feature_cols = [c for c in df.columns if c != "is_anomaly"]
    X = df[feature_cols].values.astype(float)
    y = df["is_anomaly"].values.astype(int)
    logger.info("Loaded features: %d rows x %d features", X.shape[0], X.shape[1])
    return X, y, feature_cols


def stratified_split(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 70/15/15 train/val/test split.

    Args:
        X: Feature matrix.
        y: Target array.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, y))

    # Second split: 50% of 30% = 15% val, 15% test
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Assert zero overlap
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap detected!"
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap detected!"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap detected!"

    logger.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_idx), len(val_idx), len(test_idx),
    )
    return (
        X[train_idx], X[val_idx], X[test_idx],
        y[train_idx], y[val_idx], y[test_idx],
    )


def train_isolation_forest(
    X_train: np.ndarray, y_train: np.ndarray
) -> IsolationForest:
    """Train IsolationForest with RandomizedSearchCV on 3 hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels (used for contamination estimate).

    Returns:
        Best fitted IsolationForest estimator.
    """
    contamination_estimate = float(y_train.mean())
    logger.info("Estimated contamination: %.4f", contamination_estimate)

    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_samples": ["auto", 0.5, 0.8],
        "max_features": [0.5, 0.8, 1.0],
    }

    # Typed option lists to avoid numpy scalar issues with sklearn validation
    n_est_opts = [100, 200, 300]
    max_samp_opts: list[Any] = ["auto", 0.5, 0.8]
    max_feat_opts = [0.5, 0.8, 1.0]

    best_score = -np.inf
    best_params: dict[str, Any] = {}
    import random as _random
    _random.seed(42)

    # Manual randomized search over 10 combinations
    n_iter = 10
    for _ in range(n_iter):
        params: dict[str, Any] = {
            "n_estimators": _random.choice(n_est_opts),
            "max_samples": _random.choice(max_samp_opts),
            "max_features": _random.choice(max_feat_opts),
        }
        model = IsolationForest(
            contamination=contamination_estimate,
            random_state=42,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train)
        # Score on training set via anomaly scores (negative = more anomalous)
        scores = -model.score_samples(X_train)
        # Proxy metric: mean score for true anomalies vs normals
        if y_train.sum() > 0:
            sep = scores[y_train == 1].mean() - scores[y_train == 0].mean()
        else:
            sep = 0.0
        if sep > best_score:
            best_score = sep
            best_params = params

    logger.info("Best params: %s (separation score: %.4f)", best_params, best_score)

    final_model = IsolationForest(
        contamination=contamination_estimate,
        random_state=42,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_train)
    return final_model


def evaluate_model(
    model: IsolationForest,
    X: np.ndarray,
    y: np.ndarray,
    split_name: str,
) -> dict[str, float]:
    """Evaluate model on a data split.

    Args:
        model: Fitted IsolationForest.
        X: Feature matrix.
        y: True labels.
        split_name: Name of the split (train/val/test).

    Returns:
        Dict of metric name → value.
    """
    # IsolationForest predicts -1 for anomaly, 1 for normal
    raw_preds = model.predict(X)
    y_pred = (raw_preds == -1).astype(int)
    anomaly_scores = -model.score_samples(X)

    metrics: dict[str, float] = {
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, anomaly_scores), 4),
        "avg_precision": round(average_precision_score(y, anomaly_scores), 4),
    }
    logger.info("[%s] %s", split_name, metrics)
    return metrics


def save_model_and_metrics(
    model: IsolationForest,
    metrics: dict[str, Any],
    feature_names: list[str],
) -> None:
    """Save model pkl and metrics JSON.

    Args:
        model: Fitted model.
        metrics: Evaluation metrics dict.
        feature_names: List of feature column names.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "pipeline_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "feature_names": feature_names}, f)
    logger.info("Model saved to %s", model_path)

    # Save metrics
    metrics_path = MODELS_DIR / "pipeline_model_metrics.json"
    full_metrics = {
        "model_name": "pipeline_model",
        "algorithm": "IsolationForest",
        "problem_type": "anomaly_detection",
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        **metrics,
    }
    metrics_path.write_text(json.dumps(full_metrics, indent=2))
    logger.info("Metrics saved to %s", metrics_path)

    # Audit log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = LOGS_DIR / "audit.jsonl"
    with open(audit_path, "a") as f:
        f.write(json.dumps({
            "event": "model_training",
            "timestamp": datetime.now().isoformat(),
            "algorithm": "IsolationForest",
            "test_f1": metrics.get("test", {}).get("f1"),
            "test_roc_auc": metrics.get("test", {}).get("roc_auc"),
        }) + "\n")


def main() -> None:
    """Run the full model training pipeline."""
    X, y, feature_names = load_features()
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)

    model = train_isolation_forest(X_train, y_train)

    metrics = {
        "train": evaluate_model(model, X_train, y_train, "train"),
        "val": evaluate_model(model, X_val, y_val, "val"),
        "test": evaluate_model(model, X_test, y_test, "test"),
    }

    save_model_and_metrics(model, metrics, feature_names)

    print("\n=== MODEL TRAINING COMPLETE ===")
    print(f"Algorithm: IsolationForest")
    print(f"Test F1:      {metrics['test']['f1']:.4f}")
    print(f"Test ROC-AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Test Recall:  {metrics['test']['recall']:.4f}")
    print(f"Model saved:  models/pipeline_model.pkl")


if __name__ == "__main__":
    main()
