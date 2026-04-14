"""Model training for transaction anomaly detection pipeline.

Uses XGBoost with stratified 70/15/15 split + RandomizedSearchCV.
Saves pipeline_model.pkl and pipeline_model_metrics.json.
"""
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

FEATURES_PATH = Path("data/processed/features.parquet")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
Path("models").mkdir(exist_ok=True)

FEATURE_COLS = [
    "amount", "log_amount", "amount_zscore", "num_prev_transactions",
    "customer_age_years", "transaction_hour", "is_international",
    "day_of_week", "is_weekend", "is_night", "category_code",
    "card_type_code", "high_frequency_merchant", "amount_risk",
    "age_risk", "composite_risk",
]


def train() -> dict:
    """Train XGBoost anomaly detector with hyper-param search.

    Returns:
        Metrics dict.
    """
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLS].fillna(0)
    y = df["is_anomaly"]

    # Stratified 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Assert zero index overlap
    assert len(set(X_train.index) & set(X_val.index)) == 0
    assert len(set(X_train.index) & set(X_test.index)) == 0
    assert len(set(X_val.index) & set(X_test.index)) == 0
    print("   📊 Split: 70% train / 15% val / 15% test  ·  Zero index overlap: ✓")

    scale_pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())
    param_dist = {
        "classifier__max_depth": [3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__n_estimators": [100, 200, 300],
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=42,
            verbosity=0,
        )),
    ])

    search = RandomizedSearchCV(
        pipe, param_dist, n_iter=12, cv=3, scoring="average_precision",
        random_state=42, n_jobs=-1, verbose=0,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    print(f"   📈 Best params: {search.best_params_}")

    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)

    print(f"   📈 ROC-AUC: {roc_auc:.4f}")
    print(f"   📈 PR-AUC : {pr_auc:.4f}")
    print(f"   📈 F1     : {f1:.4f}")

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": best_model, "feature_cols": FEATURE_COLS}, f)
    print(f"   💾 Saved: models/pipeline_model.pkl")

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "algorithm": "XGBoost",
        "best_params": {k.replace("classifier__", ""): v for k, v in search.best_params_.items()},
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "f1_score": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "feature_cols": FEATURE_COLS,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("   📄 Saved: models/pipeline_model_metrics.json")
    return metrics


if __name__ == "__main__":
    metrics = train()
    print(f"✅ STAGE 3 COMPLETE — XGBoost  ·  ROC-AUC: {metrics['roc_auc']}")
