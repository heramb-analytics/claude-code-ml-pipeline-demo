"""Feature engineering for transaction anomaly detection.

Reads clean.parquet, engineers features, saves features.parquet
and feature_schema.json.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_PATH = Path("data/processed/clean.parquet")
FEATURES_PATH = Path("data/processed/features.parquet")
SCHEMA_PATH = Path("data/processed/feature_schema.json")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from clean transaction data.

    Args:
        df: Clean transaction DataFrame.

    Returns:
        DataFrame with engineered features.
    """
    feat = df.copy()

    # Time-based features
    feat["day_of_week"] = pd.to_datetime(feat["timestamp"]).dt.dayofweek
    feat["is_weekend"] = feat["day_of_week"].isin([5, 6]).astype(int)
    feat["is_night"] = feat["transaction_hour"].isin(range(22, 24)).astype(int) | \
                       feat["transaction_hour"].isin(range(0, 6)).astype(int)

    # Amount features
    feat["log_amount"] = np.log1p(feat["amount"])
    feat["amount_zscore"] = (feat["amount"] - feat["amount"].mean()) / feat["amount"].std()

    # Category encoding
    cat_map = {"grocery": 0, "dining": 1, "retail": 2, "online": 3, "electronics": 4, "travel": 5}
    feat["category_code"] = feat["merchant_category"].map(cat_map).fillna(-1).astype(int)

    # Card type encoding
    card_map = {"debit": 0, "credit": 1, "prepaid": 2}
    feat["card_type_code"] = feat["card_type"].map(card_map).fillna(0).astype(int)

    # Transaction velocity proxy
    feat["high_frequency_merchant"] = (feat["num_prev_transactions"] > 50).astype(int)

    # Risk scoring
    feat["amount_risk"] = (feat["amount"] > feat["amount"].quantile(0.95)).astype(int)
    feat["age_risk"] = ((feat["customer_age_years"] < 25) | (feat["customer_age_years"] > 70)).astype(int)

    # Combined risk score
    feat["composite_risk"] = (
        feat["amount_risk"] * 2 +
        feat["is_international"] * 1.5 +
        feat["is_night"] * 1 +
        feat["age_risk"] * 0.5
    )

    return feat


def run_feature_engineering() -> pd.DataFrame:
    """Run feature engineering pipeline.

    Returns:
        Engineered features DataFrame.
    """
    df = pd.read_parquet(CLEAN_PATH)
    feat_df = engineer_features(df)

    feat_df.to_parquet(FEATURES_PATH, index=False)

    feature_cols = [c for c in feat_df.columns if c not in ["transaction_id", "timestamp", "merchant_id"]]
    schema = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features": len(feature_cols),
        "features": {col: str(feat_df[col].dtype) for col in feature_cols},
        "engineered_features": [
            "day_of_week", "is_weekend", "is_night", "log_amount",
            "amount_zscore", "category_code", "card_type_code",
            "high_frequency_merchant", "amount_risk", "age_risk", "composite_risk",
        ],
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    return feat_df


if __name__ == "__main__":
    feat_df = run_feature_engineering()
    schema = json.loads(SCHEMA_PATH.read_text())
    print(f"   ✅ Subagent A done — {schema['n_features']} features engineered")
