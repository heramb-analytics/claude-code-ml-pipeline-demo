"""Feature engineering for transaction anomaly detection.

This module reads data/processed/clean.parquet, engineers a rich feature set,
fits all scalers and encoders strictly on the training split (70 % of data),
and writes:
  - data/processed/features.parquet  — full feature matrix + is_anomaly label
  - data/processed/feature_schema.json — feature catalogue
  - logs/audit.jsonl (appended)       — run metadata
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_CLEAN = ROOT / "data" / "processed" / "clean.parquet"
OUT_FEATURES = ROOT / "data" / "processed" / "features.parquet"
OUT_SCHEMA = ROOT / "data" / "processed" / "feature_schema.json"
AUDIT_LOG = ROOT / "logs" / "audit.jsonl"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_audit(record: dict[str, Any]) -> None:
    """Append a JSON-Lines record to the audit log.

    Args:
        record: Arbitrary key/value pairs to serialise as one JSONL line.
    """
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _build_feature_schema(df: pd.DataFrame, ohe_merchant: list[str],
                           ohe_card: list[str]) -> list[dict[str, Any]]:
    """Build a feature catalogue for every column in *df*.

    Args:
        df: The final feature matrix (labels excluded).
        ohe_merchant: One-hot-encoded merchant_category column names.
        ohe_card: One-hot-encoded card_type column names.

    Returns:
        List of dicts with keys: name, dtype, description, and either
        *mean* (numeric) or *options* (categorical / binary OHE).
    """
    schema: list[dict[str, Any]] = []

    descriptions: dict[str, str] = {
        "amount_log1p": "Natural log(1 + amount) — compresses right-skewed distribution",
        "amount_zscore": "Z-score of amount fitted on train split",
        "amount_iqr_flag": "1 if amount is an IQR outlier (> Q3 + 1.5*IQR on train)",
        "amount_to_mean_ratio": "amount / train-split mean amount",
        "num_prev_transactions": "Number of prior transactions for this card",
        "customer_age_years": "Age of the card-holder in years",
        "is_international": "1 if the transaction crossed a national border",
        "is_weekend": "1 if the transaction occurred on Saturday or Sunday",
        "is_night_hour": "1 if transaction_hour is between 22:00 and 05:59 inclusive",
        "hour_sin": "Sine encoding of transaction_hour (cyclic, 24-h period)",
        "hour_cos": "Cosine encoding of transaction_hour (cyclic, 24-h period)",
        "amount_x_international": "Interaction: amount * is_international",
        "amount_x_night": "Interaction: amount * is_night_hour",
    }

    for col in df.columns:
        entry: dict[str, Any] = {"name": col, "dtype": str(df[col].dtype)}

        # OHE merchant
        if col in ohe_merchant:
            category = col.replace("merchant_cat_", "")
            entry["description"] = f"One-hot: merchant_category == '{category}'"
            entry["options"] = [0, 1]
        # OHE card
        elif col in ohe_card:
            ctype = col.replace("card_type_", "")
            entry["description"] = f"One-hot: card_type == '{ctype}'"
            entry["options"] = [0, 1]
        # Binary flags
        elif col in {"amount_iqr_flag", "is_international", "is_weekend", "is_night_hour"}:
            entry["description"] = descriptions.get(col, col)
            entry["options"] = [0, 1]
        # Numeric
        else:
            entry["description"] = descriptions.get(col, col)
            entry["mean"] = round(float(df[col].mean()), 6)

        schema.append(entry)

    return schema


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def engineer_features(input_path: Path = RAW_CLEAN) -> pd.DataFrame:
    """Load clean data, engineer features, persist outputs and return the matrix.

    All scalers / statistics are fitted **only** on the 70 % training split to
    prevent data leakage; the derived statistics are then applied to the full
    dataset before saving.

    Args:
        input_path: Path to the cleaned parquet file.

    Returns:
        Full feature DataFrame including the *is_anomaly* label column.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        AssertionError: If the output has fewer columns than the input.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    log.info("Loading %s", input_path)
    df_raw = pd.read_parquet(input_path)
    input_rows = len(df_raw)
    input_cols = df_raw.shape[1]
    log.info("Loaded %d rows × %d columns", input_rows, input_cols)

    # ------------------------------------------------------------------
    # 1. Train / rest split indices (70 %)  — stratified on is_anomaly
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    anomaly_idx = df_raw.index[df_raw["is_anomaly"] == 1].tolist()
    normal_idx = df_raw.index[df_raw["is_anomaly"] == 0].tolist()

    rng.shuffle(anomaly_idx)
    rng.shuffle(normal_idx)

    n_train_anomaly = math.floor(0.70 * len(anomaly_idx))
    n_train_normal = math.floor(0.70 * len(normal_idx))

    train_idx = set(anomaly_idx[:n_train_anomaly] + normal_idx[:n_train_normal])
    train_mask = df_raw.index.isin(train_idx)

    log.info("Train rows: %d  |  Hold-out rows: %d",
             train_mask.sum(), (~train_mask).sum())

    # ------------------------------------------------------------------
    # 2. Extract target and drop non-feature columns
    # ------------------------------------------------------------------
    label = df_raw["is_anomaly"].copy()
    df = df_raw.drop(columns=["transaction_id", "merchant_id", "is_anomaly"])

    # ------------------------------------------------------------------
    # 3. Temporal features (derived from timestamp, then drop it)
    # ------------------------------------------------------------------
    ts = pd.to_datetime(df["timestamp"])
    df["is_weekend"] = ts.dt.dayofweek.ge(5).astype(int)
    df = df.drop(columns=["timestamp"])

    # ------------------------------------------------------------------
    # 4. Cyclic hour encoding
    # ------------------------------------------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)

    # ------------------------------------------------------------------
    # 5. Night-hour flag  (22:00 – 05:59)
    # ------------------------------------------------------------------
    df["is_night_hour"] = df["transaction_hour"].apply(
        lambda h: 1 if h >= 22 or h <= 5 else 0
    )

    # ------------------------------------------------------------------
    # 6. Numeric amount features — fit statistics on TRAIN only
    # ------------------------------------------------------------------
    train_amount = df.loc[train_mask, "amount"]

    # log1p
    df["amount_log1p"] = np.log1p(df["amount"])

    # Z-score (train mean/std)
    amt_mean = train_amount.mean()
    amt_std = train_amount.std(ddof=1)
    df["amount_zscore"] = (df["amount"] - amt_mean) / amt_std

    # IQR flag (train Q1/Q3)
    q1 = train_amount.quantile(0.25)
    q3 = train_amount.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    df["amount_iqr_flag"] = (df["amount"] > upper_fence).astype(int)

    # ratio to train mean
    df["amount_to_mean_ratio"] = df["amount"] / amt_mean

    # ------------------------------------------------------------------
    # 7. Interaction features
    # ------------------------------------------------------------------
    df["amount_x_international"] = df["amount"] * df["is_international"]
    df["amount_x_night"] = df["amount"] * df["is_night_hour"]

    # ------------------------------------------------------------------
    # 8. One-hot encode merchant_category and card_type
    # ------------------------------------------------------------------
    merchant_dummies = pd.get_dummies(
        df["merchant_category"], prefix="merchant_cat", dtype=int
    )
    card_dummies = pd.get_dummies(
        df["card_type"], prefix="card_type", dtype=int
    )

    ohe_merchant_cols = merchant_dummies.columns.tolist()
    ohe_card_cols = card_dummies.columns.tolist()

    df = pd.concat([df, merchant_dummies, card_dummies], axis=1)
    df = df.drop(columns=["merchant_category", "card_type", "transaction_hour"])

    # ------------------------------------------------------------------
    # 9. StandardScaler on continuous numerics — fit on train only
    # ------------------------------------------------------------------
    continuous_cols = [
        "amount", "amount_log1p", "amount_zscore", "amount_to_mean_ratio",
        "num_prev_transactions", "customer_age_years",
        "hour_sin", "hour_cos", "amount_x_international", "amount_x_night",
    ]
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, continuous_cols])
    df[continuous_cols] = scaler.transform(df[continuous_cols])

    # ------------------------------------------------------------------
    # 10. Re-attach label
    # ------------------------------------------------------------------
    df["is_anomaly"] = label.values

    # ------------------------------------------------------------------
    # 11. Sanity assertions
    # ------------------------------------------------------------------
    assert df.shape[1] > input_cols, (
        f"Output columns ({df.shape[1]}) must exceed input columns ({input_cols})"
    )
    assert df.isnull().sum().sum() == 0, "Feature matrix contains NaN values"
    log.info("Feature matrix: %d rows × %d columns (label included)", *df.shape)

    # ------------------------------------------------------------------
    # 12. Persist features.parquet
    # ------------------------------------------------------------------
    OUT_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FEATURES, index=False)
    log.info("Saved features → %s", OUT_FEATURES)

    # ------------------------------------------------------------------
    # 13. Build and persist feature_schema.json
    # ------------------------------------------------------------------
    feature_df = df.drop(columns=["is_anomaly"])
    schema = _build_feature_schema(feature_df, ohe_merchant_cols, ohe_card_cols)

    OUT_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SCHEMA.open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)
    log.info("Saved schema  → %s  (%d features)", OUT_SCHEMA, len(schema))

    # ------------------------------------------------------------------
    # 14. Audit log
    # ------------------------------------------------------------------
    audit_record = {
        "event": "feature_engineering",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_rows": input_rows,
        "output_features": len(schema),
        "output_rows": len(df),
        "train_rows": int(train_mask.sum()),
        "output_path": str(OUT_FEATURES),
        "schema_path": str(OUT_SCHEMA),
    }
    _append_audit(audit_record)
    log.info("Audit entry appended → %s", AUDIT_LOG)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = engineer_features()
    print(f"\nDone. features.parquet shape: {result.shape}")
    print(f"Columns: {result.columns.tolist()}")
