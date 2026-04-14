"""Feature engineering module for transaction anomaly detection pipeline.

This module loads cleaned transaction data, engineers domain-specific features,
and saves the resulting feature set as a parquet file alongside a feature schema JSON.
"""

import json
import math
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[2]

DATA_PROCESSED = ROOT / "data" / "processed"
LOGS_DIR = ROOT / "logs"

INPUT_PARQUET = DATA_PROCESSED / "clean.parquet"
OUTPUT_PARQUET = DATA_PROCESSED / "features.parquet"
OUTPUT_SCHEMA = DATA_PROCESSED / "feature_schema.json"
LOG_FILE = LOGS_DIR / "feature_engineering.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _log_event(event: dict) -> None:
    """Append a JSON Lines log entry to the feature engineering log file.

    Args:
        event: Dictionary containing log fields. A 'timestamp' key is added
               automatically if not present.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def load_clean_data(path: Path) -> pd.DataFrame:
    """Load the cleaned parquet file from disk.

    Args:
        path: Absolute path to the clean parquet file.

    Returns:
        DataFrame containing the raw cleaned transaction records.

    Raises:
        FileNotFoundError: If the parquet file does not exist at *path*.
    """
    if not path.exists():
        raise FileNotFoundError(f"Clean parquet not found: {path}")
    df = pd.read_parquet(path)
    _log_event({
        "stage": "load",
        "status": "ok",
        "rows": len(df),
        "columns": list(df.columns),
    })
    log.info("Loaded %d rows × %d columns from %s", len(df), df.shape[1], path)
    return df


def engineer_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive amount-based features.

    Features created:
        - log_amount: log1p transformation of raw amount.
        - amount_zscore: z-score of amount within each merchant_category group.
        - is_high_value: 1 if amount exceeds the 95th-percentile threshold, else 0.
        - amount_per_age: amount divided by (customer_age_years + 1) to prevent
          division by zero.

    Args:
        df: DataFrame that must contain 'amount', 'merchant_category', and
            'customer_age_years' columns.

    Returns:
        DataFrame with the four new columns appended.
    """
    df = df.copy()

    df["log_amount"] = np.log1p(df["amount"])

    grp = df.groupby("merchant_category")["amount"]
    df["amount_zscore"] = (df["amount"] - grp.transform("mean")) / (
        grp.transform("std").replace(0, 1)
    )

    p95 = df["amount"].quantile(0.95)
    df["is_high_value"] = (df["amount"] > p95).astype(int)

    df["amount_per_age"] = df["amount"] / (df["customer_age_years"] + 1)

    _log_event({
        "stage": "amount_features",
        "status": "ok",
        "p95_threshold": float(p95),
        "high_value_count": int(df["is_high_value"].sum()),
    })
    log.info("Engineered amount features (p95=%.2f, high_value=%d)", p95, df["is_high_value"].sum())
    return df


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive time-based cyclical and categorical features.

    Features created:
        - hour_sin: sine of the transaction hour mapped to a 24-hour circle.
        - hour_cos: cosine of the transaction hour mapped to a 24-hour circle.
        - is_night: 1 if transaction_hour is in {22,23,0,1,2,3,4,5}, else 0.
        - is_weekend: 1 if the transaction date falls on Saturday or Sunday, else 0.

    Args:
        df: DataFrame containing 'transaction_hour' (int) and 'timestamp'
            (datetime64) columns.

    Returns:
        DataFrame with the four new columns appended.
    """
    df = df.copy()

    hours = df["transaction_hour"].astype(float)
    df["hour_sin"] = np.sin(2 * math.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * math.pi * hours / 24)

    night_hours = {22, 23, 0, 1, 2, 3, 4, 5}
    df["is_night"] = df["transaction_hour"].isin(night_hours).astype(int)

    df["is_weekend"] = (pd.to_datetime(df["timestamp"]).dt.dayofweek >= 5).astype(int)

    _log_event({
        "stage": "time_features",
        "status": "ok",
        "night_transactions": int(df["is_night"].sum()),
        "weekend_transactions": int(df["is_weekend"].sum()),
    })
    log.info("Engineered time features (night=%d, weekend=%d)", df["is_night"].sum(), df["is_weekend"].sum())
    return df


def engineer_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode high-cardinality categorical columns.

    Features created:
        - merchant_category_encoded: integer label encoding of 'merchant_category'.
        - card_type_encoded: integer label encoding of 'card_type'.

    Args:
        df: DataFrame containing 'merchant_category' and 'card_type' string columns.

    Returns:
        DataFrame with the two new encoded columns appended.
    """
    df = df.copy()

    le_cat = LabelEncoder()
    df["merchant_category_encoded"] = le_cat.fit_transform(
        df["merchant_category"].astype(str)
    )

    le_card = LabelEncoder()
    df["card_type_encoded"] = le_card.fit_transform(df["card_type"].astype(str))

    _log_event({
        "stage": "categorical_features",
        "status": "ok",
        "merchant_category_classes": list(le_cat.classes_),
        "card_type_classes": list(le_card.classes_),
    })
    log.info(
        "Label-encoded merchant_category (%d classes) and card_type (%d classes)",
        len(le_cat.classes_),
        len(le_card.classes_),
    )
    return df


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove high-cardinality identifier columns not useful for modelling.

    Drops 'transaction_id' and 'merchant_id' from the DataFrame.

    Args:
        df: Input DataFrame potentially containing identifier columns.

    Returns:
        DataFrame with identifier columns removed.
    """
    cols_to_drop = [c for c in ["transaction_id", "merchant_id"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    _log_event({
        "stage": "drop_identifiers",
        "status": "ok",
        "dropped_columns": cols_to_drop,
    })
    log.info("Dropped identifier columns: %s", cols_to_drop)
    return df


def save_features(df: pd.DataFrame, parquet_path: Path, schema_path: Path) -> None:
    """Persist the feature DataFrame as parquet and write a feature schema JSON.

    The schema JSON maps each column name (excluding 'is_anomaly') to its pandas
    dtype string, enabling downstream consumers to validate the feature set.

    Args:
        df: Feature DataFrame to save. Must contain an 'is_anomaly' target column.
        parquet_path: Destination path for the parquet file.
        schema_path: Destination path for the feature schema JSON.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    log.info("Saved features.parquet → %s  (%d rows × %d cols)", parquet_path, *df.shape)

    schema = {
        col: str(df[col].dtype)
        for col in df.columns
        if col != "is_anomaly"
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    log.info("Saved feature_schema.json → %s  (%d features)", schema_path, len(schema))

    _log_event({
        "stage": "save",
        "status": "ok",
        "parquet_path": str(parquet_path),
        "schema_path": str(schema_path),
        "feature_count": len(schema),
        "row_count": len(df),
        "columns_saved": list(df.columns),
    })


def run_feature_engineering() -> pd.DataFrame:
    """Execute the full feature engineering pipeline end-to-end.

    Steps:
        1. Load clean.parquet from data/processed/.
        2. Engineer amount-based features (log_amount, amount_zscore,
           is_high_value, amount_per_age).
        3. Engineer time-based features (hour_sin, hour_cos, is_night, is_weekend).
        4. Engineer categorical features (merchant_category_encoded, card_type_encoded).
        5. Drop identifier columns (transaction_id, merchant_id).
        6. Save features.parquet and feature_schema.json to data/processed/.

    Returns:
        The final feature DataFrame including the 'is_anomaly' label column.
    """
    _log_event({"stage": "start", "status": "started", "input": str(INPUT_PARQUET)})
    log.info("=== Feature Engineering Pipeline START ===")

    df = load_clean_data(INPUT_PARQUET)
    df = engineer_amount_features(df)
    df = engineer_time_features(df)
    df = engineer_categorical_features(df)
    df = drop_identifier_columns(df)
    save_features(df, OUTPUT_PARQUET, OUTPUT_SCHEMA)

    feature_cols = [c for c in df.columns if c != "is_anomaly"]
    _log_event({
        "stage": "complete",
        "status": "success",
        "total_features": len(feature_cols),
        "total_rows": len(df),
    })
    log.info("=== Feature Engineering Pipeline COMPLETE — %d features ===", len(feature_cols))
    return df


if __name__ == "__main__":
    df_features = run_feature_engineering()
    feature_cols = [c for c in df_features.columns if c != "is_anomaly"]
    print(
        f"\nFeature engineering complete.\n"
        f"  Output  : {OUTPUT_PARQUET}\n"
        f"  Schema  : {OUTPUT_SCHEMA}\n"
        f"  Rows    : {len(df_features)}\n"
        f"  Features ({len(feature_cols)}): {feature_cols}"
    )
