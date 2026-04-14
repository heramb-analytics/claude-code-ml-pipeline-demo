"""Data ingestion and validation for transaction anomaly detection pipeline.

Reads raw CSV, runs 10 quality assertions, self-heals issues, and outputs
a clean Parquet file to data/processed/clean.parquet.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

RAW_PATH = Path("data/raw/transactions.csv")
PROCESSED_PATH = Path("data/processed/clean.parquet")
QUALITY_REPORT_PATH = Path("logs/quality_report.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityError(Exception):
    """Raised when a critical data quality check fails and cannot be auto-healed."""


def _log_check(check_num: int, check_name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    """Log result of a single quality check.

    Args:
        check_num: Sequential check number (1-10).
        check_name: Human-readable name for the check.
        passed: Whether the check passed.
        detail: Optional detail message.

    Returns:
        Check result dict.
    """
    status = "passed" if passed else "FAILED"
    msg = f"   {'✓' if passed else '✗'} Check {check_num}/10: {check_name} — {status}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return {
        "check_num": check_num,
        "check_name": check_name,
        "passed": passed,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_ingestion() -> pd.DataFrame:
    """Run full ingestion + validation pipeline.

    Returns:
        Cleaned DataFrame saved to data/processed/clean.parquet.

    Raises:
        DataQualityError: If critical checks fail after auto-healing attempts.
    """
    Path("logs").mkdir(exist_ok=True)
    Path("data/processed").mkdir(exist_ok=True)

    df = pd.read_csv(RAW_PATH, parse_dates=["timestamp"])
    results: list[dict[str, Any]] = []

    # Check 1: File non-empty
    passed = len(df) > 0
    results.append(_log_check(1, "file_non_empty", passed, f"{len(df)} rows"))
    if not passed:
        raise DataQualityError("Raw file is empty.")

    # Check 2: Required columns present
    required_cols = {
        "transaction_id", "timestamp", "merchant_id",
        "merchant_category", "card_type", "amount",
        "num_prev_transactions", "customer_age_years",
        "transaction_hour", "is_international", "is_anomaly",
    }
    missing = required_cols - set(df.columns)
    passed = len(missing) == 0
    results.append(_log_check(2, "required_columns_present", passed, f"{len(required_cols)} cols present" if not missing else f"missing: {missing}"))
    if not passed:
        raise DataQualityError(f"Required columns missing: {missing}")

    # Check 3: No duplicate transaction IDs
    dup_count = df["transaction_id"].duplicated().sum()
    passed = dup_count == 0
    if not passed:
        print(f"   ✗ Check 3/10: no_duplicate_ids — FAILED — auto-fixing {dup_count} duplicates...")
        df = df.drop_duplicates(subset=["transaction_id"], keep="first")
        dup_count = 0
        passed = True
    results.append(_log_check(3, "no_duplicate_ids", passed, f"{dup_count} duplicates"))

    # Check 4: Amount non-negative
    neg_count = (df["amount"] < 0).sum()
    passed = neg_count == 0
    if not passed:
        print(f"   ✗ Check 4/10: amount_non_negative — FAILED — auto-fixing {neg_count} negatives...")
        df = df[df["amount"] >= 0].copy()
        passed = True
    results.append(_log_check(4, "amount_non_negative", passed, f"{neg_count} negatives removed"))

    # Check 5: Null values below 10% threshold
    null_pct = df.isnull().mean().max()
    passed = null_pct < 0.10
    if not passed:
        print("   ✗ Check 5/10: null_threshold — FAILED — auto-fixing by dropping high-null rows...")
        df = df.dropna(thresh=int(len(df.columns) * 0.8))
        null_pct = df.isnull().mean().max()
        passed = null_pct < 0.10
    results.append(_log_check(5, "null_threshold", passed, f"max null pct: {null_pct:.2%}"))

    # Check 6: Timestamp valid range
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    passed = pd.Timestamp("2020-01-01") <= min_ts and max_ts <= pd.Timestamp("2030-01-01")
    results.append(_log_check(6, "timestamp_valid_range", passed, f"{min_ts} to {max_ts}"))

    # Check 7: is_anomaly binary
    unique_labels = set(df["is_anomaly"].unique())
    passed = unique_labels <= {0, 1}
    if not passed:
        print("   ✗ Check 7/10: is_anomaly_binary — FAILED — auto-fixing by binarising...")
        df["is_anomaly"] = (df["is_anomaly"] != 0).astype(int)
        passed = True
    results.append(_log_check(7, "is_anomaly_binary", passed, f"labels: {sorted(unique_labels)}"))

    # Check 8: Anomaly rate in plausible range
    anomaly_rate = df["is_anomaly"].mean()
    passed = 0.001 <= anomaly_rate <= 0.30
    results.append(_log_check(8, "anomaly_rate_plausible", passed, f"{anomaly_rate:.2%}"))

    # Check 9: Amount within reasonable bounds
    extreme_count = (df["amount"] > 1_000_000).sum()
    passed = extreme_count == 0
    if not passed:
        print(f"   ✗ Check 9/10: amount_bounds — FAILED — capping {extreme_count} extreme values...")
        df.loc[df["amount"] > 1_000_000, "amount"] = 1_000_000
        passed = True
    p999 = df["amount"].quantile(0.999)
    results.append(_log_check(9, "amount_bounds", passed, f"99.9th pct: ${p999:,.2f}"))

    # Check 10: Categorical columns have expected values
    valid_cats = {"grocery", "electronics", "travel", "dining", "retail", "online"}
    actual_cats = set(df["merchant_category"].unique())
    unexpected = actual_cats - valid_cats
    passed = len(unexpected) == 0
    if not passed:
        print(f"   ✗ Check 10/10: valid_categories — FAILED — dropping unexpected categories...")
        df = df[df["merchant_category"].isin(valid_cats)].copy()
        passed = True
    results.append(_log_check(10, "valid_categories", passed, f"cats: {sorted(actual_cats & valid_cats)}"))

    # Save outputs
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"   💾 Saved: data/processed/clean.parquet ({len(df)} rows, {len(df.columns)} cols)")

    passed_count = sum(1 for r in results if r["passed"])
    quality_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": str(RAW_PATH),
        "output_file": str(PROCESSED_PATH),
        "total_checks": 10,
        "passed_checks": passed_count,
        "rows_out": len(df),
        "anomaly_rate": float(df["is_anomaly"].mean()),
        "checks": results,
    }
    import numpy as np
    def _serialize(obj):
        if isinstance(obj, (np.bool_, np.integer)): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        return str(obj)
    QUALITY_REPORT_PATH.write_text(json.dumps(quality_report, indent=2, default=_serialize))
    print("   📄 Saved: logs/quality_report.json")
    return df


if __name__ == "__main__":
    df = run_ingestion()
