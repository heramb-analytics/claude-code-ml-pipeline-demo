"""Transaction data ingestion and validation pipeline.

Reads raw CSV, runs 10 quality assertions, outputs clean.parquet.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
LOGS_DIR = Path("logs")


class DataQualityError(Exception):
    """Raised when a critical data quality assertion fails."""


def load_raw_transactions() -> pd.DataFrame:
    """Load raw transactions CSV from data/raw/.

    Returns:
        DataFrame with raw transaction data.

    Raises:
        FileNotFoundError: If no CSV file found in data/raw/.
    """
    files = list(DATA_RAW.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_RAW}")
    path = files[0]
    logger.info("Loading %s", path)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    logger.info("Loaded %d rows x %d cols", len(df), len(df.columns))
    return df


def run_quality_assertions(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run 10 data quality assertions on the raw dataframe.

    Args:
        df: Raw transaction DataFrame.

    Returns:
        List of assertion result dicts.

    Raises:
        DataQualityError: If any critical assertion fails.
    """
    results: list[dict[str, Any]] = []

    def record(name: str, passed: bool, severity: str, message: str, rows_affected: int = 0) -> None:
        results.append({
            "check_name": name,
            "passed": passed,
            "severity": severity,
            "message": message,
            "rows_affected": rows_affected,
        })
        status = "PASS" if passed else "FAIL"
        logger.info("[%s] %s — %s", status, name, message)
        if not passed and severity == "critical":
            raise DataQualityError(f"Critical check failed: {name} — {message}")

    # 1. Required columns present
    required_cols = [
        "transaction_id", "timestamp", "merchant_id", "merchant_category",
        "card_type", "amount", "num_prev_transactions", "customer_age_years",
        "transaction_hour", "is_international", "is_anomaly",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    record("required_columns", len(missing) == 0, "critical",
           f"Missing: {missing}" if missing else "All required columns present")

    # 2. No duplicate transaction IDs
    n_dupes = df["transaction_id"].duplicated().sum()
    record("no_duplicate_ids", n_dupes == 0, "critical",
           f"{n_dupes} duplicate transaction_id(s) found", int(n_dupes))

    # 3. No null values in key columns
    key_cols = ["transaction_id", "timestamp", "amount", "is_anomaly"]
    null_counts = df[key_cols].isnull().sum()
    total_nulls = int(null_counts.sum())
    record("no_nulls_in_key_cols", total_nulls == 0, "critical",
           f"Nulls found: {null_counts.to_dict()}" if total_nulls else "No nulls in key columns",
           total_nulls)

    # 4. Amount > 0
    non_positive = int((df["amount"] <= 0).sum())
    record("amount_positive", non_positive == 0, "critical",
           f"{non_positive} rows with amount <= 0", non_positive)

    # 5. Amount < 1,000,000 (business rule)
    over_limit = int((df["amount"] >= 1_000_000).sum())
    record("amount_under_1m", over_limit == 0, "critical",
           f"{over_limit} rows with amount >= 1M", over_limit)

    # 6. Valid timestamp range (2020–2030)
    invalid_ts = int(((df["timestamp"].dt.year < 2020) | (df["timestamp"].dt.year > 2030)).sum())
    record("timestamp_range_valid", invalid_ts == 0, "critical",
           f"{invalid_ts} rows with timestamp outside 2020–2030", invalid_ts)

    # 7. is_anomaly is binary (0 or 1)
    invalid_target = int((~df["is_anomaly"].isin([0, 1])).sum())
    record("target_binary", invalid_target == 0, "warning",
           f"{invalid_target} non-binary is_anomaly values", invalid_target)

    # 8. Merchant category in known set
    known_cats = {"grocery", "electronics", "travel", "dining", "retail", "online"}
    unknown_cats = int((~df["merchant_category"].isin(known_cats)).sum())
    record("merchant_category_valid", unknown_cats == 0, "warning",
           f"{unknown_cats} rows with unknown merchant_category", unknown_cats)

    # 9. transaction_hour in [0, 23]
    bad_hours = int(((df["transaction_hour"] < 0) | (df["transaction_hour"] > 23)).sum())
    record("transaction_hour_valid", bad_hours == 0, "warning",
           f"{bad_hours} rows with invalid transaction_hour", bad_hours)

    # 10. Merchant ID format: MER followed by digits
    invalid_merch = int((~df["merchant_id"].str.match(r"^MER\d+$")).sum())
    record("merchant_id_format", invalid_merch == 0, "warning",
           f"{invalid_merch} merchant_ids with invalid format", invalid_merch)

    passed = sum(r["passed"] for r in results)
    logger.info("Quality assertions: %d/%d passed", passed, len(results))
    return results


def clean_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal cleaning and save clean.parquet.

    Args:
        df: Validated raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    # Drop rows with any nulls in required columns
    before = len(df)
    df = df.dropna(subset=["transaction_id", "timestamp", "amount", "is_anomaly"])
    if len(df) < before:
        logger.warning("Dropped %d rows with nulls", before - len(df))

    # Enforce types
    df["amount"] = df["amount"].astype(float)
    df["is_anomaly"] = df["is_anomaly"].astype(int)
    df["is_international"] = df["is_international"].astype(int)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / "clean.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("Saved clean.parquet: %d rows x %d cols → %s", len(df), len(df.columns), out_path)
    return df


def save_quality_report(results: list[dict[str, Any]]) -> None:
    """Persist quality assertion results to logs/quality_report.json.

    Args:
        results: List of assertion result dicts.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_timestamp": datetime.now(tz=None).isoformat(),
        "total_checks": len(results),
        "passed": sum(r["passed"] for r in results),
        "failed": sum(not r["passed"] for r in results),
        "checks": results,
    }

    def _convert(obj: Any) -> Any:
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        raise TypeError(f"Not serializable: {type(obj)}")

    path = LOGS_DIR / "quality_report.json"
    path.write_text(json.dumps(report, indent=2, default=_convert))
    logger.info("Quality report saved to %s", path)


def main() -> None:
    """Run the full ingestion pipeline."""
    df = load_raw_transactions()
    results = run_quality_assertions(df)
    clean_df = clean_and_save(df)
    save_quality_report(results)
    print(f"\nIngestion complete: {len(clean_df)} rows → data/processed/clean.parquet")
    print(f"Quality checks: {sum(r['passed'] for r in results)}/{len(results)} passed")


if __name__ == "__main__":
    main()
