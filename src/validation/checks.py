"""Data quality validation checks for transaction anomaly detection data.

Runs 12 assertion-style checks against clean.parquet and saves results to
logs/validation_report.json. Critical checks raise DataQualityError on failure;
warning checks log the issue and continue.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
LOGS_DIR = PROJECT_ROOT / "logs"
VALIDATION_REPORT = LOGS_DIR / "validation_report.json"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom exception
# ─────────────────────────────────────────────────────────────────────────────
class DataQualityError(Exception):
    """Raised when a critical data quality check fails.

    Args:
        check_name: Identifier of the failing check.
        message: Human-readable description of the failure.
        rows_affected: Number of rows that triggered the failure.
    """

    def __init__(self, check_name: str, message: str, rows_affected: int = 0) -> None:
        self.check_name = check_name
        self.rows_affected = rows_affected
        super().__init__(f"[{check_name}] {message} (rows_affected={rows_affected})")


# ─────────────────────────────────────────────────────────────────────────────
# Type alias for check results
# ─────────────────────────────────────────────────────────────────────────────
CheckResult = dict[str, Any]

# Known valid merchant categories
KNOWN_MERCHANT_CATEGORIES: frozenset[str] = frozenset(
    {"dining", "grocery", "retail", "travel", "electronics", "online"}
)

# ─────────────────────────────────────────────────────────────────────────────
# CRITICAL CHECKS (1 – 6)
# ─────────────────────────────────────────────────────────────────────────────


def check_required_columns(df: pd.DataFrame) -> CheckResult:
    """Check that all required columns are present in the dataframe.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If any required column is missing.
    """
    check_name = "check_required_columns"
    required = {
        "transaction_id",
        "timestamp",
        "merchant_id",
        "merchant_category",
        "card_type",
        "amount",
        "num_prev_transactions",
        "customer_age_years",
        "transaction_hour",
        "is_international",
        "is_anomaly",
    }
    missing = required - set(df.columns)
    passed = len(missing) == 0
    rows_affected = len(df) if not passed else 0
    message = (
        "All required columns present."
        if passed
        else f"Missing columns: {sorted(missing)}"
    )
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": rows_affected,
    }
    if not passed:
        raise DataQualityError(check_name, message, rows_affected)
    return result


def check_dtypes(df: pd.DataFrame) -> CheckResult:
    """Check that amount is float and is_anomaly is integer.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If dtype constraints are violated.
    """
    check_name = "check_dtypes"
    issues: list[str] = []

    if not pd.api.types.is_float_dtype(df["amount"]):
        issues.append(f"'amount' expected float, got {df['amount'].dtype}")

    if not pd.api.types.is_integer_dtype(df["is_anomaly"]):
        issues.append(f"'is_anomaly' expected int, got {df['is_anomaly'].dtype}")

    passed = len(issues) == 0
    rows_affected = 0 if passed else len(df)
    message = "Dtype constraints satisfied." if passed else "; ".join(issues)
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": rows_affected,
    }
    if not passed:
        raise DataQualityError(check_name, message, rows_affected)
    return result


def check_no_nulls_key_cols(df: pd.DataFrame) -> CheckResult:
    """Check there are no null values in transaction_id, amount, or is_anomaly.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If any null is found in a key column.
    """
    check_name = "check_no_nulls_key_cols"
    key_cols = ["transaction_id", "amount", "is_anomaly"]
    null_counts = {col: int(df[col].isnull().sum()) for col in key_cols}
    total_nulls = sum(null_counts.values())
    passed = total_nulls == 0
    message = (
        "No nulls in key columns."
        if passed
        else f"Null counts — {null_counts}"
    )
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": total_nulls,
    }
    if not passed:
        raise DataQualityError(check_name, message, total_nulls)
    return result


def check_amount_positive(df: pd.DataFrame) -> CheckResult:
    """Check that all transaction amounts are strictly greater than zero.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If any amount is <= 0.
    """
    check_name = "check_amount_positive"
    non_positive = int((df["amount"] <= 0).sum())
    passed = non_positive == 0
    message = (
        "All amounts are positive."
        if passed
        else f"{non_positive} row(s) have amount <= 0."
    )
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": non_positive,
    }
    if not passed:
        raise DataQualityError(check_name, message, non_positive)
    return result


def check_timestamp_valid(df: pd.DataFrame) -> CheckResult:
    """Check that all timestamps fall within the valid range 2020–2030.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If any timestamp is outside the expected range.
    """
    check_name = "check_timestamp_valid"
    ts = pd.to_datetime(df["timestamp"])
    lower = pd.Timestamp("2020-01-01")
    upper = pd.Timestamp("2030-12-31 23:59:59")
    out_of_range = int(((ts < lower) | (ts > upper)).sum())
    passed = out_of_range == 0
    message = (
        f"All timestamps in valid range [{lower.date()}, {upper.date()}]."
        if passed
        else f"{out_of_range} timestamp(s) outside 2020-2030."
    )
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": out_of_range,
    }
    if not passed:
        raise DataQualityError(check_name, message, out_of_range)
    return result


def check_categories_valid(df: pd.DataFrame) -> CheckResult:
    """Check that merchant_category only contains values from the known set.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.

    Raises:
        DataQualityError: If any unknown category is found.
    """
    check_name = "check_categories_valid"
    unknown_mask = ~df["merchant_category"].isin(KNOWN_MERCHANT_CATEGORIES)
    rows_affected = int(unknown_mask.sum())
    passed = rows_affected == 0
    unknown_vals = sorted(df.loc[unknown_mask, "merchant_category"].unique().tolist())
    message = (
        f"All merchant categories are valid: {sorted(KNOWN_MERCHANT_CATEGORIES)}."
        if passed
        else f"{rows_affected} row(s) have unknown categories: {unknown_vals}"
    )
    result: CheckResult = {
        "check_name": check_name,
        "passed": passed,
        "severity": "critical",
        "message": message,
        "rows_affected": rows_affected,
    }
    if not passed:
        raise DataQualityError(check_name, message, rows_affected)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# WARNING CHECKS (7 – 12)
# ─────────────────────────────────────────────────────────────────────────────


def check_amount_mean_within_3std(df: pd.DataFrame) -> CheckResult:
    """Check that the mean transaction amount is within 3 standard deviations
    of the expected mid-range (500).

    A very large or very small mean indicates a data drift or processing error.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_amount_mean_within_3std"
    mean_amount = float(df["amount"].mean())
    std_amount = float(df["amount"].std())
    # Expected centre for typical transaction data
    expected_centre = 500.0
    deviation = abs(mean_amount - expected_centre)
    passed = deviation <= 3 * std_amount
    message = (
        f"Mean amount {mean_amount:.2f} is within 3 std ({3 * std_amount:.2f}) "
        f"of expected centre {expected_centre}."
        if passed
        else f"Mean amount {mean_amount:.2f} deviates {deviation:.2f} > "
             f"3*std ({3 * std_amount:.2f}) from expected centre {expected_centre}."
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": 0,
    }


def check_no_infinite_values(df: pd.DataFrame) -> CheckResult:
    """Check that no numeric column contains infinite values.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_no_infinite_values"
    numeric_cols = df.select_dtypes(include="number")
    inf_counts = {
        col: int(np.isinf(numeric_cols[col]).sum())
        for col in numeric_cols.columns
    }
    total_inf = sum(inf_counts.values())
    passed = total_inf == 0
    message = (
        "No infinite values found in numeric columns."
        if passed
        else f"{total_inf} infinite value(s) found: {inf_counts}"
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": total_inf,
    }


def check_no_nan_values(df: pd.DataFrame) -> CheckResult:
    """Check that no column contains NaN values.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_no_nan_values"
    nan_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    total_nan = sum(nan_counts.values())
    passed = total_nan == 0
    message = (
        "No NaN values found in any column."
        if passed
        else f"{total_nan} NaN value(s) found: "
             f"{ {k: v for k, v in nan_counts.items() if v > 0} }"
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": total_nan,
    }


def check_transaction_amount_under_1m(df: pd.DataFrame) -> CheckResult:
    """Check that no transaction amount exceeds or equals 1,000,000.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_transaction_amount_under_1m"
    over_limit = int((df["amount"] >= 1_000_000).sum())
    passed = over_limit == 0
    message = (
        "All transaction amounts are below 1,000,000."
        if passed
        else f"{over_limit} transaction(s) have amount >= 1,000,000."
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": over_limit,
    }


def check_merchant_id_format(df: pd.DataFrame) -> CheckResult:
    """Check that all merchant IDs match the pattern ^MER\\d+.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_merchant_id_format"
    pattern = re.compile(r"^MER\d+$")
    bad_mask = ~df["merchant_id"].astype(str).str.match(pattern)
    rows_affected = int(bad_mask.sum())
    passed = rows_affected == 0
    bad_samples = df.loc[bad_mask, "merchant_id"].unique()[:5].tolist()
    message = (
        "All merchant IDs match pattern ^MER\\d+."
        if passed
        else f"{rows_affected} merchant ID(s) do not match ^MER\\d+. "
             f"Samples: {bad_samples}"
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": rows_affected,
    }


def check_anomaly_rate_reasonable(df: pd.DataFrame) -> CheckResult:
    """Check that the anomaly rate is between 0.1% and 20%.

    Rates outside this range suggest labelling or sampling errors.

    Args:
        df: Input dataframe to validate.

    Returns:
        CheckResult dict with check_name, passed, severity, message, rows_affected.
    """
    check_name = "check_anomaly_rate_reasonable"
    anomaly_rate = float(df["is_anomaly"].mean())
    lower_bound = 0.001  # 0.1%
    upper_bound = 0.20   # 20%
    passed = lower_bound <= anomaly_rate <= upper_bound
    message = (
        f"Anomaly rate {anomaly_rate:.4%} is within acceptable range "
        f"[{lower_bound:.1%}, {upper_bound:.0%}]."
        if passed
        else f"Anomaly rate {anomaly_rate:.4%} is outside acceptable range "
             f"[{lower_bound:.1%}, {upper_bound:.0%}]."
    )
    if not passed:
        warnings.warn(message)
        logger.warning("[%s] %s", check_name, message)
    return {
        "check_name": check_name,
        "passed": passed,
        "severity": "warning",
        "message": message,
        "rows_affected": 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

_ALL_CHECKS = [
    check_required_columns,
    check_dtypes,
    check_no_nulls_key_cols,
    check_amount_positive,
    check_timestamp_valid,
    check_categories_valid,
    check_amount_mean_within_3std,
    check_no_infinite_values,
    check_no_nan_values,
    check_transaction_amount_under_1m,
    check_merchant_id_format,
    check_anomaly_rate_reasonable,
]


def run_all_checks(df: pd.DataFrame | None = None) -> list[CheckResult]:
    """Run all 12 data quality checks and persist results to logs/.

    Args:
        df: Optional dataframe to validate. Loads clean.parquet if not supplied.

    Returns:
        List of CheckResult dicts, one per check (12 total).

    Raises:
        DataQualityError: Re-raised from any critical check that fails.
    """
    if df is None:
        logger.info("Loading %s", CLEAN_PARQUET)
        df = pd.read_parquet(CLEAN_PARQUET)

    logger.info("Running %d checks on %d rows × %d columns.", len(_ALL_CHECKS), *df.shape)
    results: list[CheckResult] = []

    for check_fn in _ALL_CHECKS:
        logger.info("  → %s", check_fn.__name__)
        result = check_fn(df)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        logger.info("    %s — %s", status, result["message"])

    # Persist results
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "failed": sum(1 for r in results if not r["passed"]),
        "checks": results,
    }
    VALIDATION_REPORT.write_text(json.dumps(report, indent=2, default=str))
    logger.info(
        "Validation report saved to %s  [%d/%d passed].",
        VALIDATION_REPORT,
        report["passed"],
        report["total_checks"],
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_checks()

    print()
    print("=" * 60)
    print(f"  DATA QUALITY VALIDATION — {len(results)} checks")
    print("=" * 60)
    passed_count = 0
    failed_count = 0
    for r in results:
        icon = "PASS" if r["passed"] else "FAIL"
        sev = r["severity"].upper()
        print(f"  [{icon}] [{sev:8s}] {r['check_name']}")
        if not r["passed"]:
            print(f"           {r['message']}")
        if r["passed"]:
            passed_count += 1
        else:
            failed_count += 1
    print("-" * 60)
    print(f"  Total  : {len(results)}")
    print(f"  Passed : {passed_count}")
    print(f"  Failed : {failed_count}")
    print("=" * 60)
    print(f"  Report : {VALIDATION_REPORT}")
    print("=" * 60)
