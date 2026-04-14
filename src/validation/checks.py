"""Data validation checks — 12 checks on engineered features."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/processed/features.parquet")
VALIDATION_REPORT_PATH = Path("logs/validation_report.json")


class DataQualityError(Exception):
    """Raised when a critical validation check fails."""


def _check(n: int, name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    """Record and print a single validation check result.

    Args:
        n: Check number.
        name: Check name.
        passed: Result.
        detail: Optional detail string.

    Returns:
        Check result dict.
    """
    icon = "✓" if passed else "✗"
    print(f"   {icon} Check {n}/12: {name} — {'passed' if passed else 'FAILED'}" +
          (f" — {detail}" if detail else ""))
    return {
        "check_num": n, "check_name": name, "passed": passed,
        "detail": detail, "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_validation() -> dict[str, Any]:
    """Run 12 validation checks on feature data.

    Returns:
        Validation report dict.
    """
    Path("logs").mkdir(exist_ok=True)
    df = pd.read_parquet(FEATURES_PATH)
    results: list[dict[str, Any]] = []

    # 1: Row count sufficient
    results.append(_check(1, "row_count_sufficient", len(df) >= 1000, f"{len(df):,} rows"))

    # 2: No all-null columns
    null_cols = [c for c in df.columns if df[c].isnull().all()]
    results.append(_check(2, "no_all_null_columns", len(null_cols) == 0, f"null cols: {null_cols}"))

    # 3: Target column exists
    results.append(_check(3, "target_col_exists", "is_anomaly" in df.columns))

    # 4: Numeric features have finite values
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols].values).sum()
    results.append(_check(4, "no_infinite_values", inf_count == 0, f"{inf_count} inf values"))

    # 5: log_amount non-negative
    results.append(_check(5, "log_amount_non_negative",
                          (df["log_amount"] >= 0).all(), f"min: {df['log_amount'].min():.4f}"))

    # 6: amount_zscore reasonable range
    z_min, z_max = df["amount_zscore"].min(), df["amount_zscore"].max()
    results.append(_check(6, "zscore_range_reasonable",
                          z_min > -50 and z_max < 50, f"range: [{z_min:.2f}, {z_max:.2f}]"))

    # 7: Category codes valid
    valid_codes = set(range(-1, 6))
    invalid = ~df["category_code"].isin(valid_codes)
    results.append(_check(7, "category_codes_valid", not invalid.any(),
                          f"{invalid.sum()} invalid codes"))

    # 8: Anomaly class imbalance check
    anomaly_rate = df["is_anomaly"].mean()
    results.append(_check(8, "class_imbalance_expected",
                          0.001 <= anomaly_rate <= 0.30, f"rate: {anomaly_rate:.2%}"))

    # 9: day_of_week in 0-6
    valid_dow = df["day_of_week"].between(0, 6).all()
    results.append(_check(9, "day_of_week_valid", valid_dow,
                          f"range: [{df['day_of_week'].min()}, {df['day_of_week'].max()}]"))

    # 10: is_weekend binary
    binary_weekend = df["is_weekend"].isin([0, 1]).all()
    results.append(_check(10, "is_weekend_binary", binary_weekend))

    # 11: composite_risk non-negative
    results.append(_check(11, "composite_risk_non_negative",
                          (df["composite_risk"] >= 0).all(),
                          f"min: {df['composite_risk'].min():.2f}"))

    # 12: Feature schema matches expected
    expected_engineered = {
        "day_of_week", "is_weekend", "is_night", "log_amount",
        "amount_zscore", "category_code", "card_type_code",
        "high_frequency_merchant", "amount_risk", "age_risk", "composite_risk",
    }
    actual = set(df.columns)
    missing = expected_engineered - actual
    results.append(_check(12, "engineered_features_present", len(missing) == 0,
                          f"missing: {missing}" if missing else "all present"))

    passed_count = sum(1 for r in results if r["passed"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_checks": 12,
        "passed_checks": passed_count,
        "rows": len(df),
        "checks": results,
    }

    def _ser(obj: Any) -> Any:
        if isinstance(obj, (bool,)): return obj
        if hasattr(obj, "item"): return obj.item()
        return str(obj)

    VALIDATION_REPORT_PATH.write_text(json.dumps(report, indent=2, default=_ser))
    return report


if __name__ == "__main__":
    report = run_validation()
    passed = report["passed_checks"]
    print(f"   ✅ Subagent C done — {passed}/12 validation checks passed")
