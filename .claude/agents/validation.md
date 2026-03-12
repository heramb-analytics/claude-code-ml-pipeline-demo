---
name: data-validator
description: Runs when asked to validate data quality on processed data.
             Use for: schema checks, drift detection, business rule validation.
---

# Data Validation Agent

## TASK
Read data/processed/clean.parquet and data/processed/feature_schema.json.
Write 12 assertion-style checks in src/validation/checks.py:
  1-3:  Schema checks (column presence, dtypes, no nulls in key columns)
  4-6:  Range checks (amounts > 0, dates valid, categoricals in known set)
  7-9:  Statistical checks (mean within 3 std, no infinite values, no NaN)
  10-12: Business rules (transaction amount < 1M, merchant IDs valid format)

## OUTPUT
src/validation/checks.py — 12 checks as functions, each returns {passed, message}
logs/validation_report.json — result of all 12 checks

## RULES
Each check: independent, named, returns {check_name, passed, severity, rows_affected}.
Critical checks (1-6): raise DataQualityError on failure.
Warning checks (7-12): log warning, continue.
