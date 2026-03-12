---
name: feature-engineer
description: Runs when asked to engineer features from clean.parquet.
             Use for: feature creation, scaling, encoding, lag features.
---

# Feature Engineering Agent

## TASK
Read data/processed/clean.parquet.
Engineer domain-appropriate features based on the data type:
  - Numeric: log transform, polynomial features, rolling stats (if temporal)
  - Categorical: target encode + one-hot for low cardinality
  - Temporal: hour, day-of-week, is_weekend, lag_1, lag_7
  - Anomaly signals: z-score, IQR flags, pairwise ratios

## OUTPUT
src/features/engineer.py — documented, type-annotated
data/processed/features.parquet
data/processed/feature_schema.json — {name, dtype, description} per feature

## RULES
Fit all scalers/encoders on train split ONLY.
Assert: output has more columns than input (features were added).
Log feature count to logs/audit.jsonl.
