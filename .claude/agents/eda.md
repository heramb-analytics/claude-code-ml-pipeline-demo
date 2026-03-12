---
name: eda-analyst
description: Runs when asked to create exploratory data analysis charts.
             Use for: distributions, correlations, class balance, time trends.
---

# EDA Analysis Agent

## TASK
Read data/processed/clean.parquet.
Create exactly 5 matplotlib figures saved to reports/figures/:
  1. figures/01_target_distribution.png — class/value balance
  2. figures/02_feature_correlations.png — heatmap of top 15 features
  3. figures/03_missing_values.png — bar chart of null %
  4. figures/04_amount_distribution.png — histogram of numeric target
  5. figures/05_temporal_trends.png — if temporal column exists, else feature boxplots

## OUTPUT
reports/eda_report.py — reproducible, parameterised script
5 PNG files in reports/figures/

## RULES
All charts: 1200x800px, dpi=150, tight_layout, saved as PNG.
Each chart title must include the dataset name and date.
