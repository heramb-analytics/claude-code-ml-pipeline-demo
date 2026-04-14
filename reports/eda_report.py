"""EDA report generation — 5 charts saved to reports/figures/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FEATURES_PATH = Path("data/processed/features.parquet")
FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def run_eda() -> None:
    """Generate 5 EDA charts from features data."""
    df = pd.read_parquet(FEATURES_PATH)

    # Chart 1: Amount distribution by anomaly label
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [(0, "steelblue"), (1, "crimson")]:
        subset = df[df["is_anomaly"] == label]["amount"]
        ax.hist(subset, bins=60, alpha=0.6, color=color,
                label=f"{'Anomaly' if label else 'Normal'} (n={len(subset):,})")
    ax.set_xlabel("Transaction Amount ($)")
    ax.set_ylabel("Count")
    ax.set_title("Transaction Amount Distribution: Normal vs Anomaly")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_amount_distribution.png", dpi=120)
    plt.close(fig)
    print("   ✅ Chart 1/5: amount_distribution saved")

    # Chart 2: Anomaly rate by merchant category
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_rates = df.groupby("merchant_category")["is_anomaly"].mean().sort_values(ascending=False)
    bars = ax.bar(cat_rates.index, cat_rates.values * 100, color=sns.color_palette("muted", len(cat_rates)))
    ax.set_xlabel("Merchant Category")
    ax.set_ylabel("Anomaly Rate (%)")
    ax.set_title("Anomaly Rate by Merchant Category")
    for bar, val in zip(bars, cat_rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_anomaly_by_category.png", dpi=120)
    plt.close(fig)
    print("   ✅ Chart 2/5: anomaly_by_category saved")

    # Chart 3: Transaction hour heatmap
    fig, ax = plt.subplots(figsize=(12, 4))
    hourly = df.groupby(["transaction_hour", "is_anomaly"]).size().unstack(fill_value=0)
    hourly.columns = ["Normal", "Anomaly"]
    hourly_pct = hourly.div(hourly.sum(axis=1), axis=0)
    sns.heatmap(hourly_pct.T, ax=ax, cmap="YlOrRd", fmt=".1%", annot=True,
                linewidths=0.5, cbar_kws={"label": "Proportion"})
    ax.set_title("Transaction Distribution by Hour (Normal vs Anomaly)")
    ax.set_xlabel("Hour of Day")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_hourly_heatmap.png", dpi=120)
    plt.close(fig)
    print("   ✅ Chart 3/5: hourly_heatmap saved")

    # Chart 4: Correlation matrix of numeric features
    fig, ax = plt.subplots(figsize=(10, 8))
    num_cols = ["amount", "log_amount", "num_prev_transactions", "customer_age_years",
                "transaction_hour", "is_international", "composite_risk", "is_anomaly"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm", center=0,
                annot=True, fmt=".2f", square=True, linewidths=0.5)
    ax.set_title("Feature Correlation Matrix")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_correlation_matrix.png", dpi=120)
    plt.close(fig)
    print("   ✅ Chart 4/5: correlation_matrix saved")

    # Chart 5: Composite risk score vs anomaly
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color, name in [(0, "steelblue", "Normal"), (1, "crimson", "Anomaly")]:
        subset = df[df["is_anomaly"] == label]["composite_risk"]
        ax.hist(subset, bins=30, alpha=0.7, color=color,
                label=f"{name} (n={len(subset):,})", density=True)
    ax.set_xlabel("Composite Risk Score")
    ax.set_ylabel("Density")
    ax.set_title("Composite Risk Score Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_risk_score_distribution.png", dpi=120)
    plt.close(fig)
    print("   ✅ Chart 5/5: risk_score_distribution saved")


if __name__ == "__main__":
    run_eda()
    print("   ✅ Subagent B done — 5 charts saved to reports/figures/")
