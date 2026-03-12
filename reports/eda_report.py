"""EDA Report — Transactions Anomaly Detection.

Generates 5 exploratory data analysis charts from the cleaned transactions dataset.
All charts are saved to reports/figures/ as PNG files.

Usage:
    python reports/eda_report.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "clean.parquet"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "Transactions"
REPORT_DATE = "2026-03-13"
TITLE_SUFFIX = f"{DATASET_NAME} — {REPORT_DATE}"

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    """Load cleaned parquet file.

    Args:
        path: Absolute path to the parquet file.

    Returns:
        DataFrame with transaction records.
    """
    df = pd.read_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Chart 1 — Target distribution
# ---------------------------------------------------------------------------
def plot_target_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart showing class balance of is_anomaly (0 vs 1).

    Args:
        df: Transaction DataFrame.
        out_dir: Directory to save the figure.

    Returns:
        Path to saved PNG.
    """
    out_path = out_dir / "01_target_distribution.png"

    counts = df["is_anomaly"].value_counts().sort_index()
    labels = ["Normal (0)", "Anomaly (1)"]
    total = counts.sum()
    percentages = (counts / total * 100).round(2)

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=1.5)

    for bar, count, pct in zip(bars, counts.values, percentages.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({pct}%)",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_title(f"Target Distribution — {TITLE_SUFFIX}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Class", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Chart 2 — Feature correlations heatmap
# ---------------------------------------------------------------------------
def plot_feature_correlations(df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of the Pearson correlation matrix for all numeric columns.

    Args:
        df: Transaction DataFrame.
        out_dir: Directory to save the figure.

    Returns:
        Path to saved PNG.
    """
    out_path = out_dir / "02_feature_correlations.png"

    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show lower triangle + diagonal
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title(f"Feature Correlation Matrix — {TITLE_SUFFIX}", fontsize=16, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Chart 3 — Missing values
# ---------------------------------------------------------------------------
def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar chart of null percentage per column.

    Args:
        df: Transaction DataFrame.
        out_dir: Directory to save the figure.

    Returns:
        Path to saved PNG.
    """
    out_path = out_dir / "03_missing_values.png"

    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#DD8452" if v > 0 else "#4C72B0" for v in null_pct.values]
    bars = ax.bar(null_pct.index, null_pct.values, color=colors, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, null_pct.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title(f"Missing Values per Column — {TITLE_SUFFIX}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Column", fontsize=13)
    ax.set_ylabel("Missing %", fontsize=13)
    ax.set_ylim(0, max(null_pct.max() * 1.2, 5))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Chart 4 — Amount distribution (log1p)
# ---------------------------------------------------------------------------
def plot_amount_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Histogram of log1p(amount) with separate colours for normal vs anomaly.

    Args:
        df: Transaction DataFrame.
        out_dir: Directory to save the figure.

    Returns:
        Path to saved PNG.
    """
    out_path = out_dir / "04_amount_distribution.png"

    df = df.copy()
    df["log_amount"] = np.log1p(df["amount"])

    fig, ax = plt.subplots(figsize=(12, 8))

    normal = df.loc[df["is_anomaly"] == 0, "log_amount"]
    anomaly = df.loc[df["is_anomaly"] == 1, "log_amount"]

    bins = np.linspace(df["log_amount"].min(), df["log_amount"].max(), 50)

    ax.hist(normal, bins=bins, alpha=0.65, color="#4C72B0", label="Normal (0)", edgecolor="white")
    ax.hist(anomaly, bins=bins, alpha=0.75, color="#DD8452", label="Anomaly (1)", edgecolor="white")

    ax.set_title(f"log1p(Amount) Distribution — {TITLE_SUFFIX}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("log1p(Amount)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Chart 5 — Temporal trends by hour
# ---------------------------------------------------------------------------
def plot_temporal_trends(df: pd.DataFrame, out_dir: Path) -> Path:
    """Line chart of transaction count by hour of day, coloured by anomaly vs normal.

    Args:
        df: Transaction DataFrame.
        out_dir: Directory to save the figure.

    Returns:
        Path to saved PNG.
    """
    out_path = out_dir / "05_temporal_trends.png"

    hourly = (
        df.groupby(["transaction_hour", "is_anomaly"])
        .size()
        .reset_index(name="count")
    )

    normal_h = hourly[hourly["is_anomaly"] == 0].set_index("transaction_hour")["count"]
    anomaly_h = hourly[hourly["is_anomaly"] == 1].set_index("transaction_hour")["count"]

    # Reindex to ensure all 24 hours present
    all_hours = pd.RangeIndex(0, 24)
    normal_h = normal_h.reindex(all_hours, fill_value=0)
    anomaly_h = anomaly_h.reindex(all_hours, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(all_hours, normal_h.values, marker="o", color="#4C72B0", linewidth=2.5,
            markersize=6, label="Normal (0)")
    ax.plot(all_hours, anomaly_h.values, marker="s", color="#DD8452", linewidth=2.5,
            markersize=6, label="Anomaly (1)", linestyle="--")

    ax.fill_between(all_hours, normal_h.values, alpha=0.15, color="#4C72B0")
    ax.fill_between(all_hours, anomaly_h.values, alpha=0.20, color="#DD8452")

    ax.set_title(f"Transaction Count by Hour of Day — {TITLE_SUFFIX}", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Hour of Day (0–23)", fontsize=13)
    ax.set_ylabel("Transaction Count", fontsize=13)
    ax.set_xticks(range(0, 24))
    ax.legend(fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}  ({out_path.stat().st_size:,} bytes)")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run all 5 EDA charts and confirm file sizes."""
    print(f"Loading data from {DATA_PATH} ...")
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns.\n")

    saved_paths = [
        plot_target_distribution(df, FIGURES_DIR),
        plot_feature_correlations(df, FIGURES_DIR),
        plot_missing_values(df, FIGURES_DIR),
        plot_amount_distribution(df, FIGURES_DIR),
        plot_temporal_trends(df, FIGURES_DIR),
    ]

    print("\n--- EDA Report Summary ---")
    all_ok = True
    for p in saved_paths:
        size = p.stat().st_size
        status = "OK" if size > 0 else "EMPTY — ERROR"
        if size == 0:
            all_ok = False
        print(f"  {p.name}: {size:,} bytes  [{status}]")

    if all_ok:
        print(f"\nAll 5 PNGs saved successfully to {FIGURES_DIR}")
    else:
        raise RuntimeError("One or more output files are empty. Check chart generation.")


if __name__ == "__main__":
    main()
