"""EDA report script for transaction anomaly detection pipeline.

Generates 5 exploratory data analysis charts from the cleaned transaction data
and saves them to reports/figures/.

Usage:
    python reports/eda_report.py
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before other matplotlib imports

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean.parquet"
FIGURES_DIR = ROOT / "reports" / "figures"


def load_data() -> pd.DataFrame:
    """Load the cleaned transaction parquet file.

    Returns:
        DataFrame with transaction records.
    """
    return pd.read_parquet(DATA_PATH)


def plot_amount_distribution(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot histogram of transaction amounts colored by anomaly label.

    Uses a log scale on the x-axis. Normal transactions are shown in blue,
    anomalies in red.

    Args:
        df: DataFrame containing ``amount`` and ``is_anomaly`` columns.
        out_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    normal = df.loc[df["is_anomaly"] == 0, "amount"]
    anomaly = df.loc[df["is_anomaly"] == 1, "amount"]

    bins = 50
    ax.hist(normal, bins=bins, color="steelblue", alpha=0.7, label="Normal")
    ax.hist(anomaly, bins=bins, color="crimson", alpha=0.7, label="Anomaly")

    ax.set_xscale("log")
    ax.set_xlabel("Transaction Amount (log scale)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Transaction Amount Distribution by Anomaly Label", fontsize=14)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out_path = out_dir / "01_amount_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_anomaly_by_category(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot anomaly rate per merchant category as a bar chart sorted descending.

    Args:
        df: DataFrame containing ``merchant_category`` and ``is_anomaly`` columns.
        out_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved PNG file.
    """
    anomaly_rate = (
        df.groupby("merchant_category")["is_anomaly"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    anomaly_rate.columns = ["merchant_category", "anomaly_rate"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        anomaly_rate["merchant_category"],
        anomaly_rate["anomaly_rate"],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Merchant Category", fontsize=12)
    ax.set_ylabel("Anomaly Rate", fontsize=12)
    ax.set_title("Anomaly Rate by Merchant Category (Descending)", fontsize=14)
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.1%}")
    )

    plt.tight_layout()
    out_path = out_dir / "02_anomaly_by_category.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_transaction_hour_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot transaction counts by hour (0-23) as a bar chart with anomalies highlighted.

    Normal and anomaly counts are stacked so anomalies are visually distinguishable.

    Args:
        df: DataFrame containing ``transaction_hour`` and ``is_anomaly`` columns.
        out_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved PNG file.
    """
    hours = list(range(24))

    hour_normal = df[df["is_anomaly"] == 0].groupby("transaction_hour").size()
    hour_anomaly = df[df["is_anomaly"] == 1].groupby("transaction_hour").size()

    normal_counts = [int(hour_normal.get(h, 0)) for h in hours]
    anomaly_counts = [int(hour_anomaly.get(h, 0)) for h in hours]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(hours, normal_counts, color="steelblue", label="Normal", alpha=0.8)
    ax.bar(
        hours,
        anomaly_counts,
        bottom=normal_counts,
        color="crimson",
        label="Anomaly",
        alpha=0.8,
    )

    ax.set_xlabel("Transaction Hour (0–23)", fontsize=12)
    ax.set_ylabel("Number of Transactions", fontsize=12)
    ax.set_title("Transaction Count by Hour with Anomalies Highlighted", fontsize=14)
    ax.set_xticks(hours)
    ax.legend(fontsize=11)

    plt.tight_layout()
    out_path = out_dir / "03_transaction_hour_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_amount_boxplot_by_cardtype(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot boxplot of transaction amount by card type with a log-scaled y-axis.

    Args:
        df: DataFrame containing ``card_type`` and ``amount`` columns.
        out_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved PNG file.
    """
    card_types = sorted(df["card_type"].unique())
    data_by_card = [
        df.loc[df["card_type"] == ct, "amount"].dropna().values
        for ct in card_types
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        data_by_card,
        labels=card_types,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="crimson", linewidth=2),
    )

    ax.set_yscale("log")
    ax.set_xlabel("Card Type", fontsize=12)
    ax.set_ylabel("Transaction Amount (log scale)", fontsize=12)
    ax.set_title("Transaction Amount Distribution by Card Type", fontsize=14)
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out_path = out_dir / "04_amount_boxplot_by_cardtype.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot Pearson correlation heatmap for numeric columns using seaborn.

    Args:
        df: DataFrame with mixed types; numeric columns are auto-selected.
        out_dir: Directory where the PNG will be saved.

    Returns:
        Path to the saved PNG file.
    """
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title("Pearson Correlation Heatmap — Numeric Features", fontsize=14)

    plt.tight_layout()
    out_path = out_dir / "05_correlation_heatmap.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    """Run all 5 EDA charts and save them to reports/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    saved_files = [
        plot_amount_distribution(df, FIGURES_DIR),
        plot_anomaly_by_category(df, FIGURES_DIR),
        plot_transaction_hour_heatmap(df, FIGURES_DIR),
        plot_amount_boxplot_by_cardtype(df, FIGURES_DIR),
        plot_correlation_heatmap(df, FIGURES_DIR),
    ]

    print("EDA report complete. 5 charts saved:")
    for path in saved_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
