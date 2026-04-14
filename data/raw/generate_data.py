"""Generate synthetic transaction dataset for anomaly detection demo."""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10000
ANOMALY_RATE = 0.02

timestamps = pd.date_range("2024-01-01", periods=N, freq="5min")
merchant_ids = [f"MER{str(i).zfill(5)}" for i in np.random.randint(1, 500, N)]
categories = np.random.choice(
    ["grocery", "electronics", "travel", "dining", "retail", "online"],
    N, p=[0.30, 0.10, 0.10, 0.20, 0.20, 0.10]
)
card_types = np.random.choice(["credit", "debit", "prepaid"], N, p=[0.55, 0.35, 0.10])

# Normal transactions
amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=N)
amounts = np.clip(amounts, 1.0, 5000.0)

# Inject anomalies
n_anomalies = int(N * ANOMALY_RATE)
anomaly_idx = np.random.choice(N, n_anomalies, replace=False)
amounts[anomaly_idx] = np.random.uniform(8000, 50000, n_anomalies)

is_fraud = np.zeros(N, dtype=int)
is_fraud[anomaly_idx] = 1

df = pd.DataFrame({
    "transaction_id": [f"TXN{str(i).zfill(8)}" for i in range(N)],
    "timestamp": timestamps,
    "merchant_id": merchant_ids,
    "merchant_category": categories,
    "card_type": card_types,
    "amount": np.round(amounts, 2),
    "num_prev_transactions": np.random.randint(0, 200, N),
    "customer_age_years": np.random.randint(18, 80, N),
    "transaction_hour": timestamps.hour,
    "is_international": np.random.choice([0, 1], N, p=[0.85, 0.15]),
    "is_anomaly": is_fraud,
})

out = Path(__file__).parent / "transactions.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(f"Anomaly rate: {is_fraud.mean():.2%}")
print(df.dtypes)
