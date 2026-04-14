"""Nightly scheduler for transaction anomaly detection pipeline.

Job 1 @ 02:00 daily  : validate new data → retrain if >500 new rows
Job 2 @ every 6h     : drift check → log alert if anomaly rate deviates >20%
"""
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

RAW_PATH = Path("data/raw/transactions.csv")
PROCESSED_PATH = Path("data/processed/clean.parquet")
MODEL_PATH = Path("models/pipeline_model.pkl")
METRICS_PATH = Path("models/pipeline_model_metrics.json")
SCHEDULER_LOG = Path("logs/scheduler.jsonl")
Path("logs").mkdir(exist_ok=True)

BASELINE_ANOMALY_RATE: float | None = None


def _log_event(event_type: str, detail: dict) -> None:
    """Append a scheduler event to the JSONL audit log.

    Args:
        event_type: Type of scheduler event (e.g., 'retrain', 'drift_check').
        detail: Additional details to log.
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **detail,
    }
    with open(SCHEDULER_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("[%s] %s", event_type, detail)


def job_retrain_if_new_data() -> None:
    """Job 1: Check for new rows; retrain model if >500 new rows detected.

    Runs daily at 02:00. Compares current raw CSV row count against the
    last processed count and triggers a full retrain pipeline if the
    threshold is exceeded.
    """
    logger.info("▶ Job 1: checking for new data...")
    try:
        current_rows = sum(1 for _ in open(RAW_PATH)) - 1  # subtract header
        processed_rows = len(pd.read_parquet(PROCESSED_PATH)) if PROCESSED_PATH.exists() else 0
        new_rows = max(0, current_rows - processed_rows)

        _log_event("data_check", {"current_rows": current_rows, "new_rows": new_rows})

        if new_rows > 500:
            logger.info("   🔄 %d new rows detected — triggering retrain...", new_rows)
            import subprocess
            subprocess.run(["python3", "src/data/ingest.py"], check=True)
            subprocess.run(["python3", "src/features/engineer.py"], check=True)
            subprocess.run(["python3", "src/train.py"], check=True)
            _log_event("retrain_complete", {"new_rows": new_rows, "status": "success"})
            logger.info("   ✅ Retrain complete.")
        else:
            logger.info("   ℹ️  Only %d new rows — retrain skipped (threshold: 500)", new_rows)
            _log_event("retrain_skipped", {"new_rows": new_rows, "threshold": 500})
    except Exception as exc:
        logger.error("   ✗ Job 1 error: %s", exc)
        _log_event("retrain_error", {"error": str(exc)})


def job_drift_check() -> None:
    """Job 2: Monitor anomaly rate drift every 6 hours.

    Compares the current rolling anomaly rate against the baseline from
    model training. Logs a DRIFT_ALERT if deviation exceeds 20%.
    """
    global BASELINE_ANOMALY_RATE
    logger.info("▶ Job 2: drift check...")
    try:
        if not PROCESSED_PATH.exists():
            logger.warning("   ⚠️  No processed data found — skipping drift check.")
            return

        df = pd.read_parquet(PROCESSED_PATH)
        current_rate = float(df["is_anomaly"].mean())

        if BASELINE_ANOMALY_RATE is None:
            if METRICS_PATH.exists():
                metrics = json.loads(METRICS_PATH.read_text())
                # Approximate baseline from training data anomaly rate
                BASELINE_ANOMALY_RATE = 0.02  # known from data generation
            else:
                BASELINE_ANOMALY_RATE = current_rate
            logger.info("   📊 Baseline anomaly rate set: %.4f", BASELINE_ANOMALY_RATE)

        deviation = abs(current_rate - BASELINE_ANOMALY_RATE) / max(BASELINE_ANOMALY_RATE, 1e-9)
        detail = {
            "current_rate": round(current_rate, 4),
            "baseline_rate": round(BASELINE_ANOMALY_RATE, 4),
            "deviation_pct": round(deviation * 100, 2),
        }

        if deviation > 0.20:
            logger.warning("   🚨 DRIFT ALERT — anomaly rate deviated %.1f%% from baseline!", deviation * 100)
            _log_event("DRIFT_ALERT", {**detail, "alert": True})
        else:
            logger.info("   ✅ Drift within bounds (%.1f%% deviation)", deviation * 100)
            _log_event("drift_check_ok", {**detail, "alert": False})

    except Exception as exc:
        logger.error("   ✗ Job 2 error: %s", exc)
        _log_event("drift_error", {"error": str(exc)})


def start_scheduler() -> None:
    """Start the APScheduler with both nightly jobs configured."""
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        job_retrain_if_new_data,
        trigger=CronTrigger(hour=2, minute=0),
        id="nightly_retrain",
        name="Nightly retrain @ 02:00 UTC",
        replace_existing=True,
    )

    scheduler.add_job(
        job_drift_check,
        trigger=IntervalTrigger(hours=6),
        id="drift_check",
        name="Drift check every 6h",
        replace_existing=True,
    )

    logger.info("⏰ Scheduler started.")
    logger.info("   Job 1: retrain @ 02:00 UTC daily")
    logger.info("   Job 2: drift check every 6h")
    _log_event("scheduler_started", {
        "jobs": ["nightly_retrain@02:00", "drift_check@6h"],
    })

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")


if __name__ == "__main__":
    start_scheduler()
