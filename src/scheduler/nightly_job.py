"""Nightly APScheduler jobs for automated model maintenance.

Job 1 @ 02:00 daily  — validate new data, retrain if >500 new rows, auto-promote.
Job 2 @ every 6h     — drift check, create JIRA ticket if anomaly rate deviates >20%.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

DRIFT_LOG = LOGS_DIR / "drift_checks.jsonl"
RETRAIN_LOG = LOGS_DIR / "retrain_history.jsonl"

# Baseline anomaly rate from initial training (2.0%)
BASELINE_ANOMALY_RATE = 0.02
DRIFT_THRESHOLD = 0.20  # 20% relative deviation triggers alert
MIN_NEW_ROWS_FOR_RETRAIN = 500


def _log_event(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON-Lines record to a log file.

    Args:
        path: Target .jsonl file path.
        record: Dict to serialize.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _load_model() -> dict[str, Any]:
    """Load the current production model bundle.

    Returns:
        Dict with 'model' and 'feature_names' keys.

    Raises:
        FileNotFoundError: If model pkl does not exist.
    """
    pkl = MODELS_DIR / "pipeline_model.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"Model not found: {pkl}")
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _load_metrics() -> dict[str, Any]:
    """Load current model metrics JSON.

    Returns:
        Metrics dict.
    """
    path = MODELS_DIR / "pipeline_model_metrics.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


# ---------------------------------------------------------------------------
# Job 1: Nightly validation + conditional retrain
# ---------------------------------------------------------------------------

def job_nightly_retrain() -> None:
    """Validate new data and retrain if >500 new rows accumulated.

    Checks for new CSV files in data/raw/ that haven't been processed.
    If new rows exceed MIN_NEW_ROWS_FOR_RETRAIN, triggers full retrain pipeline.
    """
    logger.info("=== Job 1: Nightly validation + retrain check ===")

    # Count new raw data rows
    csv_files = list(DATA_RAW.glob("*.csv"))
    total_raw_rows = 0
    for f in csv_files:
        try:
            total_raw_rows += sum(1 for _ in open(f)) - 1  # subtract header
        except Exception as e:
            logger.warning("Could not count rows in %s: %s", f, e)

    # Count already processed rows
    processed_parquet = DATA_PROCESSED / "clean.parquet"
    processed_rows = 0
    if processed_parquet.exists():
        try:
            processed_rows = len(pd.read_parquet(processed_parquet))
        except Exception as e:
            logger.warning("Could not read clean.parquet: %s", e)

    new_rows = max(0, total_raw_rows - processed_rows)
    logger.info("Raw rows: %d | Processed rows: %d | New rows: %d",
                total_raw_rows, processed_rows, new_rows)

    record: dict[str, Any] = {
        "event": "nightly_retrain_check",
        "timestamp": datetime.now().isoformat(),
        "total_raw_rows": total_raw_rows,
        "processed_rows": processed_rows,
        "new_rows": new_rows,
        "retrain_triggered": False,
    }

    if new_rows >= MIN_NEW_ROWS_FOR_RETRAIN:
        logger.info("New rows (%d) >= threshold (%d) — triggering retrain",
                    new_rows, MIN_NEW_ROWS_FOR_RETRAIN)
        record["retrain_triggered"] = True

        try:
            # Re-run ingest
            result = subprocess.run(
                ["python", str(ROOT / "src" / "data" / "ingest.py")],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode != 0:
                logger.error("Ingest failed:\n%s", result.stderr)
                record["retrain_status"] = "ingest_failed"
                _log_event(RETRAIN_LOG, record)
                return
            logger.info("Ingest complete")

            # Re-run feature engineering
            result = subprocess.run(
                ["python", str(ROOT / "src" / "features" / "engineer.py")],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode != 0:
                logger.error("Feature engineering failed:\n%s", result.stderr)
                record["retrain_status"] = "feature_eng_failed"
                _log_event(RETRAIN_LOG, record)
                return
            logger.info("Feature engineering complete")

            # Re-run model training
            result = subprocess.run(
                ["python", str(ROOT / "src" / "models" / "train.py")],
                capture_output=True, text=True, cwd=ROOT
            )
            if result.returncode != 0:
                logger.error("Training failed:\n%s", result.stderr)
                record["retrain_status"] = "training_failed"
                _log_event(RETRAIN_LOG, record)
                return

            new_metrics = _load_metrics()
            logger.info("Retrain complete — new metrics: %s",
                        new_metrics.get("test", {}))
            record["retrain_status"] = "success"
            record["new_metrics"] = new_metrics.get("test", {})

        except Exception as e:
            logger.exception("Retrain pipeline error: %s", e)
            record["retrain_status"] = "error"
            record["error"] = str(e)
    else:
        logger.info("New rows (%d) below threshold (%d) — skipping retrain",
                    new_rows, MIN_NEW_ROWS_FOR_RETRAIN)
        record["retrain_status"] = "skipped"

    _log_event(RETRAIN_LOG, record)
    logger.info("=== Job 1 complete ===")


# ---------------------------------------------------------------------------
# Job 2: Drift detection every 6 hours
# ---------------------------------------------------------------------------

def job_drift_check() -> None:
    """Check for anomaly rate drift vs baseline. Create JIRA ticket if >20% deviation.

    Loads recent predictions from logs/predictions.jsonl and computes the
    rolling anomaly rate. If it deviates from BASELINE_ANOMALY_RATE by more
    than DRIFT_THRESHOLD (20%), logs a warning and attempts to create a JIRA ticket.
    """
    logger.info("=== Job 2: Drift check ===")

    predictions_log = LOGS_DIR / "predictions.jsonl"
    record: dict[str, Any] = {
        "event": "drift_check",
        "timestamp": datetime.now().isoformat(),
        "baseline_anomaly_rate": BASELINE_ANOMALY_RATE,
        "drift_detected": False,
    }

    if not predictions_log.exists():
        logger.info("No predictions log found — skipping drift check")
        record["status"] = "no_predictions"
        _log_event(DRIFT_LOG, record)
        return

    # Load last 500 predictions
    lines = predictions_log.read_text().strip().split("\n")
    lines = [l for l in lines if l.strip()]
    recent = lines[-500:]

    if not recent:
        logger.info("Empty predictions log — skipping drift check")
        record["status"] = "empty_log"
        _log_event(DRIFT_LOG, record)
        return

    anomaly_count = 0
    total_count = 0
    for line in recent:
        try:
            pred = json.loads(line)
            total_count += 1
            if pred.get("is_anomaly", 0) == 1 or pred.get("prediction") == "ANOMALY":
                anomaly_count += 1
        except json.JSONDecodeError:
            continue

    if total_count == 0:
        logger.info("No valid predictions parsed — skipping")
        record["status"] = "parse_error"
        _log_event(DRIFT_LOG, record)
        return

    current_rate = anomaly_count / total_count
    relative_deviation = abs(current_rate - BASELINE_ANOMALY_RATE) / BASELINE_ANOMALY_RATE

    logger.info(
        "Anomaly rate: %.4f (baseline: %.4f, deviation: %.2f%%)",
        current_rate, BASELINE_ANOMALY_RATE, relative_deviation * 100
    )

    record.update({
        "total_predictions_checked": total_count,
        "anomaly_count": anomaly_count,
        "current_anomaly_rate": round(current_rate, 6),
        "relative_deviation": round(relative_deviation, 4),
        "status": "ok",
    })

    if relative_deviation > DRIFT_THRESHOLD:
        logger.warning(
            "DRIFT DETECTED: rate=%.4f baseline=%.4f deviation=%.1f%% > threshold=%.0f%%",
            current_rate, BASELINE_ANOMALY_RATE,
            relative_deviation * 100, DRIFT_THRESHOLD * 100,
        )
        record["drift_detected"] = True
        record["status"] = "drift_detected"
        _create_drift_jira_ticket(current_rate, relative_deviation, total_count)
    else:
        logger.info("No drift detected — anomaly rate within acceptable range")

    _log_event(DRIFT_LOG, record)
    logger.info("=== Job 2 complete ===")


def _create_drift_jira_ticket(
    current_rate: float,
    relative_deviation: float,
    sample_size: int,
) -> None:
    """Attempt to create a JIRA drift alert ticket via subprocess call.

    Args:
        current_rate: Current observed anomaly rate.
        relative_deviation: Relative deviation from baseline.
        sample_size: Number of predictions analysed.
    """
    try:
        import requests as req

        jira_url = os.environ.get("JIRA_URL", "")
        jira_token = os.environ.get("JIRA_TOKEN", "")
        jira_email = os.environ.get("JIRA_EMAIL", "")
        project_key = os.environ.get("JIRA_PROJECT_KEY", "KAN")

        if not all([jira_url, jira_token, jira_email]):
            logger.warning(
                "JIRA credentials not configured — logging drift alert locally only"
            )
            drift_alert_path = LOGS_DIR / "drift_alerts.jsonl"
            _log_event(drift_alert_path, {
                "event": "drift_alert",
                "timestamp": datetime.now().isoformat(),
                "current_rate": current_rate,
                "relative_deviation": relative_deviation,
                "sample_size": sample_size,
                "action": "would_create_jira_ticket",
            })
            return

        payload = {
            "fields": {
                "project": {"key": project_key},
                "summary": (
                    f"[DRIFT ALERT] Anomaly rate {current_rate:.2%} "
                    f"({relative_deviation*100:.1f}% deviation from baseline)"
                ),
                "description": (
                    f"Drift detected in production model.\n\n"
                    f"- Baseline rate: {BASELINE_ANOMALY_RATE:.2%}\n"
                    f"- Current rate: {current_rate:.2%}\n"
                    f"- Relative deviation: {relative_deviation*100:.1f}%\n"
                    f"- Sample size: {sample_size} predictions\n"
                    f"- Detected at: {datetime.now().isoformat()}\n\n"
                    f"Action required: investigate and consider retraining."
                ),
                "issuetype": {"name": "Bug"},
                "priority": {"name": "High"},
            }
        }
        resp = req.post(
            f"{jira_url}/rest/api/2/issue",
            json=payload,
            auth=(jira_email, jira_token),
            timeout=10,
        )
        if resp.status_code in (200, 201):
            ticket_key = resp.json().get("key", "unknown")
            logger.info("Created JIRA drift ticket: %s", ticket_key)
        else:
            logger.error("JIRA ticket creation failed: %d %s", resp.status_code, resp.text)

    except Exception as e:
        logger.exception("Failed to create JIRA drift ticket: %s", e)


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the APScheduler with both jobs."""
    scheduler = BlockingScheduler(timezone="UTC")

    # Job 1: daily at 02:00 UTC
    scheduler.add_job(
        job_nightly_retrain,
        trigger="cron",
        hour=2,
        minute=0,
        id="nightly_retrain",
        name="Nightly validation + retrain",
        misfire_grace_time=3600,
    )

    # Job 2: every 6 hours
    scheduler.add_job(
        job_drift_check,
        trigger="interval",
        hours=6,
        id="drift_check",
        name="Drift detection",
        misfire_grace_time=600,
    )

    logger.info("Scheduler starting — jobs: nightly_retrain @ 02:00 UTC, drift_check @ 6h interval")
    logger.info("Press Ctrl+C to stop")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
