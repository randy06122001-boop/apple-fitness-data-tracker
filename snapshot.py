"""
snapshot.py – Weekly snapshot persistence and week-over-week comparison.

Saves a structured JSON snapshot after every weekly run so that historical
data accumulates over time.  Provides helpers to load history, retrieve
the previous week's snapshot, and compute week-over-week deltas.

Snapshot file location: ``<OBSIDIAN_VAULT_PATH>/weekly_snapshots.json``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import config

logger = logging.getLogger(__name__)

SNAPSHOT_FILE: Path = config.OBSIDIAN_VAULT_PATH / "weekly_snapshots.json"


# ────────────────────────────────────────────────────────────────
# Build a snapshot from analysis results
# ────────────────────────────────────────────────────────────────

def build_snapshot(
    analysis_results: Dict[str, Any],
    report_date: str,
) -> Dict[str, Any]:
    """Create a snapshot dict from the current analysis run.

    Parameters
    ----------
    analysis_results:
        The dict produced by ``dataclasses.asdict(AnalysisResults)``.
    report_date:
        ISO-8601 date string (``YYYY-MM-DD``).

    Returns
    -------
    dict
        A flat snapshot ready to be appended to the history file.
    """
    trend_summary = analysis_results.get("trend_summary", {})
    rolling_averages = analysis_results.get("rolling_averages", {})
    anomalies = analysis_results.get("anomalies", {})
    sleep_analysis = analysis_results.get("sleep_analysis", {})
    workout_analysis = analysis_results.get("workout_analysis", {})

    # ── Build per-metric summary ──
    metrics_snapshot: Dict[str, Dict[str, Any]] = {}
    for metric_key, trend_info in trend_summary.items():
        if not isinstance(trend_info, dict):
            continue

        current_val = trend_info.get("current_value")
        avg_90 = trend_info.get("avg_90d")
        direction = trend_info.get("direction", "unknown")
        unit = config.METRIC_UNITS.get(metric_key, "")

        # Compute 7-day average from rolling data if available
        rolling_df = rolling_averages.get(metric_key)
        avg_7d = None
        min_val = None
        max_val = None
        if isinstance(rolling_df, pd.DataFrame) and not rolling_df.empty:
            last_7 = rolling_df["raw"].tail(7)
            if not last_7.empty:
                avg_7d = round(float(last_7.mean()), 2)
                min_val = round(float(last_7.min()), 2)
                max_val = round(float(last_7.max()), 2)
        elif isinstance(rolling_df, dict) and "raw" in rolling_df:
            # asdict converts DataFrame to dict
            raw_vals = list(rolling_df["raw"].values())
            if raw_vals:
                last_7 = raw_vals[-7:]
                last_7_clean = [v for v in last_7 if v is not None and not (isinstance(v, float) and v != v)]
                if last_7_clean:
                    avg_7d = round(sum(last_7_clean) / len(last_7_clean), 2)
                    min_val = round(min(last_7_clean), 2)
                    max_val = round(max(last_7_clean), 2)

        metrics_snapshot[metric_key] = {
            "avg": avg_7d if avg_7d is not None else current_val,
            "current": current_val,
            "min": min_val,
            "max": max_val,
            "avg_90d": avg_90,
            "direction": direction,
            "unit": unit,
        }

    # ── Anomaly count ──
    anomaly_count = 0
    if isinstance(anomalies, dict) and "date" in anomalies:
        anomaly_count = len(anomalies["date"])
    elif isinstance(anomalies, list):
        anomaly_count = len(anomalies)

    # ── Workout stats ──
    workout_stats = workout_analysis.get("stats", {})
    workout_count = workout_stats.get("total_workouts", 0)
    workout_duration_hrs = workout_stats.get("total_duration_hrs", 0)

    # ── Sleep stats ──
    sleep_stats = sleep_analysis.get("stats", {})
    sleep_avg_hrs = sleep_stats.get("mean_hrs")

    # ── Run number ──
    existing = load_snapshots()
    run_number = len(existing) + 1

    snapshot = {
        "week": report_date,
        "generated_at": datetime.now().isoformat(),
        "run_number": run_number,
        "metrics": metrics_snapshot,
        "anomaly_count": anomaly_count,
        "workout_count": workout_count,
        "workout_duration_hrs": workout_duration_hrs,
        "sleep_avg_hrs": sleep_avg_hrs,
    }

    return snapshot


# ────────────────────────────────────────────────────────────────
# Save / Load
# ────────────────────────────────────────────────────────────────

def save_snapshot(snapshot: Dict[str, Any]) -> Path:
    """Append a snapshot to the history file and return the file path."""
    existing = load_snapshots()
    existing.append(snapshot)

    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(
        json.dumps(existing, indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "Saved snapshot #%d for week %s → %s",
        snapshot.get("run_number", "?"),
        snapshot.get("week", "?"),
        SNAPSHOT_FILE,
    )
    return SNAPSHOT_FILE


def load_snapshots() -> List[Dict[str, Any]]:
    """Load the full snapshot history. Returns an empty list if no file exists."""
    if not SNAPSHOT_FILE.exists():
        return []
    try:
        data = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        logger.warning("Snapshot file is not a list – returning empty history.")
        return []
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read snapshot file – returning empty history.")
        return []


def get_previous_snapshot() -> Optional[Dict[str, Any]]:
    """Return the most recent snapshot from history, or None if no history."""
    snapshots = load_snapshots()
    if not snapshots:
        return None
    return snapshots[-1]


# ────────────────────────────────────────────────────────────────
# Week-over-Week Deltas
# ────────────────────────────────────────────────────────────────

def compute_wow_deltas(
    current: Dict[str, Any],
    previous: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Compute week-over-week deltas for every metric.

    Parameters
    ----------
    current:
        This week's snapshot dict.
    previous:
        Last week's snapshot dict.

    Returns
    -------
    dict
        Metric key → {delta, pct_change, direction, unit, current_avg, previous_avg}
    """
    deltas: Dict[str, Dict[str, Any]] = {}

    curr_metrics = current.get("metrics", {})
    prev_metrics = previous.get("metrics", {})

    for metric_key, curr_info in curr_metrics.items():
        prev_info = prev_metrics.get(metric_key)
        if prev_info is None:
            continue

        curr_avg = curr_info.get("avg")
        prev_avg = prev_info.get("avg")

        if curr_avg is None or prev_avg is None:
            continue

        try:
            curr_avg = float(curr_avg)
            prev_avg = float(prev_avg)
        except (TypeError, ValueError):
            continue

        delta = round(curr_avg - prev_avg, 2)
        pct = round((delta / prev_avg) * 100, 1) if prev_avg != 0 else 0.0

        if abs(delta) < 0.01:
            direction = "unchanged"
        elif delta > 0:
            direction = "up"
        else:
            direction = "down"

        deltas[metric_key] = {
            "delta": delta,
            "pct_change": pct,
            "direction": direction,
            "unit": curr_info.get("unit", ""),
            "current_avg": curr_avg,
            "previous_avg": prev_avg,
        }

    return deltas


def is_monthly_due(snapshots: List[Dict[str, Any]]) -> bool:
    """Return True if a monthly rollup should be generated (every 4th run)."""
    if not snapshots:
        return False
    latest = snapshots[-1]
    run_number = latest.get("run_number", len(snapshots))
    return run_number % 4 == 0


def get_monthly_snapshots(snapshots: List[Dict[str, Any]], count: int = 4) -> List[Dict[str, Any]]:
    """Return the last `count` snapshots for a monthly rollup."""
    return snapshots[-count:] if len(snapshots) >= count else snapshots
