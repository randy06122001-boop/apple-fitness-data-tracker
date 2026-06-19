"""
analyzer.py – Data analysis engine for parsed Apple Watch health data.

Consumes DataFrames produced by the parser (read from CACHE_DIR Parquet files)
and returns an AnalysisResults dataclass that other modules (AI coach,
Obsidian exporter) can consume directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

import config

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Result container
# ────────────────────────────────────────────────────────────────

@dataclass
class AnalysisResults:
    """Container for every analysis output produced by :func:`analyze`."""

    rolling_averages: dict[str, pd.DataFrame] = field(default_factory=dict)
    """Metric name → DataFrame with date index and one column per window."""

    correlations: dict[str, dict[str, float]] = field(default_factory=dict)
    """Named relationship → {'r': …, 'p': …, 'n': …}."""

    anomalies: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Rows: date, metric, value, rolling_mean, rolling_std, expected_low,
    expected_high, deviation_sigma, severity."""

    trend_summary: dict[str, Any] = field(default_factory=dict)
    """Per-metric trend direction, current vs 90-day avg, recent anomalies."""

    sleep_analysis: dict[str, Any] = field(default_factory=dict)
    """Nightly duration, per-stage breakdowns (when available)."""

    workout_analysis: dict[str, Any] = field(default_factory=dict)
    """Frequency by type, avg duration, weekly training load."""


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

_KEY_METRICS: list[str] = [
    # Cardiovascular
    "resting_heart_rate",
    "heart_rate",
    "hrv",
    "vo2max",
    "walking_heart_rate_avg",
    # Activity
    "step_count",
    "active_energy",
    "exercise_minutes",
    # Sleep (derived)
    "sleep_duration",
    # Body & Environment
    "wrist_temperature",
    "environmental_audio",
    # Ultra-Specific
    "running_power",
    "ground_contact_time",
    "vertical_oscillation",
    "water_temperature",
    "underwater_depth",
]

# Which metrics are "higher is better" vs "lower is better"
_HIGHER_IS_BETTER: set[str] = {
    "hrv", "vo2max", "step_count", "active_energy",
    "sleep_duration", "exercise_minutes", "running_power",
}
_LOWER_IS_BETTER: set[str] = {
    "resting_heart_rate", "walking_heart_rate_avg",
    "ground_contact_time",
}


def _safe_pearsonr(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Compute Pearson *r* between two series, aligning by index.

    Returns a dict with keys ``r``, ``p``, and ``n``.  If fewer than 5
    paired observations exist, returns NaN values instead of unreliable
    statistics.
    """
    combined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(combined)
    if n < 5:
        return {"r": float("nan"), "p": float("nan"), "n": n}
    r_val, p_val = stats.pearsonr(combined["x"], combined["y"])
    return {"r": round(float(r_val), 4), "p": round(float(p_val), 6), "n": n}


def _daily_aggregate(df: pd.DataFrame, value_col: str = "value", agg_func: str = "mean") -> pd.Series:
    """Collapse an intra-day DataFrame to a daily series using the specified aggregation.

    Checks for ``date``, ``startDate``, or ``creationDate`` columns
    (in that priority order).  Returns a Series indexed by date.
    """
    if df.empty or value_col not in df.columns:
        return pd.Series(dtype=float)

    work = df.copy()

    # Find the best date column
    date_col = None
    for candidate in ("date", "startDate", "creationDate"):
        if candidate in work.columns:
            date_col = candidate
            break

    if date_col is not None:
        work["_date"] = pd.to_datetime(work[date_col], errors="coerce").dt.normalize()
        work = work.dropna(subset=["_date"])
        work = work.set_index("_date")
    elif isinstance(work.index, pd.DatetimeIndex):
        pass  # already has a datetime index
    else:
        return pd.Series(dtype=float)

    # Ensure value column is numeric
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    return work[value_col].resample("D").agg(agg_func).dropna()


def _classify_trend(series: pd.Series, window: int = 30) -> str:
    """Classify the recent *window*-day trend as improving/declining/stable.

    Uses a simple linear regression slope normalised by the series std.
    A normalised slope magnitude < 0.10 is considered **stable**.
    """
    recent = series.dropna().tail(window)
    if len(recent) < 7:
        return "insufficient_data"

    x = np.arange(len(recent), dtype=float)
    y = recent.values.astype(float)
    slope, _, _, _, _ = stats.linregress(x, y)

    std = np.std(y)
    if std == 0:
        return "stable"

    norm_slope = slope / std
    if abs(norm_slope) < 0.10:
        return "stable"
    return "increasing" if norm_slope > 0 else "decreasing"


def _trend_direction_label(metric: str, direction: str) -> str:
    """Convert raw direction into a human-friendly label for a given metric.

    For metrics where *higher* is better, ``increasing`` → ``improving``.
    """
    if direction in ("stable", "insufficient_data"):
        return direction
    if metric in _HIGHER_IS_BETTER:
        return "improving" if direction == "increasing" else "declining"
    if metric in _LOWER_IS_BETTER:
        return "improving" if direction == "decreasing" else "declining"
    # Unknown polarity – just echo direction.
    return direction


# ────────────────────────────────────────────────────────────────
# 1. Rolling Averages
# ────────────────────────────────────────────────────────────────

def compute_rolling_averages(
    daily_series: dict[str, pd.Series],
    windows: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Compute rolling averages for every key metric.

    Parameters
    ----------
    daily_series:
        Metric name → daily-mean ``pd.Series`` (DatetimeIndex).
    windows:
        List of window sizes in days.  Defaults to
        :pydata:`config.ROLLING_WINDOWS`.

    Returns
    -------
    dict mapping metric name → DataFrame whose columns are
    ``raw``, ``rolling_7``, ``rolling_30``, ``rolling_90``, etc.
    """
    if windows is None:
        windows = config.ROLLING_WINDOWS

    results: dict[str, pd.DataFrame] = {}
    for metric, series in daily_series.items():
        if series.empty:
            continue
        frame = series.to_frame(name="raw")
        for w in windows:
            frame[f"rolling_{w}"] = series.rolling(window=w, min_periods=max(1, w // 3)).mean()
        results[metric] = frame
    return results


# ────────────────────────────────────────────────────────────────
# 2. Correlations
# ────────────────────────────────────────────────────────────────

def compute_correlations(
    daily_series: dict[str, pd.Series],
    data: dict[str, pd.DataFrame],
) -> dict[str, dict[str, float]]:
    """Identify relationships between health metrics.

    Analyses performed:

    * **workout_intensity_vs_next_day_hrv** – mean workout heart-rate on
      day *d* vs HRV on day *d+1*.
    * **sleep_duration_vs_next_day_rhr** – total sleep on night *d* vs
      RHR on day *d+1*.
    * **evening_workout_vs_sleep_quality** – presence / avg HR of workouts
      after 18:00 vs that night's total sleep duration.
    * **wrist_temp_vs_rhr** – daily wrist-temperature deviation vs daily
      RHR.
    """
    correlations: dict[str, dict[str, float]] = {}

    # ── Workout intensity vs next-day HRV ──
    hrv_series = daily_series.get("hrv", pd.Series(dtype=float))
    workout_df = data.get("workouts", pd.DataFrame())

    if not workout_df.empty and not hrv_series.empty and "duration" in workout_df.columns:
        work = workout_df.copy()
        if "startDate" in work.columns:
            work["date"] = pd.to_datetime(work["startDate"]).dt.normalize()
        elif "date" in work.columns:
            work["date"] = pd.to_datetime(work["date"]).dt.normalize()
        else:
            work["date"] = pd.NaT

        # Use average heart rate if available, else fall back to duration as proxy
        intensity_col = "averageHeartRate" if "averageHeartRate" in work.columns else "duration"
        workout_intensity = work.groupby("date")[intensity_col].mean()
        workout_intensity.index = pd.to_datetime(workout_intensity.index)

        # Shift HRV back by 1 day so we compare workout day → next-day HRV
        hrv_next = hrv_series.copy()
        hrv_next.index = hrv_next.index - pd.Timedelta(days=1)

        correlations["workout_intensity_vs_next_day_hrv"] = _safe_pearsonr(
            workout_intensity, hrv_next,
        )

    # ── Sleep duration vs next-day RHR ──
    sleep_dur = daily_series.get("sleep_duration", pd.Series(dtype=float))
    rhr_series = daily_series.get("resting_heart_rate", pd.Series(dtype=float))

    if not sleep_dur.empty and not rhr_series.empty:
        rhr_next = rhr_series.copy()
        rhr_next.index = rhr_next.index - pd.Timedelta(days=1)
        correlations["sleep_duration_vs_next_day_rhr"] = _safe_pearsonr(
            sleep_dur, rhr_next,
        )

    # ── Evening workout timing vs sleep quality ──
    if not workout_df.empty and not sleep_dur.empty:
        work = workout_df.copy()
        if "startDate" in work.columns:
            work["start_dt"] = pd.to_datetime(work["startDate"])
        elif "date" in work.columns:
            work["start_dt"] = pd.to_datetime(work["date"])
        else:
            work["start_dt"] = pd.NaT

        work = work.dropna(subset=["start_dt"])
        evening = work[work["start_dt"].dt.hour >= 18].copy()
        if not evening.empty:
            evening["date"] = evening["start_dt"].dt.normalize()
            intensity_col = (
                "averageHeartRate" if "averageHeartRate" in evening.columns else "duration"
            )
            eve_intensity = evening.groupby("date")[intensity_col].mean()
            eve_intensity.index = pd.to_datetime(eve_intensity.index)
            correlations["evening_workout_vs_sleep_quality"] = _safe_pearsonr(
                eve_intensity, sleep_dur,
            )

    # ── Wrist temperature deviations vs RHR changes ──
    wrist_temp = daily_series.get("wrist_temperature", pd.Series(dtype=float))
    if not wrist_temp.empty and not rhr_series.empty:
        # Deviation from personal baseline (30-day rolling mean)
        temp_baseline = wrist_temp.rolling(30, min_periods=7).mean()
        temp_deviation = wrist_temp - temp_baseline
        rhr_baseline = rhr_series.rolling(30, min_periods=7).mean()
        rhr_change = rhr_series - rhr_baseline
        correlations["wrist_temp_deviation_vs_rhr_change"] = _safe_pearsonr(
            temp_deviation, rhr_change,
        )

    return correlations


# ────────────────────────────────────────────────────────────────
# 3. Anomaly Detection
# ────────────────────────────────────────────────────────────────

def detect_anomalies(
    daily_series: dict[str, pd.Series],
    threshold: float | None = None,
) -> pd.DataFrame:
    """Flag daily values that deviate significantly from the 30-day rolling mean.

    Parameters
    ----------
    daily_series:
        Metric name → daily-mean ``pd.Series``.
    threshold:
        Number of standard deviations to trigger an anomaly.
        Defaults to :pydata:`config.ANOMALY_STD_THRESHOLD`.

    Returns
    -------
    DataFrame with columns: ``date``, ``metric``, ``value``,
    ``rolling_mean``, ``rolling_std``, ``expected_low``,
    ``expected_high``, ``deviation_sigma``, ``severity``.
    """
    if threshold is None:
        threshold = config.ANOMALY_STD_THRESHOLD

    records: list[dict[str, Any]] = []

    for metric, series in daily_series.items():
        if series.empty or len(series) < 10:
            continue

        rolling_mean = series.rolling(30, min_periods=7).mean()
        rolling_std = series.rolling(30, min_periods=7).std()

        for date, value in series.items():
            mu = rolling_mean.get(date)
            sigma = rolling_std.get(date)
            if pd.isna(mu) or pd.isna(sigma) or sigma == 0:
                continue

            deviation = abs(value - mu) / sigma
            if deviation >= threshold:
                severity = (
                    "critical" if deviation >= threshold + 1.5
                    else "high" if deviation >= threshold + 0.5
                    else "moderate"
                )
                records.append({
                    "date": date,
                    "metric": metric,
                    "value": round(float(value), 2),
                    "rolling_mean": round(float(mu), 2),
                    "rolling_std": round(float(sigma), 2),
                    "expected_low": round(float(mu - threshold * sigma), 2),
                    "expected_high": round(float(mu + threshold * sigma), 2),
                    "deviation_sigma": round(float(deviation), 2),
                    "severity": severity,
                })

    if not records:
        return pd.DataFrame(columns=[
            "date", "metric", "value", "rolling_mean", "rolling_std",
            "expected_low", "expected_high", "deviation_sigma", "severity",
        ])
    return pd.DataFrame(records).sort_values("date", ascending=False).reset_index(drop=True)


# ────────────────────────────────────────────────────────────────
# 4. Trend Summary
# ────────────────────────────────────────────────────────────────

def build_trend_summary(
    daily_series: dict[str, pd.Series],
    anomalies_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a structured trend summary for every key metric.

    Returns
    -------
    dict keyed by metric name, each value being::

        {
            "direction": "improving" | "declining" | "stable" | "insufficient_data",
            "current_value": <float>,
            "avg_90d": <float>,
            "pct_vs_90d": <float>,       # percent difference
            "recent_anomalies": [...]     # anomalies in last 14 days
        }
    """
    summary: dict[str, Any] = {}
    cutoff_14d = pd.Timestamp.now().normalize() - pd.Timedelta(days=14)
    if not anomalies_df.empty and hasattr(anomalies_df["date"].dtype, "tz") and anomalies_df["date"].dtype.tz is not None:
        cutoff_14d = cutoff_14d.tz_localize(anomalies_df["date"].dtype.tz)

    for metric, series in daily_series.items():
        if series.empty:
            summary[metric] = {
                "direction": "insufficient_data",
                "current_value": None,
                "avg_90d": None,
                "pct_vs_90d": None,
                "recent_anomalies": [],
            }
            continue

        raw_direction = _classify_trend(series, window=30)
        direction = _trend_direction_label(metric, raw_direction)

        current = float(series.dropna().iloc[-1])
        tail_90 = series.dropna().tail(90)
        avg_90 = float(tail_90.mean()) if len(tail_90) > 0 else float("nan")
        pct = round((current - avg_90) / avg_90 * 100, 1) if avg_90 else None

        # Recent anomalies for this metric
        recent_anom: list[dict[str, Any]] = []
        if not anomalies_df.empty:
            mask = (anomalies_df["metric"] == metric) & (anomalies_df["date"] >= cutoff_14d)
            for _, row in anomalies_df.loc[mask].iterrows():
                recent_anom.append(row.to_dict())

        summary[metric] = {
            "direction": direction,
            "current_value": round(current, 2),
            "avg_90d": round(avg_90, 2),
            "pct_vs_90d": pct,
            "recent_anomalies": recent_anom,
        }

    return summary


# ────────────────────────────────────────────────────────────────
# 5. Sleep Analysis
# ────────────────────────────────────────────────────────────────

_SLEEP_STAGE_MAP: dict[int, str] = {
    0: "in_bed",
    1: "asleep_unspecified",
    2: "awake",
    3: "core",
    4: "deep",
    5: "rem",
}


def analyze_sleep(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Analyse sleep data, returning nightly duration and stage breakdowns.

    Parameters
    ----------
    data:
        The full parsed-data dict.  Looks for key ``"sleep_analysis"``.

    Returns
    -------
    dict with:
        * ``nightly_duration`` – Series indexed by date (hours).
        * ``stage_breakdown`` – DataFrame (date × stage) in minutes, or
          empty if stage info is unavailable.
        * ``avg_duration_7d`` / ``avg_duration_30d`` – rolling averages.
        * ``stats`` – overall mean, median, std of nightly duration.
    """
    result: dict[str, Any] = {
        "nightly_duration": pd.Series(dtype=float),
        "stage_breakdown": pd.DataFrame(),
        "avg_duration_7d": pd.Series(dtype=float),
        "avg_duration_30d": pd.Series(dtype=float),
        "stats": {},
    }

    sleep_df = data.get("sleep_analysis", pd.DataFrame())
    if sleep_df.empty:
        logger.info("No sleep_analysis data found – skipping sleep analysis.")
        return result

    work = sleep_df.copy()

    # ── Parse timestamps ──
    for col in ("startDate", "endDate"):
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")

    if "startDate" not in work.columns or "endDate" not in work.columns:
        logger.warning("sleep_analysis missing startDate/endDate columns.")
        return result

    work["duration_hrs"] = (
        (work["endDate"] - work["startDate"]).dt.total_seconds() / 3600
    )

    # Assign each record to the *night of* date (the calendar day the
    # sleep session started; sessions starting before 18:00 are attributed
    # to the previous night).
    work["night"] = work["startDate"].dt.normalize()
    early_mask = work["startDate"].dt.hour < 18
    work.loc[early_mask, "night"] = work.loc[early_mask, "night"] - pd.Timedelta(days=1)

    # ── Filter to actual sleep (exclude "in_bed" / "awake" if stage data exists) ──
    has_stages = "value" in work.columns and work["value"].nunique() > 2
    if has_stages:
        # Map integer codes to stage names
        work["stage"] = work["value"].map(_SLEEP_STAGE_MAP).fillna("unknown")
        sleep_stages = work[~work["stage"].isin(["in_bed", "awake"])]
    else:
        sleep_stages = work  # treat everything as generic sleep

    # ── Nightly total duration ──
    nightly = sleep_stages.groupby("night")["duration_hrs"].sum().sort_index()
    nightly.index.name = "date"
    result["nightly_duration"] = nightly

    # ── Stage breakdown ──
    if has_stages:
        stage_minutes = (
            sleep_stages
            .assign(duration_min=sleep_stages["duration_hrs"] * 60)
            .groupby(["night", "stage"])["duration_min"]
            .sum()
            .unstack(fill_value=0)
            .sort_index()
        )
        stage_minutes.index.name = "date"
        result["stage_breakdown"] = stage_minutes

    # ── Rolling averages ──
    if not nightly.empty:
        result["avg_duration_7d"] = nightly.rolling(7, min_periods=3).mean()
        result["avg_duration_30d"] = nightly.rolling(30, min_periods=7).mean()
        result["stats"] = {
            "mean_hrs": round(float(nightly.mean()), 2),
            "median_hrs": round(float(nightly.median()), 2),
            "std_hrs": round(float(nightly.std()), 2),
            "min_hrs": round(float(nightly.min()), 2),
            "max_hrs": round(float(nightly.max()), 2),
            "total_nights": int(len(nightly)),
        }

    return result


# ────────────────────────────────────────────────────────────────
# 6. Workout Analysis
# ────────────────────────────────────────────────────────────────

def analyze_workouts(data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Summarise workout frequency, duration, and weekly training load.

    Parameters
    ----------
    data:
        The full parsed-data dict.  Looks for key ``"workouts"``.

    Returns
    -------
    dict with:
        * ``frequency_by_type`` – Series of workout counts per type.
        * ``avg_duration_by_type`` – Series of mean duration (min) per type.
        * ``weekly_training_load`` – DataFrame with week, total_duration,
          session_count, and estimated load.
        * ``stats`` – overall totals.
    """
    result: dict[str, Any] = {
        "frequency_by_type": pd.Series(dtype=int),
        "avg_duration_by_type": pd.Series(dtype=float),
        "weekly_training_load": pd.DataFrame(),
        "stats": {},
    }

    workout_df = data.get("workouts", pd.DataFrame())
    if workout_df.empty:
        logger.info("No workout data found – skipping workout analysis.")
        return result

    work = workout_df.copy()

    # ── Normalise columns ──
    if "workoutActivityType" in work.columns:
        type_col = "workoutActivityType"
    elif "type" in work.columns:
        type_col = "type"
    else:
        type_col = None

    if "duration" in work.columns:
        work["duration_min"] = pd.to_numeric(work["duration"], errors="coerce")
    elif "totalDuration" in work.columns:
        work["duration_min"] = pd.to_numeric(work["totalDuration"], errors="coerce")
    else:
        work["duration_min"] = float("nan")

    # Parse start date for weekly grouping
    if "startDate" in work.columns:
        work["start_dt"] = pd.to_datetime(work["startDate"], errors="coerce")
    elif "date" in work.columns:
        work["start_dt"] = pd.to_datetime(work["date"], errors="coerce")
    else:
        work["start_dt"] = pd.NaT

    # ── Frequency & duration by type ──
    if type_col:
        # Strip the Apple HK prefix for readability
        work["activity"] = (
            work[type_col]
            .astype(str)
            .str.replace("HKWorkoutActivityType", "", regex=False)
        )
        result["frequency_by_type"] = work["activity"].value_counts().sort_values(ascending=False)
        result["avg_duration_by_type"] = (
            work.groupby("activity")["duration_min"]
            .mean()
            .round(1)
            .sort_values(ascending=False)
        )

    # ── Weekly training load ──
    work = work.dropna(subset=["start_dt"])
    if not work.empty:
        work["week"] = work["start_dt"].dt.to_period("W").apply(lambda p: p.start_time)

        weekly = work.groupby("week").agg(
            total_duration_min=("duration_min", "sum"),
            session_count=("duration_min", "count"),
            avg_hr=("averageHeartRate", "mean") if "averageHeartRate" in work.columns else ("duration_min", "count"),
        ).round(1)

        # Simple training-load proxy: duration × avg HR (TRIMP-like), or
        # just total duration if HR data is unavailable.
        if "averageHeartRate" in work.columns:
            weekly["estimated_load"] = (
                weekly["total_duration_min"] * weekly["avg_hr"] / 100
            ).round(1)
        else:
            weekly["estimated_load"] = weekly["total_duration_min"]

        result["weekly_training_load"] = weekly.sort_index()

    # ── Overall stats ──
    result["stats"] = {
        "total_workouts": int(len(work)),
        "total_duration_hrs": round(float(work["duration_min"].sum()) / 60, 1),
        "avg_duration_min": round(float(work["duration_min"].mean()), 1) if not work["duration_min"].isna().all() else None,
        "unique_types": int(work["activity"].nunique()) if "activity" in work.columns else 0,
    }

    return result


# ────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────

def _build_daily_series(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Convert raw DataFrames into daily-mean Series for each key metric.

    Processes ALL metrics present in the parsed data dict, not just a
    hardcoded list.  Also derives ``sleep_duration`` from sleep analysis.
    """
    series: dict[str, pd.Series] = {}

    # Process every metric key present in the data
    skip_keys = {"sleep_analysis", "workouts", "sleep_duration"}
    
    # Metrics that should be summed rather than averaged
    cumulative_metrics = {
        "step_count", 
        "active_energy", 
        "basal_energy_burned", 
        "exercise_minutes", 
        "distance_walking_running",
        "distance_cycling",
        "flights_climbed",
        "apple_stand_time"
    }
    
    for metric, df in data.items():
        if metric in skip_keys:
            continue
        if not df.empty:
            agg = "sum" if metric in cumulative_metrics else "mean"
            s = _daily_aggregate(df, agg_func=agg)
            if not s.empty:
                series[metric] = s

    # Derive nightly sleep duration from sleep_analysis
    sleep_df = data.get("sleep_analysis", pd.DataFrame())
    if not sleep_df.empty:
        sleep_info = analyze_sleep(data)
        nightly = sleep_info.get("nightly_duration", pd.Series(dtype=float))
        if not nightly.empty:
            nightly.index = pd.to_datetime(nightly.index)
            series["sleep_duration"] = nightly

    return series


def analyze(data: dict[str, pd.DataFrame]) -> AnalysisResults:
    """Run the full analysis pipeline on parsed Apple Health data.

    Parameters
    ----------
    data:
        Dictionary mapping metric keys (matching ``config.METRIC_TYPES``
        keys and ``"workouts"``) to DataFrames.  Each DataFrame is
        expected to have at least ``date`` and ``value`` columns (records)
        or appropriate workout/sleep columns.

    Returns
    -------
    AnalysisResults
        Dataclass containing rolling averages, correlations, anomalies,
        trend summaries, sleep analysis, and workout analysis.
    """
    logger.info("Starting analysis pipeline …")

    # 0. Build daily series for key metrics
    daily_series = _build_daily_series(data)
    logger.info("Built daily series for %d metrics: %s", len(daily_series), list(daily_series.keys()))

    # 1. Rolling averages
    rolling = compute_rolling_averages(daily_series)
    logger.info("Computed rolling averages for %d metrics.", len(rolling))

    # 2. Correlations
    correlations = compute_correlations(daily_series, data)
    logger.info("Computed %d correlation analyses.", len(correlations))

    # 3. Anomaly detection
    anomalies = detect_anomalies(daily_series)
    logger.info("Detected %d anomalies.", len(anomalies))

    # 4. Trend summary
    trend = build_trend_summary(daily_series, anomalies)
    logger.info("Built trend summary for %d metrics.", len(trend))

    # 5. Sleep analysis
    sleep = analyze_sleep(data)
    logger.info("Sleep analysis complete – %d nights.", sleep.get("stats", {}).get("total_nights", 0))

    # 6. Workout analysis
    workouts = analyze_workouts(data)
    logger.info(
        "Workout analysis complete – %d workouts across %d types.",
        workouts.get("stats", {}).get("total_workouts", 0),
        workouts.get("stats", {}).get("unique_types", 0),
    )

    return AnalysisResults(
        rolling_averages=rolling,
        correlations=correlations,
        anomalies=anomalies,
        trend_summary=trend,
        sleep_analysis=sleep,
        workout_analysis=workouts,
    )
