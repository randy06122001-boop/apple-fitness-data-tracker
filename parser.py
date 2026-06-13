"""
Streaming XML parser for Apple Health export.xml files.

Uses xml.etree.ElementTree.iterparse for memory-efficient parsing of
multi-gigabyte export files. Extracts Record and Workout elements,
filters to the metric types defined in config.METRIC_TYPES, and caches
the results as Parquet files so subsequent runs skip the expensive parse.

Main entry point: parse_export() → dict[str, pd.DataFrame]
"""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

import config

# ──────────────────────────────────────────────
# Internal constants
# ──────────────────────────────────────────────

# Attributes we extract from <Record> elements.
_RECORD_ATTRS: list[str] = [
    "type",
    "sourceName",
    "unit",
    "value",
    "creationDate",
    "startDate",
    "endDate",
]

# Attributes we extract from <Workout> elements.
_WORKOUT_ATTRS: list[str] = [
    "workoutActivityType",
    "duration",
    "durationUnit",
    "totalDistance",
    "totalDistanceUnit",
    "totalEnergyBurned",
    "totalEnergyBurnedUnit",
    "sourceName",
    "creationDate",
    "startDate",
    "endDate",
]

# Build a reverse lookup: XML type string → short metric name.
_TYPE_TO_METRIC: dict[str, str] = {v: k for k, v in config.METRIC_TYPES.items()}


# ──────────────────────────────────────────────
# Date / value conversion helpers
# ──────────────────────────────────────────────

def _parse_date(raw: str | None) -> pd.Timestamp | None:
    """Convert an Apple Health date string to a pandas Timestamp.

    Apple Health uses the format ``2024-01-15 08:30:00 -0600``.
    Returns *None* for missing or unparseable dates so the caller can
    decide how to handle them.
    """
    if not raw:
        return None
    try:
        return pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None


def _parse_float(raw: str | None) -> float | None:
    """Attempt to convert a string value to float.

    Returns *None* when the value is missing or cannot be converted
    (e.g. categorical values like sleep-analysis states).
    """
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# Cache management
# ──────────────────────────────────────────────

def _parquet_path(metric_name: str) -> Path:
    """Return the expected Parquet cache path for a given metric."""
    return config.CACHE_DIR / f"{metric_name}.parquet"


def check_cache() -> bool:
    """Return True if cached Parquet files exist and are newer than export.xml.

    The check verifies that:
    1. The export.xml file exists (otherwise there's nothing to parse).
    2. Every metric defined in ``config.METRIC_TYPES`` **plus** the
       workouts file has a corresponding ``.parquet`` in ``CACHE_DIR``.
    3. Each Parquet file's modification time is strictly newer than
       the export.xml modification time.

    If any of these conditions fail the cache is considered stale.
    """
    export_path: Path = config.EXPORT_XML_PATH
    if not export_path.exists():
        print(f"[parser] export.xml not found at {export_path}")
        return False

    export_mtime: float = export_path.stat().st_mtime

    # All expected Parquet files (metrics + workouts).
    expected_files: list[Path] = [
        _parquet_path(name) for name in config.METRIC_TYPES
    ]
    expected_files.append(_parquet_path("workouts"))

    for pq_path in expected_files:
        if not pq_path.exists():
            return False
        if pq_path.stat().st_mtime <= export_mtime:
            return False

    print("[parser] Cache is up-to-date – skipping re-parse.")
    return True


# ──────────────────────────────────────────────
# Loading from cache
# ──────────────────────────────────────────────

def _load_from_cache() -> dict[str, pd.DataFrame]:
    """Read every cached Parquet file and return as a dict of DataFrames."""
    frames: dict[str, pd.DataFrame] = {}
    for metric_name in config.METRIC_TYPES:
        pq = _parquet_path(metric_name)
        if pq.exists():
            frames[metric_name] = pd.read_parquet(pq)
    workout_pq = _parquet_path("workouts")
    if workout_pq.exists():
        frames["workouts"] = pd.read_parquet(workout_pq)
    return frames


# ──────────────────────────────────────────────
# Core streaming parser
# ──────────────────────────────────────────────

def _stream_parse(export_path: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Stream-parse the XML and return raw dicts for records and workouts.

    Returns
    -------
    records_by_metric : dict[str, list[dict]]
        Keyed by short metric name (e.g. ``"resting_heart_rate"``).
    workouts : list[dict]
        Raw workout attribute dicts.
    """
    records_by_metric: dict[str, list[dict[str, Any]]] = {
        name: [] for name in config.METRIC_TYPES
    }
    workouts: list[dict[str, Any]] = []

    # Counter for progress reporting.
    total_elements: int = 0
    kept_records: int = 0
    skipped_records: int = 0

    print(f"[parser] Starting streaming parse of {export_path}")
    print(f"[parser] Filtering to {len(config.METRIC_TYPES)} metric types + Workouts")
    t0: float = time.perf_counter()

    # Use iterparse with 'end' events so each element is fully populated
    # before we inspect it.  We call elem.clear() aggressively and
    # delete handled elements from the root to keep memory flat.
    context = ET.iterparse(str(export_path), events=("end",))

    for event, elem in context:
        total_elements += 1

        # ── Progress heartbeat every 500 000 elements ──
        if total_elements % 500_000 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[parser]   … processed {total_elements:,} XML elements "
                f"({elapsed:.1f}s elapsed, {kept_records:,} records kept)"
            )

        # ── Handle <Record> ──
        if elem.tag == "Record":
            record_type: str | None = elem.get("type")
            if record_type and record_type in _TYPE_TO_METRIC:
                metric_name = _TYPE_TO_METRIC[record_type]
                row: dict[str, Any] = {}
                for attr in _RECORD_ATTRS:
                    row[attr] = elem.get(attr)
                records_by_metric[metric_name].append(row)
                kept_records += 1
            else:
                skipped_records += 1
            elem.clear()
            continue

        # ── Handle <Workout> ──
        if elem.tag == "Workout":
            row = {}
            for attr in _WORKOUT_ATTRS:
                row[attr] = elem.get(attr)
            workouts.append(row)
            kept_records += 1
            elem.clear()
            continue

        # Free memory for elements we don't care about.
        elem.clear()

    elapsed = time.perf_counter() - t0
    print(f"[parser] Finished parsing in {elapsed:.1f}s")
    print(f"[parser]   Total XML elements: {total_elements:,}")
    print(f"[parser]   Kept records:       {kept_records:,}")
    print(f"[parser]   Skipped records:    {skipped_records:,}")
    print(f"[parser]   Workouts found:     {len(workouts):,}")

    return records_by_metric, workouts


# ──────────────────────────────────────────────
# DataFrame construction
# ──────────────────────────────────────────────

_DATE_COLUMNS: list[str] = ["creationDate", "startDate", "endDate"]

# Numeric columns that should be cast to float in workout DataFrames.
_WORKOUT_NUMERIC_COLS: list[str] = [
    "duration",
    "totalDistance",
    "totalEnergyBurned",
]


def _build_record_df(rows: list[dict[str, Any]], metric_name: str) -> pd.DataFrame:
    """Convert raw record dicts into a cleaned DataFrame.

    * Dates → ``pd.Timestamp``
    * ``value`` → ``float`` (where possible)
    * Drops rows where *all* date columns are null (completely malformed).
    """
    if not rows:
        return pd.DataFrame(columns=_RECORD_ATTRS)

    df = pd.DataFrame(rows)

    # Convert date columns.
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_date)

    # Convert value to float.
    if "value" in df.columns:
        df["value"] = df["value"].apply(_parse_float)

    # Drop rows where every date is missing (malformed entries).
    date_cols_present = [c for c in _DATE_COLUMNS if c in df.columns]
    if date_cols_present:
        df.dropna(subset=date_cols_present, how="all", inplace=True)

    # Sort by startDate when available.
    if "startDate" in df.columns:
        df.sort_values("startDate", inplace=True, ignore_index=True)

    print(f"[parser]   {metric_name}: {len(df):,} records")
    return df


def _build_workout_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert raw workout dicts into a cleaned DataFrame."""
    if not rows:
        return pd.DataFrame(columns=_WORKOUT_ATTRS)

    df = pd.DataFrame(rows)

    # Convert date columns.
    for col in _DATE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_date)

    # Convert numeric columns.
    for col in _WORKOUT_NUMERIC_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_float)

    # Drop rows where every date is missing.
    date_cols_present = [c for c in _DATE_COLUMNS if c in df.columns]
    if date_cols_present:
        df.dropna(subset=date_cols_present, how="all", inplace=True)

    # Sort by startDate when available.
    if "startDate" in df.columns:
        df.sort_values("startDate", inplace=True, ignore_index=True)

    print(f"[parser]   workouts: {len(df):,} records")
    return df


# ──────────────────────────────────────────────
# Saving to cache
# ──────────────────────────────────────────────

def _save_to_cache(frames: dict[str, pd.DataFrame]) -> None:
    """Persist every DataFrame as a Parquet file in ``config.CACHE_DIR``."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        dest = _parquet_path(name)
        df.to_parquet(dest, index=False)
    print(f"[parser] Saved {len(frames)} Parquet files to {config.CACHE_DIR}")


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def parse_export(*, force: bool = False) -> dict[str, pd.DataFrame]:
    """Parse the Apple Health export.xml and return DataFrames by metric.

    Parameters
    ----------
    force : bool
        If *True*, ignore the cache and re-parse even when it's fresh.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are short metric names (e.g. ``"resting_heart_rate"``,
        ``"workouts"``).  Values are cleaned DataFrames ready for
        analysis.

    Raises
    ------
    FileNotFoundError
        If ``config.EXPORT_XML_PATH`` does not point to an existing file.
    """
    export_path: Path = config.EXPORT_XML_PATH

    # ── Cache shortcut ──
    if not force and check_cache():
        return _load_from_cache()

    # ── Validate input ──
    if not export_path.exists():
        raise FileNotFoundError(
            f"Apple Health export not found at {export_path}. "
            f"Set the HEALTH_EXPORT_PATH environment variable or edit config.py."
        )

    # ── Stream-parse ──
    records_by_metric, raw_workouts = _stream_parse(export_path)

    # ── Build DataFrames ──
    print("[parser] Building DataFrames …")
    frames: dict[str, pd.DataFrame] = {}
    for metric_name, rows in records_by_metric.items():
        frames[metric_name] = _build_record_df(rows, metric_name)

    frames["workouts"] = _build_workout_df(raw_workouts)

    # ── Persist cache ──
    _save_to_cache(frames)

    return frames


# ──────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse Apple Health export.xml")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-parse the XML.",
    )
    args = ap.parse_args()

    data = parse_export(force=args.force)
    print()
    print("=" * 52)
    print("  Summary of parsed data")
    print("=" * 52)
    for name, df in sorted(data.items()):
        date_range = ""
        if "startDate" in df.columns and not df.empty:
            first = df["startDate"].min()
            last = df["startDate"].max()
            if pd.notna(first) and pd.notna(last):
                date_range = f"  ({first:%Y-%m-%d} → {last:%Y-%m-%d})"
        print(f"  {name:30s}  {len(df):>8,} rows{date_range}")
    print("=" * 52)
