"""
Obsidian Vault Exporter for Apple Watch Ultra Data Analyzer.

Exports analysis results and AI coaching insights to an Obsidian vault
as richly-formatted Markdown files with YAML frontmatter, [[wiki-links]],
#tags, and callout blocks.

Folder structure created under config.OBSIDIAN_VAULT_PATH / config.OBSIDIAN_HEALTH_FOLDER:
    Summaries/     – weekly and monthly summary notes
    Metrics/       – per-metric tracking pages
    AI_Coaching/   – AI recommendations log
    Anomalies/     – flagged anomaly reports
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


def _get_anomalies_list(analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    import pandas as pd
    anomalies = analysis_results.get("anomalies")
    if anomalies is None:
        return []
    if isinstance(anomalies, pd.DataFrame):
        if anomalies.empty:
            return []
        records = anomalies.to_dict(orient="records")
        for r in records:
            if "deviation_sigma" in r and "deviation" not in r:
                r["deviation"] = r["deviation_sigma"]
        return records
    elif isinstance(anomalies, dict):
        if not anomalies:
            return []
        if "date" in anomalies:
            df = pd.DataFrame(anomalies)
            records = df.to_dict(orient="records")
            for r in records:
                if "deviation_sigma" in r and "deviation" not in r:
                    r["deviation"] = r["deviation_sigma"]
            return records
        records = []
        for metric, entries in anomalies.items():
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        entry_copy = entry.copy()
                        entry_copy["metric"] = metric
                        records.append(entry_copy)
        return records
    elif isinstance(anomalies, list):
        return anomalies
    return []

# ──────────────────────────────────────────────
# PATH HELPERS
# ──────────────────────────────────────────────

_BASE_DIR: Path = config.OBSIDIAN_VAULT_PATH / config.OBSIDIAN_HEALTH_FOLDER

SUMMARIES_DIR: Path = _BASE_DIR / "Summaries"
METRICS_DIR: Path = _BASE_DIR / "Metrics"
CHARTS_DIR: Path = METRICS_DIR / "Charts"
AI_COACHING_DIR: Path = _BASE_DIR / "AI_Coaching"
ANOMALIES_DIR: Path = _BASE_DIR / "Anomalies"

_ALL_DIRS: List[Path] = [SUMMARIES_DIR, METRICS_DIR, CHARTS_DIR, AI_COACHING_DIR, ANOMALIES_DIR]


def _ensure_directories() -> None:
    """Create the full folder hierarchy if it doesn't already exist."""
    for directory in _ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", directory)


# ──────────────────────────────────────────────
# TREND HELPERS
# ──────────────────────────────────────────────

_TREND_EMOJI = {
    "up": "⬆️",
    "down": "⬇️",
    "flat": "➡️",
}


def _trend_arrow(trend: str) -> str:
    """Return an emoji arrow for the given trend direction string.

    Recognised inputs (case-insensitive): up, down, flat/stable/unchanged.
    Falls back to ➡️ for anything unexpected.
    """
    key = trend.strip().lower()
    if key in ("up", "increasing", "rising"):
        return _TREND_EMOJI["up"]
    if key in ("down", "decreasing", "falling"):
        return _TREND_EMOJI["down"]
    return _TREND_EMOJI["flat"]


def _metric_display_name(raw_key: str) -> str:
    """Convert a snake_case metric key to a human-readable title.

    Example: ``resting_heart_rate`` → ``Resting Heart Rate``
    """
    return raw_key.replace("_", " ").title()


def _sanitise_filename(name: str) -> str:
    """Strip characters that are unsafe for filenames on Windows / macOS."""
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


# ──────────────────────────────────────────────
# SAFE FILE I/O
# ──────────────────────────────────────────────

def safe_append(filepath: Path, content: str) -> None:
    """Safely append to a file, ensuring there's a separating newline."""
    if not filepath.exists():
        _write_full(filepath, content)
        return
    current = filepath.read_text(encoding="utf-8")
    if not current.endswith("\n"):
        content = "\n" + content
    with filepath.open("a", encoding="utf-8") as f:
        f.write(content)


def _write_full(filepath: Path, content: str) -> None:
    """Overwrite *filepath* entirely with *content* (used for fresh notes)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        filepath.write_text(content, encoding="utf-8")
        logger.info("Wrote file: %s", filepath)
    except OSError:
        logger.exception("Failed to write to %s", filepath)
        raise


def _has_date_row(content: str, date: str) -> bool:
    return bool(re.search(r"^\|\s*" + re.escape(date) + r"\s*\|", content, re.MULTILINE))

def _replace_date_row(content: str, date: str, new_row: str) -> str:
    pattern = r"^\|\s*" + re.escape(date) + r"\s*\|.*$"
    return re.sub(pattern, new_row, content, count=1, flags=re.MULTILINE)

def _has_coaching_entry(content: str, date: str) -> bool:
    return f"## {date}" in content

def _replace_coaching_entry(content: str, date: str, new_entry: str) -> str:
    # Match from ## YYYY-MM-DD up to the next ---
    pattern = r"##\s*" + re.escape(date) + r"\n.*?^---$"
    return re.sub(pattern, new_entry.strip(), content, count=1, flags=re.MULTILINE | re.DOTALL)

# ──────────────────────────────────────────────
# YAML FRONTMATTER
# ──────────────────────────────────────────────

def _frontmatter(fields: Dict[str, Any]) -> str:
    """Build a YAML frontmatter block from a flat dictionary.

    Lists are rendered in ``[inline, flow]`` style to keep the block compact.
    """
    lines: List[str] = ["---"]
    for key, value in fields.items():
        if isinstance(value, list):
            items = ", ".join(str(v) for v in value)
            lines.append(f"{key}: [{items}]")
        elif isinstance(value, bool):
            lines.append(f"{key}: {'true' if value else 'false'}")
        else:
            lines.append(f"{key}: {value}")
    lines.append("---")
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────
# WEEKLY SUMMARY
# ──────────────────────────────────────────────

def _extract_metrics_overview(
    analysis_results: Dict[str, Any],
    wow_deltas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Build a Markdown bullet list of key metrics with trend arrows and WoW deltas."""
    metrics: Dict[str, Any] = analysis_results.get("trend_summary", {})
    if not metrics:
        return "_No metric data available._\n"

    lines: List[str] = []
    for key, info in metrics.items():
        if not isinstance(info, dict) or info.get("direction") == "insufficient_data":
            continue
        display = _metric_display_name(key)
        latest = info.get("current_value", "–")
        unit = config.METRIC_UNITS.get(key, "")
        trend = _trend_arrow(info.get("direction", "flat"))
        avg_90 = info.get("avg_90d")

        line = f"- **{display}**: {latest} {unit} {trend}"

        # Add week-over-week delta if available
        if wow_deltas and key in wow_deltas:
            delta_info = wow_deltas[key]
            delta = delta_info.get("delta", 0)
            pct = delta_info.get("pct_change", 0)
            if delta_info.get("direction") == "unchanged":
                line += "  _(unchanged from last week)_"
            else:
                sign = "+" if delta > 0 else ""
                line += f"  _({sign}{delta} {unit} / {sign}{pct}% from last week)_"

        details: List[str] = []
        if avg_90 is not None:
            details.append(f"90-day avg {avg_90}")
        if details:
            line += "  (" + " · ".join(details) + ")"

        # Append a wiki-link to the dedicated metric page.
        metric_page = _sanitise_filename(display)
        line += f"  → [[{metric_page}]]"
        lines.append(line)

    return "\n".join(lines) + "\n"


def _extract_anomalies_section(analysis_results: Dict[str, Any]) -> str:
    """Build a Markdown section listing anomalies detected this week.

    Each anomaly dict is expected to have at least ``metric``, ``date``,
    ``value``, and optionally ``deviation`` / ``message``.
    """
    anomalies = _get_anomalies_list(analysis_results)
    if not anomalies:
        return "> [!tip] All Clear\n> No anomalies detected this week. Keep it up! 💪\n"

    lines: List[str] = ["> [!warning] Anomalies Detected"]
    for a in anomalies:
        metric = _metric_display_name(a.get("metric", "Unknown"))
        date = a.get("date", "?")
        value = a.get("value", "?")
        unit = a.get("unit", "")
        deviation = a.get("deviation")
        msg = a.get("message", "")

        detail = f"> - **{metric}** on {date}: {value} {unit}"
        if deviation is not None:
            detail += f" ({deviation:+.1f}σ)"
        if msg:
            detail += f" – {msg}"
        lines.append(detail)

    return "\n".join(lines) + "\n"


def generate_weekly_summary(
    analysis_results: Dict[str, Any],
    ai_summary: str,
    ai_recommendations: str,
    report_date: str,
    wow_deltas: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """Create (or overwrite) a Weekly Summary note and return its path."""
    filename = f"Weekly Summary {report_date}.md"
    filepath = SUMMARIES_DIR / filename

    fm = _frontmatter({
        "title": f"Weekly Health Summary – {report_date}",
        "date": report_date,
        "tags": ["health", "weekly-review", "ai-coach"],
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    overview = _extract_metrics_overview(analysis_results, wow_deltas)
    anomalies = _extract_anomalies_section(analysis_results)

    # Build WoW comparison section
    wow_section = ""
    if wow_deltas:
        wow_lines = ["## 📈 Week-over-Week Changes\n"]
        for key, delta_info in wow_deltas.items():
            display = _metric_display_name(key)
            delta = delta_info.get("delta", 0)
            pct = delta_info.get("pct_change", 0)
            unit = delta_info.get("unit", "")
            curr = delta_info.get("current_avg", "–")
            prev = delta_info.get("previous_avg", "–")
            direction = delta_info.get("direction", "unchanged")

            if direction == "up":
                emoji = "🔺"
            elif direction == "down":
                emoji = "🔻"
            else:
                emoji = "➖"

            sign = "+" if delta > 0 else ""
            wow_lines.append(
                f"| {display} | {prev} {unit} | {curr} {unit} | {emoji} {sign}{delta} ({sign}{pct}%) |"
            )

        # Insert table header before rows
        header = "| Metric | Last Week | This Week | Change |\n| ------ | --------: | --------: | ------ |"
        wow_lines.insert(1, header)
        wow_lines.append("")
        wow_section = "\n".join(wow_lines) + "\n"

    body_parts: List[str] = [
        fm,
        f"# 📊 Weekly Health Summary – {report_date}\n",
        "## Key Metrics Overview\n",
        overview,
        "---\n",
    ]

    if wow_section:
        body_parts.append(wow_section)
        body_parts.append("---\n")

    body_parts.extend([
        "## 🤖 AI Health Summary\n",
        ai_summary.strip() + "\n",
        "---\n",
        "## 💡 AI Recommendations\n",
        ai_recommendations.strip() + "\n",
        "---\n",
        "## ⚠️ Anomalies\n",
        anomalies,
        "---\n",
        "## Related Notes\n",
        f"- [[Coaching Log]] – full AI coaching history",
        f"- [[Anomaly Report {report_date}]] – detailed anomaly breakdown",
        "",
    ])
    content = "\n".join(body_parts)

    _write_full(filepath, content)
    return filepath


# ──────────────────────────────────────────────
# METRIC TRACKING PAGES
# ──────────────────────────────────────────────

def _build_metric_table_row(date: str, value: Any, unit: str, trend: str) -> str:
    """Return a single Markdown table row for a metric data-point."""
    arrow = _trend_arrow(trend)
    return f"| {date} | {value} | {unit} | {arrow} {trend} |"


def generate_metric_chart(metric_key: str, metric_df: pd.DataFrame, report_date: str) -> Path:
    """Generate a line chart for the metric showing raw values and 7-day average over last 90 days."""
    display = _metric_display_name(metric_key)
    filename = _sanitise_filename(display) + f"_{report_date}.png"
    filepath = CHARTS_DIR / filename
    
    if metric_df.empty or "raw" not in metric_df.columns:
        return None
        
    # Take up to last 90 days
    df = metric_df.tail(90).copy()
    if len(df) < 2:
        return None
        
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df["raw"], marker='o', linestyle='-', alpha=0.4, label='Daily Value', color='#4CAF50')
    if "rolling_7" in df.columns:
        plt.plot(df.index, df["rolling_7"], linestyle='-', linewidth=2, label='7-Day Avg', color='#2E7D32')
        
    plt.title(f"{display} - Last 90 Days")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=120)
    plt.close()
    return filepath

def generate_metric_page(
    metric_key: str,
    metric_info: Dict[str, Any],
    report_date: str,
    metric_df: Optional[pd.DataFrame] = None,
) -> Path:
    """Create or append to a per-metric tracking page.

    If the file already exists, the new data-point row is appended to the
    table; otherwise a fresh page with YAML frontmatter and an initial
    table header is created.

    Parameters
    ----------
    metric_key:
        Snake-case key from ``config.METRIC_TYPES`` (e.g. ``resting_heart_rate``).
    metric_info:
        Dict from trend_summary
    report_date:
        ``YYYY-MM-DD`` date string.
    metric_df:
        DataFrame with daily history (raw, rolling_7, etc).
    """
    display = _metric_display_name(metric_key)
    filename = _sanitise_filename(display) + ".md"
    filepath = METRICS_DIR / filename

    latest = metric_info.get("current_value", "–")
    trend = metric_info.get("direction", "flat")
    avg_90 = metric_info.get("avg_90d")
    unit = config.METRIC_UNITS.get(metric_key, "")

    new_row = _build_metric_table_row(report_date, latest, unit, trend)
    
    # Generate chart
    chart_path = None
    if metric_df is not None and not metric_df.empty:
        chart_path = generate_metric_chart(metric_key, metric_df, report_date)

    if filepath.exists():
        # Read the existing content
        existing = filepath.read_text(encoding="utf-8")

        # Update the "Current Trend" line if present.
        updated = _update_current_trend(existing, trend)

        if _has_date_row(updated, report_date):
            updated = _replace_date_row(updated, report_date, new_row)
            filepath.write_text(updated, encoding="utf-8")
            logger.info("Updated existing data-point row for %s in %s", report_date, filepath)
        else:
            # Append the row just before the end of file.
            safe_append(filepath, "")  # ensure trailing newline
            filepath.write_text(updated.rstrip("\n") + "\n" + new_row + "\n", encoding="utf-8")
            logger.info("Appended data-point to %s", filepath)
    else:
        # Build a brand-new metric page and backfill data log
        fm = _frontmatter({
            "title": display,
            "metric_key": metric_key,
            "tags": ["health", "metric", metric_key.replace("_", "-")],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })

        averages_lines: List[str] = []
        if avg_90 is not None:
            averages_lines.append(f"- **90-day average**: {avg_90} {unit}")
        averages_block = "\n".join(averages_lines) if averages_lines else "_Not enough data yet._"
        
        chart_block = f"![[{chart_path.parent.name}/{chart_path.name}]]\n" if chart_path else ""

        # Backfill history table
        history_rows = []
        if metric_df is not None and not metric_df.empty and "raw" in metric_df.columns:
            df_recent = metric_df.tail(90)
            for date_idx, row in df_recent.iterrows():
                val = round(float(row["raw"]), 2)
                d_str = date_idx.strftime("%Y-%m-%d")
                history_rows.append(_build_metric_table_row(d_str, val, unit, "flat")) # Approximate trend as flat for past

        if not history_rows:
            history_rows.append(new_row)

        body_parts: List[str] = [
            fm,
            f"# {display}\n",
            f"**Current Trend**: {_trend_arrow(trend)} {trend.capitalize()}\n",
            chart_block,
            "## Rolling Averages\n",
            averages_block + "\n",
            "## Data Log\n",
            "| Date | Value | Unit | Trend |",
            "| ---- | ----: | ---- | ----- |",
            *history_rows,
            "",
        ]
        content = "\n".join(body_parts)
        _write_full(filepath, content)

    return filepath


def _update_current_trend(content: str, new_trend: str) -> str:
    """Replace the ``**Current Trend**`` line in an existing metric page."""
    arrow = _trend_arrow(new_trend)
    replacement = f"**Current Trend**: {arrow} {new_trend.capitalize()}"
    # Match the existing trend line regardless of previous emoji / text.
    pattern = r"\*\*Current Trend\*\*:.*"
    updated, n = re.subn(pattern, replacement, content, count=1)
    if n == 0:
        logger.debug("No 'Current Trend' line found – leaving content unchanged.")
    return updated


def generate_all_metric_pages(
    analysis_results: Dict[str, Any],
    report_date: str,
) -> List[Path]:
    """Iterate over all metrics in the results and create/update their pages."""
    metrics: Dict[str, Any] = analysis_results.get("trend_summary", {})
    rolling: Dict[str, pd.DataFrame] = analysis_results.get("rolling_averages", {})
    paths: List[Path] = []
    for metric_key, metric_info in metrics.items():
        if not isinstance(metric_info, dict) or metric_info.get("direction") == "insufficient_data":
            logger.warning("Skipping metric entry: %s", metric_key)
            continue
        try:
            metric_df = rolling.get(metric_key)
            path = generate_metric_page(metric_key, metric_info, report_date, metric_df)
            paths.append(path)
        except Exception:
            logger.exception("Failed to generate metric page for %s", metric_key)
    return paths


# ──────────────────────────────────────────────
# AI COACHING LOG
# ──────────────────────────────────────────────

def generate_coaching_log(
    ai_summary: str,
    ai_recommendations: str,
    report_date: str,
) -> Path:
    """Append a dated entry to the running AI Coaching Log.

    Each call adds a new ``## YYYY-MM-DD`` section so the file grows into
    a chronological history of every coaching run.
    """
    filepath = AI_COACHING_DIR / "Coaching Log.md"

    entry_lines: List[str] = [
        f"## {report_date}\n",
        "### Health Summary\n",
        ai_summary.strip() + "\n",
        "### Recommendations\n",
        ai_recommendations.strip() + "\n",
        f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n",
        "---\n",
    ]
    entry = "\n".join(entry_lines)

    if filepath.exists():
        existing = filepath.read_text(encoding="utf-8")
        if _has_coaching_entry(existing, report_date):
            updated = _replace_coaching_entry(existing, report_date, entry)
            filepath.write_text(updated, encoding="utf-8")
            logger.info("Replaced existing coaching log entry for %s", report_date)
        else:
            safe_append(filepath, entry)
    else:
        fm = _frontmatter({
            "title": "AI Coaching Log",
            "tags": ["health", "ai-coach", "coaching-log"],
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        header = fm + "\n# 🧠 AI Coaching Log\n\nA chronological record of AI-generated health insights.\n\n---\n\n"
        _write_full(filepath, header + entry)

    return filepath


# ──────────────────────────────────────────────
# ANOMALY REPORTS
# ──────────────────────────────────────────────

def generate_anomaly_report(
    analysis_results: Dict[str, Any],
    report_date: str,
) -> Optional[Path]:
    """Create a detailed anomaly report note for the given date.

    Returns ``None`` if there are no anomalies to report.
    """
    anomalies = _get_anomalies_list(analysis_results)
    if not anomalies:
        logger.info("No anomalies to report for %s", report_date)
        return None

    filename = f"Anomaly Report {report_date}.md"
    filepath = ANOMALIES_DIR / filename

    fm = _frontmatter({
        "title": f"Anomaly Report – {report_date}",
        "date": report_date,
        "tags": ["health", "anomaly", "review"],
        "anomaly_count": len(anomalies),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    body_parts: List[str] = [
        fm,
        f"# ⚠️ Anomaly Report – {report_date}\n",
        f"**Total anomalies detected**: {len(anomalies)}\n",
        "| # | Metric | Date | Value | Unit | Deviation | Note |",
        "| - | ------ | ---- | ----: | ---- | --------: | ---- |",
    ]

    for idx, a in enumerate(anomalies, start=1):
        metric = _metric_display_name(a.get("metric", "Unknown"))
        date = a.get("date", "–")
        value = a.get("value", "–")
        unit = a.get("unit", "")
        deviation = a.get("deviation")
        dev_str = f"{deviation:+.2f}σ" if deviation is not None else "–"
        msg = a.get("message", "")
        body_parts.append(f"| {idx} | [[{_sanitise_filename(metric)}\\|{metric}]] | {date} | {value} | {unit} | {dev_str} | {msg} |")

    body_parts.append("")  # trailing newline after table

    # Add per-anomaly callout detail blocks.
    body_parts.append("## Detail Breakdown\n")
    for a in anomalies:
        metric = _metric_display_name(a.get("metric", "Unknown"))
        date = a.get("date", "–")
        value = a.get("value", "–")
        unit = a.get("unit", "")
        deviation = a.get("deviation")
        msg = a.get("message", "")

        callout_type = "warning" if deviation is not None and abs(deviation) >= config.ANOMALY_STD_THRESHOLD else "info"
        body_parts.append(f"> [!{callout_type}] {metric} – {date}")
        body_parts.append(f"> **Recorded value**: {value} {unit}")
        if deviation is not None:
            body_parts.append(f"> **Deviation**: {deviation:+.2f} standard deviations from rolling mean")
        if msg:
            body_parts.append(f"> **Note**: {msg}")
        body_parts.append("")  # blank line between callouts

    body_parts.append("---\n")
    body_parts.append(f"← Back to [[Weekly Summary {report_date}]]\n")

    content = "\n".join(body_parts)
    _write_full(filepath, content)
    return filepath


# ──────────────────────────────────────────────
# MONTHLY ROLLUP
# ──────────────────────────────────────────────

MONTHLY_DIR: Path = _BASE_DIR / "Summaries"  # Reuse Summaries folder


def generate_monthly_summary(
    snapshots: List[Dict[str, Any]],
    report_date: str,
) -> Optional[Path]:
    """Generate a monthly rollup note from the last 4 weekly snapshots.

    Returns None if fewer than 2 snapshots are available.
    """
    if len(snapshots) < 2:
        logger.info("Not enough snapshots for a monthly rollup (need at least 2).")
        return None

    filename = f"Monthly Summary {report_date}.md"
    filepath = MONTHLY_DIR / filename

    fm = _frontmatter({
        "title": f"Monthly Health Summary – {report_date}",
        "date": report_date,
        "tags": ["health", "monthly-review", "rollup"],
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "weeks_covered": len(snapshots),
    })

    # ── Aggregate metrics across the month's snapshots ──
    all_metrics: Dict[str, List[float]] = {}
    for snap in snapshots:
        for metric_key, metric_info in snap.get("metrics", {}).items():
            avg_val = metric_info.get("avg")
            if avg_val is not None:
                all_metrics.setdefault(metric_key, []).append(float(avg_val))

    metric_table_lines = [
        "| Metric | Month Avg | Best Week | Worst Week | Unit |",
        "| ------ | --------: | --------: | ---------: | ---- |",
    ]
    for metric_key, values in all_metrics.items():
        display = _metric_display_name(metric_key)
        unit = config.METRIC_UNITS.get(metric_key, "")
        month_avg = round(sum(values) / len(values), 2)
        best = round(max(values), 2)
        worst = round(min(values), 2)
        metric_table_lines.append(
            f"| {display} | {month_avg} | {best} | {worst} | {unit} |"
        )

    # ── Aggregate totals ──
    total_anomalies = sum(s.get("anomaly_count", 0) for s in snapshots)
    total_workouts = sum(s.get("workout_count", 0) for s in snapshots)
    total_workout_hrs = round(sum(s.get("workout_duration_hrs", 0) for s in snapshots), 1)
    sleep_avgs = [s.get("sleep_avg_hrs") for s in snapshots if s.get("sleep_avg_hrs") is not None]
    avg_sleep = round(sum(sleep_avgs) / len(sleep_avgs), 2) if sleep_avgs else None

    # ── Weekly summary links ──
    week_links = []
    for snap in snapshots:
        week_date = snap.get("week", "?")
        week_links.append(f"- [[Weekly Summary {week_date}]]")

    body_parts: List[str] = [
        fm,
        f"# 📅 Monthly Health Summary – {report_date}\n",
        f"_Covering {len(snapshots)} weekly reports_\n",
        "## Monthly Metrics Overview\n",
        "\n".join(metric_table_lines) + "\n",
        "---\n",
        "## Monthly Totals\n",
        f"- **Total Anomalies Flagged**: {total_anomalies}",
        f"- **Total Workouts**: {total_workouts}",
        f"- **Total Workout Time**: {total_workout_hrs} hrs",
    ]

    if avg_sleep is not None:
        body_parts.append(f"- **Average Sleep**: {avg_sleep} hrs/night")

    body_parts.extend([
        "",
        "---\n",
        "## Weekly Reports\n",
        "\n".join(week_links),
        "",
    ])

    content = "\n".join(body_parts)
    _write_full(filepath, content)
    logger.info("Generated monthly summary: %s", filepath)
    return filepath


# ──────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────

def export_to_obsidian(
    analysis_results: Dict[str, Any],
    ai_summary: str,
    ai_recommendations: str,
    report_date: str,
    wow_deltas: Optional[Dict[str, Dict[str, Any]]] = None,
    monthly_snapshots: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Export all analysis artefacts to the Obsidian vault.

    This is the single entry point that orchestrates every export step:

    1. Ensure the directory tree exists.
    2. Write/overwrite the Weekly Summary note (with WoW deltas).
    3. Create or update per-metric tracking pages.
    4. Append to the AI Coaching Log.
    5. Write an Anomaly Report (if anomalies are present).
    6. Generate a Monthly Summary (if monthly_snapshots is provided).
    """
    logger.info("Starting Obsidian export for %s …", report_date)
    _ensure_directories()

    summary_path = generate_weekly_summary(
        analysis_results, ai_summary, ai_recommendations, report_date, wow_deltas
    )

    metric_paths = generate_all_metric_pages(analysis_results, report_date)

    coaching_path = generate_coaching_log(ai_summary, ai_recommendations, report_date)

    anomaly_path = generate_anomaly_report(analysis_results, report_date)

    monthly_path = None
    if monthly_snapshots:
        monthly_path = generate_monthly_summary(monthly_snapshots, report_date)

    result = {
        "weekly_summary": summary_path,
        "metric_pages": metric_paths,
        "coaching_log": coaching_path,
        "anomaly_report": anomaly_path,
        "monthly_summary": monthly_path,
    }
    logger.info(
        "Obsidian export complete: %d metric pages, anomaly report %s",
        len(metric_paths),
        "written" if anomaly_path else "skipped (none)",
    )
    return result


# ──────────────────────────────────────────────
# CLI CONVENIENCE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    """Quick smoke-test with synthetic data."""
    logging.basicConfig(level=logging.DEBUG)

    sample_results: Dict[str, Any] = {
        "metrics": {
            "resting_heart_rate": {
                "latest": 52,
                "unit": "bpm",
                "trend": "down",
                "7d_avg": 53.2,
                "30d_avg": 54.1,
            },
            "hrv": {
                "latest": 42,
                "unit": "ms",
                "trend": "up",
                "7d_avg": 39.8,
                "30d_avg": 37.5,
            },
            "step_count": {
                "latest": 9432,
                "unit": "steps",
                "trend": "flat",
                "7d_avg": 8900,
                "30d_avg": 8750,
            },
        },
        "anomalies": [
            {
                "metric": "resting_heart_rate",
                "date": "2026-06-10",
                "value": 68,
                "unit": "bpm",
                "deviation": 2.3,
                "message": "Unusually high – check for illness or stress.",
            },
        ],
    }

    sample_summary = (
        "Your cardiovascular fitness continues to improve. Resting heart "
        "rate is trending downward while HRV is climbing — both positive "
        "signs of adaptation to your current training load."
    )

    sample_recs = (
        "1. Consider adding a dedicated Zone-2 session on recovery days.\n"
        "2. Your step count has plateaued — try a post-lunch walk.\n"
        "3. Monitor the elevated resting HR reading from June 10."
    )

    output = export_to_obsidian(
        analysis_results=sample_results,
        ai_summary=sample_summary,
        ai_recommendations=sample_recs,
        report_date="2026-06-12",
    )

    for key, value in output.items():
        print(f"{key}: {value}")
