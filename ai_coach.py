"""
ai_coach.py — Ollama-powered AI Health Coach for Apple Watch data.

Communicates with a locally-running Ollama instance to generate
natural-language health summaries, actionable recommendations,
and interactive Q&A grounded in the user's real metrics.
"""

from __future__ import annotations

import json
import logging
import textwrap
from typing import Any, Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# SYSTEM PROMPT — defines the LLM's persona and constraints
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = textwrap.dedent("""\
    You are an expert personal health coach and data analyst specialising in
    wearable health data from the Apple Watch Ultra.

    Ground rules:
    • Always reference the specific numbers, dates, and trends provided in the
      user's data context.  Never invent data points.
    • When you identify a trend (improving, declining, stable), cite the actual
      metric values that support your conclusion.
    • Use clear, encouraging language — avoid medical jargon unless you explain it.
    • If data is insufficient to draw a conclusion, say so honestly.
    • You are NOT a licensed physician.  Preface clinical-sounding advice with a
      reminder that the user should consult their healthcare provider for medical
      decisions.
    • Format your output in clean Markdown with headers, bullet points, and bold
      text where it improves readability.
""")

# ──────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────

def _generate_endpoint() -> str:
    """Return the full URL for Ollama's /api/generate endpoint."""
    base = config.OLLAMA_BASE_URL.rstrip("/")
    return f"{base}/api/generate"


def _build_payload(prompt: str, *, system: str = SYSTEM_PROMPT) -> Dict[str, Any]:
    """Build the JSON body expected by the Ollama generate API."""
    return {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
    }


def _call_ollama(prompt: str, *, system: str = SYSTEM_PROMPT) -> str:
    """Send a prompt to Ollama and return the generated text.

    Raises
    ------
    OllamaUnavailableError
        If the Ollama server cannot be reached or returns an HTTP error.
    """
    url = _generate_endpoint()
    payload = _build_payload(prompt, system=system)

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}. "
            "Is `ollama serve` running?"
        ) from exc
    except requests.Timeout as exc:
        raise OllamaUnavailableError(
            "Ollama request timed out after 120 s. The model may be loading "
            "or the prompt may be too large."
        ) from exc
    except requests.HTTPError as exc:
        raise OllamaUnavailableError(
            f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
        ) from exc

    data = response.json()
    return data.get("response", "").strip()


# ──────────────────────────────────────────────────────────────
# CUSTOM EXCEPTION
# ──────────────────────────────────────────────────────────────

class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server is unreachable or unhealthy."""


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def check_ollama_available() -> bool:
    """Ping the Ollama server and return *True* if it responds.

    Prints a user-friendly warning to the console when the server is
    unreachable so callers can degrade gracefully.
    """
    try:
        resp = requests.get(
            f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        logger.info("Ollama is available at %s", config.OLLAMA_BASE_URL)
        return True
    except requests.ConnectionError:
        msg = (
            f"⚠️  Ollama is not reachable at {config.OLLAMA_BASE_URL}.\n"
            "   AI coaching features will be skipped.\n"
            "   To enable them, start Ollama with:  ollama serve"
        )
        print(msg)
        logger.warning(msg)
        return False
    except requests.Timeout:
        msg = (
            f"⚠️  Ollama at {config.OLLAMA_BASE_URL} timed out.\n"
            "   AI coaching features will be skipped."
        )
        print(msg)
        logger.warning(msg)
        return False
    except requests.HTTPError as exc:
        msg = (
            f"⚠️  Ollama returned an error: {exc}\n"
            "   AI coaching features will be skipped."
        )
        print(msg)
        logger.warning(msg)
        return False


# ──────────────────────────────────────────────────────────────
# PROMPT FORMATTING HELPERS
# ──────────────────────────────────────────────────────────────

def _format_trends(analysis_results: Dict[str, Any]) -> str:
    """Convert the ``trend_summary`` section into a human-readable block."""
    trends: Dict[str, Any] = analysis_results.get("trend_summary", {})
    if not trends:
        return "No trend data available."

    lines: List[str] = []
    for metric, info in trends.items():
        if isinstance(info, dict):
            direction = info.get("direction", "unknown")
            current = info.get("current_value")
            avg_90 = info.get("avg_90d")
            pct_change = info.get("pct_vs_90d")

            part = f"• **{metric}**: trend is **{direction}**"
            if current is not None:
                part += f" (current: {current:.1f}"
                if avg_90 is not None:
                    part += f" vs 90d avg: {avg_90:.1f}"
                part += ")"
            if pct_change is not None:
                part += f", {pct_change:+.1f}% vs 90d avg"
            lines.append(part)
        else:
            lines.append(f"• **{metric}**: {info}")
    return "\n".join(lines)


def _format_anomalies(analysis_results: Dict[str, Any]) -> str:
    """Convert the ``anomalies`` section into a human-readable block."""
    import pandas as pd
    anomalies = analysis_results.get("anomalies")
    
    if anomalies is None:
        return "No anomalies detected."
        
    if isinstance(anomalies, pd.DataFrame):
        if anomalies.empty:
            return "No anomalies detected."
        anomalies_list = anomalies.to_dict(orient="records")
    elif isinstance(anomalies, dict):
        if not anomalies:
            return "No anomalies detected."
        if "date" in anomalies:
            anomalies_df = pd.DataFrame(anomalies)
            anomalies_list = anomalies_df.to_dict(orient="records")
        else:
            lines: List[str] = []
            for metric, entries in anomalies.items():
                if isinstance(entries, list):
                    lines.append(f"• **{metric}** — {len(entries)} anomalous reading(s):")
                    for entry in entries[:5]:
                        if isinstance(entry, dict):
                            date = entry.get("date", "?")
                            value = entry.get("value", "?")
                            zscore = entry.get("z_score")
                            detail = f"  – {date}: value={value}"
                            if zscore is not None:
                                detail += f" (z-score {zscore:+.2f})"
                            lines.append(detail)
                        else:
                            lines.append(f"  – {entry}")
                else:
                    lines.append(f"• **{metric}**: {entries}")
            return "\n".join(lines)
    elif isinstance(anomalies, list):
        if not anomalies:
            return "No anomalies detected."
        anomalies_list = anomalies
    else:
        return "No anomalies detected."

    lines = []
    from collections import defaultdict
    grouped = defaultdict(list)
    for entry in anomalies_list:
        grouped[entry.get("metric", "Unknown")].append(entry)
        
    for metric, entries in grouped.items():
        lines.append(f"• **{metric}** — {len(entries)} anomalous reading(s):")
        for entry in entries[:5]:
            date = entry.get("date", "?")
            value = entry.get("value", "?")
            zscore = entry.get("deviation_sigma")
            if zscore is None:
                zscore = entry.get("z_score")
            detail = f"  – {date}: value={value}"
            if zscore is not None:
                try:
                    detail += f" (z-score {float(zscore):+.2f})"
                except Exception:
                    pass
            lines.append(detail)
            
    return "\n".join(lines)


def _format_correlations(analysis_results: Dict[str, Any]) -> str:
    """Convert the ``correlations`` section into a human-readable block."""
    correlations: Dict[str, Any] = analysis_results.get("correlations", {})
    if not correlations:
        return "No correlation data available."

    lines: List[str] = []
    for pair, info in correlations.items():
        if isinstance(info, dict):
            r = info.get("r")
            interpretation = info.get("interpretation", "")
            part = f"• **{pair}**: r = {r:.3f}" if r is not None else f"• **{pair}**"
            if interpretation:
                part += f" ({interpretation})"
            lines.append(part)
        else:
            lines.append(f"• **{pair}**: {info}")
    return "\n".join(lines)


def _format_summary_stats(analysis_results: Dict[str, Any]) -> str:
    """Render high-level summary statistics when present."""
    stats: Dict[str, Any] = analysis_results.get("summary_stats", {})
    if not stats:
        return ""

    lines: List[str] = ["### Summary Statistics"]
    for metric, values in stats.items():
        if isinstance(values, dict):
            parts = ", ".join(f"{k}={v}" for k, v in values.items())
            lines.append(f"• **{metric}**: {parts}")
        else:
            lines.append(f"• **{metric}**: {values}")
    return "\n".join(lines)


def _build_data_context(analysis_results: Dict[str, Any]) -> str:
    """Assemble a full data-context block from all analysis sections."""
    date_range = analysis_results.get("date_range", {})
    start = date_range.get("start", "unknown")
    end = date_range.get("end", "unknown")

    sections = [
        f"## Data Context  (date range: {start} to {end})",
        "",
        "### Trends",
        _format_trends(analysis_results),
        "",
        "### Anomalies",
        _format_anomalies(analysis_results),
        "",
        "### Correlations",
        _format_correlations(analysis_results),
    ]

    stats_block = _format_summary_stats(analysis_results)
    if stats_block:
        sections.extend(["", stats_block])

    return "\n".join(sections)


# ──────────────────────────────────────────────────────────────
# CORE PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────

def generate_health_summary(analysis_results: Dict[str, Any]) -> str:
    """Generate a natural-language summary of the user's health trends.

    Parameters
    ----------
    analysis_results:
        Structured dict produced by ``analyzer.py``.  Expected top-level
        keys: ``trends``, ``anomalies``, ``correlations``,
        ``summary_stats``, ``date_range``.

    Returns
    -------
    str
        Markdown-formatted health summary, or a fallback message if Ollama
        is unavailable.
    """
    if not check_ollama_available():
        return (
            "⚠️  AI summary unavailable — Ollama is not running.\n"
            "Start Ollama (`ollama serve`) and try again."
        )

    context = _build_data_context(analysis_results)

    prompt = textwrap.dedent(f"""\
        Below is the user's Apple Watch health data analysis for the most
        recent period.

        {context}

        ---

        Based on this data, please provide a comprehensive health summary that:
        1. Highlights the most notable trends (improving or declining) and
           cites the specific numbers.
        2. Calls out any anomalies and explains their potential significance.
        3. Notes interesting correlations between metrics.
        4. Gives an overall "health trajectory" assessment in 1-2 sentences.

        Keep the tone supportive and data-driven.
    """)

    try:
        return _call_ollama(prompt)
    except OllamaUnavailableError as exc:
        logger.error("Health summary generation failed: %s", exc)
        return f"⚠️  Could not generate AI summary: {exc}"


def generate_recommendations(analysis_results: Dict[str, Any]) -> str:
    """Generate 3-5 actionable, data-backed health recommendations.

    Parameters
    ----------
    analysis_results:
        Same structured dict as ``generate_health_summary``.

    Returns
    -------
    str
        Markdown list of recommendations, or a fallback message.
    """
    if not check_ollama_available():
        return (
            "⚠️  AI recommendations unavailable — Ollama is not running.\n"
            "Start Ollama (`ollama serve`) and try again."
        )

    context = _build_data_context(analysis_results)

    prompt = textwrap.dedent(f"""\
        Below is the user's Apple Watch health data analysis.

        {context}

        ---

        Based on this data, provide **3 to 5 specific, actionable
        recommendations** to improve the user's health and fitness.

        Requirements for each recommendation:
        • Reference the user's actual numbers (e.g. "Your resting heart
          rate averaged 62 bpm over the past 30 days…").
        • Explain *why* the recommendation matters, linking it to the
          trend or anomaly you observed.
        • Suggest a concrete, practical lifestyle or training adjustment
          (not vague advice like "exercise more").
        • Where relevant, suggest measurable targets the user can track
          on their Apple Watch.

        Format each recommendation as a numbered section with a bold title.
    """)

    try:
        return _call_ollama(prompt)
    except OllamaUnavailableError as exc:
        logger.error("Recommendation generation failed: %s", exc)
        return f"⚠️  Could not generate AI recommendations: {exc}"


def chat(question: str, analysis_results: Dict[str, Any]) -> str:
    """Interactive Q&A — answer a natural-language question about the data.

    Parameters
    ----------
    question:
        Free-form question from the user (e.g. "Why did my HRV drop last
        week?").
    analysis_results:
        Structured analysis dict for data context.

    Returns
    -------
    str
        The LLM's answer grounded in the user's data, or a fallback.
    """
    if not check_ollama_available():
        return (
            "⚠️  AI chat unavailable — Ollama is not running.\n"
            "Start Ollama (`ollama serve`) and try again."
        )

    context = _build_data_context(analysis_results)

    prompt = textwrap.dedent(f"""\
        The user is asking a question about their Apple Watch health data.
        Here is their full data context:

        {context}

        ---

        **User question:** {question}

        Answer the question using ONLY the data context above.  Cite
        specific values, dates, and trends where possible.  If the data
        does not contain enough information to answer, say so clearly
        rather than guessing.
    """)

    try:
        return _call_ollama(prompt)
    except OllamaUnavailableError as exc:
        logger.error("Chat query failed: %s", exc)
        return f"⚠️  Could not get AI response: {exc}"


# ──────────────────────────────────────────────────────────────
# CLI DEMO
# ──────────────────────────────────────────────────────────────

def _demo() -> None:
    """Quick smoke-test with synthetic analysis results."""
    sample_results: Dict[str, Any] = {
        "date_range": {"start": "2026-05-13", "end": "2026-06-12"},
        "trends": {
            "resting_heart_rate": {
                "direction": "declining",
                "current_avg": 58.3,
                "previous_avg": 61.7,
                "percent_change": -5.5,
                "period": "30 days",
            },
            "step_count": {
                "direction": "stable",
                "current_avg": 9842,
                "previous_avg": 9780,
                "percent_change": 0.6,
                "period": "30 days",
            },
            "hrv": {
                "direction": "improving",
                "current_avg": 48.2,
                "previous_avg": 42.1,
                "percent_change": 14.5,
                "period": "30 days",
            },
        },
        "anomalies": {
            "heart_rate": [
                {"date": "2026-06-01", "value": 142, "z_score": 3.1},
                {"date": "2026-06-05", "value": 138, "z_score": 2.8},
            ],
        },
        "correlations": {
            "sleep_vs_hrv": {"r": 0.67, "interpretation": "moderate positive"},
            "steps_vs_active_energy": {"r": 0.91, "interpretation": "strong positive"},
        },
        "summary_stats": {
            "resting_heart_rate": {"mean": 59.8, "min": 52, "max": 68},
            "step_count": {"mean": 9810, "min": 3200, "max": 18400},
        },
    }

    print("=" * 60)
    print("  AI Health Coach — Demo")
    print("=" * 60)

    if not check_ollama_available():
        print("\nOllama is not running.  Exiting demo.\n")
        return

    print("\n📊 Generating health summary …\n")
    summary = generate_health_summary(sample_results)
    print(summary)

    print("\n" + "─" * 60)
    print("\n💡 Generating recommendations …\n")
    recs = generate_recommendations(sample_results)
    print(recs)

    print("\n" + "─" * 60)
    print("\n💬 Chat demo: 'Why is my HRV improving?'\n")
    answer = chat("Why is my HRV improving?", sample_results)
    print(answer)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
