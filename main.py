#!/usr/bin/env python3
"""
Apple Watch Ultra Data Analyzer — Main Entry Point
===================================================
Orchestrates the full pipeline:
  1. Parse the Apple Health export.xml
  2. Analyze trends, correlations, and anomalies
  3. Generate AI-powered health insights via Ollama
  4. Export everything to your Obsidian vault

Usage:
    python main.py                   # Full pipeline (parse → analyze → AI → Obsidian)
    python main.py --skip-ai         # Skip Ollama AI features
    python main.py --force-parse     # Force re-parse even if cache is fresh
    python main.py --chat            # Interactive chat mode after analysis
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import datetime

import config
from parser import parse_export
from analyzer import analyze
from ai_coach import (
    check_ollama_available,
    generate_health_summary,
    generate_recommendations,
    chat,
)
from obsidian_export import export_to_obsidian

# Reconfigure stdout to support UTF-8 encoding in Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass



# ──────────────────────────────────────────────
# BANNER
# ──────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║   ⌚  Apple Watch Ultra Data Analyzer                    ║
║   🤖  AI-Powered Health Coach · Obsidian Integration     ║
╚══════════════════════════════════════════════════════════╝
"""


def print_banner() -> None:
    print(BANNER)


# ──────────────────────────────────────────────
# PIPELINE STAGES
# ──────────────────────────────────────────────
def stage_parse(force: bool = False) -> dict:
    """Stage 1: Parse the Apple Health export.xml."""
    print("\n" + "=" * 60)
    print("  📥  STAGE 1: Parsing Apple Health Data")
    print("=" * 60)

    data = parse_export(force=force)

    total_records = sum(len(df) for df in data.values())
    print(f"\n  📊 Loaded {len(data)} metric types with {total_records:,} total records.")
    return data


def stage_analyze(data: dict) -> dict:
    """Stage 2: Analyze trends, correlations, and anomalies."""
    print("\n" + "=" * 60)
    print("  📈  STAGE 2: Analyzing Trends & Patterns")
    print("=" * 60)

    results = analyze(data)
    results_dict = asdict(results)

    # Print a quick summary
    trend_summary = results_dict.get("trend_summary", {})
    anomalies = results_dict.get("anomalies", {})
    correlations = results_dict.get("correlations", {})

    print(f"\n  📋 Trend directions computed for {len(trend_summary)} metrics.")
    print(f"  🔗 {len(correlations)} correlation analyses performed.")

    # Count anomalies
    if isinstance(anomalies, dict) and "date" in anomalies:
        n_anomalies = len(anomalies["date"])
    elif isinstance(anomalies, list):
        n_anomalies = len(anomalies)
    else:
        n_anomalies = 0
    print(f"  ⚠️  {n_anomalies} anomalies flagged.")

    return results_dict


def stage_ai_coach(analysis_results: dict) -> tuple[str, str]:
    """Stage 3: Generate AI health summary and recommendations."""
    print("\n" + "=" * 60)
    print("  🤖  STAGE 3: AI Health Coach (Ollama)")
    print("=" * 60)

    if not check_ollama_available():
        print("\n  ⚠️  Ollama is not running. Skipping AI features.")
        print("  💡 Start Ollama with: ollama serve")
        print(f"  💡 Then pull a model: ollama pull {config.OLLAMA_MODEL}")
        return "", ""

    print(f"\n  🧠 Using model: {config.OLLAMA_MODEL}")
    print("  ⏳ Generating health summary...")
    ai_summary = generate_health_summary(analysis_results)
    print("  ✅ Summary generated.")

    print("  ⏳ Generating recommendations...")
    ai_recommendations = generate_recommendations(analysis_results)
    print("  ✅ Recommendations generated.")

    # Print a preview
    print("\n  ── AI Summary Preview ──")
    preview = ai_summary[:300] + "..." if len(ai_summary) > 300 else ai_summary
    for line in preview.split("\n"):
        print(f"  │ {line}")

    print("\n  ── Recommendations Preview ──")
    preview = ai_recommendations[:300] + "..." if len(ai_recommendations) > 300 else ai_recommendations
    for line in preview.split("\n"):
        print(f"  │ {line}")

    return ai_summary, ai_recommendations


def stage_obsidian_export(
    analysis_results: dict,
    ai_summary: str,
    ai_recommendations: str,
) -> None:
    """Stage 4: Export everything to Obsidian."""
    print("\n" + "=" * 60)
    print("  📝  STAGE 4: Exporting to Obsidian Vault")
    print("=" * 60)

    report_date = datetime.now().strftime("%Y-%m-%d")
    export_to_obsidian(
        analysis_results=analysis_results,
        ai_summary=ai_summary,
        ai_recommendations=ai_recommendations,
        report_date=report_date,
    )


def interactive_chat(analysis_results: dict) -> None:
    """Interactive chat loop for asking questions about your health data."""
    print("\n" + "=" * 60)
    print("  💬  Interactive Chat Mode")
    print("=" * 60)
    print("  Ask questions about your health data. Type 'quit' to exit.\n")

    if not check_ollama_available():
        print("  ⚠️  Ollama is not running. Cannot start chat mode.")
        return

    while True:
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("  👋 Goodbye!")
            break

        print("  ⏳ Thinking...")
        response = chat(question, analysis_results)
        print(f"\n  🤖 Coach: {response}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apple Watch Ultra Data Analyzer — AI-Powered Health Coach",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Force re-parsing even if cached data is fresh.",
    )
    parser.add_argument(
        "--skip-ai",
        action="store_true",
        help="Skip the Ollama AI coaching stage.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enter interactive chat mode after the analysis pipeline.",
    )
    parser.add_argument(
        "--skip-obsidian",
        action="store_true",
        help="Skip exporting to Obsidian vault.",
    )
    args = parser.parse_args()

    print_banner()

    # Validate paths
    if not config.EXPORT_XML_PATH.exists():
        print(f"  ❌ Export file not found: {config.EXPORT_XML_PATH}")
        print("  💡 Export your data from:")
        print("     iPhone → Health App → Profile → Export All Health Data")
        print(f"  💡 Then place export.xml at: {config.EXPORT_XML_PATH}")
        print(f"  💡 Or set the HEALTH_EXPORT_PATH environment variable.")
        sys.exit(1)

    # ── Stage 1: Parse ──
    data = stage_parse(force=args.force_parse)

    if not data:
        print("  ❌ No health data found in the export. Exiting.")
        sys.exit(1)

    # ── Stage 2: Analyze ──
    analysis_results = stage_analyze(data)

    # ── Stage 3: AI Coach ──
    ai_summary = ""
    ai_recommendations = ""
    if not args.skip_ai:
        ai_summary, ai_recommendations = stage_ai_coach(analysis_results)

    # ── Stage 4: Obsidian Export ──
    if not args.skip_obsidian:
        stage_obsidian_export(analysis_results, ai_summary, ai_recommendations)

    # ── Optional: Interactive Chat ──
    if args.chat:
        interactive_chat(analysis_results)

    # ── Done ──
    print("\n" + "=" * 60)
    print("  🎉  Pipeline Complete!")
    print("=" * 60)
    print(f"  📂 Obsidian vault: {config.OBSIDIAN_VAULT_PATH}")
    print(f"  📊 Cached data:    {config.CACHE_DIR}")
    print(f"  🤖 AI model:       {config.OLLAMA_MODEL}")
    print()


if __name__ == "__main__":
    main()
