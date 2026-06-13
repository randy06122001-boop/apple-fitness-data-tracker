"""
Configuration settings for the Apple Watch Ultra Data Analyzer.
Edit these paths and preferences to match your setup.
"""

import os
from pathlib import Path

# ──────────────────────────────────────────────
# DATA INPUT
# ──────────────────────────────────────────────
# Path to your Apple Health export.xml file.
# Export from: iPhone → Health App → Profile Picture → Export All Health Data
EXPORT_XML_PATH = Path(os.environ.get(
    "HEALTH_EXPORT_PATH",
    r"D:\project\export\apple_health_export\export.xml"
))

# Local cache: parsed data is stored here so you don't re-parse the XML every run.
CACHE_DIR = Path(os.environ.get(
    "HEALTH_CACHE_DIR",
    r"D:\project\export\apple_health_export\.cache"
))

# ──────────────────────────────────────────────
# OBSIDIAN VAULT
# ──────────────────────────────────────────────
# Root path of your Obsidian vault.
OBSIDIAN_VAULT_PATH = Path(os.environ.get(
    "OBSIDIAN_VAULT_PATH",
    r"D:\project\ObsidianVault"
))

# Subfolder inside the vault where health data notes will be written.
OBSIDIAN_HEALTH_FOLDER = "Health_Data"

# ──────────────────────────────────────────────
# OLLAMA (Local AI)
# ──────────────────────────────────────────────
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")

# ──────────────────────────────────────────────
# ANALYSIS SETTINGS
# ──────────────────────────────────────────────
# Rolling average windows (in days)
ROLLING_WINDOWS = [7, 30, 90]

# Anomaly detection: flag values that deviate more than this many
# standard deviations from the rolling mean.
ANOMALY_STD_THRESHOLD = 2.0

# ──────────────────────────────────────────────
# METRIC DEFINITIONS
# ──────────────────────────────────────────────
# Apple Health XML record type identifiers we care about.
METRIC_TYPES = {
    # ── Cardiovascular ──
    "resting_heart_rate":      "HKQuantityTypeIdentifierRestingHeartRate",
    "heart_rate":              "HKQuantityTypeIdentifierHeartRate",
    "hrv":                     "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "vo2max":                  "HKQuantityTypeIdentifierVO2Max",
    "walking_heart_rate_avg":  "HKQuantityTypeIdentifierWalkingHeartRateAverage",

    # ── Activity ──
    "step_count":              "HKQuantityTypeIdentifierStepCount",
    "active_energy":           "HKQuantityTypeIdentifierActiveEnergyBurned",
    "exercise_minutes":        "HKQuantityTypeIdentifierAppleExerciseTime",

    # ── Sleep ──
    "sleep_analysis":          "HKCategoryTypeIdentifierSleepAnalysis",

    # ── Body & Environment ──
    "wrist_temperature":       "HKQuantityTypeIdentifierAppleWalkingSteadiness",  # placeholder
    "environmental_audio":     "HKQuantityTypeIdentifierEnvironmentalAudioExposure",

    # ── Ultra-Specific ──
    "running_power":           "HKQuantityTypeIdentifierRunningPower",
    "ground_contact_time":     "HKQuantityTypeIdentifierRunningGroundContactTime",
    "vertical_oscillation":    "HKQuantityTypeIdentifierRunningVerticalOscillation",
    "water_temperature":       "HKQuantityTypeIdentifierWaterTemperature",
    "underwater_depth":        "HKQuantityTypeIdentifierUnderwaterDepth",
}

# Workout type identifier
WORKOUT_TYPE = "HKWorkoutActivityType"
