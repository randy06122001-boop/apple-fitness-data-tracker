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
# COMPOSITE SCORE SETTINGS
# ──────────────────────────────────────────────
# Weights for each score's sub-components (must sum to 1.0).
SCORE_WEIGHTS = {
    "recovery": {"hrv": 0.40, "rhr": 0.30, "sleep": 0.30},
    "sleep": {"duration": 0.40, "deep": 0.25, "rem": 0.20, "consistency": 0.15},
    "strain": {"trimp": 0.60, "active_energy": 0.25, "exercise_minutes": 0.15},
}

# Optimal sleep duration range (hours).
SLEEP_OPTIMAL_RANGE = (7.0, 9.0)

# Target sleep stage fractions (of total sleep time).
DEEP_SLEEP_TARGET = (0.15, 0.20)
REM_SLEEP_TARGET = (0.20, 0.25)

# Rolling window for computing personal baselines used in scoring.
SCORE_BASELINE_WINDOW = 30

# Score zone thresholds (inclusive lower bounds).
SCORE_ZONES = {"green": 67, "yellow": 34}

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

# ──────────────────────────────────────────────
# METRIC DISPLAY UNITS
# ──────────────────────────────────────────────
METRIC_UNITS = {
    "resting_heart_rate":      "bpm",
    "heart_rate":              "bpm",
    "hrv":                     "ms",
    "vo2max":                  "mL/kg/min",
    "walking_heart_rate_avg":  "bpm",
    "step_count":              "steps",
    "active_energy":           "kcal",
    "exercise_minutes":        "min",
    "sleep_duration":          "hrs",
    "sleep_analysis":          "hrs",
    "wrist_temperature":       "°C",
    "environmental_audio":     "dBASPL",
    "running_power":           "W",
    "ground_contact_time":     "ms",
    "vertical_oscillation":    "cm",
    "water_temperature":       "°C",
    "underwater_depth":        "m",
}

# Units for composite scores
METRIC_UNITS.update({
    "recovery_score": "/100",
    "sleep_score": "/100",
    "strain_score": "/100",
})

# ──────────────────────────────────────────────
# SCORING THRESHOLDS
# ──────────────────────────────────────────────
SCORE_ZONES = {
    "green": 67,
    "yellow": 34,
}

SLEEP_OPTIMAL_RANGE = (7.0, 9.0)

SCORE_WEIGHTS = {
    "recovery": {"hrv": 0.40, "rhr": 0.30, "sleep": 0.30},
    "sleep": {"duration": 0.40, "deep": 0.25, "rem": 0.20, "consistency": 0.15},
    "sleep_no_stages": {"duration": 0.65, "consistency": 0.35},
    "strain": {"trimp": 0.60, "active_energy": 0.25, "exercise_minutes": 0.15},
}
