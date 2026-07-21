"""
Composite health scoring module for Recovery, Sleep, and Strain.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp a value between a minimum and maximum."""
    return max(min_val, min(val, max_val))


def _zscore_to_subscore(z: float) -> float:
    """Map a z-score to a 0-100 score where z=0 is 50."""
    if pd.isna(z):
        return 0.0
    return _clamp((z + 3) / 6 * 100.0, 0.0, 100.0)


def _sleep_duration_subscore(hours: float) -> float:
    """Calculate sleep duration subscore based on optimal range."""
    if pd.isna(hours):
        return 0.0
    optimal_min, optimal_max = getattr(config, 'SLEEP_OPTIMAL_RANGE', (7.0, 9.0))
    
    if optimal_min <= hours <= optimal_max:
        return 100.0
    elif hours < optimal_min:
        # Linear ramp from 0 at 4 hrs to 100 at optimal_min
        if hours <= 4.0:
            return 0.0
        return _clamp((hours - 4.0) / (optimal_min - 4.0) * 100.0, 0.0, 100.0)
    else:
        # Linear ramp from 100 at optimal_max to 50 at 11 hrs
        if hours >= 11.0:
            return 50.0
        return _clamp(100.0 - (hours - optimal_max) / (11.0 - optimal_max) * 50.0, 0.0, 100.0)


def _get_zone(score: float) -> str:
    """Return the color zone for a given score."""
    if pd.isna(score):
        return 'unknown'
    zones = getattr(config, 'SCORE_ZONES', {'green': 67, 'yellow': 34})
    if score >= zones.get('green', 67):
        return 'green'
    elif score >= zones.get('yellow', 34):
        return 'yellow'
    return 'red'


def compute_recovery_score(daily_series: dict[str, pd.Series], sleep_analysis: dict[str, Any]) -> dict[str, Any]:
    """Compute a daily Recovery Score (0-100)."""
    if 'hrv' not in daily_series or 'resting_heart_rate' not in daily_series:
        logger.warning("Missing required series ('hrv', 'resting_heart_rate') for recovery score.")
        return {}
    if not sleep_analysis or 'nightly_duration' not in sleep_analysis:
        logger.warning("Missing sleep 'nightly_duration' for recovery score.")
        return {}
    
    hrv = daily_series['hrv']
    rhr = daily_series['resting_heart_rate']
    sleep_dur = sleep_analysis['nightly_duration']
    
    # Shift sleep forward by 1 day: night-of date D reflects in recovery for
    # day D+1 (i.e., last night's sleep → this morning's readiness).
    sleep_dur = sleep_dur.copy()
    sleep_dur.index = sleep_dur.index + pd.Timedelta(days=1)
    
    df = pd.DataFrame({'hrv': hrv, 'rhr': rhr, 'sleep_dur': sleep_dur})
    df = df.sort_index()
    
    # Calculate rolling stats (30 days)
    hrv_mean = df['hrv'].rolling(window=30, min_periods=3).mean()
    hrv_std = df['hrv'].rolling(window=30, min_periods=3).std()
    
    rhr_mean = df['rhr'].rolling(window=30, min_periods=3).mean()
    rhr_std = df['rhr'].rolling(window=30, min_periods=3).std()
    
    # Fill zero stds to avoid division by zero
    hrv_std = hrv_std.replace(0, np.nan).fillna(1.0)
    rhr_std = rhr_std.replace(0, np.nan).fillna(1.0)
    
    hrv_z = (df['hrv'] - hrv_mean) / hrv_std
    rhr_z = ((df['rhr'] - rhr_mean) / rhr_std) * -1.0  # invert RHR
    
    hrv_sub = hrv_z.apply(_zscore_to_subscore)
    rhr_sub = rhr_z.apply(_zscore_to_subscore)
    sleep_sub = df['sleep_dur'].apply(_sleep_duration_subscore)
    
    weights = getattr(config, 'SCORE_WEIGHTS', {}).get('recovery', {"hrv": 0.40, "rhr": 0.30, "sleep": 0.30})
    daily_score = (hrv_sub * weights['hrv'] + rhr_sub * weights['rhr'] + sleep_sub * weights['sleep'])
    
    daily_score = daily_score.dropna()
    if daily_score.empty:
        return {}
        
    current_val = float(daily_score.iloc[-1])
    return {
        'daily': daily_score,
        'current': current_val,
        'avg_7d': float(daily_score.tail(7).mean()),
        'avg_30d': float(daily_score.tail(30).mean()),
        'zone': _get_zone(current_val),
        'components': {
            'hrv': float(hrv_sub.dropna().iloc[-1]) if not hrv_sub.dropna().empty else 0.0,
            'rhr': float(rhr_sub.dropna().iloc[-1]) if not rhr_sub.dropna().empty else 0.0,
            'sleep': float(sleep_sub.dropna().iloc[-1]) if not sleep_sub.dropna().empty else 0.0,
        }
    }


def compute_sleep_score(sleep_analysis: dict[str, Any]) -> dict[str, Any]:
    """Compute a daily Sleep Score (0-100)."""
    if not sleep_analysis or 'nightly_duration' not in sleep_analysis:
        logger.warning("Missing 'nightly_duration' for sleep score.")
        return {}
        
    duration_series = sleep_analysis['nightly_duration']
    stage_breakdown = sleep_analysis.get('stage_breakdown', pd.DataFrame())
    
    df = pd.DataFrame({'duration': duration_series})
    df['duration_sub'] = df['duration'].apply(_sleep_duration_subscore)
    df['consistency'] = df['duration'].rolling(window=7, min_periods=3).std()
    
    def _consistency_sub(std: float) -> float:
        if pd.isna(std):
            return 50.0
        if std <= 0.5:
            return 100.0
        if std >= 2.5:
            return 0.0
        return _clamp(100.0 - (std - 0.5) / (2.5 - 0.5) * 100.0, 0.0, 100.0)
        
    df['consistency_sub'] = df['consistency'].apply(_consistency_sub)
    
    has_stages = not stage_breakdown.empty and 'deep' in stage_breakdown.columns and 'rem' in stage_breakdown.columns
    
    if has_stages:
        df = df.join(stage_breakdown[['deep', 'rem']])
        # Convert minutes to percentage of total duration
        df['deep_pct'] = df['deep'] / (df['duration'] * 60.0) * 100.0
        df['rem_pct'] = df['rem'] / (df['duration'] * 60.0) * 100.0
        
        def _deep_sub(pct: float) -> float:
            if pd.isna(pct): return 0.0
            if 15.0 <= pct <= 20.0: return 100.0
            if pct < 15.0: return _clamp((pct / 15.0) * 100.0, 0.0, 100.0)
            return _clamp(100.0 - (pct - 20.0) / (40.0 - 20.0) * 30.0, 0.0, 100.0)
            
        def _rem_sub(pct: float) -> float:
            if pd.isna(pct): return 0.0
            if 20.0 <= pct <= 25.0: return 100.0
            if pct < 20.0: return _clamp((pct / 20.0) * 100.0, 0.0, 100.0)
            return _clamp(100.0 - (pct - 25.0) / (45.0 - 25.0) * 30.0, 0.0, 100.0)
            
        df['deep_sub'] = df['deep_pct'].apply(_deep_sub)
        df['rem_sub'] = df['rem_pct'].apply(_rem_sub)
        
        weights = getattr(config, 'SCORE_WEIGHTS', {}).get('sleep', {"duration": 0.40, "deep": 0.25, "rem": 0.20, "consistency": 0.15})
        daily_score = (
            df['duration_sub'] * weights['duration'] + 
            df['deep_sub'] * weights['deep'] + 
            df['rem_sub'] * weights['rem'] + 
            df['consistency_sub'] * weights['consistency']
        )
    else:
        weights = getattr(config, 'SCORE_WEIGHTS', {}).get('sleep_no_stages', {"duration": 0.65, "consistency": 0.35})
        daily_score = (
            df['duration_sub'] * weights['duration'] + 
            df['consistency_sub'] * weights['consistency']
        )
        
    daily_score = daily_score.dropna()
    if daily_score.empty:
        return {}
        
    current_val = float(daily_score.iloc[-1])
    components = {
        'duration': float(df['duration_sub'].dropna().iloc[-1]) if not df['duration_sub'].dropna().empty else 0.0,
        'consistency': float(df['consistency_sub'].dropna().iloc[-1]) if not df['consistency_sub'].dropna().empty else 0.0
    }
    if has_stages:
        components['deep'] = float(df['deep_sub'].dropna().iloc[-1]) if not df['deep_sub'].dropna().empty else 0.0
        components['rem'] = float(df['rem_sub'].dropna().iloc[-1]) if not df['rem_sub'].dropna().empty else 0.0
        
    return {
        'daily': daily_score,
        'current': current_val,
        'avg_7d': float(daily_score.tail(7).mean()),
        'avg_30d': float(daily_score.tail(30).mean()),
        'zone': _get_zone(current_val),
        'components': components
    }


def compute_strain_score(daily_series: dict[str, pd.Series], workout_analysis: dict[str, Any], data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Compute a daily Strain Score (0-100)."""
    req_keys = ['active_energy', 'exercise_minutes', 'resting_heart_rate']
    if not all(k in daily_series for k in req_keys):
        logger.warning(f"Missing required daily series for strain score: {req_keys}")
        return {}
    
    rhr_series = daily_series['resting_heart_rate']
    rhr_rolling = rhr_series.rolling(30, min_periods=1).mean()
    
    if 'heart_rate' in daily_series and not daily_series['heart_rate'].dropna().empty:
        max_hr = daily_series['heart_rate'].max()
    else:
        max_hr = 190.0
    if pd.isna(max_hr): 
        max_hr = 190.0
        
    trimp_daily = pd.Series(0.0, index=rhr_series.index)
    
    if 'workouts' in data and not data['workouts'].empty:
        workouts = data['workouts'].copy()
        if 'startDate' in workouts.columns:
            workouts['startDate'] = pd.to_datetime(workouts['startDate'], errors='coerce')
            workouts['date'] = workouts['startDate'].dt.floor('D')
            for date, group in workouts.groupby('date'):
                if date not in rhr_rolling.index:
                    continue
                rhr = rhr_rolling.loc[date]
                if pd.isna(rhr):
                    rhr = rhr_series.mean() if not pd.isna(rhr_series.mean()) else 60.0
                
                day_trimp = 0.0
                for _, w in group.iterrows():
                    avg_hr = w.get('averageHeartRate', rhr)
                    if pd.isna(avg_hr):
                        avg_hr = rhr
                    dur_min = w.get('duration', 0.0)
                    if pd.isna(dur_min):
                        continue
                    
                    denom = max_hr - rhr
                    hrr = (avg_hr - rhr) / denom if denom > 0 else 0.0
                    hrr = _clamp(hrr, 0.0, 1.0)
                    
                    y = 0.64 * np.exp(1.92 * hrr)
                    trimp = dur_min * hrr * y
                    day_trimp += trimp
                    
                trimp_daily.loc[date] = day_trimp

    # Scale against personal 90-day max TRIMP
    trimp_max_90d = trimp_daily.rolling(90, min_periods=1).max()
    trimp_max_90d = trimp_max_90d.replace(0, 1.0)
    trimp_sub = (trimp_daily / trimp_max_90d * 100.0).clip(0, 100)
    
    # Active energy
    energy = daily_series['active_energy']
    p5_energy = energy.rolling(90, min_periods=3).quantile(0.05).fillna(0)
    p95_energy = energy.rolling(90, min_periods=3).quantile(0.95).fillna(1)
    diff_energy = (p95_energy - p5_energy).replace(0, 1)
    energy_sub = ((energy - p5_energy) / diff_energy * 100.0).clip(0, 100).fillna(0)
    
    # Exercise minutes
    ex_min = daily_series['exercise_minutes']
    p5_ex = ex_min.rolling(90, min_periods=3).quantile(0.05).fillna(0)
    p95_ex = ex_min.rolling(90, min_periods=3).quantile(0.95).fillna(1)
    diff_ex = (p95_ex - p5_ex).replace(0, 1)
    ex_sub = ((ex_min - p5_ex) / diff_ex * 100.0).clip(0, 100).fillna(0)
    
    df = pd.DataFrame({
        'trimp_sub': trimp_sub,
        'energy_sub': energy_sub,
        'ex_sub': ex_sub
    }).sort_index()
    
    weights = getattr(config, 'SCORE_WEIGHTS', {}).get('strain', {"trimp": 0.60, "active_energy": 0.25, "exercise_minutes": 0.15})
    daily_score = (
        df['trimp_sub'] * weights['trimp'] +
        df['energy_sub'] * weights['active_energy'] +
        df['ex_sub'] * weights['exercise_minutes']
    ).fillna(0)
    
    daily_score = daily_score.dropna()
    if daily_score.empty:
        return {}
        
    current_val = float(daily_score.iloc[-1])
    return {
        'daily': daily_score,
        'current': current_val,
        'avg_7d': float(daily_score.tail(7).mean()),
        'avg_30d': float(daily_score.tail(30).mean()),
        'zone': _get_zone(current_val),
        'components': {
            'trimp': float(df['trimp_sub'].iloc[-1]) if not df['trimp_sub'].empty else 0.0,
            'active_energy': float(df['energy_sub'].iloc[-1]) if not df['energy_sub'].empty else 0.0,
            'exercise_minutes': float(df['ex_sub'].iloc[-1]) if not df['ex_sub'].empty else 0.0,
        }
    }


def compute_all_scores(daily_series: dict[str, pd.Series], sleep_analysis: dict[str, Any], workout_analysis: dict[str, Any], anomalies: dict[str, Any], data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Orchestrate the computation of all composite scores.
    Returns a dictionary with 'recovery', 'sleep', and 'strain' scores.
    """
    return {
        'recovery': compute_recovery_score(daily_series, sleep_analysis),
        'sleep': compute_sleep_score(sleep_analysis),
        'strain': compute_strain_score(daily_series, workout_analysis, data),
    }
