"""
aqi_calculator.py
=================
Implements the official **Indian National Ambient Air Quality Index (AQI)**
breakpoints and sub-index formulae as notified by CPCB.

Reference:
    CPCB, "National Air Quality Index", Sep 2014.
    https://cpcb.nic.in/displaypdf.php?id=bmFxaS9OQVFJX0J1bGxldGluXzE3MDcyMDE0LnBkZg==

Usage:
    from src.utils.aqi_calculator import compute_indian_aqi, compute_sub_index

    result = compute_indian_aqi({
        "pm25": 75.0,
        "pm10": 120.0,
        "no2": 80.0,
        "so2": 40.0,
        "o3": 60.0,
        "co": 1.5,
    })
    print(result)
    # {'aqi': 151, 'dominant_pollutant': 'pm25', 'category': 'Moderate', ...}
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AQI breakpoints  (C_low, C_high, I_low, I_high)
# ---------------------------------------------------------------------------

BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "pm25": [  # 24-h avg µg/m³
        (0.0, 30.0, 0, 50),
        (30.0, 60.0, 51, 100),
        (60.0, 90.0, 101, 200),
        (90.0, 120.0, 201, 300),
        (120.0, 250.0, 301, 400),
        (250.0, 500.0, 401, 500),
    ],
    "pm10": [  # 24-h avg µg/m³
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 250.0, 101, 200),
        (250.0, 350.0, 201, 300),
        (350.0, 430.0, 301, 400),
        (430.0, 600.0, 401, 500),
    ],
    "no2": [  # 24-h avg µg/m³
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 180.0, 101, 200),
        (180.0, 280.0, 201, 300),
        (280.0, 400.0, 301, 400),
        (400.0, 800.0, 401, 500),
    ],
    "so2": [  # 24-h avg µg/m³
        (0.0, 40.0, 0, 50),
        (40.0, 80.0, 51, 100),
        (80.0, 380.0, 101, 200),
        (380.0, 800.0, 201, 300),
        (800.0, 1600.0, 301, 400),
        (1600.0, 2100.0, 401, 500),
    ],
    "o3": [  # 8-h avg µg/m³
        (0.0, 50.0, 0, 50),
        (50.0, 100.0, 51, 100),
        (100.0, 168.0, 101, 200),
        (168.0, 208.0, 201, 300),
        (208.0, 748.0, 301, 400),
        (748.0, 1000.0, 401, 500),
    ],
    "co": [  # 8-h avg mg/m³
        (0.0, 1.0, 0, 50),
        (1.0, 2.0, 51, 100),
        (2.0, 10.0, 101, 200),
        (10.0, 17.0, 201, 300),
        (17.0, 34.0, 301, 400),
        (34.0, 50.0, 401, 500),
    ],
}

AQI_CATEGORIES: list[tuple[int, int, str, str]] = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Satisfactory", "#ffff00"),
    (101, 200, "Moderate", "#ff7e00"),
    (201, 300, "Poor", "#ff0000"),
    (301, 400, "Very Poor", "#8f3f97"),
    (401, 500, "Severe", "#7e0023"),
]


def compute_sub_index(pollutant: str, concentration: float) -> float | None:
    """
    Compute the AQI sub-index for a single pollutant.

    Parameters
    ----------
    pollutant : str
        One of "pm25", "pm10", "no2", "so2", "o3", "co".
    concentration : float
        24-hour (or 8-hour for O3/CO) average concentration.

    Returns
    -------
    float or None
        Sub-index value, or None if *concentration* is NaN or out of range.
    """
    if np.isnan(concentration) or concentration < 0:
        return None

    bps = BREAKPOINTS.get(pollutant.lower())
    if bps is None:
        raise ValueError(f"Unknown pollutant: {pollutant!r}")

    for c_lo, c_hi, i_lo, i_hi in bps:
        if c_lo <= concentration <= c_hi:
            sub_index = (i_hi - i_lo) / (c_hi - c_lo) * (concentration - c_lo) + i_lo
            return round(sub_index, 1)

    if concentration > bps[-1][1]:
        return 500.0

    return None


def compute_indian_aqi(pollutant_dict: dict[str, float]) -> dict:
    """
    Compute the overall Indian AQI from a dict of pollutant concentrations.

    Parameters
    ----------
    pollutant_dict : dict
        Keys: any subset of {"pm25", "pm10", "no2", "so2", "o3", "co"}.
        Values: concentration in the units specified in BREAKPOINTS.
        Missing or NaN values are silently skipped.

    Returns
    -------
    dict with keys:
        aqi               – overall AQI (int), or None if no valid sub-indices.
        dominant_pollutant – pollutant driving the AQI.
        category          – AQI category string.
        color             – hex color for the category.
        sub_indices       – dict of individual sub-indices.
    """
    sub_indices: dict[str, float] = {}
    for poll, conc in pollutant_dict.items():
        if conc is None or (isinstance(conc, float) and np.isnan(conc)):
            continue
        si = compute_sub_index(poll, float(conc))
        if si is not None:
            sub_indices[poll] = si

    if not sub_indices:
        return {
            "aqi": None,
            "dominant_pollutant": None,
            "category": "Unknown",
            "color": "#cccccc",
            "sub_indices": {},
        }

    dominant = max(sub_indices, key=sub_indices.__getitem__)
    aqi_value = int(round(sub_indices[dominant]))

    category, color = "Unknown", "#cccccc"
    for lo, hi, cat, col in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            category, color = cat, col
            break

    return {
        "aqi": aqi_value,
        "dominant_pollutant": dominant,
        "category": category,
        "color": color,
        "sub_indices": sub_indices,
    }


def compute_aqi_series(df: pd.DataFrame) -> pd.Series:
    """
    Vectorised AQI computation over a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least one of the pollutant columns:
        pm25, pm10, no2, so2, o3, co.

    Returns
    -------
    pd.Series
        AQI values (int), index aligned with *df*.
    """
    available_cols = [c for c in ["pm25", "pm10", "no2", "so2", "o3", "co"] if c in df.columns]

    aqi_vals = []
    for _, row in df[available_cols].iterrows():
        result = compute_indian_aqi(row.to_dict())
        aqi_vals.append(result["aqi"])

    return pd.Series(aqi_vals, index=df.index, name="aqi")


def aqi_category(aqi_value: int | None) -> str:
    """Return the AQI category string for a numeric AQI value."""
    if aqi_value is None:
        return "Unknown"
    for lo, hi, cat, _ in AQI_CATEGORIES:
        if lo <= aqi_value <= hi:
            return cat
    return "Severe" if aqi_value > 400 else "Unknown"


if __name__ == "__main__":
    sample = {"pm25": 75.0, "pm10": 120.0, "no2": 80.0, "so2": 40.0, "o3": 60.0, "co": 1.5}
    result = compute_indian_aqi(sample)
    print(f"AQI: {result['aqi']} | {result['category']} | Dominant: {result['dominant_pollutant']}")
    print("Sub-indices:", result["sub_indices"])
