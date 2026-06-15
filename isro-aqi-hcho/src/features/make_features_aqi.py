"""
make_features_aqi.py
====================
Feature engineering for the AQI prediction pipeline.

Takes the raw aqi_training_dataset.csv and produces a clean feature matrix
ready for model training:
  - Handles missing values (median imputation per season/region)
  - Adds derived features (wind speed, stability index)
  - Standardises features
  - Optionally creates spatial lag features

Usage:
    python -m src.features.make_features_aqi \\
        --input data/processed/aqi_training_dataset.csv \\
        --output data/processed/aqi_features.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_FEATURES = [
    "no2_column", "so2_column", "co_column", "o3_column",
    "hcho_column", "insat_aod",
    "t2m", "rh2m", "u10", "v10", "tp", "sp", "blh",
]

TARGETS = ["pm25_target", "aqi_target"]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional physically motivated features.

    New columns
    -----------
    wind_speed      – sqrt(u10² + v10²)  [m/s]
    wind_dir        – atan2(v10, u10) in degrees
    stability_idx   – blh / (t2m - 273.15 + 1)  (proxy for mixing)
    no2_hcho_ratio  – NO2/HCHO column ratio
    season_sin      – sin(2π DOY / 365), captures annual cycle
    season_cos      – cos(2π DOY / 365)
    """
    df = df.copy()

    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        df["wind_dir"] = np.degrees(np.arctan2(df["v10"], df["u10"]))

    if "blh" in df.columns and "t2m" in df.columns:
        df["stability_idx"] = df["blh"] / (df["t2m"] - 272.15 + 1).replace(0, np.nan)

    if "no2_column" in df.columns and "hcho_column" in df.columns:
        df["no2_hcho_ratio"] = df["no2_column"] / (df["hcho_column"] + 1e-6)

    if "date" in df.columns:
        dts = pd.to_datetime(df["date"])
        doy = dts.dt.day_of_year
        df["season_sin"] = np.sin(2 * np.pi * doy / 365)
        df["season_cos"] = np.cos(2 * np.pi * doy / 365)

    return df


def impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Impute missing values using column medians.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]

    Returns
    -------
    pd.DataFrame with NaNs replaced.
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            logger.debug("  Imputed %d NaN in %s with median %.4f", n_missing, col, median_val)
    return df


def clip_outliers(df: pd.DataFrame, feature_cols: list[str], n_sigma: float = 5.0) -> pd.DataFrame:
    """
    Clip values beyond *n_sigma* standard deviations to reduce influence of extreme outliers.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    n_sigma : float

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].dtype.kind in "fc":
            mean, std = df[col].mean(), df[col].std()
            lo, hi = mean - n_sigma * std, mean + n_sigma * std
            df[col] = df[col].clip(lo, hi)
    return df


def make_features(
    input_csv: str | Path,
    output_csv: str | Path,
    scaler_path: str | Path | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full feature engineering pipeline for AQI training.

    Parameters
    ----------
    input_csv : str | Path
        aqi_training_dataset.csv
    output_csv : str | Path
        Output feature CSV.
    scaler_path : str | Path | None
        Where to save/load the fitted scaler (pickle).
    fit_scaler : bool
        If True, fit a new StandardScaler on the data and save it.
        If False, load from *scaler_path*.

    Returns
    -------
    (feature DataFrame, fitted StandardScaler)
    """
    df = pd.read_csv(input_csv)
    logger.info("Loaded %d rows from %s", len(df), input_csv)

    df = add_derived_features(df)

    all_features = BASE_FEATURES + ["wind_speed", "wind_dir", "stability_idx", "no2_hcho_ratio",
                                    "season_sin", "season_cos"]
    available_features = [c for c in all_features if c in df.columns]

    df = impute_missing(df, available_features)
    df = clip_outliers(df, available_features)

    # Drop rows still missing a target
    target_available = [c for c in TARGETS if c in df.columns]
    if target_available:
        df = df.dropna(subset=target_available[:1])

    if fit_scaler:
        scaler = StandardScaler()
        df[available_features] = scaler.fit_transform(df[available_features])
        if scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)
            logger.info("Scaler saved to %s", scaler_path)
    else:
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            df[available_features] = scaler.transform(df[available_features])
        else:
            logger.warning("Scaler not found; skipping normalisation.")
            scaler = StandardScaler()

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Feature matrix saved: %d rows, %d features → %s", len(df), len(available_features), output_csv)
    return df, scaler


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="AQI feature engineering")
    parser.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    parser.add_argument("--output", default="data/processed/aqi_features.csv")
    parser.add_argument("--scaler", default="models/baseline/scaler.pkl")
    args = parser.parse_args()
    make_features(args.input, args.output, args.scaler, fit_scaler=True)


if __name__ == "__main__":
    main()
