"""
make_features_aqi.py
====================
V3 feature engineering pipeline for AQI prediction.

New in V3
---------
* ``add_temporal_features``   — month sin/cos, weekday, is_weekend, quarter
* ``add_rolling_features``    — 3-day and 7-day rolling means per grid cell
* ``add_spatial_context``     — 3×3 neighbourhood average via scipy.ndimage
* Parquet read/write via ``storage_format`` config flag
* Scaler saved with ``joblib`` (replaces pickle)
* All derived-feature names documented in a centralised constant

Usage::

    python -m src.features.make_features_aqi \\
        --input  data/processed/aqi_training_dataset.csv \\
        --output data/processed/aqi_features.csv \\
        --rolling --spatial_context
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Feature lists
# ──────────────────────────────────────────────────────────────────────────────

BASE_FEATURES: list[str] = [
    "no2_column", "so2_column", "co_column", "o3_column",
    "hcho_column", "insat_aod",
    "t2m", "rh2m", "u10", "v10", "tp", "sp", "blh",
]

DERIVED_FEATURES: list[str] = [
    "wind_speed", "wind_dir", "stability_idx", "no2_hcho_ratio",
    "season_sin", "season_cos",
    "doy_sin", "doy_cos", "month_sin", "month_cos",
    "weekday", "is_weekend", "quarter",
]

TARGETS: list[str] = ["pm25_target", "aqi_target"]

ROLLING_COLS: list[str] = [
    "no2_column", "insat_aod", "hcho_column",
    "pm25_target", "t2m", "rh2m",
]
ROLLING_WINDOWS: list[int] = [3, 7]

SPATIAL_CONTEXT_COLS: list[str] = ["no2_column", "insat_aod", "hcho_column"]
SPATIAL_WINDOW: int = 3   # 3×3 neighbourhood


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Physically motivated derived features (unchanged from V2)
# ──────────────────────────────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute physically motivated scalar features.

    New columns
    -----------
    wind_speed      — sqrt(u10² + v10²)  [m s⁻¹]
    wind_dir        — atan2(v10, u10) in degrees
    stability_idx   — BLH / (T2m − 272 + 1)  (proxy for atmospheric mixing)
    no2_hcho_ratio  — NO₂/HCHO column ratio
    season_sin/cos  — annual cycle encoded as sin/cos of DOY
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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Temporal features  [V3 new]
# ──────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar-derived temporal features.

    New columns
    -----------
    doy_sin / doy_cos   — day-of-year annual cycle (redundant with season_sin/cos
                          but kept separately for clarity)
    month_sin / cos     — monthly cycle
    weekday             — 0=Monday … 6=Sunday
    is_weekend          — 1 for Saturday/Sunday, 0 otherwise
    quarter             — calendar quarter (1–4)
    """
    if "date" not in df.columns:
        return df

    df = df.copy()
    dts = pd.to_datetime(df["date"])
    doy = dts.dt.day_of_year
    month = dts.dt.month

    df["doy_sin"]    = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"]    = np.cos(2 * np.pi * doy / 365)
    df["month_sin"]  = np.sin(2 * np.pi * month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * month / 12)
    df["weekday"]    = dts.dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)
    df["quarter"]    = dts.dt.quarter.astype(np.int8)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Rolling means per grid cell  [V3 new]
# ──────────────────────────────────────────────────────────────────────────────

def add_rolling_features(
    df: pd.DataFrame,
    cell_col: str = "cell_id",
    roll_cols: list[str] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Add rolling-mean features per grid cell.

    The DataFrame is sorted by ``(cell_id, date)`` before computing rolling
    statistics so that inter-cell bleeding does not occur.

    New columns
    -----------
    ``{col}_roll{W}d`` for each col in *roll_cols* and W in *windows*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``date`` and ideally ``cell_id``.
    cell_col : str
        Column used to group cells (default: ``cell_id``).
    roll_cols : list[str] | None
        Feature columns to smooth. Defaults to ``ROLLING_COLS``.
    windows : list[int] | None
        Rolling windows in days. Defaults to ``ROLLING_WINDOWS`` (3, 7).
    """
    if roll_cols is None:
        roll_cols = ROLLING_COLS
    if windows is None:
        windows = ROLLING_WINDOWS

    if "date" not in df.columns:
        logger.warning("rolling features require 'date' column; skipping.")
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    has_cell = cell_col in df.columns
    sort_cols = [cell_col, "date"] if has_cell else ["date"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    avail = [c for c in roll_cols if c in df.columns]
    if not avail:
        return df

    for col in avail:
        for w in windows:
            new_col = f"{col}_roll{w}d"
            if has_cell:
                df[new_col] = (
                    df.groupby(cell_col, sort=False)[col]
                    .transform(lambda x: x.rolling(w, min_periods=1).mean())
                )
            else:
                df[new_col] = df[col].rolling(w, min_periods=1).mean()

    n_new = len(avail) * len(windows)
    logger.info("Rolling features: %d new columns (%s-day windows)", n_new, windows)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Spatial context features  [V3 new]
# ──────────────────────────────────────────────────────────────────────────────

def add_spatial_context(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    window_size: int = SPATIAL_WINDOW,
) -> pd.DataFrame:
    """
    Add spatial neighbourhood average features.

    For each date, pivot the grid to a 2-D raster, apply a
    ``scipy.ndimage.uniform_filter`` (equivalent to a box-car mean over the
    *window_size* × *window_size* neighbourhood), and merge back.

    New columns
    -----------
    ``{col}_ctx{W}x{W}`` for each col in *feature_cols*.

    Parameters
    ----------
    df : pd.DataFrame
        Must have ``lat``, ``lon`` columns.
    feature_cols : list[str] | None
        Defaults to ``SPATIAL_CONTEXT_COLS``.
    window_size : int
        Neighbourhood size in grid cells.
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        logger.warning("scipy not available; skipping spatial context features.")
        return df

    if feature_cols is None:
        feature_cols = SPATIAL_CONTEXT_COLS

    if "lat" not in df.columns or "lon" not in df.columns:
        logger.warning("Spatial context requires lat/lon; skipping.")
        return df

    avail = [c for c in feature_cols if c in df.columns]
    if not avail:
        return df

    df = df.copy()
    suffix = f"_ctx{window_size}x{window_size}"

    date_col = "date" if "date" in df.columns else None
    dates = [None] if date_col is None else df[date_col].unique()

    result_frames: list[pd.DataFrame] = []
    for date in dates:
        day_df = df if date_col is None else df[df[date_col] == date]

        smooth_parts: list[pd.DataFrame] = []
        for col in avail:
            piv = day_df.pivot_table(index="lat", columns="lon", values=col, aggfunc="mean")
            arr = np.nan_to_num(piv.values.astype(np.float32))
            smoothed = uniform_filter(arr, size=window_size, mode="nearest")
            flat = (
                pd.DataFrame(smoothed, index=piv.index, columns=piv.columns)
                .reset_index()
                .melt(id_vars="lat", var_name="lon", value_name=col + suffix)
            )
            flat["lon"] = flat["lon"].astype(float)
            smooth_parts.append(flat)

        if smooth_parts:
            merged_smooth = smooth_parts[0]
            for part in smooth_parts[1:]:
                merged_smooth = merged_smooth.merge(part, on=["lat", "lon"], how="outer")
            day_out = day_df.merge(merged_smooth, on=["lat", "lon"], how="left")
        else:
            day_out = day_df

        result_frames.append(day_out)

    out = pd.concat(result_frames, ignore_index=True) if result_frames else df
    n_new = len(avail)
    logger.info("Spatial context features: %d new columns (window=%dx%d)", n_new, window_size, window_size)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5: Missing-value imputation and outlier clipping
# ──────────────────────────────────────────────────────────────────────────────

def impute_missing(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Impute missing values using column medians."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            if n_missing > 0:
                logger.debug("  Imputed %d NaN in '%s' with median %.4f", n_missing, col, median_val)
    return df


def clip_outliers(df: pd.DataFrame, feature_cols: list[str], n_sigma: float = 5.0) -> pd.DataFrame:
    """Clip values beyond *n_sigma* standard deviations from the column mean."""
    df = df.copy()
    for col in feature_cols:
        if col in df.columns and df[col].dtype.kind in "fc":
            mean, std = df[col].mean(), df[col].std()
            lo, hi = mean - n_sigma * std, mean + n_sigma * std
            df[col] = df[col].clip(lo, hi)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read(path: Path) -> pd.DataFrame:
    """Read CSV or Parquet based on file extension."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write(df: pd.DataFrame, path: Path, storage_format: str = "csv") -> None:
    """Write DataFrame as CSV or Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if storage_format == "parquet":
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
        logger.info("Saved %d rows → %s", len(df), out)
    else:
        df.to_csv(path, index=False)
        logger.info("Saved %d rows → %s", len(df), path)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def make_features(
    input_csv: str | Path,
    output_csv: str | Path,
    scaler_path: str | Path | None = None,
    fit_scaler: bool = True,
    add_rolling: bool = True,
    add_spatial: bool = False,
    storage_format: str = "csv",
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Full V3 feature engineering pipeline.

    Pipeline stages
    ---------------
    1. Load data (CSV or Parquet)
    2. add_derived_features — wind speed, stability index, season sin/cos
    3. add_temporal_features — month sin/cos, weekday, is_weekend, quarter
    4. add_rolling_features  — 3-day & 7-day rolling means per cell [optional]
    5. add_spatial_context   — 3×3 neighbourhood mean [optional, slow on full grid]
    6. impute_missing        — median imputation per column
    7. clip_outliers         — clip at ±5σ
    8. StandardScaler        — fit (training) or transform (inference)
    9. Write output

    Parameters
    ----------
    input_csv : str | Path
        ``data/processed/aqi_training_dataset.csv``
    output_csv : str | Path
        Destination feature file.
    scaler_path : str | Path | None
        Joblib file to save/load the fitted scaler.
    fit_scaler : bool
        True = fit scaler on this data; False = load from *scaler_path*.
    add_rolling : bool
        Whether to compute rolling means (requires sorted data per cell_id).
    add_spatial : bool
        Whether to compute spatial context features (slow on full India grid).
    storage_format : str
        ``"csv"`` or ``"parquet"``.

    Returns
    -------
    (feature_df, scaler)
    """
    input_path = Path(input_csv)
    logger.info("Loading data from %s …", input_path)
    df = _read(input_path)
    logger.info("  Loaded: %d rows × %d cols", len(df), len(df.columns))

    # Stage 1: Derived physical features
    df = add_derived_features(df)

    # Stage 2: Temporal features
    df = add_temporal_features(df)

    # Stage 3: Rolling features (per cell)
    if add_rolling:
        df = add_rolling_features(df)

    # Stage 4: Spatial context (optional — slow on full grid)
    if add_spatial:
        df = add_spatial_context(df)

    # Stage 5: Collect all available features
    all_features = (
        BASE_FEATURES
        + DERIVED_FEATURES
        + [f"{c}_roll{w}d" for c in ROLLING_COLS for w in ROLLING_WINDOWS if add_rolling]
        + [f"{c}_ctx{SPATIAL_WINDOW}x{SPATIAL_WINDOW}" for c in SPATIAL_CONTEXT_COLS if add_spatial]
    )
    available_features = [c for c in all_features if c in df.columns]
    logger.info("  Feature columns: %d", len(available_features))

    # Stage 6: Imputation + clipping
    df = impute_missing(df, available_features)
    df = clip_outliers(df, available_features)

    # Drop rows still missing a target
    target_available = [c for c in TARGETS if c in df.columns]
    if target_available:
        before = len(df)
        df = df.dropna(subset=target_available[:1])
        logger.info("  Dropped %d rows with missing target", before - len(df))

    # Stage 7: Scaling
    if fit_scaler:
        scaler = StandardScaler()
        df[available_features] = scaler.fit_transform(df[available_features])
        if scaler_path:
            sp = Path(scaler_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, sp)
            logger.info("Scaler saved → %s", sp)
    else:
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            df[available_features] = scaler.transform(df[available_features])
            logger.info("Scaler loaded from %s", scaler_path)
        else:
            logger.warning("Scaler not found; returning unscaled features.")
            scaler = StandardScaler()

    _write(df, Path(output_csv), storage_format)
    logger.info(
        "Feature matrix: %d rows × %d features → %s",
        len(df), len(available_features), output_csv,
    )
    return df, scaler


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AQI V3 feature engineering")
    parser.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    parser.add_argument("--output", default="data/processed/aqi_features.csv")
    parser.add_argument("--scaler", default="models/baseline/scaler.joblib")
    parser.add_argument("--rolling", action="store_true", default=True,
                        help="Add rolling means per grid cell (default: on)")
    parser.add_argument("--no_rolling", dest="rolling", action="store_false")
    parser.add_argument("--spatial_context", action="store_true",
                        help="Add 3×3 spatial context features (slow on full grid)")
    parser.add_argument("--format", dest="storage_format", choices=["csv", "parquet"],
                        default="csv")
    args = parser.parse_args()

    from src.utils.logging_utils import setup_logging
    setup_logging(log_file="logs/feature_engineering.log")

    make_features(
        args.input, args.output, args.scaler,
        fit_scaler=True,
        add_rolling=args.rolling,
        add_spatial=args.spatial_context,
        storage_format=args.storage_format,
    )


if __name__ == "__main__":
    main()
