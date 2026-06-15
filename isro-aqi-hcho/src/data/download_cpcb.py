"""
download_cpcb.py
================
Download and process ground-truth air quality data from the
Central Pollution Control Board (CPCB) CAAQM portal.

Portal: https://airquality.cpcb.gov.in/ccr/#/caaqm-dashboard-all/caaqm-landing/caaqm-data-repository

Instructions
------------
1. Visit the portal and log in (registration may be required).
2. Navigate to "Data Repository" → select station(s), parameter(s),
   frequency "24 hours / Daily", and date range.
3. Export as CSV and save under  data/raw/cpcb/.
4. Run this script to aggregate and clean the raw CSVs.

Usage:
    python -m src.data.download_cpcb \\
        --input_dir data/raw/cpcb \\
        --output data/processed/cpcb_daily.csv \\
        --start 2019-01-01 \\
        --end 2022-12-31
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.grid_definition import assign_cell_id
from src.utils.aqi_calculator import compute_aqi_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected raw CSV column mapping  (CPCB portal → standard names)
# ---------------------------------------------------------------------------
RAW_COL_MAP = {
    "Station": "station_name",
    "StationId": "station_id",
    "City": "city",
    "State": "state",
    "Latitude": "lat",
    "Longitude": "lon",
    "From Date": "datetime",
    "To Date": "datetime_end",
    "PM2.5 (ug/m3)": "pm25",
    "PM10 (ug/m3)": "pm10",
    "NO2 (ug/m3)": "no2",
    "SO2 (ug/m3)": "so2",
    "Ozone (ug/m3)": "o3",
    "CO (mg/m3)": "co",
}

POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co"]


def load_station_metadata(csv_path: str | Path) -> pd.DataFrame:
    """
    Load a reference file of CPCB station metadata.

    The file should have columns: station_id, station_name, city, state, lat, lon.
    A minimal example is bundled in data/raw/cpcb/stations_metadata.csv.

    Parameters
    ----------
    csv_path : str | Path
        Path to the metadata CSV.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_csv(csv_path)


def read_raw_cpcb_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Read a single CPCB export CSV (hourly or sub-daily measurements)
    and rename columns to standard names.

    Parameters
    ----------
    filepath : str | Path
        Path to one CPCB CSV file.

    Returns
    -------
    pd.DataFrame with standardised column names.
    """
    df = pd.read_csv(filepath, skiprows=0, low_memory=False)
    # Rename known columns
    df = df.rename(columns={k: v for k, v in RAW_COL_MAP.items() if k in df.columns})

    # Parse datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
        df["date"] = df["datetime"].dt.date.astype(str)

    # Coerce pollutant columns to numeric
    for col in POLLUTANT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sub-daily (hourly) measurements to daily means.

    At least 75% of hourly records must be present for a daily mean to be valid
    (following WMO/CPCB conventions). Otherwise the daily value is set to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Standardised sub-daily dataframe with columns including ``date``,
        station metadata, and pollutant columns.

    Returns
    -------
    pd.DataFrame
        One row per (station_id, date) with daily-mean pollutant values.
    """
    group_cols = ["station_id", "station_name", "city", "state", "lat", "lon", "date"]
    group_cols = [c for c in group_cols if c in df.columns]

    numeric_cols = [c for c in POLLUTANT_COLS if c in df.columns]

    def _agg(grp: pd.DataFrame) -> dict:
        result: dict = {}
        for col in numeric_cols:
            vals = grp[col].dropna()
            completeness = len(vals) / max(len(grp), 1)
            result[col] = vals.mean() if completeness >= 0.50 else np.nan
        return result

    agg_rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update(_agg(grp))
        agg_rows.append(row)

    daily = pd.DataFrame(agg_rows)
    return daily


def compute_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``aqi_observed`` column derived from ground pollutant values."""
    df = df.copy()
    df["aqi_observed"] = compute_aqi_series(df)
    return df


def process_cpcb_directory(
    input_dir: str | Path,
    output_path: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Process all CPCB CSV files in *input_dir* into a single daily dataset.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing raw CPCB CSV exports.
    output_path : str | Path
        Where to save the final ``cpcb_daily.csv``.
    start_date, end_date : str | None
        ISO date strings to filter the output (inclusive).
    resolution : float
        Grid resolution for snapping stations to grid cells.

    Returns
    -------
    pd.DataFrame
        The processed daily CPCB dataset.
    """
    input_dir = Path(input_dir)
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        logger.warning("No CSV files found in %s", input_dir)
        logger.info("Please download CPCB data and place CSV files in %s", input_dir)
        return pd.DataFrame()

    logger.info("Found %d CSV file(s) in %s", len(csv_files), input_dir)

    dfs = []
    for fpath in csv_files:
        try:
            df_raw = read_raw_cpcb_csv(fpath)
            df_daily = aggregate_to_daily(df_raw)
            dfs.append(df_daily)
            logger.info("  Processed %s: %d rows", fpath.name, len(df_daily))
        except Exception as exc:
            logger.error("  Failed to process %s: %s", fpath.name, exc)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    if start_date:
        combined = combined[combined["date"] >= pd.Timestamp(start_date)]
    if end_date:
        combined = combined[combined["date"] <= pd.Timestamp(end_date)]

    # Snap stations to grid cells
    if "lat" in combined.columns and "lon" in combined.columns:
        combined = assign_cell_id(combined, resolution=resolution)

    # Compute observed AQI
    combined = compute_aqi_column(combined)

    # Sort and save
    combined = combined.sort_values(["station_id", "date"]).reset_index(drop=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Saved %d daily records to %s", len(combined), output_path)

    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Process CPCB air quality data")
    parser.add_argument("--input_dir", default="data/raw/cpcb")
    parser.add_argument("--output", default="data/processed/cpcb_daily.csv")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    process_cpcb_directory(args.input_dir, args.output, args.start, args.end)


if __name__ == "__main__":
    main()
