"""
download_reanalysis.py
======================
Download ERA5 reanalysis meteorological data over India via the
Copernicus Climate Data Store (CDS) API.

Setup
-----
1. Register at https://cds.climate.copernicus.eu
2. Install the CDS API key in ~/.cdsapirc :
       url: https://cds.climate.copernicus.eu/api/v2
       key: <UID>:<API-KEY>
3. Install: pip install cdsapi

Alternative sources:
  IMDAA  – https://nwp.ncmrwf.gov.in/reanalysis   (requires NCMRWF registration)
  MERRA-2 – https://disc.gsfc.nasa.gov/datasets?project=MERRA-2

Variables downloaded
--------------------
  2m_temperature            → t2m   (K)
  2m_dewpoint_temperature   → d2m   (K)
  10m_u_component_of_wind   → u10   (m/s)
  10m_v_component_of_wind   → v10   (m/s)
  total_precipitation        → tp    (m/day)
  surface_pressure           → sp    (Pa)
  boundary_layer_height      → blh   (m)

Usage:
    python -m src.data.download_reanalysis \\
        --start 2019-01-01 --end 2019-12-31 \\
        --output_dir data/raw/reanalysis \\
        --output_csv data/processed/era5_daily.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# India bounding box for CDS API  [north, west, south, east]
INDIA_AREA = [37.5, 68.0, 8.0, 97.5]
GRID_STEP = 0.1

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "surface_pressure",
    "boundary_layer_height",
]

VAR_SHORT_NAMES = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "total_precipitation": "tp",
    "surface_pressure": "sp",
    "boundary_layer_height": "blh",
}


def compute_relative_humidity(t_k: np.ndarray, d_k: np.ndarray) -> np.ndarray:
    """
    Compute relative humidity from 2 m temperature and dew-point (both in K).

    Uses the Magnus approximation.

    Returns
    -------
    np.ndarray
        Relative humidity in percent [0–100].
    """
    t_c = t_k - 273.15
    d_c = d_k - 273.15
    e_sat = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    e_dew = 6.112 * np.exp(17.67 * d_c / (d_c + 243.5))
    rh = 100.0 * e_dew / e_sat
    return np.clip(rh, 0.0, 100.0)


def download_era5_monthly(
    year: int,
    month: int,
    output_dir: str | Path,
) -> Path:
    """
    Download one month of ERA5 hourly data over India for all variables.

    Parameters
    ----------
    year : int
    month : int
    output_dir : str | Path

    Returns
    -------
    Path to downloaded NetCDF file.
    """
    try:
        import cdsapi  # type: ignore
    except ImportError:
        raise RuntimeError("cdsapi not installed. Run: pip install cdsapi")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"era5_{year}_{month:02d}.nc"

    if out_file.exists():
        logger.info("Already exists: %s", out_file.name)
        return out_file

    c = cdsapi.Client()
    logger.info("Downloading ERA5 %d-%02d …", year, month)

    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": ERA5_VARIABLES,
            "year": str(year),
            "month": f"{month:02d}",
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": INDIA_AREA,
            "grid": [GRID_STEP, GRID_STEP],
            "format": "netcdf",
        },
        str(out_file),
    )
    return out_file


def process_era5_netcdf(nc_path: str | Path) -> pd.DataFrame:
    """
    Read an ERA5 NetCDF, compute daily means, and return a flat DataFrame.

    Parameters
    ----------
    nc_path : str | Path

    Returns
    -------
    pd.DataFrame with columns: lat, lon, date, t2m, d2m, u10, v10, tp, sp, blh, rh2m.
    """
    import xarray as xr

    ds = xr.open_dataset(nc_path)

    # Rename variables to short names
    rename_map: dict = {}
    for long, short in VAR_SHORT_NAMES.items():
        for possible in [long, short, short.upper()]:
            if possible in ds:
                rename_map[possible] = short
                break

    ds = ds.rename(rename_map)

    # Resample to daily means
    ds_daily = ds.resample(time="1D").mean()

    # Flatten to DataFrame
    rows: list[dict] = []
    times = ds_daily["time"].values

    for t_idx, t in enumerate(times):
        date_str = str(t)[:10]
        for lat_val in ds_daily["latitude"].values:
            for lon_val in ds_daily["longitude"].values:
                row: dict = {"lat": round(float(lat_val), 4), "lon": round(float(lon_val), 4), "date": date_str}
                for var in VAR_SHORT_NAMES.values():
                    if var in ds_daily:
                        val = float(ds_daily[var].sel(
                            latitude=lat_val, longitude=lon_val, time=t, method="nearest"
                        ).values)
                        row[var] = val
                # Compute relative humidity
                if "t2m" in row and "d2m" in row:
                    row["rh2m"] = float(compute_relative_humidity(
                        np.array([row["t2m"]]), np.array([row["d2m"]])
                    )[0])
                rows.append(row)

    return pd.DataFrame(rows)


def download_and_process_era5(
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Download ERA5 data for all months in [start_date, end_date] and process.

    Parameters
    ----------
    start_date, end_date : str
        ISO date strings.
    output_dir : str | Path
        Directory for raw NetCDF files.
    output_csv : str | Path
        Path for the output daily CSV.

    Returns
    -------
    pd.DataFrame
    """
    months = pd.date_range(start_date, end_date, freq="MS")
    all_dfs: list[pd.DataFrame] = []

    for month_ts in months:
        try:
            nc_path = download_era5_monthly(month_ts.year, month_ts.month, output_dir)
            df_month = process_era5_netcdf(nc_path)
            all_dfs.append(df_month)
            logger.info("  Processed %d-%02d: %d rows", month_ts.year, month_ts.month, len(df_month))
        except Exception as exc:
            logger.error("  Failed %d-%02d: %s", month_ts.year, month_ts.month, exc)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = combined[
        (combined["date"] >= pd.Timestamp(start_date)) &
        (combined["date"] <= pd.Timestamp(end_date))
    ]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    logger.info("Saved %d rows to %s", len(combined), output_csv)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 reanalysis data")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output_dir", default="data/raw/reanalysis")
    parser.add_argument("--output_csv", default="data/processed/era5_daily.csv")
    args = parser.parse_args()
    download_and_process_era5(args.start, args.end, args.output_dir, args.output_csv)


if __name__ == "__main__":
    main()
