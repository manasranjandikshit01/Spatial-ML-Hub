"""
download_insat_aod.py
=====================
Load and process INSAT-3D Aerosol Optical Depth (AOD) data from MOSDAC.

Portal: https://www.mosdac.gov.in/insat-3d-data-products

Manual download instructions
-----------------------------
1. Visit https://www.mosdac.gov.in and register / log in.
2. Navigate to Data → INSAT-3D → Derived Products → AOD.
3. Select the date range and download HDF5/NetCDF files.
4. Place the downloaded files under  data/raw/insat_aod/.

This script then reads those files, reprojects to the common 0.1° lat/lon
grid over India, and outputs daily GeoTIFF / CSV files.

Usage:
    python -m src.data.download_insat_aod \\
        --input_dir data/raw/insat_aod \\
        --output_dir data/interim/insat_aod \\
        --output_csv data/processed/insat_aod_daily.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def read_insat_aod_hdf(filepath: str | Path) -> pd.DataFrame:
    """
    Read an INSAT-3D AOD HDF5 file and return a flat DataFrame.

    INSAT-3D AOD files are typically HDF5 with datasets:
      /AOD (2D array)
      /Latitude (2D or 1D)
      /Longitude (2D or 1D)
      /QF or /QualityFlag (optional)

    Parameters
    ----------
    filepath : str | Path
        Path to one HDF5 or NetCDF file.

    Returns
    -------
    pd.DataFrame with columns: lat, lon, insat_aod, date.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise RuntimeError("h5py not installed. Run: pip install h5py")

    filepath = Path(filepath)
    rows: list[dict] = []

    with h5py.File(filepath, "r") as f:
        # Try to locate AOD and coordinate datasets
        aod_keys = [k for k in f.keys() if "aod" in k.lower() or "AOD" in k]
        lat_keys = [k for k in f.keys() if "lat" in k.lower()]
        lon_keys = [k for k in f.keys() if "lon" in k.lower()]

        if not aod_keys:
            logger.warning("No AOD dataset found in %s; keys: %s", filepath.name, list(f.keys()))
            return pd.DataFrame()

        aod = np.array(f[aod_keys[0]])
        lats = np.array(f[lat_keys[0]]) if lat_keys else None
        lons = np.array(f[lon_keys[0]]) if lon_keys else None

        # Fill / scale
        fill_value = f[aod_keys[0]].attrs.get("_FillValue", -9999)
        scale = f[aod_keys[0]].attrs.get("scale_factor", 1.0)
        aod = aod.astype(float)
        aod[aod == fill_value] = np.nan
        aod *= float(scale)

        if lats is None or lons is None:
            logger.error("Could not locate lat/lon in %s", filepath.name)
            return pd.DataFrame()

        if lats.ndim == 1 and aod.ndim == 2:
            lons_2d, lats_2d = np.meshgrid(lons, lats)
        else:
            lats_2d, lons_2d = lats, lons

        flat_aod = aod.ravel()
        flat_lats = lats_2d.ravel()
        flat_lons = lons_2d.ravel()

        valid = ~np.isnan(flat_aod)
        for lat, lon, val in zip(flat_lats[valid], flat_lons[valid], flat_aod[valid]):
            # Filter to India bbox
            if 8.0 <= lat <= 37.5 and 68.0 <= lon <= 97.5:
                rows.append({"lat": round(float(lat), 4), "lon": round(float(lon), 4), "insat_aod": float(val)})

    df = pd.DataFrame(rows)
    # Extract date from filename  (INSAT files often have YYYYMMDD in name)
    import re
    m = re.search(r"(\d{8})", filepath.name)
    df["date"] = m.group(1) if m else "unknown"
    return df


def read_insat_aod_netcdf(filepath: str | Path) -> pd.DataFrame:
    """
    Read an INSAT-3D AOD NetCDF file.

    Parameters
    ----------
    filepath : str | Path

    Returns
    -------
    pd.DataFrame with columns: lat, lon, insat_aod, date.
    """
    import xarray as xr

    ds = xr.open_dataset(filepath)

    aod_vars = [v for v in ds.data_vars if "aod" in v.lower()]
    if not aod_vars:
        raise KeyError(f"No AOD variable found. Available: {list(ds.data_vars)}")

    aod_var = aod_vars[0]
    da = ds[aod_var]

    if "lat" in ds.coords:
        lats, lons = ds["lat"].values, ds["lon"].values
    elif "latitude" in ds.coords:
        lats, lons = ds["latitude"].values, ds["longitude"].values
    else:
        raise KeyError("Cannot find lat/lon coordinates")

    aod_vals = da.values
    if aod_vals.ndim == 3:
        aod_vals = aod_vals[0]

    if lats.ndim == 1:
        lons_2d, lats_2d = np.meshgrid(lons, lats)
    else:
        lats_2d, lons_2d = lats, lons

    flat_aod = aod_vals.ravel().astype(float)
    flat_lats = lats_2d.ravel()
    flat_lons = lons_2d.ravel()

    valid = ~np.isnan(flat_aod)
    india_mask = (flat_lats[valid] >= 8.0) & (flat_lats[valid] <= 37.5) & \
                 (flat_lons[valid] >= 68.0) & (flat_lons[valid] <= 97.5)

    df = pd.DataFrame({
        "lat": np.round(flat_lats[valid][india_mask], 4),
        "lon": np.round(flat_lons[valid][india_mask], 4),
        "insat_aod": flat_aod[valid][india_mask],
    })

    import re
    m = re.search(r"(\d{8})", Path(filepath).name)
    df["date"] = m.group(1) if m else "unknown"
    return df


def regrid_to_common_grid(
    df: pd.DataFrame,
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Snap AOD values to the common 0.1° grid by averaging within each cell.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain lat, lon, insat_aod, date columns.
    resolution : float
        Target grid spacing.

    Returns
    -------
    pd.DataFrame on the target grid.
    """
    from src.data.grid_definition import assign_cell_id

    df = assign_cell_id(df, resolution=resolution)
    return (
        df.groupby(["cell_id", "cell_lat", "cell_lon", "date"], as_index=False)["insat_aod"]
        .mean()
        .rename(columns={"cell_lat": "lat", "cell_lon": "lon"})
    )


def process_insat_directory(
    input_dir: str | Path,
    output_csv: str | Path,
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Process all INSAT AOD files in *input_dir* to a single daily gridded CSV.

    Parameters
    ----------
    input_dir : str | Path
        Directory with raw HDF5/NetCDF files.
    output_csv : str | Path
        Output file path.
    resolution : float
        Target grid spacing.

    Returns
    -------
    pd.DataFrame
    """
    input_dir = Path(input_dir)
    files = list(input_dir.glob("*.h5")) + list(input_dir.glob("*.nc")) + \
            list(input_dir.glob("*.hdf"))

    if not files:
        logger.warning("No INSAT files found in %s", input_dir)
        return pd.DataFrame()

    dfs = []
    for fpath in sorted(files):
        try:
            if fpath.suffix in (".h5", ".hdf"):
                df = read_insat_aod_hdf(fpath)
            else:
                df = read_insat_aod_netcdf(fpath)
            if not df.empty:
                df = regrid_to_common_grid(df, resolution)
                dfs.append(df)
                logger.info("  Processed %s: %d cells", fpath.name, len(df))
        except Exception as exc:
            logger.error("  Failed %s: %s", fpath.name, exc)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    logger.info("Saved %d rows to %s", len(combined), output_csv)
    return combined


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Process INSAT-3D AOD data")
    parser.add_argument("--input_dir", default="data/raw/insat_aod")
    parser.add_argument("--output_csv", default="data/processed/insat_aod_daily.csv")
    args = parser.parse_args()
    process_insat_directory(args.input_dir, args.output_csv)


if __name__ == "__main__":
    main()
