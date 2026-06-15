"""
download_tropomi.py
===================
Download Sentinel-5P TROPOMI Level-3 data for NO2, SO2, CO, O3, and HCHO
over India, using Google Earth Engine (preferred) or direct DLR download.

Earth Engine datasets:
  COPERNICUS/S5P/OFFL/L3_NO2
  COPERNICUS/S5P/OFFL/L3_SO2
  COPERNICUS/S5P/OFFL/L3_CO
  COPERNICUS/S5P/OFFL/L3_O3
  COPERNICUS/S5P/OFFL/L3_HCHO

DLR portal: https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/

Requirements (Google Earth Engine route):
    pip install earthengine-api
    earthengine authenticate

Usage:
    # Google Earth Engine
    python -m src.data.download_tropomi \\
        --method gee \\
        --start 2019-01-01 --end 2019-12-31 \\
        --output_dir data/raw/tropomi

    # DLR HTTP download
    python -m src.data.download_tropomi \\
        --method dlr \\
        --pollutant NO2 \\
        --start 2019-01-01 --end 2019-01-31 \\
        --output_dir data/raw/tropomi
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Band names for each pollutant in GEE
# ---------------------------------------------------------------------------
GEE_COLLECTIONS = {
    "NO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_NO2",
        "band": "tropospheric_NO2_column_number_density",
        "qa_band": "qa_value",
        "qa_threshold": 0.75,
        "scale_factor": 1e6,   # mol/m² → µmol/m²  (for readability)
        "output_col": "no2_column",
    },
    "SO2": {
        "collection": "COPERNICUS/S5P/OFFL/L3_SO2",
        "band": "SO2_column_number_density",
        "qa_band": "qa_value",
        "qa_threshold": 0.5,
        "scale_factor": 1e6,
        "output_col": "so2_column",
    },
    "CO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_CO",
        "band": "CO_column_number_density",
        "qa_band": "qa_value",
        "qa_threshold": 0.5,
        "scale_factor": 1e3,
        "output_col": "co_column",
    },
    "O3": {
        "collection": "COPERNICUS/S5P/OFFL/L3_O3",
        "band": "O3_column_number_density",
        "qa_band": "qa_value",
        "qa_threshold": 0.5,
        "scale_factor": 1e3,
        "output_col": "o3_column",
    },
    "HCHO": {
        "collection": "COPERNICUS/S5P/OFFL/L3_HCHO",
        "band": "tropospheric_HCHO_column_number_density",
        "qa_band": "qa_value",
        "qa_threshold": 0.5,
        "scale_factor": 1e6,
        "output_col": "hcho_column",
    },
}

INDIA_BBOX = [68.0, 8.0, 97.5, 37.5]   # [west, south, east, north]
GRID_RESOLUTION = 0.1                   # degrees


def download_gee(
    pollutants: list[str],
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    project_id: str | None = None,
) -> None:
    """
    Export TROPOMI daily gridded data for *pollutants* using Google Earth Engine.

    The function reduces each collection to a daily mean on the 0.1° grid,
    applies the QA mask, and exports to Drive (then download manually) or
    saves a CSV if running in notebook mode.

    Parameters
    ----------
    pollutants : list[str]
        Subset of ["NO2", "SO2", "CO", "O3", "HCHO"].
    start_date, end_date : str
        ISO date strings.
    output_dir : str | Path
        Where to save exported files.
    project_id : str | None
        GEE Cloud project ID (required for recent ee versions).
    """
    try:
        import ee  # type: ignore
    except ImportError:
        raise RuntimeError(
            "earthengine-api not installed. Run: pip install earthengine-api"
        )

    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception as exc:
        raise RuntimeError(
            f"Earth Engine initialisation failed: {exc}\n"
            "Run: earthengine authenticate"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    india_geom = ee.Geometry.Rectangle(INDIA_BBOX)

    for poll in pollutants:
        info = GEE_COLLECTIONS[poll]
        logger.info("Processing %s from %s …", poll, info["collection"])

        collection = (
            ee.ImageCollection(info["collection"])
            .filterDate(start_date, end_date)
            .filterBounds(india_geom)
            .select([info["band"], info["qa_band"]])
        )

        def mask_and_select(img: "ee.Image") -> "ee.Image":
            qa = img.select(info["qa_band"])
            mask = qa.gte(info["qa_threshold"])
            return img.select(info["band"]).updateMask(mask).multiply(info["scale_factor"])

        masked = collection.map(mask_and_select)

        # Reduce each day's mosaic to a mean
        dates = pd.date_range(start_date, end_date, freq="D")
        rows: list[dict] = []

        for date in dates:
            date_str = date.strftime("%Y-%m-%d")
            next_str = (date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            day_img = masked.filterDate(date_str, next_str).mean()

            region = day_img.sample(
                region=india_geom,
                scale=int(GRID_RESOLUTION * 111_000),  # approx metres
                numPixels=5000,
                seed=42,
            )

            try:
                points = region.getInfo()["features"]
                for pt in points:
                    props = pt["properties"]
                    coords = pt["geometry"]["coordinates"]
                    rows.append({
                        "lon": round(coords[0], 4),
                        "lat": round(coords[1], 4),
                        "date": date_str,
                        info["output_col"]: props.get(info["band"]),
                    })
            except Exception as exc:
                logger.warning("  %s %s: %s", poll, date_str, exc)
                time.sleep(2)

        if rows:
            out_csv = output_dir / f"tropomi_{poll.lower()}_{start_date}_{end_date}.csv"
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            logger.info("  Saved %d rows to %s", len(rows), out_csv)


def download_dlr(
    pollutant: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path,
    username: str = "",
    password: str = "",
) -> None:
    """
    Download TROPOMI L3 NetCDF files from the DLR geoservice portal.

    Portal: https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/

    Parameters
    ----------
    pollutant : str
        One of the keys in GEE_COLLECTIONS, e.g. "NO2".
    start_date, end_date : str
        ISO date strings.
    output_dir : str | Path
        Where to save raw NetCDF files.
    username, password : str
        DLR portal credentials (register at geoservice.dlr.de).
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"https://download.geoservice.dlr.de/S5P_TROPOMI/files/L3/{pollutant}/"
    logger.info("Listing DLR files for %s …", pollutant)

    try:
        resp = requests.get(base_url, auth=(username, password), timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logger.error("DLR listing failed: %s", exc)
        logger.info(
            "Please register at https://geoservice.dlr.de and set credentials."
        )
        return

    # Parse file links – simple HTML scraping (DLR uses Apache autoindex)
    import re
    dates = pd.date_range(start_date, end_date, freq="D")
    date_strings = {d.strftime("%Y%m%d") for d in dates}

    file_links = re.findall(r'href="(S5P_OFFL_L3_[^"]+\.nc)"', resp.text)
    for fname in file_links:
        if any(ds in fname for ds in date_strings):
            url = base_url + fname
            out_path = output_dir / fname
            if out_path.exists():
                continue
            logger.info("  Downloading %s …", fname)
            with requests.get(url, auth=(username, password), stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)


def read_tropomi_netcdf(nc_path: str | Path, pollutant: str) -> pd.DataFrame:
    """
    Read a single TROPOMI L3 NetCDF file and return a flat DataFrame.

    Parameters
    ----------
    nc_path : str | Path
        Path to the NetCDF file.
    pollutant : str
        Key in GEE_COLLECTIONS (e.g. "NO2").

    Returns
    -------
    pd.DataFrame with columns: lat, lon, date, <output_col>.
    """
    import xarray as xr

    info = GEE_COLLECTIONS[pollutant]
    ds = xr.open_dataset(nc_path)

    band_names = list(ds.data_vars)
    matching = [b for b in band_names if info["band"].lower() in b.lower()]
    if not matching:
        raise KeyError(f"Band {info['band']!r} not found in {nc_path}. Available: {band_names}")

    data_var = matching[0]
    arr = ds[data_var].values * info["scale_factor"]

    lats = ds["lat"].values if "lat" in ds else ds["latitude"].values
    lons = ds["lon"].values if "lon" in ds else ds["longitude"].values

    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = arr[i, j] if arr.ndim == 2 else arr[0, i, j]
            if not np.isnan(val):
                rows.append({"lat": lat, "lon": lon, info["output_col"]: val})

    df = pd.DataFrame(rows)
    # Extract date from filename
    import re
    m = re.search(r"(\d{8})", Path(nc_path).name)
    df["date"] = m.group(1) if m else "unknown"

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Download TROPOMI satellite data")
    parser.add_argument("--method", choices=["gee", "dlr"], default="gee")
    parser.add_argument(
        "--pollutants", nargs="+", default=["NO2", "SO2", "CO", "O3", "HCHO"],
        help="Pollutants to download (space-separated)"
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--output_dir", default="data/raw/tropomi")
    parser.add_argument("--gee_project", default=None, help="GEE Cloud project ID")
    parser.add_argument("--dlr_user", default="", help="DLR username")
    parser.add_argument("--dlr_pass", default="", help="DLR password")
    args = parser.parse_args()

    if args.method == "gee":
        download_gee(args.pollutants, args.start, args.end, args.output_dir, args.gee_project)
    else:
        for poll in args.pollutants:
            download_dlr(poll, args.start, args.end, args.output_dir, args.dlr_user, args.dlr_pass)


if __name__ == "__main__":
    main()
