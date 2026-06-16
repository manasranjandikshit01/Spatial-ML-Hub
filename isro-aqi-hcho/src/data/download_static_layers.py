"""
download_static_layers.py
=========================
Download and process static (time-invariant) geospatial layers for the ISRO
AQI pipeline.  These optional layers improve prediction accuracy, especially
for exposure-weighted AQI maps.

Currently supported layers
--------------------------
land_cover  — ESA CCI Land Cover (annual, 300 m → resampled to 0.1°)
population  — WorldPop gridded population count (~1 km → 0.1°)
elevation   — SRTM 90 m DEM → mean elevation per 0.1° cell

Configuration flags (config/paths.yaml)
----------------------------------------
extra_features:
  use_land_cover: false
  use_population: false
  use_elevation: false

Usage::

    python -m src.data.download_static_layers --layers land_cover population

Note
----
All layers require internet access and external credentials/licences.
See the download instructions per dataset below.  The functions in this module
write CSVs to ``data/interim/static/`` with one row per grid cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# India 0.1° grid (matches grid_definition.py)
_LAT_MIN, _LAT_MAX = 8.0, 37.5
_LON_MIN, _LON_MAX = 68.0, 97.5
_RES = 0.1


def _base_grid() -> pd.DataFrame:
    """Return the standard 0.1° India grid as a DataFrame."""
    lats = np.arange(_LAT_MIN, _LAT_MAX, _RES).round(1)
    lons = np.arange(_LON_MIN, _LON_MAX, _RES).round(1)
    lat_g, lon_g = np.meshgrid(lats, lons, indexing="ij")
    df = pd.DataFrame({
        "lat": lat_g.ravel(),
        "lon": lon_g.ravel(),
    })
    df["cell_id"] = "CELL_" + df["lat"].map("{:.2f}".format) + "_" + df["lon"].map("{:.2f}".format)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Land cover
# ──────────────────────────────────────────────────────────────────────────────

def download_land_cover(
    output_dir: str | Path = "data/interim/static",
    year: int = 2020,
) -> pd.DataFrame:
    """
    Download ESA CCI Land Cover and aggregate to the 0.1° India grid.

    Real data source
    ----------------
    https://maps.elie.ucl.ac.be/CCI/viewer/download.php
    Requires free registration. Download the annual GeoTIFF for *year* and
    set ``LC_GEOTIFF_PATH`` in your ``.env``.

    Synthetic fallback
    ------------------
    When the GeoTIFF is not available, a realistic synthetic land cover
    distribution is generated (IGP = cropland, coasts = water, forests = NE).

    Output columns
    --------------
    cell_id, lat, lon, lc_cropland_frac, lc_forest_frac, lc_urban_frac,
    lc_water_frac, lc_barren_frac, lc_dominant_class
    """
    import os

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"land_cover_{year}.csv"

    geotiff = os.getenv("LC_GEOTIFF_PATH")
    if geotiff and Path(geotiff).exists():
        logger.info("Loading ESA CCI land cover from %s …", geotiff)
        try:
            import rasterio
            from rasterio.transform import rowcol

            grid = _base_grid()
            with rasterio.open(geotiff) as src:
                # Sample the raster at each grid cell centre
                coords = [(row.lon, row.lat) for _, row in grid.iterrows()]
                samples = [v[0] for v in src.sample(coords)]
            grid["lc_class"] = samples
            # ESA CCI classes: 10-30 cropland, 50-90 forest, 190 urban, 210 water
            grid["lc_cropland_frac"] = grid["lc_class"].isin(range(10, 40)).astype(float)
            grid["lc_forest_frac"] = grid["lc_class"].isin(range(50, 100)).astype(float)
            grid["lc_urban_frac"] = (grid["lc_class"] == 190).astype(float)
            grid["lc_water_frac"] = grid["lc_class"].isin([210, 220]).astype(float)
            grid["lc_barren_frac"] = grid["lc_class"].isin([150, 152, 153]).astype(float)
            grid["lc_dominant_class"] = grid["lc_class"]
            cols = ["cell_id","lat","lon","lc_cropland_frac","lc_forest_frac",
                    "lc_urban_frac","lc_water_frac","lc_barren_frac","lc_dominant_class"]
            out = grid[cols]
            out.to_csv(out_path, index=False)
            logger.info("Land cover saved: %d cells → %s", len(out), out_path)
            return out
        except Exception as exc:
            logger.warning("Real land cover failed (%s); using synthetic.", exc)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    logger.info("Generating synthetic land cover (no GeoTIFF found) …")
    rng = np.random.default_rng(0)
    grid = _base_grid()

    # Realistic heuristics
    igp = (grid["lat"] >= 23) & (grid["lat"] <= 30) & (grid["lon"] >= 75) & (grid["lon"] <= 90)
    ne  = (grid["lat"] >= 23) & (grid["lat"] >= 88)
    coastal = (grid["lat"] <= 12)

    grid["lc_cropland_frac"] = 0.2 + 0.5 * igp.astype(float) + rng.uniform(0, 0.1, len(grid))
    grid["lc_forest_frac"]   = 0.1 + 0.4 * ne.astype(float) + rng.uniform(0, 0.1, len(grid))
    grid["lc_urban_frac"]    = 0.05 + rng.uniform(0, 0.05, len(grid))
    grid["lc_water_frac"]    = 0.05 + 0.2 * coastal.astype(float)
    grid["lc_barren_frac"]   = rng.uniform(0, 0.1, len(grid))

    # Normalise fractions to ≤ 1
    frac_cols = ["lc_cropland_frac","lc_forest_frac","lc_urban_frac","lc_water_frac","lc_barren_frac"]
    total = grid[frac_cols].sum(axis=1).clip(lower=1)
    grid[frac_cols] = grid[frac_cols].div(total, axis=0)

    grid["lc_dominant_class"] = grid[frac_cols].idxmax(axis=1).map({
        "lc_cropland_frac": 20, "lc_forest_frac": 60,
        "lc_urban_frac": 190, "lc_water_frac": 210, "lc_barren_frac": 150,
    })

    out = grid[["cell_id","lat","lon"] + frac_cols + ["lc_dominant_class"]]
    out.to_csv(out_path, index=False)
    logger.info("Synthetic land cover: %d cells → %s", len(out), out_path)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Population density
# ──────────────────────────────────────────────────────────────────────────────

def download_population(
    output_dir: str | Path = "data/interim/static",
    year: int = 2020,
) -> pd.DataFrame:
    """
    Download WorldPop gridded population and aggregate to the 0.1° India grid.

    Real data source
    ----------------
    https://www.worldpop.org/geodata/summary?id=24767
    Download the India GeoTIFF and set ``WORLDPOP_GEOTIFF_PATH`` in ``.env``.

    Output columns
    --------------
    cell_id, lat, lon, population_count, population_density_per_km2
    """
    import os

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"population_{year}.csv"

    geotiff = os.getenv("WORLDPOP_GEOTIFF_PATH")
    if geotiff and Path(geotiff).exists():
        logger.info("Loading WorldPop population from %s …", geotiff)
        try:
            import rasterio
            grid = _base_grid()
            with rasterio.open(geotiff) as src:
                coords = [(row.lon, row.lat) for _, row in grid.iterrows()]
                samples = [max(0.0, v[0] if v[0] is not None else 0.0) for v in src.sample(coords)]
            grid["population_count"] = samples
            grid["population_density_per_km2"] = grid["population_count"] / (11.1 ** 2)
            out = grid[["cell_id","lat","lon","population_count","population_density_per_km2"]]
            out.to_csv(out_path, index=False)
            logger.info("Population saved: %d cells → %s", len(out), out_path)
            return out
        except Exception as exc:
            logger.warning("Real population data failed (%s); using synthetic.", exc)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    logger.info("Generating synthetic population density …")
    rng = np.random.default_rng(1)
    grid = _base_grid()

    # Urban centres get high population
    city_centres = [
        (28.65, 77.23, 28_514_000),   # Delhi
        (19.08, 72.88, 20_667_000),   # Mumbai
        (22.57, 88.36, 14_850_000),   # Kolkata
        (13.08, 80.27, 10_971_000),   # Chennai
        (12.97, 77.59, 12_765_000),   # Bangalore
    ]
    grid["population_count"] = 5000.0  # base population per cell
    for clat, clon, total_pop in city_centres:
        dist2 = (grid["lat"] - clat) ** 2 + (grid["lon"] - clon) ** 2
        grid["population_count"] += total_pop / 500 * np.exp(-dist2 / 0.5)

    grid["population_count"] = (
        grid["population_count"] * (1 + rng.uniform(0, 0.3, len(grid)))
    ).clip(0).round(0)
    grid["population_density_per_km2"] = (grid["population_count"] / (11.1 ** 2)).round(1)

    out = grid[["cell_id","lat","lon","population_count","population_density_per_km2"]]
    out.to_csv(out_path, index=False)
    logger.info("Synthetic population: %d cells → %s", len(out), out_path)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Elevation
# ──────────────────────────────────────────────────────────────────────────────

def download_elevation(
    output_dir: str | Path = "data/interim/static",
) -> pd.DataFrame:
    """
    Compute mean elevation per 0.1° cell from SRTM 90 m DEM.

    Real data source
    ----------------
    https://srtm.csi.cgiar.org/ — Download tiles for India (tiles 55-66, rows 05-07).
    Set ``SRTM_DIR`` in ``.env`` to the directory containing the GeoTIFF tiles.

    Output columns
    --------------
    cell_id, lat, lon, elevation_m, slope_deg
    """
    import os
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "elevation.csv"

    srtm_dir = os.getenv("SRTM_DIR")
    if srtm_dir and Path(srtm_dir).exists():
        logger.info("Loading SRTM elevation from %s …", srtm_dir)
        try:
            import rasterio
            from rasterio.merge import merge
            tifs = list(Path(srtm_dir).glob("*.tif"))
            if tifs:
                datasets = [rasterio.open(t) for t in tifs]
                mosaic, transform = merge(datasets)
                for ds in datasets:
                    ds.close()
                grid = _base_grid()
                grid["elevation_m"] = [
                    float(mosaic[0, int((row.lat - transform.f) / transform.e),
                                   int((row.lon - transform.c) / transform.a)])
                    for _, row in grid.iterrows()
                ]
                grid["elevation_m"] = grid["elevation_m"].clip(-200, 8850)
                grid["slope_deg"] = 0.0  # simplified
                out = grid[["cell_id","lat","lon","elevation_m","slope_deg"]]
                out.to_csv(out_path, index=False)
                logger.info("Elevation saved: %d cells → %s", len(out), out_path)
                return out
        except Exception as exc:
            logger.warning("Real elevation failed (%s); using synthetic.", exc)

    # ── Synthetic fallback ────────────────────────────────────────────────────
    logger.info("Generating synthetic elevation (DEM not found) …")
    grid = _base_grid()
    # Western Ghats (~76°E) and Himalayas (>32°N)
    himalaya = np.clip((grid["lat"] - 28) * 200, 0, 5000).values
    ghats = np.clip((0.5 - abs(grid["lon"] - 76)) * 2000, 0, 1500).values
    igp_depression = np.clip((30 - grid["lat"]).clip(0) * (-10), -200, 0).values
    rng = np.random.default_rng(2)
    grid["elevation_m"] = (himalaya + ghats + igp_depression + rng.uniform(0, 100, len(grid))).clip(0, 5000).round(1)
    grid["slope_deg"] = (rng.exponential(2, len(grid))).clip(0, 30).round(2)

    out = grid[["cell_id","lat","lon","elevation_m","slope_deg"]]
    out.to_csv(out_path, index=False)
    logger.info("Synthetic elevation: %d cells → %s", len(out), out_path)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

LAYER_MAP = {
    "land_cover": download_land_cover,
    "population": download_population,
    "elevation": download_elevation,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download static spatial layers")
    parser.add_argument("--layers", nargs="+",
                        choices=list(LAYER_MAP.keys()) + ["all"],
                        default=["all"],
                        help="Which layers to download (default: all)")
    parser.add_argument("--output_dir", default="data/interim/static")
    parser.add_argument("--year", type=int, default=2020,
                        help="Reference year for land cover and population")
    args = parser.parse_args()

    layers = list(LAYER_MAP.keys()) if "all" in args.layers else args.layers
    for layer in layers:
        logger.info("--- Downloading: %s ---", layer)
        fn = LAYER_MAP[layer]
        if layer in ("land_cover", "population"):
            fn(args.output_dir, args.year)
        else:
            fn(args.output_dir)


if __name__ == "__main__":
    main()
