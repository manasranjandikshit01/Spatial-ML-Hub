"""
build_dataset_aqi.py
====================
Joins all data sources (CPCB ground truth, TROPOMI satellite columns,
INSAT-3D AOD, ERA5 reanalysis) on a common spatial grid and daily time step
to produce the AQI model training dataset.

Outputs:
    data/processed/grid_daily_features.csv   — gridded satellite + met features
    data/processed/aqi_training_dataset.csv  — joined with CPCB targets

Usage:
    python -m src.data.build_dataset_aqi
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.grid_definition import assign_cell_id, get_india_grid
from src.utils.aqi_calculator import compute_aqi_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
SATELLITE_COLS = ["no2_column", "so2_column", "co_column", "o3_column", "hcho_column", "insat_aod"]
MET_COLS = ["t2m", "rh2m", "u10", "v10", "tp", "sp", "blh"]
POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "o3", "co"]
TARGET_COLS = ["pm25_target", "aqi_target"]


def load_config(path: str = "config/paths.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_tropomi_csvs(raw_dir: str | Path) -> pd.DataFrame:
    """
    Load all TROPOMI CSV files (one per pollutant) from *raw_dir* and merge.

    Each CSV has columns: lat, lon, date, <pollutant_col>.
    """
    raw_dir = Path(raw_dir)
    dfs: list[pd.DataFrame] = []

    pollutant_files = {
        "no2_column": list(raw_dir.glob("*no2*.csv")),
        "so2_column": list(raw_dir.glob("*so2*.csv")),
        "co_column": list(raw_dir.glob("*co*.csv")),
        "o3_column": list(raw_dir.glob("*o3*.csv")),
        "hcho_column": list(raw_dir.glob("*hcho*.csv")),
    }

    for col, files in pollutant_files.items():
        if not files:
            logger.warning("No files found for %s", col)
            continue
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        # Keep only needed cols
        keep = [c for c in ["lat", "lon", "date", "cell_id", col] if c in df.columns]
        dfs.append(df[keep])

    if not dfs:
        logger.warning("No TROPOMI data found in %s", raw_dir)
        return pd.DataFrame()

    # Merge all pollutants on (lat, lon, date)
    merged = dfs[0]
    for df in dfs[1:]:
        on_cols = [c for c in ["lat", "lon", "date", "cell_id"] if c in merged.columns and c in df.columns]
        merged = merged.merge(df, on=on_cols, how="outer")

    return merged


def build_grid_daily_features(
    tropomi_dir: str | Path,
    insat_csv: str | Path | None,
    era5_csv: str | Path | None,
    firms_csv: str | Path | None,
    output_csv: str | Path,
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Build the gridded daily feature matrix by joining all satellite and met sources.

    Parameters
    ----------
    tropomi_dir : str | Path
        Directory with processed TROPOMI CSVs (one per pollutant).
    insat_csv : str | Path | None
        Path to processed INSAT AOD daily CSV.
    era5_csv : str | Path | None
        Path to processed ERA5 daily CSV.
    firms_csv : str | Path | None
        Path to processed FIRMS fire count CSV.
    output_csv : str | Path
        Output file path.
    resolution : float
        Grid resolution in degrees.

    Returns
    -------
    pd.DataFrame
        Gridded daily feature dataset.
    """
    logger.info("Loading TROPOMI data …")
    df = load_tropomi_csvs(tropomi_dir)

    if df.empty:
        logger.warning("No satellite data available; building synthetic demo dataset.")
        df = _build_synthetic_features(resolution)

    if "cell_id" not in df.columns:
        df = assign_cell_id(df, resolution=resolution)

    df["date"] = pd.to_datetime(df["date"])

    # ---- INSAT AOD ----
    if insat_csv and Path(insat_csv).exists():
        logger.info("Merging INSAT AOD …")
        insat = pd.read_csv(insat_csv)
        insat["date"] = pd.to_datetime(insat["date"])
        if "cell_id" not in insat.columns:
            insat = assign_cell_id(insat, resolution=resolution)
        insat = insat[["cell_id", "date", "insat_aod"]].copy()
        df = df.merge(insat, on=["cell_id", "date"], how="left")
    else:
        logger.warning("INSAT AOD not found; column will be NaN.")
        if "insat_aod" not in df.columns:
            df["insat_aod"] = np.nan

    # ---- ERA5 ----
    if era5_csv and Path(era5_csv).exists():
        logger.info("Merging ERA5 reanalysis …")
        era5 = pd.read_csv(era5_csv)
        era5["date"] = pd.to_datetime(era5["date"])
        if "cell_id" not in era5.columns:
            era5 = assign_cell_id(era5, lat_col="lat", lon_col="lon", resolution=resolution)
        keep_met = ["cell_id", "date"] + [c for c in MET_COLS if c in era5.columns]
        df = df.merge(era5[keep_met], on=["cell_id", "date"], how="left")
    else:
        logger.warning("ERA5 data not found; met columns will be NaN.")
        for col in MET_COLS:
            if col not in df.columns:
                df[col] = np.nan

    # ---- FIRMS ----
    if firms_csv and Path(firms_csv).exists():
        logger.info("Merging FIRMS fire counts …")
        firms = pd.read_csv(firms_csv)
        firms["date"] = pd.to_datetime(firms["date"])
        if "cell_id" not in firms.columns:
            firms = assign_cell_id(firms, resolution=resolution)
        firms = firms[["cell_id", "date", "fire_count"]].copy()
        df = df.merge(firms, on=["cell_id", "date"], how="left")
        df["fire_count"] = df["fire_count"].fillna(0)
    else:
        df["fire_count"] = 0

    # Final column ordering
    base_cols = ["cell_id", "lat", "lon", "date"]
    feature_cols = SATELLITE_COLS + MET_COLS + ["fire_count"]
    ordered_cols = base_cols + [c for c in feature_cols if c in df.columns]
    df = df[[c for c in ordered_cols if c in df.columns]].copy()

    # Sort
    df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Saved grid_daily_features: %d rows, %d cols → %s", len(df), len(df.columns), output_csv)
    return df


def build_aqi_training_dataset(
    cpcb_csv: str | Path,
    grid_features_csv: str | Path,
    output_csv: str | Path,
) -> pd.DataFrame:
    """
    Join CPCB ground observations with gridded satellite/met features to produce
    the model training dataset.

    Each CPCB station row is matched to the nearest grid cell on the same date.

    Parameters
    ----------
    cpcb_csv : str | Path
        Path to data/processed/cpcb_daily.csv.
    grid_features_csv : str | Path
        Path to data/processed/grid_daily_features.csv.
    output_csv : str | Path
        Output path.

    Returns
    -------
    pd.DataFrame
    """
    logger.info("Building AQI training dataset …")

    cpcb = pd.read_csv(cpcb_csv)
    grid = pd.read_csv(grid_features_csv)

    cpcb["date"] = pd.to_datetime(cpcb["date"])
    grid["date"] = pd.to_datetime(grid["date"])

    if "cell_id" not in cpcb.columns:
        from src.data.grid_definition import assign_cell_id
        cpcb = assign_cell_id(cpcb)

    # Join on (cell_id, date)
    merged = cpcb.merge(grid, on=["cell_id", "date"], how="inner", suffixes=("_cpcb", "_sat"))

    # Rename targets
    for col in POLLUTANT_COLS:
        if col in merged.columns:
            merged[f"{col}_target"] = merged[col]

    # Compute AQI target
    target_df = merged[[c for c in POLLUTANT_COLS if c in merged.columns]]
    merged["aqi_target"] = compute_aqi_series(target_df)

    # Keep lat/lon from the grid side
    if "lat_sat" in merged.columns:
        merged["lat"] = merged["lat_sat"]
        merged["lon"] = merged["lon_sat"]

    feature_cols = [c for c in SATELLITE_COLS + MET_COLS if c in merged.columns]
    id_cols = ["station_id", "station_name", "city", "state", "cell_id", "date", "lat", "lon"]
    target_col_list = [c for c in [f"{p}_target" for p in POLLUTANT_COLS] + ["aqi_target"] if c in merged.columns]

    output_df = merged[[c for c in id_cols + feature_cols + target_col_list if c in merged.columns]]
    output_df = output_df.sort_values(["date", "station_id"]).reset_index(drop=True)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info("AQI training dataset: %d samples, %d features → %s", len(output_df), len(feature_cols), output_csv)
    return output_df


# ---------------------------------------------------------------------------
# Synthetic demo data (used when real data is not yet available)
# ---------------------------------------------------------------------------
def _build_synthetic_features(
    resolution: float = 0.1,
    start: str = "2019-01-01",
    end: str = "2022-12-31",
    n_cells: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a small synthetic feature dataset for demonstration purposes."""
    rng = np.random.default_rng(seed)
    grid = get_india_grid(resolution).sample(n_cells, random_state=42).reset_index(drop=True)
    dates = pd.date_range(start, end, freq="D")

    rows: list[dict] = []
    for date in dates:
        doy = date.day_of_year / 365.0
        for _, cell in grid.iterrows():
            lat_norm = (cell["lat"] - 8.0) / 30.0
            rows.append({
                "cell_id": cell["cell_id"],
                "lat": cell["lat"],
                "lon": cell["lon"],
                "date": date.strftime("%Y-%m-%d"),
                "no2_column": max(0, rng.normal(50 + 20 * lat_norm, 15)),
                "so2_column": max(0, rng.normal(30 + 10 * lat_norm, 10)),
                "co_column":  max(0, rng.normal(1.5 + 0.5 * lat_norm, 0.4)),
                "o3_column":  max(0, rng.normal(80 + 10 * np.sin(doy * 2 * np.pi), 15)),
                "hcho_column": max(0, rng.normal(40 + 20 * (1 - lat_norm), 12)),
                "insat_aod": max(0, rng.normal(0.4 + 0.2 * lat_norm, 0.15)),
                "t2m": 288 + 15 * np.sin(doy * 2 * np.pi) + rng.normal(0, 3),
                "rh2m": np.clip(rng.normal(55 + 30 * np.sin(doy * 2 * np.pi + 1), 15), 0, 100),
                "u10": rng.normal(2.0, 3.0),
                "v10": rng.normal(1.0, 2.0),
                "tp": max(0, rng.exponential(0.003)),
                "sp": rng.normal(95000, 2000),
                "blh": max(200, rng.normal(1200 + 500 * np.sin(doy * 2 * np.pi), 300)),
                "fire_count": int(max(0, rng.poisson(0.5 if 9 <= date.month <= 11 else 0.1))),
            })

    return pd.DataFrame(rows)


def generate_synthetic_cpcb(
    grid_csv: str | Path,
    output_csv: str | Path,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic CPCB dataset aligned to the grid features CSV."""
    rng = np.random.default_rng(seed)
    grid = pd.read_csv(grid_csv)
    grid["date"] = pd.to_datetime(grid["date"])

    # Use a random 20-cell subset as "stations"
    station_cells = grid["cell_id"].unique()
    station_cells = rng.choice(station_cells, size=min(20, len(station_cells)), replace=False)
    station_grid = grid[grid["cell_id"].isin(station_cells)].copy()

    def aod_to_pm25(row: pd.Series) -> float:
        aod = row.get("insat_aod", 0.4)
        no2 = row.get("no2_column", 50) / 100
        t = row.get("t2m", 295)
        rh = row.get("rh2m", 60)
        return max(5, aod * 120 + no2 * 30 + (100 - t) * 0.5 + rh * 0.3 + rng.normal(0, 5))

    rows: list[dict] = []
    for _, row in station_grid.iterrows():
        pm25 = aod_to_pm25(row)
        pm10 = pm25 * rng.uniform(1.5, 2.5)
        no2 = max(0, rng.normal(60, 20))
        so2 = max(0, rng.normal(30, 12))
        o3 = max(0, rng.normal(80, 20))
        co = max(0, rng.normal(1.5, 0.5))
        rows.append({
            "station_id": f"ST_{row['cell_id']}",
            "station_name": f"Station {row['cell_id'][:12]}",
            "city": "Demo City",
            "state": "Demo State",
            "lat": row["lat"],
            "lon": row["lon"],
            "cell_id": row["cell_id"],
            "date": row["date"],
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "no2": round(no2, 2),
            "so2": round(so2, 2),
            "o3": round(o3, 2),
            "co": round(co, 3),
        })

    df = pd.DataFrame(rows)
    from src.utils.aqi_calculator import compute_aqi_series
    df["aqi_observed"] = compute_aqi_series(df)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Synthetic CPCB: %d rows → %s", len(df), output_csv)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build AQI training dataset")
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for demo")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        paths = cfg["processed"]
        raw = cfg["raw"]
    except FileNotFoundError:
        cfg = {}
        raw = {"tropomi": "data/raw/tropomi", "firms": "data/raw/firms"}
        paths = {
            "grid_daily_features": "data/processed/grid_daily_features.csv",
            "cpcb_daily": "data/processed/cpcb_daily.csv",
            "aqi_training": "data/processed/aqi_training_dataset.csv",
        }

    grid_csv = paths["grid_daily_features"]
    cpcb_csv = paths.get("cpcb_daily", "data/processed/cpcb_daily.csv")
    training_csv = paths["aqi_training"]

    if args.synthetic or not Path(raw.get("tropomi", "data/raw/tropomi")).exists():
        logger.info("Generating synthetic feature data for demo …")
        grid_df = _build_synthetic_features()
        Path(grid_csv).parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(grid_csv, index=False)
        cpcb_df = generate_synthetic_cpcb(grid_csv, cpcb_csv)
    else:
        grid_df = build_grid_daily_features(
            raw["tropomi"],
            cfg.get("processed", {}).get("insat_aod"),
            cfg.get("processed", {}).get("era5"),
            cfg.get("processed", {}).get("firms"),
            grid_csv,
        )
        cpcb_df = pd.read_csv(cpcb_csv) if Path(cpcb_csv).exists() else pd.DataFrame()

    if not cpcb_df.empty and not grid_df.empty:
        build_aqi_training_dataset(cpcb_csv, grid_csv, training_csv)

    logger.info("Done.")


if __name__ == "__main__":
    main()
