"""
build_dataset_hcho.py
=====================
Build the HCHO + fire gridded daily dataset used for hotspot detection
and correlation analysis.

Joins:
  - TROPOMI HCHO column
  - FIRMS fire counts
  - ERA5 wind, BLH, precipitation
  - Season labels

Output: data/processed/hcho_fire_daily_grid.csv

Schema:
    cell_id, lat, lon, date, hcho_column, fire_count,
    u10, v10, blh, tp, season

Usage:
    python -m src.data.build_dataset_hcho
    python -m src.data.build_dataset_hcho --synthetic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.grid_definition import assign_cell_id, get_india_grid

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEASON_MAP = {
    1: "winter", 2: "winter",
    3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon",
    12: "winter",
}

HCHO_TRANSPORT_COLS = ["u10", "v10", "blh", "tp"]


def label_season(dates: pd.Series) -> pd.Series:
    """Map a Series of datetime-like values to season strings."""
    months = pd.to_datetime(dates).dt.month
    return months.map(SEASON_MAP)


def build_hcho_fire_dataset(
    tropomi_dir: str | Path,
    firms_csv: str | Path | None,
    era5_csv: str | Path | None,
    output_csv: str | Path,
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Join HCHO, fire counts, and met data into a single gridded daily CSV.

    Parameters
    ----------
    tropomi_dir : str | Path
        Directory with TROPOMI processed CSVs; HCHO must be present.
    firms_csv : str | Path | None
    era5_csv : str | Path | None
    output_csv : str | Path
    resolution : float

    Returns
    -------
    pd.DataFrame
    """
    # ---- HCHO ----
    hcho_files = list(Path(tropomi_dir).glob("*hcho*.csv"))
    if not hcho_files:
        logger.warning("No HCHO files found; generating synthetic data.")
        return _build_synthetic_hcho_dataset(output_csv)

    hcho = pd.concat([pd.read_csv(f) for f in hcho_files], ignore_index=True)
    hcho["date"] = pd.to_datetime(hcho["date"])
    if "cell_id" not in hcho.columns:
        hcho = assign_cell_id(hcho, resolution=resolution)

    df = hcho[["cell_id", "lat", "lon", "date", "hcho_column"]].copy()

    # ---- FIRMS ----
    if firms_csv and Path(firms_csv).exists():
        firms = pd.read_csv(firms_csv)
        firms["date"] = pd.to_datetime(firms["date"])
        if "cell_id" not in firms.columns:
            firms = assign_cell_id(firms, resolution=resolution)
        df = df.merge(firms[["cell_id", "date", "fire_count"]], on=["cell_id", "date"], how="left")
        df["fire_count"] = df["fire_count"].fillna(0).astype(int)
    else:
        df["fire_count"] = 0

    # ---- ERA5 ----
    if era5_csv and Path(era5_csv).exists():
        era5 = pd.read_csv(era5_csv)
        era5["date"] = pd.to_datetime(era5["date"])
        if "cell_id" not in era5.columns:
            era5 = assign_cell_id(era5, resolution=resolution)
        keep = ["cell_id", "date"] + [c for c in HCHO_TRANSPORT_COLS if c in era5.columns]
        df = df.merge(era5[keep], on=["cell_id", "date"], how="left")
    else:
        for col in HCHO_TRANSPORT_COLS:
            df[col] = np.nan

    # ---- Season ----
    df["season"] = label_season(df["date"])

    df = df.sort_values(["date", "cell_id"]).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("HCHO-fire dataset: %d rows → %s", len(df), output_csv)
    return df


def _build_synthetic_hcho_dataset(
    output_csv: str | Path,
    resolution: float = 0.1,
    start: str = "2019-01-01",
    end: str = "2022-12-31",
    n_cells: int = 300,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Generate a synthetic HCHO + fire dataset for demonstration.

    The synthetic data mimics seasonal biomass burning patterns:
    - Post-monsoon (Oct–Nov): elevated fire + HCHO over NW India (Punjab/Haryana)
    - Pre-monsoon (Mar–May): elevated fire + HCHO over NE India / Central forests
    """
    rng = np.random.default_rng(seed)
    grid = get_india_grid(resolution).sample(n_cells, random_state=42).reset_index(drop=True)
    dates = pd.date_range(start, end, freq="D")

    rows: list[dict] = []
    for date in dates:
        month = date.month
        doy = date.day_of_year / 365.0
        for _, cell in grid.iterrows():
            lat, lon = cell["lat"], cell["lon"]

            # Biomass burning zones
            is_igp = 23 <= lat <= 30 and 75 <= lon <= 90
            is_ne = 23 <= lat <= 28 and 90 <= lon <= 97.5
            is_central = 18 <= lat <= 25 and 77 <= lon <= 87

            base_hcho = 40.0

            # Post-monsoon crop residue burning (Punjab/Haryana – IGP)
            if month in (10, 11) and 28 <= lat <= 32 and 73 <= lon <= 78:
                fire_rate = rng.poisson(8)
                base_hcho += 60 + rng.normal(0, 10)
            # Pre-monsoon forest fires (NE + Central)
            elif month in (3, 4, 5) and (is_ne or is_central):
                fire_rate = rng.poisson(5)
                base_hcho += 40 + rng.normal(0, 8)
            elif is_igp:
                fire_rate = rng.poisson(1)
                base_hcho += 10
            else:
                fire_rate = rng.poisson(0.2)

            hcho = max(5, base_hcho + rng.normal(0, 8))
            u10 = rng.normal(2.5, 3.5)
            v10 = rng.normal(1.0, 2.5)
            blh = max(200, rng.normal(1200, 400))
            tp = max(0, rng.exponential(0.003 if month in (6, 7, 8, 9) else 0.0005))

            rows.append({
                "cell_id": cell["cell_id"],
                "lat": lat,
                "lon": lon,
                "date": date.strftime("%Y-%m-%d"),
                "hcho_column": round(hcho, 3),
                "fire_count": int(fire_rate),
                "u10": round(u10, 3),
                "v10": round(v10, 3),
                "blh": round(blh, 1),
                "tp": round(tp, 6),
                "season": SEASON_MAP[month],
            })

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info("Synthetic HCHO-fire dataset: %d rows → %s", len(df), output_csv)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build HCHO-fire dataset")
    parser.add_argument("--config", default="config/paths.yaml")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        raw = cfg["raw"]
        processed = cfg["processed"]
    except FileNotFoundError:
        raw = {"tropomi": "data/raw/tropomi"}
        processed = {"hcho_fire": "data/processed/hcho_fire_daily_grid.csv", "era5": None, "firms": None}

    if args.synthetic:
        _build_synthetic_hcho_dataset(processed["hcho_fire"])
    else:
        build_hcho_fire_dataset(
            raw["tropomi"],
            processed.get("firms"),
            processed.get("era5"),
            processed["hcho_fire"],
        )


if __name__ == "__main__":
    main()
