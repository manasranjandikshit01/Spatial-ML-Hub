"""
add_static_features.py
======================
Merge time-invariant static layers (land cover, population density, elevation)
into the main AQI / HCHO feature DataFrames.

This module is wired into ``build_dataset_aqi.py`` via config flags::

    extra_features:
      use_land_cover: true
      use_population: true
      use_elevation: false

Usage::

    from src.features.add_static_features import add_static_features

    df = pd.read_csv("data/processed/aqi_training_dataset.csv")
    config = yaml.safe_load(open("config/paths.yaml"))
    df_enriched = add_static_features(df, config)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

STATIC_DIR = Path("data/interim/static")

# ── Column lists for each static layer ───────────────────────────────────────
LAND_COVER_COLS = [
    "lc_cropland_frac", "lc_forest_frac", "lc_urban_frac",
    "lc_water_frac", "lc_barren_frac",
]
POPULATION_COLS = [
    "population_count", "population_density_per_km2",
]
ELEVATION_COLS = [
    "elevation_m", "slope_deg",
]


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def _load_static_csv(filename: str, required_cols: list[str]) -> pd.DataFrame | None:
    """
    Load a static-layer CSV if it exists and has the required columns.

    Returns
    -------
    pd.DataFrame or None (if file missing or invalid).
    """
    path = STATIC_DIR / filename
    if not path.exists():
        logger.warning("Static layer not found: %s — skipping.", path)
        return None
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Static layer %s missing columns %s — skipping.", filename, missing)
        return None
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Merge helpers
# ──────────────────────────────────────────────────────────────────────────────

def _merge_on_cell_id(
    df: pd.DataFrame,
    static_df: pd.DataFrame,
    feature_cols: list[str],
    layer_name: str = "",
) -> pd.DataFrame:
    """
    Left-join *static_df* columns into *df* on ``cell_id``.

    Falls back to ``(lat, lon)`` merge if ``cell_id`` is absent.
    """
    merge_col = "cell_id" if "cell_id" in df.columns and "cell_id" in static_df.columns else None
    if merge_col is None and "lat" in df.columns and "lat" in static_df.columns:
        merge_col = ["lat", "lon"]

    if merge_col is None:
        logger.warning("Cannot merge %s layer: no cell_id or lat/lon in dataset.", layer_name)
        return df

    cols_to_add = [c for c in feature_cols if c in static_df.columns and c not in df.columns]
    if not cols_to_add:
        return df

    key_cols = [merge_col] if isinstance(merge_col, str) else merge_col
    static_sub = static_df[key_cols + cols_to_add].drop_duplicates(subset=key_cols)
    merged = df.merge(static_sub, on=merge_col, how="left")
    n_matched = merged[cols_to_add[0]].notna().sum()
    logger.info("  %s: merged %d feature(s), %d/%d rows matched",
                layer_name, len(cols_to_add), n_matched, len(merged))
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def add_static_features(
    df: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Conditionally merge static spatial layers into *df*.

    The layers merged are controlled by ``extra_features`` flags in the config:

    - ``use_land_cover``  → land cover fractions per 0.1° cell
    - ``use_population``  → population count and density per km²
    - ``use_elevation``   → mean elevation and slope per cell

    Parameters
    ----------
    df : pd.DataFrame
        Main feature DataFrame (must have ``cell_id`` or ``lat``/``lon``).
    config : dict | None
        paths.yaml contents (or equivalent).  If None, all layers are skipped.

    Returns
    -------
    pd.DataFrame  Extended with static feature columns (NaN if unmatched).
    """
    if config is None:
        return df

    extra = config.get("extra_features", {})

    # Land cover
    if extra.get("use_land_cover", False):
        logger.info("Adding land cover features …")
        lc = _load_static_csv("land_cover_2020.csv", LAND_COVER_COLS)
        if lc is not None:
            df = _merge_on_cell_id(df, lc, LAND_COVER_COLS, "land_cover")

    # Population density
    if extra.get("use_population", False):
        logger.info("Adding population density features …")
        pop = _load_static_csv("population_2020.csv", POPULATION_COLS)
        if pop is not None:
            df = _merge_on_cell_id(df, pop, POPULATION_COLS, "population")

    # Elevation
    if extra.get("use_elevation", False):
        logger.info("Adding elevation features …")
        elev = _load_static_csv("elevation.csv", ELEVATION_COLS)
        if elev is not None:
            df = _merge_on_cell_id(df, elev, ELEVATION_COLS, "elevation")

    return df


def prepare_static_layers(config: dict) -> None:
    """
    Download and prepare all enabled static layers according to *config*.

    Call this once before building datasets; layers are cached to disk.

    Parameters
    ----------
    config : dict
        paths.yaml contents.
    """
    from src.data.download_static_layers import (
        download_land_cover, download_population, download_elevation,
    )
    extra = config.get("extra_features", {})

    if extra.get("use_land_cover", False):
        logger.info("Preparing land cover …")
        download_land_cover()

    if extra.get("use_population", False):
        logger.info("Preparing population density …")
        download_population()

    if extra.get("use_elevation", False):
        logger.info("Preparing elevation …")
        download_elevation()
