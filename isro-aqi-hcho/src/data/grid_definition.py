"""
grid_definition.py
==================
Defines the common spatial grid used across all datasets.

The project uses a regular lat/lon grid at 0.1° resolution covering India.
Every gridded dataset is resampled to this grid before joining.

Usage:
    from src.data.grid_definition import get_india_grid, assign_cell_id, find_nearest_cell
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "config/paths.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_india_grid(resolution: float = 0.1) -> pd.DataFrame:
    """
    Generate the regular lat/lon grid covering India.

    Parameters
    ----------
    resolution : float
        Grid spacing in degrees. Default 0.1°.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: cell_id, lat, lon (cell centres).
    """
    try:
        cfg = load_config()
        bbox = cfg["india_bbox"]
        lon_min, lon_max = bbox["lon_min"], bbox["lon_max"]
        lat_min, lat_max = bbox["lat_min"], bbox["lat_max"]
        resolution = cfg.get("grid_resolution", resolution)
    except FileNotFoundError:
        lon_min, lon_max = 68.0, 97.5
        lat_min, lat_max = 8.0, 37.5

    lats = np.arange(lat_min + resolution / 2, lat_max, resolution)
    lons = np.arange(lon_min + resolution / 2, lon_max, resolution)

    grid_lons, grid_lats = np.meshgrid(lons, lats)
    flat_lats = grid_lats.ravel()
    flat_lons = grid_lons.ravel()

    cell_ids = [
        f"CELL_{lat:.2f}_{lon:.2f}"
        for lat, lon in zip(flat_lats, flat_lons)
    ]

    return pd.DataFrame({
        "cell_id": cell_ids,
        "lat": np.round(flat_lats, 4),
        "lon": np.round(flat_lons, 4),
    })


def assign_cell_id(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    resolution: float = 0.1,
) -> pd.DataFrame:
    """
    Snap each point in *df* to the nearest grid cell and add a ``cell_id`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing latitude and longitude columns.
    lat_col, lon_col : str
        Names of the latitude and longitude columns in *df*.
    resolution : float
        Grid spacing in degrees.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with an added ``cell_id`` column.
    """
    df = df.copy()
    grid = get_india_grid(resolution)

    snapped_lats = np.round(
        np.round(df[lat_col] / resolution) * resolution, 4
    )
    snapped_lons = np.round(
        np.round(df[lon_col] / resolution) * resolution, 4
    )

    df["cell_lat"] = snapped_lats
    df["cell_lon"] = snapped_lons
    df["cell_id"] = [
        f"CELL_{lat:.2f}_{lon:.2f}"
        for lat, lon in zip(snapped_lats, snapped_lons)
    ]
    return df


def find_nearest_cell(
    lat: float,
    lon: float,
    grid: pd.DataFrame | None = None,
    resolution: float = 0.1,
) -> str:
    """
    Return the cell_id of the grid cell nearest to (lat, lon).

    Parameters
    ----------
    lat, lon : float
        Query coordinates.
    grid : pd.DataFrame | None
        Pre-computed grid (from ``get_india_grid``). If None, it is generated.
    resolution : float
        Grid spacing in degrees.

    Returns
    -------
    str
        The cell_id string.
    """
    if grid is None:
        grid = get_india_grid(resolution)

    dists = np.sqrt((grid["lat"] - lat) ** 2 + (grid["lon"] - lon) ** 2)
    return grid.loc[dists.idxmin(), "cell_id"]


def grid_to_array(
    df: pd.DataFrame,
    value_col: str,
    resolution: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert a flat gridded DataFrame column to a 2-D NumPy array.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``lat``, ``lon``, and *value_col*.
    value_col : str
        Column to reshape.
    resolution : float
        Grid spacing.

    Returns
    -------
    (array2d, lats, lons)
        2-D array of shape (n_lats, n_lons) and the corresponding 1-D
        lat / lon coordinate arrays (sorted ascending).
    """
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())

    arr = np.full((len(lats), len(lons)), np.nan)
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}

    for _, row in df.iterrows():
        i = lat_idx.get(round(row["lat"], 4))
        j = lon_idx.get(round(row["lon"], 4))
        if i is not None and j is not None:
            arr[i, j] = row[value_col]

    return arr, lats, lons


if __name__ == "__main__":
    grid = get_india_grid()
    print(f"India grid: {len(grid):,} cells at 0.1° resolution")
    print(grid.head())
