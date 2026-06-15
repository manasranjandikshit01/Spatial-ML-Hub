"""
plot_maps.py
============
Generate India AQI and pollutant maps using matplotlib / geopandas.

Functions:
  - plot_aqi_map         — gridded AQI or pollutant raster over India
  - plot_station_aqi     — CPCB station dots coloured by AQI
  - plot_seasonal_mean   — seasonal mean map for a variable
  - save_daily_maps      — batch generate daily maps for a date range

Usage:
    python -m src.visualization.plot_maps \\
        --csv data/processed/grid_daily_features.csv \\
        --variable no2_column \\
        --date 2022-06-15
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---- AQI colour scale (CPCB) ----
AQI_BOUNDS = [0, 50, 100, 200, 300, 400, 500]
AQI_COLORS = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
AQI_CMAP = mcolors.ListedColormap(AQI_COLORS)
AQI_NORM = mcolors.BoundaryNorm(AQI_BOUNDS, ncolors=len(AQI_COLORS))

INDIA_EXTENT = [68.0, 97.5, 8.0, 37.5]  # [lon_min, lon_max, lat_min, lat_max]


def _add_india_boundary(ax: plt.Axes) -> None:
    """Attempt to draw India's boundary using geopandas natural-earth data."""
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        india = world[world["name"] == "India"]
        india.boundary.plot(ax=ax, color="black", linewidth=0.8)
    except Exception:
        # Fallback: plain box
        ax.set_xlim(INDIA_EXTENT[:2])
        ax.set_ylim(INDIA_EXTENT[2:])


def plot_aqi_map(
    df: pd.DataFrame,
    variable: str = "aqi",
    date: str | None = None,
    title: str | None = None,
    out_path: str | Path | None = None,
    cmap: str | mcolors.Colormap = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot a gridded raster map of *variable* over India.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain lat, lon, and *variable* columns.
        Optionally a 'date' column; if *date* is given, df is filtered first.
    variable : str
        Column to plot.
    date : str | None
        ISO date string to filter df. If None, all rows are plotted (use seasonal means etc.)
    title : str | None
    out_path : str | Path | None
        If given, save figure to this path.
    cmap, vmin, vmax : colour scale settings.
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    plot_df = df.copy()
    if date and "date" in plot_df.columns:
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        plot_df = plot_df[plot_df["date"] == pd.Timestamp(date)]

    if variable not in plot_df.columns:
        raise KeyError(f"Column {variable!r} not found. Available: {list(plot_df.columns)}")

    if variable == "aqi":
        use_cmap = AQI_CMAP
        use_norm = AQI_NORM
        use_vmin, use_vmax = 0, 500
    else:
        use_cmap = cmap
        use_norm = None
        use_vmin = vmin if vmin is not None else plot_df[variable].quantile(0.02)
        use_vmax = vmax if vmax is not None else plot_df[variable].quantile(0.98)

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        plot_df["lon"], plot_df["lat"],
        c=plot_df[variable],
        cmap=use_cmap,
        norm=use_norm,
        vmin=use_vmin, vmax=use_vmax,
        s=8, marker="s", linewidths=0,
    )

    _add_india_boundary(ax)
    ax.set_xlim(INDIA_EXTENT[:2])
    ax.set_ylim(INDIA_EXTENT[2:])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title or f"{variable}  |  {date or 'all dates'}")

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    if variable == "aqi":
        cbar.set_label("AQI")
        cbar.set_ticks([25, 75, 150, 250, 350, 450])
        cbar.set_ticklabels(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
    else:
        cbar.set_label(variable)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved map: %s", out_path)

    return fig


def plot_station_aqi(
    station_df: pd.DataFrame,
    date: str | None = None,
    aqi_col: str = "aqi_observed",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot CPCB station AQI as coloured circles on an India map.

    Parameters
    ----------
    station_df : pd.DataFrame
        Must contain: lat, lon, *aqi_col*.
    date : str | None
    aqi_col : str
    out_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    df = station_df.copy()
    if date and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] == pd.Timestamp(date)]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df[aqi_col],
        cmap=AQI_CMAP, norm=AQI_NORM,
        s=60, edgecolors="k", linewidths=0.4, zorder=3,
    )
    _add_india_boundary(ax)
    ax.set_xlim(INDIA_EXTENT[:2])
    ax.set_ylim(INDIA_EXTENT[2:])
    ax.set_title(f"CPCB Station AQI | {date or 'all'}")
    plt.colorbar(sc, ax=ax, label="AQI")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_seasonal_mean(
    df: pd.DataFrame,
    variable: str,
    season: str,
    out_path: str | Path | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the seasonal mean of *variable* for a specific *season*.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain lat, lon, season (or date), and *variable*.
    variable : str
    season : str
        One of "winter", "pre_monsoon", "monsoon", "post_monsoon".
    out_path : str | Path | None
    """
    season_map = {
        1: "winter", 2: "winter",
        3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
        6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post_monsoon", 11: "post_monsoon", 12: "winter",
    }
    if "season" not in df.columns and "date" in df.columns:
        df = df.copy()
        df["season"] = pd.to_datetime(df["date"]).dt.month.map(season_map)

    seasonal = df[df["season"] == season].groupby(["lat", "lon"])[variable].mean().reset_index()
    title = f"{variable} – {season.replace('_', ' ').title()} mean"
    return plot_aqi_map(seasonal, variable, title=title, out_path=out_path, **kwargs)


def save_daily_maps(
    df: pd.DataFrame,
    variable: str,
    output_dir: str | Path,
    date_start: str | None = None,
    date_end: str | None = None,
) -> list[Path]:
    """
    Generate and save one map per day in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
    output_dir : str | Path
    date_start, date_end : str | None

    Returns
    -------
    list[Path]  – paths of saved images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if date_start:
        df = df[df["date"] >= pd.Timestamp(date_start)]
    if date_end:
        df = df[df["date"] <= pd.Timestamp(date_end)]

    dates = sorted(df["date"].unique())
    paths: list[Path] = []

    for d in dates:
        d_str = pd.Timestamp(d).strftime("%Y-%m-%d")
        out = output_dir / f"{variable}_{d_str}.png"
        plot_aqi_map(df, variable, d_str, out_path=out)
        plt.close("all")
        paths.append(out)

    logger.info("Generated %d daily maps for %s", len(paths), variable)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot India AQI maps")
    parser.add_argument("--csv", default="data/processed/grid_daily_features.csv")
    parser.add_argument("--variable", default="no2_column")
    parser.add_argument("--date", default=None)
    parser.add_argument("--output", default="data/processed/aqi_maps/map.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    plot_aqi_map(df, args.variable, args.date, out_path=args.output)
    plt.close("all")


if __name__ == "__main__":
    main()
