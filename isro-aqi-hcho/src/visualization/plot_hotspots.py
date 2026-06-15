"""
plot_hotspots.py
================
Visualise HCHO hotspots, fire density, and wind transport patterns.

Usage:
    python -m src.visualization.plot_hotspots \\
        --hcho_csv data/processed/hcho_fire_daily_grid.csv \\
        --hotspot_csv data/processed/hcho_hotspot_features.csv \\
        --output_dir outputs/hotspot_maps
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INDIA_EXTENT = [68.0, 97.5, 8.0, 37.5]

SEASON_LABELS = {
    "winter": "Winter (Dec–Feb)",
    "pre_monsoon": "Pre-Monsoon (Mar–May)",
    "monsoon": "Monsoon (Jun–Sep)",
    "post_monsoon": "Post-Monsoon (Oct–Nov)",
}


def _india_ax(figsize: tuple[int, int] = (10, 8)) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure with India extent pre-set."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        import geopandas as gpd
        world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        india = world[world["name"] == "India"]
        india.boundary.plot(ax=ax, color="black", linewidth=0.9)
    except Exception:
        pass
    ax.set_xlim(INDIA_EXTENT[:2])
    ax.set_ylim(INDIA_EXTENT[2:])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return fig, ax


def plot_hcho_hotspot_map(
    seasonal_stats: pd.DataFrame,
    season: str = "post_monsoon",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (11, 8),
) -> plt.Figure:
    """
    Map of seasonal mean HCHO with hotspot cells outlined.

    Parameters
    ----------
    seasonal_stats : pd.DataFrame
        Output of make_hcho_features – must have lat, lon, season,
        mean_hcho, is_hotspot, cluster.
    season : str
    out_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    df = seasonal_stats[seasonal_stats["season"] == season].copy()
    if df.empty:
        logger.warning("No data for season %r", season)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for {season}", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = _india_ax(figsize)

    # All cells – HCHO background
    sc = ax.scatter(
        df["lon"], df["lat"],
        c=df["mean_hcho"],
        cmap="YlOrRd",
        vmin=df["mean_hcho"].quantile(0.05),
        vmax=df["mean_hcho"].quantile(0.98),
        s=12, marker="s", linewidths=0, alpha=0.85,
    )
    plt.colorbar(sc, ax=ax, label="Mean HCHO column (µmol/m²)", fraction=0.03, pad=0.04)

    # Hotspot cells – outlined
    hotspots = df[df["is_hotspot"]]
    if not hotspots.empty:
        ax.scatter(
            hotspots["lon"], hotspots["lat"],
            s=30, marker="s", facecolors="none", edgecolors="blue",
            linewidths=0.7, label=f"Hotspot cells (≥90th pct, n={len(hotspots)})",
        )

    ax.set_title(f"HCHO Hotspots – {SEASON_LABELS.get(season, season)}")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Saved hotspot map: %s", out_path)

    return fig


def plot_fire_density_map(
    df: pd.DataFrame,
    season: str = "post_monsoon",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Map of fire detection density (mean daily fires per grid cell) for a season.

    Parameters
    ----------
    df : pd.DataFrame
        hcho_fire_daily_grid.csv with lat, lon, date, fire_count, season.
    season : str
    out_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    season_map = {
        1: "winter", 2: "winter",
        3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
        6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post_monsoon", 11: "post_monsoon", 12: "winter",
    }
    df = df.copy()
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["date"]).dt.month.map(season_map)

    subset = df[df["season"] == season]
    density = subset.groupby(["lat", "lon"])["fire_count"].mean().reset_index()
    density = density[density["fire_count"] > 0]

    fig, ax = _india_ax(figsize)

    if density.empty:
        ax.set_title(f"Fire Density – {SEASON_LABELS.get(season, season)}")
        ax.text(0.5, 0.5, "No fire detections", ha="center", va="center", transform=ax.transAxes)
        return fig

    sc = ax.scatter(
        density["lon"], density["lat"],
        c=density["fire_count"],
        cmap="hot_r",
        vmin=0,
        vmax=density["fire_count"].quantile(0.99),
        s=15, marker="s", linewidths=0, alpha=0.9,
    )
    plt.colorbar(sc, ax=ax, label="Mean daily fire count", fraction=0.03, pad=0.04)
    ax.set_title(f"Fire Density – {SEASON_LABELS.get(season, season)}")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_wind_transport_map(
    df: pd.DataFrame,
    date: str,
    hcho_col: str = "hcho_column",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (11, 8),
    wind_scale: float = 3.0,
    subsample: int = 5,
) -> plt.Figure:
    """
    Map of HCHO with overlaid wind vectors for a single day.

    Parameters
    ----------
    df : pd.DataFrame
        hcho_fire_daily_grid.csv with u10, v10.
    date : str
        ISO date string.
    hcho_col : str
    out_path : str | Path | None
    figsize : tuple
    wind_scale : float
        Quiver scale factor.
    subsample : int
        Plot every Nth grid point to avoid clutter.

    Returns
    -------
    matplotlib Figure
    """
    day = df[pd.to_datetime(df["date"]) == pd.Timestamp(date)].copy()
    if day.empty:
        logger.warning("No data for date %s", date)
        return plt.subplots()[0]

    fig, ax = _india_ax(figsize)

    # HCHO raster
    sc = ax.scatter(
        day["lon"], day["lat"],
        c=day[hcho_col],
        cmap="YlOrRd",
        vmin=day[hcho_col].quantile(0.05),
        vmax=day[hcho_col].quantile(0.98),
        s=12, marker="s", linewidths=0, alpha=0.8,
    )
    plt.colorbar(sc, ax=ax, label=f"{hcho_col} (µmol/m²)", fraction=0.03, pad=0.04)

    # Wind quivers (subsampled)
    if "u10" in day.columns and "v10" in day.columns:
        sub = day.iloc[::subsample]
        ax.quiver(
            sub["lon"], sub["lat"],
            sub["u10"], sub["v10"],
            scale=wind_scale * 50, width=0.002, color="navy", alpha=0.6,
            label="Wind (10 m)",
        )
        ax.legend(loc="lower right", fontsize=8)

    ax.set_title(f"HCHO + Wind Transport – {date}")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cluster_map(
    hotspot_df: pd.DataFrame,
    season: str = "post_monsoon",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (11, 8),
) -> plt.Figure:
    """
    Map of hotspot clusters with each cluster in a distinct colour.

    Parameters
    ----------
    hotspot_df : pd.DataFrame
        Hotspot feature DataFrame with lat, lon, season, cluster columns.
    season : str
    out_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    df = hotspot_df[(hotspot_df["season"] == season) & hotspot_df["is_hotspot"]].copy()
    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No hotspots found", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = _india_ax(figsize)

    clusters = sorted(df["cluster"].unique())
    cmap = plt.get_cmap("tab20", len(clusters))
    legend_patches: list[mpatches.Patch] = []

    for idx, cluster in enumerate(clusters):
        subset = df[df["cluster"] == cluster]
        color = "#888888" if cluster == -1 else cmap(idx)
        label = "Noise" if cluster == -1 else f"Cluster {cluster} (n={len(subset)})"
        ax.scatter(subset["lon"], subset["lat"], s=25, color=color, alpha=0.7, marker="o")
        legend_patches.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, title="Clusters")
    ax.set_title(f"HCHO Hotspot Clusters – {SEASON_LABELS.get(season, season)}")
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def generate_all_hotspot_plots(
    hcho_csv: str | Path,
    hotspot_csv: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate a full suite of hotspot visualisation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hcho = pd.read_csv(hcho_csv)
    hotspot = pd.read_csv(hotspot_csv) if Path(hotspot_csv).exists() else pd.DataFrame()

    for season in ["winter", "pre_monsoon", "monsoon", "post_monsoon"]:
        plot_fire_density_map(hcho, season, output_dir / f"fire_density_{season}.png")
        plt.close("all")
        if not hotspot.empty:
            plot_hcho_hotspot_map(hotspot, season, output_dir / f"hcho_hotspot_{season}.png")
            plot_cluster_map(hotspot, season, output_dir / f"hcho_clusters_{season}.png")
            plt.close("all")

    logger.info("All hotspot plots saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HCHO hotspot maps")
    parser.add_argument("--hcho_csv", default="data/processed/hcho_fire_daily_grid.csv")
    parser.add_argument("--hotspot_csv", default="data/processed/hcho_hotspot_features.csv")
    parser.add_argument("--output_dir", default="outputs/hotspot_maps")
    args = parser.parse_args()
    generate_all_hotspot_plots(args.hcho_csv, args.hotspot_csv, args.output_dir)


if __name__ == "__main__":
    main()
