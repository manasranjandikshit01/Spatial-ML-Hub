"""
plot_time_series.py
===================
Time series visualisation utilities for AQI and HCHO data.

Usage:
    python -m src.visualization.plot_time_series \\
        --csv data/processed/cpcb_daily.csv \\
        --city Delhi \\
        --variable pm25 \\
        --output outputs/delhi_pm25.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEASON_COLORS = {
    "winter": "#4575b4",
    "pre_monsoon": "#d73027",
    "monsoon": "#1a9850",
    "post_monsoon": "#f46d43",
}


def plot_city_time_series(
    df: pd.DataFrame,
    city: str,
    variable: str = "pm25",
    pred_col: str | None = None,
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 4),
    rolling: int = 7,
) -> plt.Figure:
    """
    Plot a daily time series for a single city.

    Parameters
    ----------
    df : pd.DataFrame
        CPCB daily data. Must have 'city', 'date', and *variable* columns.
    city : str
    variable : str
    pred_col : str | None
        Optional column for model predictions (plotted as dashed line).
    out_path : str | Path | None
    figsize : tuple
    rolling : int
        Rolling-mean window in days (0 to disable).

    Returns
    -------
    matplotlib Figure
    """
    subset = df[df["city"].str.lower() == city.lower()].copy()
    if subset.empty:
        logger.warning("City %r not found in dataframe.", city)
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for city: {city}", ha="center", va="center", transform=ax.transAxes)
        return fig

    subset["date"] = pd.to_datetime(subset["date"])
    daily = subset.groupby("date")[[variable]].mean()
    if pred_col and pred_col in subset.columns:
        daily[pred_col] = subset.groupby("date")[pred_col].mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(daily.index, daily[variable], color="gray", linewidth=0.8, alpha=0.6, label="Daily")

    if rolling > 1:
        rolled = daily[variable].rolling(rolling, center=True, min_periods=1).mean()
        ax.plot(daily.index, rolled, color="steelblue", linewidth=2.0, label=f"{rolling}-day mean")

    if pred_col and pred_col in daily.columns:
        ax.plot(daily.index, daily[pred_col], "--", color="tomato", linewidth=1.5, label="Predicted")

    ax.set_title(f"{city} – {variable.upper()} time series")
    ax.set_ylabel(variable)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_regional_hcho_fire(
    df: pd.DataFrame,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
    region_name: str = "Region",
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 5),
    lag_days: list[int] | None = None,
) -> plt.Figure:
    """
    Plot regional mean HCHO and fire count on twin axes, with optional
    lagged correlation annotations.

    Parameters
    ----------
    df : pd.DataFrame
        hcho_fire_daily_grid.csv with columns: lat, lon, date, hcho_column, fire_count.
    lat_min, lat_max, lon_min, lon_max : float
        Bounding box of the region.
    region_name : str
    out_path : str | Path | None
    figsize : tuple
    lag_days : list[int] | None
        Lags for which to annotate Pearson r.

    Returns
    -------
    matplotlib Figure
    """
    from scipy.stats import pearsonr

    mask = (
        (df["lat"] >= lat_min) & (df["lat"] <= lat_max) &
        (df["lon"] >= lon_min) & (df["lon"] <= lon_max)
    )
    region = df[mask].copy()
    region["date"] = pd.to_datetime(region["date"])

    daily = (
        region.groupby("date")[["hcho_column", "fire_count"]]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    fig, ax1 = plt.subplots(figsize=figsize)
    color_hcho, color_fire = "steelblue", "tomato"

    ax1.plot(daily["date"], daily["hcho_column"], color=color_hcho, linewidth=1.5, label="HCHO column")
    ax1.set_ylabel("HCHO column (µmol/m²)", color=color_hcho)
    ax1.tick_params(axis="y", labelcolor=color_hcho)

    ax2 = ax1.twinx()
    ax2.fill_between(daily["date"], daily["fire_count"], alpha=0.35, color=color_fire, label="Fire count")
    ax2.set_ylabel("Fire count (daily grid mean)", color=color_fire)
    ax2.tick_params(axis="y", labelcolor=color_fire)

    ax1.set_title(f"{region_name} – HCHO vs Fire Count")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=30, ha="right")

    # Annotate correlations
    if lag_days:
        ann_lines = []
        for lag in lag_days:
            fire_sh = daily["fire_count"].shift(lag)
            valid = ~(fire_sh.isna() | daily["hcho_column"].isna())
            if valid.sum() >= 10:
                r, p = pearsonr(daily["hcho_column"][valid], fire_sh[valid])
                ann_lines.append(f"r(lag={lag}d)={r:.3f}")
        ax1.annotate("\n".join(ann_lines), xy=(0.02, 0.95), xycoords="axes fraction",
                     fontsize=8, va="top", ha="left",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_seasonal_boxplot(
    df: pd.DataFrame,
    variable: str = "pm25",
    city: str | None = None,
    out_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Box plot of *variable* grouped by season.

    Parameters
    ----------
    df : pd.DataFrame
    variable : str
    city : str | None
        If given, filter df to this city first.
    out_path : str | Path | None
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    df = df.copy()
    if city:
        df = df[df["city"].str.lower() == city.lower()]

    df["date"] = pd.to_datetime(df["date"])
    season_map = {
        1: "Winter", 2: "Winter",
        3: "Pre-Monsoon", 4: "Pre-Monsoon", 5: "Pre-Monsoon",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon", 12: "Winter",
    }
    df["season"] = df["date"].dt.month.map(season_map)
    order = ["Winter", "Pre-Monsoon", "Monsoon", "Post-Monsoon"]
    groups = [df[df["season"] == s][variable].dropna().values for s in order]

    fig, ax = plt.subplots(figsize=figsize)
    bp = ax.boxplot(groups, labels=order, patch_artist=True, notch=False)
    colors = [SEASON_COLORS["winter"], SEASON_COLORS["pre_monsoon"],
              SEASON_COLORS["monsoon"], SEASON_COLORS["post_monsoon"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    title = f"{variable.upper()} seasonal distribution"
    if city:
        title += f" – {city}"
    ax.set_title(title)
    ax.set_ylabel(variable)
    plt.tight_layout()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AQI / HCHO time series")
    parser.add_argument("--csv", default="data/processed/cpcb_daily.csv")
    parser.add_argument("--city", default="Delhi")
    parser.add_argument("--variable", default="pm25")
    parser.add_argument("--output", default="outputs/timeseries.png")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    fig = plot_city_time_series(df, args.city, args.variable, out_path=args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
