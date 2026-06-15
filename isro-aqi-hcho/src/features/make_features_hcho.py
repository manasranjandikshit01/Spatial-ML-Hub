"""
make_features_hcho.py
=====================
Feature engineering and hotspot detection for the HCHO pipeline.

Steps:
  1. Load hcho_fire_daily_grid.csv.
  2. Compute per-cell seasonal HCHO statistics.
  3. Flag hotspot cells using percentile thresholds.
  4. Optionally cluster hotspots with DBSCAN or KMeans.
  5. Compute lagged cross-correlations between HCHO and fire count.
  6. Save enriched dataset.

Usage:
    python -m src.features.make_features_hcho \\
        --input data/processed/hcho_fire_daily_grid.csv \\
        --output data/processed/hcho_hotspot_features.csv \\
        --config config/hcho_hotspot.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/hcho_hotspot.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def label_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add a season column if not already present."""
    if "season" in df.columns:
        return df
    df = df.copy()
    season_map = {
        1: "winter", 2: "winter",
        3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
        6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
        10: "post_monsoon", 11: "post_monsoon", 12: "winter",
    }
    df["season"] = pd.to_datetime(df["date"]).dt.month.map(season_map)
    return df


def compute_seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-cell per-season mean HCHO and fire count.

    Returns
    -------
    pd.DataFrame with columns: cell_id, lat, lon, season, mean_hcho, mean_fire, max_hcho.
    """
    agg = (
        df.groupby(["cell_id", "lat", "lon", "season"])
        .agg(
            mean_hcho=("hcho_column", "mean"),
            max_hcho=("hcho_column", "max"),
            mean_fire=("fire_count", "mean"),
            total_fire=("fire_count", "sum"),
        )
        .reset_index()
    )
    return agg


def flag_hotspots(
    seasonal_stats: pd.DataFrame,
    percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Flag cells whose mean HCHO exceeds the *percentile*-th percentile
    within each season.

    Parameters
    ----------
    seasonal_stats : pd.DataFrame
        Output of ``compute_seasonal_stats``.
    percentile : float
        Threshold percentile (e.g. 90 = top 10% cells).

    Returns
    -------
    pd.DataFrame with added boolean column ``is_hotspot``.
    """
    df = seasonal_stats.copy()
    thresholds = df.groupby("season")["mean_hcho"].transform(
        lambda x: np.percentile(x, percentile)
    )
    df["is_hotspot"] = df["mean_hcho"] >= thresholds
    df["hcho_percentile"] = df.groupby("season")["mean_hcho"].rank(pct=True) * 100
    return df


def cluster_hotspots(
    hotspot_df: pd.DataFrame,
    method: str = "dbscan",
    eps: float = 1.5,
    min_samples: int = 4,
    n_clusters: int = 8,
) -> pd.DataFrame:
    """
    Cluster hotspot cells geographically.

    Parameters
    ----------
    hotspot_df : pd.DataFrame
        Rows where is_hotspot == True; must have lat, lon, mean_hcho, mean_fire.
    method : str
        "dbscan" or "kmeans".
    eps, min_samples : float, int
        DBSCAN parameters (in degrees).
    n_clusters : int
        KMeans number of clusters.

    Returns
    -------
    pd.DataFrame with an added ``cluster`` column (-1 = noise for DBSCAN).
    """
    from sklearn.cluster import DBSCAN, KMeans

    if hotspot_df.empty:
        hotspot_df["cluster"] = pd.Series(dtype=int)
        return hotspot_df

    X = hotspot_df[["lat", "lon"]].values

    if method == "dbscan":
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    else:
        k = min(n_clusters, len(hotspot_df))
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

    hotspot_df = hotspot_df.copy()
    hotspot_df["cluster"] = labels
    n_clusters_found = len(set(labels) - {-1})
    logger.info("  %s found %d clusters in %d hotspot cells", method.upper(), n_clusters_found, len(hotspot_df))
    return hotspot_df


def compute_lagged_correlation(
    df: pd.DataFrame,
    region_mask: pd.Series,
    lag_days: list[int],
    hcho_col: str = "hcho_column",
    fire_col: str = "fire_count",
) -> pd.DataFrame:
    """
    Compute Pearson correlation between HCHO and fire count at various lags
    for a regional subset of the data.

    Parameters
    ----------
    df : pd.DataFrame
    region_mask : pd.Series[bool]
        Boolean mask selecting the region.
    lag_days : list[int]
        Lags to compute (0 = same-day, 1 = fire leads HCHO by 1 day, etc.).
    hcho_col, fire_col : str

    Returns
    -------
    pd.DataFrame with columns: lag, pearson_r, p_value.
    """
    from scipy.stats import pearsonr

    regional = df[region_mask].copy()
    daily_mean = (
        regional.groupby("date")[[hcho_col, fire_col]]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    results: list[dict] = []
    for lag in lag_days:
        fire_shifted = daily_mean[fire_col].shift(lag)
        valid = ~(fire_shifted.isna() | daily_mean[hcho_col].isna())
        if valid.sum() < 10:
            results.append({"lag": lag, "pearson_r": np.nan, "p_value": np.nan})
            continue
        r, p = pearsonr(daily_mean[hcho_col][valid], fire_shifted[valid])
        results.append({"lag": lag, "pearson_r": round(r, 4), "p_value": round(p, 6)})

    return pd.DataFrame(results)


def make_hcho_features(
    input_csv: str | Path,
    output_csv: str | Path,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Full HCHO feature engineering and hotspot detection pipeline.

    Parameters
    ----------
    input_csv : str | Path
    output_csv : str | Path
    config : dict | None
        Configuration dict (from hcho_hotspot.yaml). If None, defaults are used.

    Returns
    -------
    pd.DataFrame  – seasonal stats with hotspot flags and cluster IDs.
    """
    if config is None:
        config = {}

    hotspot_cfg = config.get("hotspot", {})
    percentile = hotspot_cfg.get("percentile", 90)
    cluster_method = hotspot_cfg.get("cluster_method", "dbscan")
    dbscan_eps = hotspot_cfg.get("dbscan_eps", 1.5)
    dbscan_min = hotspot_cfg.get("dbscan_min_samples", 4)
    kmeans_k = hotspot_cfg.get("kmeans_n_clusters", 8)

    logger.info("Loading HCHO data from %s …", input_csv)
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = label_season(df)

    logger.info("Computing seasonal statistics …")
    stats = compute_seasonal_stats(df)

    logger.info("Flagging hotspots at %.0f-th percentile …", percentile)
    stats = flag_hotspots(stats, percentile)

    logger.info("Clustering hotspot cells …")
    hotspots = stats[stats["is_hotspot"]].copy()
    hotspots = cluster_hotspots(hotspots, cluster_method, dbscan_eps, dbscan_min, kmeans_k)
    stats = stats.merge(hotspots[["cell_id", "season", "cluster"]], on=["cell_id", "season"], how="left")

    # Lagged correlations per region
    lag_days = config.get("correlation", {}).get("lag_days", [0, 1, 2, 3])
    regions_cfg = config.get("correlation", {}).get("regions", {})

    corr_results: list[dict] = []
    for region_name, region_bbox in regions_cfg.items():
        mask = (
            (df["lat"] >= region_bbox["lat_min"]) & (df["lat"] <= region_bbox["lat_max"]) &
            (df["lon"] >= region_bbox["lon_min"]) & (df["lon"] <= region_bbox["lon_max"])
        )
        corr = compute_lagged_correlation(df, mask, lag_days)
        corr["region"] = region_name
        corr_results.append(corr)

    if corr_results:
        corr_df = pd.concat(corr_results, ignore_index=True)
        corr_path = Path(output_csv).parent / "hcho_fire_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        logger.info("Lagged correlations saved to %s", corr_path)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_csv, index=False)
    logger.info("HCHO hotspot features: %d rows → %s", len(stats), output_csv)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="HCHO feature engineering and hotspot detection")
    parser.add_argument("--input", default="data/processed/hcho_fire_daily_grid.csv")
    parser.add_argument("--output", default="data/processed/hcho_hotspot_features.csv")
    parser.add_argument("--config", default="config/hcho_hotspot.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found; using defaults.")
        config = {}

    make_hcho_features(args.input, args.output, config)


if __name__ == "__main__":
    main()
