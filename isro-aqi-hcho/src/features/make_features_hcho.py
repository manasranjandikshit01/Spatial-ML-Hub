"""
make_features_hcho.py
=====================
V3 feature engineering and hotspot detection for the HCHO pipeline.

What's new in V3
----------------
* ``compute_hcho_anomaly``      — daily HCHO minus per-cell seasonal mean
* ``compute_hotspot_persistence`` — fraction of days each cell is a hotspot
* ``export_hotspot_geojson``    — GeoJSON export of DBSCAN cluster polygons
* ``export_top_hotspot_regions`` — CSV of the N highest-anomaly hotspot regions

Existing pipeline (unchanged API)
----------------------------------
1. Load hcho_fire_daily_grid.csv.
2. Compute per-cell seasonal HCHO statistics.
3. Flag hotspot cells using percentile thresholds.
4. Cluster hotspots with DBSCAN or KMeans.
5. Compute lagged cross-correlations between HCHO and fire count.
6. Save enriched dataset.

Usage::

    python -m src.features.make_features_hcho \\
        --input  data/processed/hcho_fire_daily_grid.csv \\
        --output data/processed/hcho_hotspot_features.csv \\
        --config config/hcho_hotspot.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/hcho_hotspot.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def label_season(df: pd.DataFrame) -> pd.DataFrame:
    """Add a season column based on calendar month (IMD convention)."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Seasonal statistics
# ──────────────────────────────────────────────────────────────────────────────

def compute_seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-cell, per-season aggregations of HCHO and fire count.

    Returns
    -------
    pd.DataFrame with columns:
        cell_id, lat, lon, season, mean_hcho, max_hcho, mean_fire, total_fire
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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Hotspot flagging
# ──────────────────────────────────────────────────────────────────────────────

def flag_hotspots(
    seasonal_stats: pd.DataFrame,
    percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Flag cells whose mean HCHO exceeds the *percentile*-th within each season.

    Adds columns
    ------------
    is_hotspot       — bool
    hcho_percentile  — rank (0-100) within that season
    """
    df = seasonal_stats.copy()
    thresholds = df.groupby("season")["mean_hcho"].transform(
        lambda x: np.percentile(x, percentile)
    )
    df["is_hotspot"] = df["mean_hcho"] >= thresholds
    df["hcho_percentile"] = df.groupby("season")["mean_hcho"].rank(pct=True) * 100
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Clustering
# ──────────────────────────────────────────────────────────────────────────────

def cluster_hotspots(
    hotspot_df: pd.DataFrame,
    method: str = "dbscan",
    eps: float = 1.5,
    min_samples: int = 4,
    n_clusters: int = 8,
) -> pd.DataFrame:
    """
    Cluster hotspot cells geographically using DBSCAN or KMeans.

    Adds column: ``cluster``  (-1 = noise for DBSCAN).
    """
    from sklearn.cluster import DBSCAN, KMeans

    if hotspot_df.empty:
        hotspot_df = hotspot_df.copy()
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
    logger.info("  %s: %d clusters in %d hotspot cells", method.upper(), n_clusters_found, len(hotspot_df))
    return hotspot_df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 4: Lagged correlation
# ──────────────────────────────────────────────────────────────────────────────

def compute_lagged_correlation(
    df: pd.DataFrame,
    region_mask: pd.Series,
    lag_days: list[int],
    hcho_col: str = "hcho_column",
    fire_col: str = "fire_count",
) -> pd.DataFrame:
    """
    Pearson correlation between HCHO and fire count at various lags.

    Returns pd.DataFrame with columns: lag, pearson_r, p_value.
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


# ──────────────────────────────────────────────────────────────────────────────
# Phase 5 [V3 new]: HCHO anomaly — daily minus seasonal mean
# ──────────────────────────────────────────────────────────────────────────────

def compute_hcho_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the standardised HCHO anomaly for every daily observation.

    For each (cell_id, season) pair, the seasonal mean and std are computed,
    then the daily anomaly is::

        hcho_anomaly = (hcho_column − seasonal_mean) / (seasonal_std + ε)

    A raw anomaly column (``hcho_anomaly_raw``, in µmol m⁻²) is also added.

    Parameters
    ----------
    df : pd.DataFrame
        Daily HCHO grid data with columns: cell_id, date, season, hcho_column.

    Returns
    -------
    pd.DataFrame with added columns:
        hcho_anomaly_raw   — absolute departure from seasonal mean (µmol m⁻²)
        hcho_anomaly       — standardised anomaly (z-score within season × cell)
    """
    df = df.copy()
    if "season" not in df.columns:
        df = label_season(df)
    if "hcho_column" not in df.columns:
        logger.warning("'hcho_column' not found; skipping anomaly computation.")
        return df

    grp = df.groupby(["cell_id", "season"])["hcho_column"]
    df["_seasonal_mean"] = grp.transform("mean")
    df["_seasonal_std"]  = grp.transform("std").fillna(1.0)

    df["hcho_anomaly_raw"] = df["hcho_column"] - df["_seasonal_mean"]
    df["hcho_anomaly"]     = df["hcho_anomaly_raw"] / (df["_seasonal_std"] + 1e-6)

    df = df.drop(columns=["_seasonal_mean", "_seasonal_std"])
    logger.info("HCHO anomaly computed: %d rows", len(df))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6 [V3 new]: Hotspot persistence per cell
# ──────────────────────────────────────────────────────────────────────────────

def compute_hotspot_persistence(
    df: pd.DataFrame,
    threshold_percentile: float = 90.0,
) -> pd.DataFrame:
    """
    Count how many days each grid cell is a "hotspot" in each season.

    A daily observation is labelled a hotspot if its HCHO column exceeds
    the *threshold_percentile*-th across all cells for that (season, date).

    Adds columns to the seasonal stats level.

    Parameters
    ----------
    df : pd.DataFrame
        Daily grid with: cell_id, date, season, hcho_column.
    threshold_percentile : float

    Returns
    -------
    pd.DataFrame  Per-(cell_id, season) summary with columns:
        cell_id, lat, lon, season,
        hotspot_days, total_days, hotspot_fraction
    """
    df = df.copy()
    if "season" not in df.columns:
        df = label_season(df)

    df["date"] = pd.to_datetime(df["date"])

    # Per-date percentile threshold (across all cells on that day)
    def _date_threshold(group):
        return group.transform(lambda x: np.percentile(x, threshold_percentile))

    df["_day_threshold"] = df.groupby(["date", "season"])["hcho_column"].transform(
        lambda x: np.percentile(x, threshold_percentile)
    )
    df["_is_hotspot_day"] = (df["hcho_column"] >= df["_day_threshold"]).astype(int)

    persistence = (
        df.groupby(["cell_id", "lat", "lon", "season"])
        .agg(
            hotspot_days=("_is_hotspot_day", "sum"),
            total_days=("_is_hotspot_day", "count"),
        )
        .reset_index()
    )
    persistence["hotspot_fraction"] = (
        persistence["hotspot_days"] / persistence["total_days"].clip(lower=1)
    ).round(4)

    df = df.drop(columns=["_day_threshold", "_is_hotspot_day"])
    logger.info("Hotspot persistence: %d (cell, season) pairs", len(persistence))
    return persistence


# ──────────────────────────────────────────────────────────────────────────────
# Phase 7 [V3 new]: GeoJSON export of DBSCAN clusters
# ──────────────────────────────────────────────────────────────────────────────

def export_hotspot_geojson(
    clustered_df: pd.DataFrame,
    output_path: str | Path,
    season: str | None = None,
) -> None:
    """
    Export clustered hotspot cells as a GeoJSON FeatureCollection.

    Each DBSCAN cluster is exported as:
    - A MultiPoint geometry of all cells in the cluster
    - Properties: cluster_id, season, n_cells, mean_hcho, centroid_lat, centroid_lon

    Non-cluster noise cells (cluster == -1) are exported as individual points.

    Parameters
    ----------
    clustered_df : pd.DataFrame
        Output of ``cluster_hotspots`` — must have lat, lon, cluster, mean_hcho, season.
    output_path : str | Path
    season : str | None
        If provided, filter to a single season before export.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = clustered_df.copy()
    if season:
        df = df[df["season"] == season]

    features: list[dict] = []

    for cluster_id, grp in df.groupby("cluster"):
        if cluster_id == -1:
            # Export individual noise points
            for _, row in grp.iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["lon"]), float(row["lat"])],
                    },
                    "properties": {
                        "cluster_id": int(cluster_id),
                        "season": str(row.get("season", "")),
                        "mean_hcho": float(row.get("mean_hcho", 0)),
                        "hcho_percentile": float(row.get("hcho_percentile", 0)),
                        "n_cells": 1,
                        "label": "noise",
                    },
                })
        else:
            centroid_lat = float(grp["lat"].mean())
            centroid_lon = float(grp["lon"].mean())
            season_val = str(grp["season"].iloc[0]) if "season" in grp.columns else ""
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": [
                        [float(r["lon"]), float(r["lat"])]
                        for _, r in grp.iterrows()
                    ],
                },
                "properties": {
                    "cluster_id": int(cluster_id),
                    "season": season_val,
                    "n_cells": len(grp),
                    "mean_hcho": float(grp["mean_hcho"].mean()),
                    "max_hcho": float(grp["mean_hcho"].max()),
                    "centroid_lat": centroid_lat,
                    "centroid_lon": centroid_lon,
                    "hcho_percentile_mean": float(grp.get("hcho_percentile", pd.Series([0])).mean()),
                    "label": f"cluster_{cluster_id}",
                },
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "n_clusters": len(set(df["cluster"].unique()) - {-1}),
            "n_noise": int((df["cluster"] == -1).sum()),
            "season": season or "all",
        },
    }

    with open(out, "w") as f:
        json.dump(geojson, f, indent=2)

    logger.info("GeoJSON saved: %d features → %s", len(features), out)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 8 [V3 new]: Top-N hotspot regions CSV
# ──────────────────────────────────────────────────────────────────────────────

def export_top_hotspot_regions(
    stats_df: pd.DataFrame,
    output_path: str | Path,
    n: int = 10,
    rank_by: str = "mean_hcho",
) -> pd.DataFrame:
    """
    Export the top-N hotspot grid cells (by mean HCHO or anomaly) to CSV.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Seasonal stats with is_hotspot, mean_hcho, cluster, lat, lon.
    output_path : str | Path
    n : int
        Number of top cells to export (per season).
    rank_by : str
        Column to rank by (default: ``mean_hcho``).

    Returns
    -------
    pd.DataFrame  Top-N rows.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    hotspots = stats_df[stats_df.get("is_hotspot", pd.Series(True, index=stats_df.index))].copy()
    if rank_by not in hotspots.columns:
        rank_by = "mean_hcho"

    top_n = (
        hotspots
        .sort_values([rank_by], ascending=False)
        .groupby("season", group_keys=False)
        .head(n)
        .sort_values(["season", rank_by], ascending=[True, False])
        .reset_index(drop=True)
    )

    select_cols = [c for c in [
        "season", "cell_id", "lat", "lon", "cluster",
        "mean_hcho", "max_hcho", "hcho_percentile",
        "mean_fire", "total_fire", "hotspot_fraction",
    ] if c in top_n.columns]
    top_n = top_n[select_cols]

    top_n.to_csv(out, index=False)
    logger.info("Top-%d hotspot regions → %s", n, out)
    return top_n


# ──────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ──────────────────────────────────────────────────────────────────────────────

def make_hcho_features(
    input_csv: str | Path,
    output_csv: str | Path,
    config: dict | None = None,
    export_geojson: bool = True,
    export_top_regions: bool = True,
    n_top_regions: int = 10,
) -> pd.DataFrame:
    """
    Full V3 HCHO feature engineering and hotspot detection pipeline.

    Steps
    -----
    1. Load daily HCHO grid → label seasons
    2. Compute HCHO anomaly per cell per day  [V3]
    3. Seasonal stats: mean/max HCHO, fire count
    4. Flag hotspots at percentile threshold
    5. Cluster with DBSCAN/KMeans
    6. Merge hotspot persistence [V3]
    7. Save enriched seasonal stats CSV
    8. Export GeoJSON clusters [V3]
    9. Export top-N regions CSV [V3]
    10. Lagged fire–HCHO correlations

    Parameters
    ----------
    input_csv, output_csv : str | Path
    config : dict | None
        Loaded from ``config/hcho_hotspot.yaml``; defaults used if None.
    export_geojson : bool
        Save ``hcho_hotspot_clusters.geojson`` alongside the output CSV.
    export_top_regions : bool
        Save ``hcho_top_hotspot_regions.csv`` alongside the output CSV.
    n_top_regions : int
        Top-N cells per season to include in the top-regions export.

    Returns
    -------
    pd.DataFrame  Seasonal stats with hotspot flags, cluster IDs, persistence.
    """
    if config is None:
        config = {}

    hotspot_cfg = config.get("hotspot", {})
    percentile     = hotspot_cfg.get("percentile", 90)
    cluster_method = hotspot_cfg.get("cluster_method", "dbscan")
    dbscan_eps     = hotspot_cfg.get("dbscan_eps", 1.5)
    dbscan_min     = hotspot_cfg.get("dbscan_min_samples", 4)
    kmeans_k       = hotspot_cfg.get("kmeans_n_clusters", 8)

    logger.info("Loading HCHO data from %s …", input_csv)
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = label_season(df)

    # [V3] Daily HCHO anomaly
    logger.info("Computing HCHO anomaly …")
    df = compute_hcho_anomaly(df)

    # Seasonal aggregations
    logger.info("Computing seasonal statistics …")
    stats = compute_seasonal_stats(df)

    # Flag hotspots
    logger.info("Flagging hotspots at %.0f-th percentile …", percentile)
    stats = flag_hotspots(stats, percentile)

    # Cluster
    logger.info("Clustering hotspot cells (%s) …", cluster_method)
    hotspots = stats[stats["is_hotspot"]].copy()
    hotspots = cluster_hotspots(hotspots, cluster_method, dbscan_eps, dbscan_min, kmeans_k)
    stats = stats.merge(hotspots[["cell_id", "season", "cluster"]], on=["cell_id", "season"], how="left")

    # [V3] Hotspot persistence
    logger.info("Computing hotspot persistence …")
    persistence = compute_hotspot_persistence(df, threshold_percentile=percentile)
    merge_cols = ["cell_id", "lat", "lon", "season"]
    persist_cols = ["hotspot_days", "total_days", "hotspot_fraction"]
    stats = stats.merge(
        persistence[merge_cols + persist_cols],
        on=merge_cols, how="left",
    )

    # Save main output
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_path, index=False)
    logger.info("HCHO hotspot features: %d rows → %s", len(stats), out_path)

    # [V3] GeoJSON export of clusters
    if export_geojson:
        geojson_path = out_path.parent / "hcho_hotspot_clusters.geojson"
        clustered = stats[stats["is_hotspot"] & stats["cluster"].notna()].copy()
        if not clustered.empty:
            export_hotspot_geojson(clustered, geojson_path)
        else:
            logger.warning("No clustered hotspots to export as GeoJSON.")

    # [V3] Top-N hotspot regions CSV
    if export_top_regions:
        top_path = out_path.parent / "hcho_top_hotspot_regions.csv"
        export_top_hotspot_regions(stats, top_path, n=n_top_regions)

    # Lagged correlations per configured region
    lag_days    = config.get("correlation", {}).get("lag_days", [0, 1, 2, 3])
    regions_cfg = config.get("correlation", {}).get("regions", {})
    corr_results: list[pd.DataFrame] = []

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
        corr_path = out_path.parent / "hcho_fire_correlations.csv"
        corr_df.to_csv(corr_path, index=False)
        logger.info("Lagged correlations → %s", corr_path)

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="HCHO V3 feature engineering and hotspot detection")
    parser.add_argument("--input",  default="data/processed/hcho_fire_daily_grid.csv")
    parser.add_argument("--output", default="data/processed/hcho_hotspot_features.csv")
    parser.add_argument("--config", default="config/hcho_hotspot.yaml")
    parser.add_argument("--no_geojson",     dest="geojson", action="store_false",
                        help="Skip GeoJSON export")
    parser.add_argument("--no_top_regions", dest="top_regions", action="store_false",
                        help="Skip top-N hotspot regions export")
    parser.add_argument("--n_top", type=int, default=10,
                        help="Number of top regions to export (default: 10)")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.warning("Config not found at %s; using defaults.", args.config)
        config = {}

    make_hcho_features(
        args.input, args.output, config,
        export_geojson=args.geojson,
        export_top_regions=args.top_regions,
        n_top_regions=args.n_top,
    )


if __name__ == "__main__":
    main()
