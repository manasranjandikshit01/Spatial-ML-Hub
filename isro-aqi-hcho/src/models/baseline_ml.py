"""
baseline_ml.py
==============
Baseline ML models (Random Forest & Gradient Boosting) for PM2.5 and AQI
prediction from satellite + reanalysis features.

V2 improvements
---------------
* Refactored into clean ``load_aqi_training_data / train_baseline_models /
  evaluate_baseline_models`` functions for easier notebook use.
* Optional ``GridSearchCV`` hyperparameter search controlled by config.
* ``joblib`` model serialisation (replaces raw ``pickle``).
* Metrics saved to ``baseline_results.csv`` in addition to JSON.
* Domain-subset option (e.g. only IGP) for quick experiments.
* Time-split **and** group-split (by city/station) are both supported.

Usage::

    python -m src.models.baseline_ml \\
        --input data/processed/aqi_training_dataset.csv \\
        --output_dir models/baseline

    python -m src.models.baseline_ml \\
        --input data/processed/aqi_training_dataset.csv \\
        --output_dir models/baseline \\
        --hparam_search
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Feature / target definitions
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "no2_column", "so2_column", "co_column", "o3_column",
    "hcho_column", "insat_aod",
    "t2m", "rh2m", "u10", "v10", "tp", "sp", "blh",
]

TARGET_COLS: list[str] = ["pm25_target", "aqi_target"]

HYPERPARAMS: dict[str, dict] = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [10, 15, None],
        "min_samples_split": [4, 8],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_aqi_training_data(
    input_csv: str | Path,
    feature_cols: list[str] | None = None,
    target_cols: list[str] | None = None,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> pd.DataFrame:
    """
    Load the AQI training dataset from CSV with optional spatial subsetting.

    Parameters
    ----------
    input_csv : str | Path
        Path to ``aqi_training_dataset.csv`` (output of ``build_dataset_aqi.py``).
    feature_cols : list[str] | None
        Columns to validate; defaults to ``FEATURE_COLS``.
    target_cols : list[str] | None
        Target columns to validate; defaults to ``TARGET_COLS``.
    lat_min, lat_max, lon_min, lon_max : float | None
        Optional bounding-box filter (degrees).  Use ``None`` to skip.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with ``date`` as ``datetime64``.

    Raises
    ------
    FileNotFoundError
        If ``input_csv`` does not exist.
    ValueError
        If no usable features or targets are found.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if target_cols is None:
        target_cols = TARGET_COLS

    path = Path(input_csv)
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Optional spatial subset
    for col, lo, hi in [("lat", lat_min, lat_max), ("lon", lon_min, lon_max)]:
        if col in df.columns:
            if lo is not None:
                df = df[df[col] >= lo]
            if hi is not None:
                df = df[df[col] <= hi]

    feat_avail = [c for c in feature_cols if c in df.columns]
    tgt_avail = [c for c in target_cols if c in df.columns]

    if not feat_avail:
        raise ValueError(
            f"None of the expected features found.\n"
            f"  Expected: {feature_cols}\n"
            f"  Got: {list(df.columns)}"
        )
    if not tgt_avail:
        raise ValueError(
            f"None of the expected targets found.\n"
            f"  Expected: {target_cols}\n"
            f"  Got: {list(df.columns)}"
        )

    logger.info(
        "Loaded %d rows from %s  [%d features, %d targets]",
        len(df), path.name, len(feat_avail), len(tgt_avail),
    )
    return df


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """Compute RMSE, MAE, R², and Pearson-r; log and return as dict."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    metrics = {
        "model": label.split("/")[0] if "/" in label else label,
        "target": label.split("/")[1] if "/" in label else "",
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "r2": round(r2, 4),
        "pearson_r": round(r, 4),
    }
    logger.info(
        "[%s]  RMSE=%.3f  MAE=%.3f  R²=%.4f  r=%.4f",
        label, rmse, mae, r2, r,
    )
    return metrics


def train_baseline_models(
    input_csv: str | Path,
    output_dir: str | Path,
    train_end: str = "2021-12-31",
    test_start: str = "2022-01-01",
    hparam_search: bool = False,
    cv_folds: int = 3,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
) -> dict[str, dict]:
    """
    Train Random Forest and Gradient Boosting models with temporal split.

    Parameters
    ----------
    input_csv : str | Path
        Training dataset CSV.
    output_dir : str | Path
        Directory to save models, predictions, and metrics.
    train_end, test_start : str
        ISO-format date strings for temporal train/test split.
    hparam_search : bool
        If True, run ``GridSearchCV`` over ``HYPERPARAMS`` grid.
    cv_folds : int
        Number of CV folds for hyperparameter search.
    lat_min, lat_max, lon_min, lon_max : float | None
        Optional spatial domain filter.

    Returns
    -------
    dict  ``{model_name: {target: metrics_dict}}``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_aqi_training_data(
        input_csv,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
    )

    feat_avail = [c for c in FEATURE_COLS if c in df.columns]
    tgt_avail = [c for c in TARGET_COLS if c in df.columns]
    df_clean = df.dropna(subset=feat_avail + [tgt_avail[0]])

    train_mask = df_clean["date"] <= pd.Timestamp(train_end)
    test_mask = df_clean["date"] >= pd.Timestamp(test_start)
    train_df = df_clean[train_mask]
    test_df = df_clean[test_mask]

    if len(train_df) == 0:
        logger.warning("No training rows after date split — using 80/20 fallback.")
        n = len(df_clean)
        train_df = df_clean.iloc[: int(0.8 * n)]
        test_df = df_clean.iloc[int(0.8 * n):]

    logger.info("Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    X_train = train_df[feat_avail].values
    X_test = test_df[feat_avail].values

    base_models: dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42,
        ),
    }

    all_metrics: dict[str, dict] = {}
    flat_rows: list[dict] = []

    for model_name, base_model in base_models.items():
        all_metrics[model_name] = {}

        for target in tgt_avail:
            y_train = train_df[target].values
            y_test = test_df[target].values

            logger.info("▶ Training %s → %s …", model_name, target)

            if hparam_search:
                grid = GridSearchCV(
                    base_model,
                    HYPERPARAMS[model_name],
                    cv=cv_folds,
                    scoring="neg_root_mean_squared_error",
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                )
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                logger.info("  Best params: %s", grid.best_params_)
            else:
                import copy
                model = copy.deepcopy(base_model)
                model.fit(X_train, y_train)

            y_pred = np.clip(model.predict(X_test), 0, None)
            metrics = _evaluate(y_test, y_pred, f"{model_name}/{target}")
            all_metrics[model_name][target] = metrics
            flat_rows.append(metrics)

            # Save predictions CSV
            pred_df = test_df[["date"]].copy()
            if "cell_id" in test_df.columns:
                pred_df["cell_id"] = test_df["cell_id"].values
            pred_df[target] = y_test
            pred_df[f"{target}_pred"] = y_pred
            pred_df.to_csv(output_dir / f"predictions_{model_name}_{target}.csv", index=False)

            # Feature importances
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": feat_avail,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi.to_csv(
                    output_dir / f"feature_importance_{model_name}_{target}.csv",
                    index=False,
                )

            # Save model with joblib
            joblib.dump(model, output_dir / f"{model_name}_{target}.joblib")

    # Save metrics — JSON + CSV
    metrics_json = output_dir / "baseline_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(all_metrics, f, indent=2)

    results_csv = output_dir / "baseline_results.csv"
    if flat_rows:
        fieldnames = list(flat_rows[0].keys())
        with open(results_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)

    logger.info("All metrics → %s and %s", metrics_json, results_csv)
    return all_metrics


def evaluate_baseline_models(
    model_dir: str | Path,
    test_csv: str | Path,
    test_start: str = "2022-01-01",
    output_dir: str | Path | None = None,
) -> dict[str, dict]:
    """
    Evaluate all saved baseline models in *model_dir* on new test data.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing ``*.joblib`` model files.
    test_csv : str | Path
        CSV with features and targets.
    test_start : str
        Only rows on or after this date are evaluated.
    output_dir : str | Path | None
        Defaults to ``model_dir/evaluation/``.

    Returns
    -------
    dict  ``{model_target: metrics_dict}``
    """
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["date"] >= pd.Timestamp(test_start)].copy()
    if test_df.empty:
        logger.warning("No rows from %s onward; using full dataset.", test_start)
        test_df = df

    feat_avail = [c for c in FEATURE_COLS if c in test_df.columns]
    X_test = test_df[feat_avail].fillna(0).values

    all_metrics: dict[str, dict] = {}

    for model_path in sorted(model_dir.glob("*.joblib")):
        stem = model_path.stem
        target = next((t for t in TARGET_COLS if t in stem), None)
        if target is None or target not in test_df.columns:
            continue

        model = joblib.load(model_path)
        y_true = test_df[target].values
        valid = ~np.isnan(y_true)
        if valid.sum() < 10:
            logger.warning("Too few valid rows for %s; skipping.", stem)
            continue

        y_pred = np.clip(model.predict(X_test[valid]), 0, None)
        metrics = _evaluate(y_true[valid], y_pred, stem)
        all_metrics[stem] = metrics

    out_path = Path(output_dir) / "evaluation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Evaluation metrics → %s", out_path)
    return all_metrics


def load_and_predict(
    model_path: str | Path,
    input_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Load a saved joblib model and run inference on a DataFrame.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.joblib`` model file.
    input_df : pd.DataFrame
    feature_cols : list[str] | None
        Defaults to ``FEATURE_COLS``.

    Returns
    -------
    np.ndarray  of predicted values (clipped to 0).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    model = joblib.load(model_path)
    X = input_df[[c for c in feature_cols if c in input_df.columns]].fillna(0).values
    return np.clip(model.predict(X), 0, None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline AQI models (V2)")
    parser.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    parser.add_argument("--output_dir", default="models/baseline")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--test_start", default="2022-01-01")
    parser.add_argument("--hparam_search", action="store_true",
                        help="Run GridSearchCV over pre-defined hyperparameter grids")
    parser.add_argument("--igp_only", action="store_true",
                        help="Restrict to Indo-Gangetic Plain (lat 23–30, lon 75–90)")
    args = parser.parse_args()

    lat_min, lat_max, lon_min, lon_max = (None,) * 4
    if args.igp_only:
        lat_min, lat_max, lon_min, lon_max = 23.0, 30.0, 75.0, 90.0
        logger.info("Domain subset: Indo-Gangetic Plain only")

    from src.utils.logging_utils import setup_logging
    setup_logging(log_file="logs/baseline_training.log")

    metrics = train_baseline_models(
        args.input, args.output_dir,
        train_end=args.train_end, test_start=args.test_start,
        hparam_search=args.hparam_search,
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
    )
    print("\n=== Final Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
