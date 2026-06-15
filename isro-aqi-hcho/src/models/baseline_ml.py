"""
baseline_ml.py
==============
Baseline ML models (Random Forest & Gradient Boosting) for PM2.5 and AQI prediction.

Trains time-split models (train 2019–2021, test 2022) and reports
RMSE, MAE, R², and Pearson-r metrics per target.

Usage:
    python -m src.models.baseline_ml \\
        --input data/processed/aqi_training_dataset.csv \\
        --output_dir models/baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "no2_column", "so2_column", "co_column", "o3_column",
    "hcho_column", "insat_aod",
    "t2m", "rh2m", "u10", "v10", "tp", "sp", "blh",
]

TARGET_COLS = ["pm25_target", "aqi_target"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str = "") -> dict:
    """Compute RMSE, MAE, R², and Pearson-r."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    metrics = {"name": name, "rmse": round(rmse, 3), "mae": round(mae, 3),
               "r2": round(r2, 4), "pearson_r": round(pearson_r, 4)}
    logger.info("[%s] RMSE=%.3f  MAE=%.3f  R²=%.4f  r=%.4f", name, rmse, mae, r2, pearson_r)
    return metrics


def train_baseline(
    input_csv: str | Path,
    output_dir: str | Path,
    train_end: str = "2021-12-31",
    test_start: str = "2022-01-01",
) -> dict[str, dict]:
    """
    Train Random Forest and Gradient Boosting models for each target.

    Parameters
    ----------
    input_csv : str | Path
        Path to aqi_training_dataset.csv (or aqi_features.csv).
    output_dir : str | Path
        Directory to save models and metrics.
    train_end, test_start : str
        Date split for temporal cross-validation.

    Returns
    -------
    dict  {model_name: {target: metrics}}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])

    features_available = [c for c in FEATURE_COLS if c in df.columns]
    targets_available = [c for c in TARGET_COLS if c in df.columns]

    if not features_available or not targets_available:
        raise ValueError(
            f"No usable features or targets found.\n"
            f"  Features needed: {FEATURE_COLS}\n"
            f"  Found: {list(df.columns)}"
        )

    logger.info("Features: %s", features_available)
    logger.info("Targets:  %s", targets_available)

    train_mask = df["date"] <= pd.Timestamp(train_end)
    test_mask = df["date"] >= pd.Timestamp(test_start)

    df_clean = df.dropna(subset=features_available + targets_available[:1])
    train = df_clean[train_mask]
    test = df_clean[test_mask]

    if len(train) == 0:
        logger.warning("No training rows after split; using 80/20 split instead.")
        n = len(df_clean)
        train = df_clean.iloc[: int(0.8 * n)]
        test = df_clean.iloc[int(0.8 * n):]

    logger.info("Train: %d rows | Test: %d rows", len(train), len(test))

    X_train = train[features_available].values
    X_test = test[features_available].values

    models_cfg = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
        ),
    }

    all_metrics: dict[str, dict] = {}

    for model_name, model_template in models_cfg.items():
        all_metrics[model_name] = {}

        for target in targets_available:
            y_train = train[target].values
            y_test = test[target].values

            logger.info("Training %s → %s …", model_name, target)

            import copy
            model = copy.deepcopy(model_template)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)

            metrics = evaluate(y_test, y_pred, f"{model_name}/{target}")
            all_metrics[model_name][target] = metrics

            # Save predictions
            pred_df = test[["date", "cell_id"] + [target]].copy() if "cell_id" in test.columns \
                else test[["date"] + [target]].copy()
            pred_df[f"{target}_pred"] = y_pred
            pred_path = output_dir / f"predictions_{model_name}_{target}.csv"
            pred_df.to_csv(pred_path, index=False)

            # Feature importances
            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({
                    "feature": features_available,
                    "importance": model.feature_importances_,
                }).sort_values("importance", ascending=False)
                fi.to_csv(output_dir / f"feature_importance_{model_name}_{target}.csv", index=False)

            # Save model
            model_path = output_dir / f"{model_name}_{target}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

    # Save all metrics
    metrics_path = output_dir / "baseline_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return all_metrics


def load_and_predict(
    model_path: str | Path,
    input_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Load a saved baseline model and run inference on a DataFrame.

    Parameters
    ----------
    model_path : str | Path
    input_df : pd.DataFrame
    feature_cols : list[str] | None

    Returns
    -------
    np.ndarray of predictions.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    X = input_df[[c for c in feature_cols if c in input_df.columns]].values
    return np.clip(model.predict(X), 0, None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline AQI models")
    parser.add_argument("--input", default="data/processed/aqi_training_dataset.csv")
    parser.add_argument("--output_dir", default="models/baseline")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--test_start", default="2022-01-01")
    args = parser.parse_args()

    metrics = train_baseline(args.input, args.output_dir, args.train_end, args.test_start)
    print("\n=== Final Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
