"""
evaluate_aqi.py
===============
Evaluate a trained AQI model (baseline or CNN-LSTM) on the test split.

Generates:
  - Metrics table (RMSE, MAE, R², Pearson r) per pollutant
  - Scatter plots: predicted vs. observed
  - Time series plots at selected stations

Usage:
    # Baseline models
    python -m src.models.evaluate_aqi \\
        --model_type baseline \\
        --model_dir models/baseline \\
        --test_csv data/processed/aqi_training_dataset.csv

    # CNN-LSTM
    python -m src.models.evaluate_aqi \\
        --model_type cnn_lstm \\
        --model_dir models/cnn_lstm \\
        --grid_csv data/processed/grid_daily_features.csv \\
        --config config/aqi_training.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "no2_column", "so2_column", "co_column", "o3_column",
    "hcho_column", "insat_aod",
    "t2m", "rh2m", "u10", "v10", "tp", "sp", "blh",
]
TARGET_COLS = ["pm25_target", "aqi_target"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return RMSE, MAE, R², and Pearson-r."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"rmse": round(rmse, 3), "mae": round(mae, 3), "r2": round(r2, 4), "pearson_r": round(r, 4)}


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: str | Path) -> None:
    """Create a scatter plot of predicted vs. observed values."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="1:1 line")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_series(dates: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                     title: str, out_path: str | Path) -> None:
    """Create a time series comparison plot."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label="Observed", linewidth=1.5, color="black")
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.5, color="steelblue", alpha=0.8)
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_baseline(
    model_dir: str | Path,
    test_csv: str | Path,
    test_start: str = "2022-01-01",
    output_dir: str | Path | None = None,
) -> dict:
    """
    Evaluate all saved baseline models in *model_dir*.

    Parameters
    ----------
    model_dir : str | Path
    test_csv : str | Path
    test_start : str
    output_dir : str | Path | None
        Where to save plots (defaults to model_dir/evaluation/).

    Returns
    -------
    dict  {model_target: metrics}
    """
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(test_csv)
    df["date"] = pd.to_datetime(df["date"])
    test_df = df[df["date"] >= pd.Timestamp(test_start)].copy()

    if test_df.empty:
        logger.warning("No test rows found for date >= %s; using all data.", test_start)
        test_df = df

    features_available = [c for c in FEATURE_COLS if c in test_df.columns]
    X_test = test_df[features_available].fillna(0).values

    all_metrics: dict[str, dict] = {}

    for model_path in sorted(model_dir.glob("*.pkl")):
        name = model_path.stem
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Infer target from filename
        target = next((t for t in TARGET_COLS if t in name), None)
        if target is None or target not in test_df.columns:
            continue

        y_true = test_df[target].values
        valid = ~np.isnan(y_true)
        if valid.sum() < 10:
            logger.warning("Too few valid test rows for %s; skipping.", name)
            continue

        y_pred = np.clip(model.predict(X_test[valid]), 0, None)
        metrics = compute_metrics(y_true[valid], y_pred)
        all_metrics[name] = metrics
        logger.info("[%s] %s", name, metrics)

        plot_scatter(y_true[valid], y_pred, f"{name} – Scatter", Path(output_dir) / f"{name}_scatter.png")
        if "date" in test_df.columns:
            plot_time_series(
                test_df[valid]["date"].values, y_true[valid], y_pred,
                f"{name} – Time Series", Path(output_dir) / f"{name}_timeseries.png"
            )

    metrics_path = Path(output_dir) / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Evaluation metrics saved to %s", metrics_path)
    return all_metrics


def evaluate_cnn_lstm(
    model_dir: str | Path,
    grid_csv: str | Path,
    config_path: str | Path = "config/aqi_training.yaml",
    output_dir: str | Path | None = None,
) -> dict:
    """
    Evaluate the CNN-LSTM model on the test split.

    Parameters
    ----------
    model_dir : str | Path
    grid_csv : str | Path
    config_path : str | Path
    output_dir : str | Path | None

    Returns
    -------
    dict  – evaluation metrics.
    """
    import torch
    import yaml
    from src.models.cnn_lstm_aqi import CNNLSTM, AQIDataset, build_grid_arrays

    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cnn_cfg = config["cnn_lstm"]
    spatial = cnn_cfg["spatial"]
    arch = cnn_cfg["architecture"]
    train_cfg = cnn_cfg["training"]
    feature_cols = config["features"]["satellite"] + config["features"]["meteorological"]
    seq_len = config["model"]["sequence_length"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, dates = build_grid_arrays(
        grid_csv, None, feature_cols, "pm25",
        img_h=spatial["img_height"], img_w=spatial["img_width"],
    )
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    n = len(X)
    n_test = max(1, int(n * train_cfg["test_split"]))
    X_test, y_test = X[-n_test:], y[-n_test:]

    model = CNNLSTM(
        in_channels=len(feature_cols),
        img_h=spatial["img_height"],
        img_w=spatial["img_width"],
        cnn_filters=tuple(arch["cnn_filters"]),
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
        dropout=0.0,
        fc_hidden=arch["fc_hidden"],
    ).to(device)

    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_ds = AQIDataset(X_test, y_test, seq_len)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False)

    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).cpu().numpy()
            preds.append(pred)
            trues.append(yb.numpy())

    preds = np.concatenate(preds).ravel()
    trues = np.concatenate(trues).ravel()
    preds = np.clip(preds, 0, None)

    metrics = compute_metrics(trues, preds)
    logger.info("CNN-LSTM evaluation: %s", metrics)

    plot_scatter(trues, preds, "CNN-LSTM – PM2.5 Scatter", Path(output_dir) / "cnnlstm_scatter.png")

    with open(Path(output_dir) / "cnnlstm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AQI model")
    parser.add_argument("--model_type", choices=["baseline", "cnn_lstm"], default="baseline")
    parser.add_argument("--model_dir", default="models/baseline")
    parser.add_argument("--test_csv", default="data/processed/aqi_training_dataset.csv")
    parser.add_argument("--grid_csv", default="data/processed/grid_daily_features.csv")
    parser.add_argument("--config", default="config/aqi_training.yaml")
    parser.add_argument("--test_start", default="2022-01-01")
    args = parser.parse_args()

    if args.model_type == "baseline":
        metrics = evaluate_baseline(args.model_dir, args.test_csv, args.test_start)
    else:
        metrics = evaluate_cnn_lstm(args.model_dir, args.grid_csv, args.config)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
