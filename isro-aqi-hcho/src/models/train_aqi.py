"""
train_aqi.py
============
Training script for the CNN-LSTM AQI / PM2.5 model.

V2 improvements
---------------
* ``ReduceLROnPlateau`` scheduler in addition to StepLR (config-selectable).
* Early stopping patience now properly resets on improvement.
* Hyperparameter sweep loop over LR × dropout for quick ablations.
* Prediction export to CSV after training (daily mean PM2.5 per grid cell).
* Centralised logging via ``src.utils.logging_utils``.
* Robust error handling for missing data files.

Usage::

    # Quick synthetic smoke-test
    python -m src.models.train_aqi --synthetic

    # Real training
    python -m src.models.train_aqi --config config/aqi_training.yaml

    # Hyperparameter sweep (synthetic)
    python -m src.models.train_aqi --synthetic --hparam_sweep
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.cnn_lstm_aqi import CNNLSTM, AQIDataset, build_grid_arrays
from src.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config/aqi_training.yaml") -> dict:
    """Load and return the YAML training config."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Training loop helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one training epoch.

    Parameters
    ----------
    model, loader, optimizer, criterion, device : standard PyTorch objects.

    Returns
    -------
    float  Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / max(len(loader.dataset), 1)


def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Run one validation epoch.

    Returns
    -------
    float  Mean validation loss.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / max(len(loader.dataset), 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    config: dict,
    output_dir: str | Path,
    grid_csv: str | Path,
    device: torch.device | None = None,
    run_id: str = "default",
) -> dict:
    """
    Full CNN-LSTM training pipeline.

    Parameters
    ----------
    config : dict
        Loaded from ``config/aqi_training.yaml``.
    output_dir : str | Path
        Checkpoints and metrics are saved here.
    grid_csv : str | Path
        ``data/processed/grid_daily_features.csv``
    device : torch.device | None
        Auto-detected (CUDA > CPU) if None.
    run_id : str
        Tag used to name checkpoint files in sweep mode.

    Returns
    -------
    dict  Training history with ``train_loss``, ``val_loss``, ``best_epoch``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | run_id: %s", device, run_id)

    cnn_cfg = config["cnn_lstm"]
    spatial = cnn_cfg["spatial"]
    arch = cnn_cfg["architecture"]
    train_cfg = cnn_cfg["training"]

    feature_cols = (
        config["features"]["satellite"] + config["features"]["meteorological"]
    )
    seq_len: int = config["model"]["sequence_length"]
    img_h: int = spatial["img_height"]
    img_w: int = spatial["img_width"]

    # Load data ------------------------------------------------------------
    grid_path = Path(grid_csv)
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid CSV not found: {grid_path}\n"
            "Run `python -m src.data.build_dataset_aqi --synthetic` first."
        )

    logger.info("Building grid arrays from %s …", grid_path.name)
    try:
        X, y, dates = build_grid_arrays(
            grid_path, None, feature_cols, "pm25",
            img_h=img_h, img_w=img_w,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to build grid arrays: {exc}") from exc

    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    n = len(X)
    n_test = max(1, int(n * train_cfg["test_split"]))
    n_val = max(1, int(n * train_cfg["val_split"]))
    n_train = n - n_test - n_val
    logger.info("Samples — train: %d | val: %d | test: %d", n_train, n_val, n_test)

    train_ds = AQIDataset(X[:n_train], y[:n_train], seq_len)
    val_ds = AQIDataset(X[n_train: n_train + n_val], y[n_train: n_train + n_val], seq_len)

    bs = train_cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=0,
    )

    # Model ----------------------------------------------------------------
    model = CNNLSTM(
        in_channels=len(feature_cols),
        img_h=img_h,
        img_w=img_w,
        cnn_filters=tuple(arch["cnn_filters"]),
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
        dropout=arch["dropout"],
        fc_hidden=arch["fc_hidden"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CNN-LSTM trainable parameters: %s", f"{n_params:,}")

    # Optimiser + scheduler ------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler_type = train_cfg.get("scheduler", "step")
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=train_cfg.get("scheduler_gamma", 0.5),
            patience=train_cfg.get("scheduler_step", 5),
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg["scheduler_step"],
            gamma=train_cfg["scheduler_gamma"],
        )

    criterion = nn.MSELoss()

    # Training loop --------------------------------------------------------
    history: dict[str, list | int | float] = {
        "train_loss": [], "val_loss": [],
    }
    best_val = float("inf")
    patience_counter = 0
    best_epoch = 0
    ckpt_path = output_dir / f"best_model_{run_id}.pt"

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)

        if scheduler_type == "plateau":
            scheduler.step(vl_loss)
        else:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e  (%.1fs)",
            epoch, train_cfg["epochs"], tr_loss, vl_loss, current_lr, elapsed,
        )

        if vl_loss < best_val:
            best_val = vl_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                logger.info(
                    "Early stopping at epoch %d (best epoch %d  val=%.4f)",
                    epoch, best_epoch, best_val,
                )
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val

    hist_path = output_dir / f"training_history_{run_id}.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(
        "Best checkpoint: epoch %d  val_loss=%.4f  → %s",
        best_epoch, best_val, ckpt_path,
    )
    return history


def export_predictions(
    model: nn.Module,
    X: np.ndarray,
    dates: list[str],
    output_csv: str | Path,
    device: torch.device,
    seq_len: int = 7,
    batch_size: int = 16,
) -> None:
    """
    Run inference on the full grid and save daily mean PM2.5 to CSV.

    Parameters
    ----------
    model : CNNLSTM
        Trained model (already loaded with best weights).
    X : np.ndarray
        Shape ``(T, C, H, W)`` — full feature time series.
    dates : list[str]
        Date strings corresponding to the T dimension.
    output_csv : str | Path
        Destination file (e.g. ``data/processed/cnn_lstm_predictions.csv``).
    device : torch.device
    seq_len : int
        Sequence length used during training.
    batch_size : int
    """
    model.eval()
    ds = AQIDataset(X, np.zeros((len(X), X.shape[2], X.shape[3])), seq_len)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            out = model(xb.to(device)).cpu().numpy()   # (B, H, W)
            preds.append(out.reshape(len(xb), -1).mean(axis=1))  # daily mean

    pred_means = np.concatenate(preds)
    pred_dates = dates[seq_len:]  # first seq_len days have no prediction

    df_out = pd.DataFrame({"date": pred_dates, "pm25_pred_mean": pred_means.round(3)})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Exported %d daily predictions → %s", len(df_out), output_csv)


# ---------------------------------------------------------------------------
# Hyperparameter sweep
# ---------------------------------------------------------------------------

def hparam_sweep(
    config: dict,
    output_dir: str | Path,
    n_days: int = 400,
    img_size: int = 15,
) -> list[dict]:
    """
    Quick synthetic hyperparameter sweep over learning-rate × dropout.

    Parameters
    ----------
    config : dict
        Base training config (will be deep-copied per run).
    output_dir : str | Path
    n_days : int
        Number of synthetic timesteps.
    img_size : int
        Spatial grid size for synthetic runs.

    Returns
    -------
    list[dict]  Results table with run config + best val loss.
    """
    import copy
    import pandas as pd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict] = []
    lrs = [1e-3, 5e-4]
    dropouts = [0.2, 0.35]

    in_ch = len(config["features"]["satellite"]) + len(config["features"]["meteorological"])
    seq_len = config["model"]["sequence_length"]
    rng = np.random.default_rng(0)
    X_syn = rng.standard_normal((n_days, in_ch, img_size, img_size)).astype(np.float32)
    y_syn = np.abs(rng.standard_normal((n_days, img_size, img_size))).astype(np.float32) * 50

    split = int(n_days * 0.75)
    train_ds = AQIDataset(X_syn[:split], y_syn[:split], seq_len)
    val_ds = AQIDataset(X_syn[split:], y_syn[split:], seq_len)

    for lr in lrs:
        for dropout in dropouts:
            cfg = copy.deepcopy(config)
            arch = cfg["cnn_lstm"]["architecture"]
            arch["dropout"] = dropout
            cfg["cnn_lstm"]["training"]["learning_rate"] = lr
            cfg["cnn_lstm"]["training"]["epochs"] = 8
            cfg["cnn_lstm"]["training"]["batch_size"] = 16
            cfg["cnn_lstm"]["training"]["patience"] = 8

            model = CNNLSTM(
                in_channels=in_ch, img_h=img_size, img_w=img_size,
                cnn_filters=tuple(arch["cnn_filters"]),
                lstm_hidden=arch["lstm_hidden"],
                lstm_layers=arch["lstm_layers"],
                dropout=dropout,
                fc_hidden=arch["fc_hidden"],
            ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.MSELoss()
            tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
            vl_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

            best_vl = float("inf")
            for _ in range(8):
                train_epoch(model, tr_loader, opt, crit, device)
                vl = val_epoch(model, vl_loader, crit, device)
                best_vl = min(best_vl, vl)

            row = {"lr": lr, "dropout": dropout, "best_val_loss": round(best_vl, 4)}
            results.append(row)
            logger.info("Sweep  lr=%.0e  dropout=%.2f  → val=%.4f", lr, dropout, best_vl)

    out_path = Path(output_dir) / "hparam_sweep_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("best_val_loss").to_csv(out_path, index=False)
    logger.info("Sweep results → %s", out_path)
    return results


# ---------------------------------------------------------------------------
# Synthetic grid helper
# ---------------------------------------------------------------------------

def _synthetic_grid(
    n_days: int = 400,
    in_channels: int = 11,
    img_h: int = 15,
    img_w: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small synthetic grids for a quick smoke-test."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_days, in_channels, img_h, img_w)).astype(np.float32)
    y = np.abs(rng.standard_normal((n_days, img_h, img_w))).astype(np.float32) * 50
    return X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM AQI model (V2)")
    parser.add_argument("--config", default="config/aqi_training.yaml")
    parser.add_argument("--grid_csv", default="data/processed/grid_daily_features.csv")
    parser.add_argument("--output_dir", default="models/cnn_lstm")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run a synthetic smoke-test (no real data needed)")
    parser.add_argument("--hparam_sweep", action="store_true",
                        help="Run a quick LR×dropout sweep on synthetic data")
    args = parser.parse_args()

    setup_logging(log_file="logs/cnn_lstm_training.log")
    config = load_config(args.config)

    if args.hparam_sweep:
        logger.info("=== Hyperparameter Sweep (synthetic) ===")
        results = hparam_sweep(config, args.output_dir)
        import pandas as pd
        print("\n" + pd.DataFrame(results).sort_values("best_val_loss").to_string(index=False))
        return

    if args.synthetic:
        logger.info("=== Synthetic smoke-test ===")
        cnn_cfg = config["cnn_lstm"]
        cnn_cfg["spatial"]["img_height"] = 15
        cnn_cfg["spatial"]["img_width"] = 15
        cnn_cfg["training"]["epochs"] = 5
        cnn_cfg["training"]["batch_size"] = 8

        in_ch = (
            len(config["features"]["satellite"])
            + len(config["features"]["meteorological"])
        )
        X, y = _synthetic_grid(400, in_ch, 15, 15)
        seq_len = config["model"]["sequence_length"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_ds = AQIDataset(X[:300], y[:300], seq_len)
        val_ds = AQIDataset(X[300:350], y[300:350], seq_len)

        arch = cnn_cfg["architecture"]
        model = CNNLSTM(
            in_channels=in_ch, img_h=15, img_w=15,
            cnn_filters=tuple(arch["cnn_filters"]),
            lstm_hidden=arch["lstm_hidden"],
            lstm_layers=arch["lstm_layers"],
            dropout=arch["dropout"],
            fc_hidden=arch["fc_hidden"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False)

        for epoch in range(5):
            tr = train_epoch(model, loader, optimizer, criterion, device)
            vl = val_epoch(model, val_loader, criterion, device)
            logger.info("Epoch %d  train=%.4f  val=%.4f", epoch + 1, tr, vl)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "smoke_test_model.pt")
        logger.info("Smoke test complete → %s", out_dir / "smoke_test_model.pt")
        return

    history = train(config, args.output_dir, args.grid_csv)
    print("\nTraining complete.")
    print(json.dumps(history, indent=2))


import pandas as pd  # noqa: E402  (needed for export_predictions)

if __name__ == "__main__":
    main()
