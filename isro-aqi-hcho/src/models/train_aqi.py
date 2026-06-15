"""
train_aqi.py
============
Training script for the CNN-LSTM AQI model.

Usage:
    # Train with synthetic data (for testing)
    python -m src.models.train_aqi --config config/aqi_training.yaml --synthetic

    # Train with real data
    python -m src.models.train_aqi --config config/aqi_training.yaml

Checkpoints are saved under models/cnn_lstm/.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from src.models.cnn_lstm_aqi import CNNLSTM, AQIDataset, build_grid_arrays

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "config/aqi_training.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss."""
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
    return total_loss / len(loader.dataset)


def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch and return mean loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def train(
    config: dict,
    output_dir: str | Path,
    grid_csv: str | Path,
    device: torch.device | None = None,
) -> dict:
    """
    Full CNN-LSTM training pipeline.

    Parameters
    ----------
    config : dict
        Loaded aqi_training.yaml.
    output_dir : str | Path
        Where to save checkpoints and metrics.
    grid_csv : str | Path
        data/processed/grid_daily_features.csv
    device : torch.device | None
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    dict  – training history {train_loss, val_loss, best_epoch}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    cnn_cfg = config["cnn_lstm"]
    spatial = cnn_cfg["spatial"]
    arch = cnn_cfg["architecture"]
    train_cfg = cnn_cfg["training"]

    feature_cols = config["features"]["satellite"] + config["features"]["meteorological"]
    seq_len = config["model"]["sequence_length"]
    img_h = spatial["img_height"]
    img_w = spatial["img_width"]
    in_channels = len(feature_cols)

    logger.info("Building grid arrays …")
    X, y, dates = build_grid_arrays(
        grid_csv, None, feature_cols, "pm25",
        img_h=img_h, img_w=img_w,
    )

    # Impute NaNs
    X = np.nan_to_num(X, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)

    n = len(X)
    test_frac = train_cfg["test_split"]
    val_frac = train_cfg["val_split"]
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    n_train = n - n_test - n_val

    logger.info("Samples: %d train | %d val | %d test", n_train, n_val, n_test)

    train_ds = AQIDataset(X[:n_train], y[:n_train], seq_len)
    val_ds = AQIDataset(X[n_train: n_train + n_val], y[n_train: n_train + n_val], seq_len)

    bs = train_cfg["batch_size"]
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    model = CNNLSTM(
        in_channels=in_channels,
        img_h=img_h,
        img_w=img_w,
        cnn_filters=tuple(arch["cnn_filters"]),
        lstm_hidden=arch["lstm_hidden"],
        lstm_layers=arch["lstm_layers"],
        dropout=arch["dropout"],
        fc_hidden=arch["fc_hidden"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("CNN-LSTM parameters: %s", f"{n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_cfg["scheduler_step"],
        gamma=train_cfg["scheduler_gamma"],
    )
    criterion = nn.MSELoss()

    history: dict[str, list] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e  (%.1fs)",
            epoch, train_cfg["epochs"], tr_loss, vl_loss,
            optimizer.param_groups[0]["lr"], elapsed,
        )

        if vl_loss < best_val:
            best_val = vl_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                logger.info("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = best_val

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Best checkpoint: epoch %d  val_loss=%.4f  → %s", best_epoch, best_val, output_dir / "best_model.pt")
    return history


def _synthetic_grid(
    n_days: int = 400,
    in_channels: int = 11,
    img_h: int = 15,
    img_w: int = 15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate small synthetic grids for a quick smoke test."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_days, in_channels, img_h, img_w)).astype(np.float32)
    y = np.abs(rng.standard_normal((n_days, img_h, img_w))).astype(np.float32) * 50
    return X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN-LSTM AQI model")
    parser.add_argument("--config", default="config/aqi_training.yaml")
    parser.add_argument("--grid_csv", default="data/processed/grid_daily_features.csv")
    parser.add_argument("--output_dir", default="models/cnn_lstm")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (smoke test)")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.synthetic:
        logger.info("=== Synthetic smoke-test mode ===")
        cnn_cfg = config["cnn_lstm"]
        spatial = cnn_cfg["spatial"]
        # Shrink grid for fast test
        spatial["img_height"] = 15
        spatial["img_width"] = 15
        cnn_cfg["training"]["epochs"] = 5
        cnn_cfg["training"]["batch_size"] = 8

        in_ch = len(config["features"]["satellite"]) + len(config["features"]["meteorological"])
        X, y = _synthetic_grid(400, in_ch, 15, 15)

        seq_len = config["model"]["sequence_length"]
        train_ds = AQIDataset(X[:300], y[:300], seq_len)
        val_ds = AQIDataset(X[300:350], y[300:350], seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNLSTM(
            in_channels=in_ch, img_h=15, img_w=15,
            cnn_filters=tuple(cnn_cfg["architecture"]["cnn_filters"]),
            lstm_hidden=cnn_cfg["architecture"]["lstm_hidden"],
            lstm_layers=cnn_cfg["architecture"]["lstm_layers"],
            dropout=cnn_cfg["architecture"]["dropout"],
            fc_hidden=cnn_cfg["architecture"]["fc_hidden"],
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
        logger.info("Smoke test complete.")
        return

    history = train(config, args.output_dir, args.grid_csv)
    print("\nTraining complete. History:", json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
