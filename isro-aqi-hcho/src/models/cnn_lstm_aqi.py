"""
cnn_lstm_aqi.py
===============
Spatio-temporal CNN-LSTM model for PM2.5 prediction over India.

Architecture:
  - Per-timestep CNN: extracts spatial features from 2-D multi-channel grids.
  - LSTM: captures temporal evolution across T successive days.
  - FC head: outputs predicted PM2.5 for each grid cell at day T+1.

Input tensor shape: (batch, T, C, H, W)
  T = sequence length (days)
  C = number of feature channels (satellite + met)
  H, W = spatial grid dimensions

Output shape: (batch, H, W) — predicted PM2.5 surface
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Single Conv2D + BN + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SpatialEncoder(nn.Module):
    """
    CNN encoder that maps a single-day feature grid to a spatial feature map.

    Input:  (B, C, H, W)
    Output: (B, feat_dim, H, W)  where feat_dim = cnn_filters[-1]
    """

    def __init__(self, in_channels: int, filters: list[int] = (32, 64)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for out_ch in filters:
            layers.append(ConvBlock(ch, out_ch))
            ch = out_ch
        self.encoder = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM for gridded PM2.5 prediction.

    Parameters
    ----------
    in_channels : int
        Number of feature channels per day (C).
    img_h, img_w : int
        Spatial grid dimensions (H, W).
    cnn_filters : list[int]
        Output channels for each CNN block.
    lstm_hidden : int
        Number of LSTM hidden units.
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability.
    fc_hidden : int
        Hidden units in the fully-connected head.
    """

    def __init__(
        self,
        in_channels: int = 11,
        img_h: int = 30,
        img_w: int = 30,
        cnn_filters: tuple[int, ...] = (32, 64),
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        fc_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.spatial_encoder = SpatialEncoder(in_channels, list(cnn_filters))
        cnn_out_ch = self.spatial_encoder.out_channels
        cnn_feat_dim = cnn_out_ch * img_h * img_w

        self.lstm = nn.LSTM(
            input_size=cnn_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, img_h * img_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, T, C, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, H, W) — predicted PM2.5 grid for day T+1.
        """
        B, T, C, H, W = x.shape

        # Encode each time step independently
        x_flat = x.view(B * T, C, H, W)           # (B*T, C, H, W)
        feat = self.spatial_encoder(x_flat)         # (B*T, feat_ch, H, W)
        feat = feat.view(B, T, -1)                  # (B, T, feat_dim)

        # LSTM over time
        lstm_out, _ = self.lstm(feat)               # (B, T, hidden)
        last = self.dropout(lstm_out[:, -1, :])     # (B, hidden)

        # Decode to PM2.5 grid
        out = self.head(last)                       # (B, H*W)
        return out.view(B, H, W)                    # (B, H, W)


class AQIDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset for CNN-LSTM training.

    Parameters
    ----------
    grid_array : np.ndarray
        Shape (T_total, C, H, W) — full time series of feature grids.
    target_array : np.ndarray
        Shape (T_total, H, W) — full time series of PM2.5 grids.
    seq_len : int
        Length of input sequence (T).
    """

    def __init__(
        self,
        grid_array: np.ndarray,
        target_array: np.ndarray,
        seq_len: int = 7,
    ) -> None:
        assert grid_array.shape[0] == target_array.shape[0]
        self.X = torch.from_numpy(grid_array.astype(np.float32))
        self.y = torch.from_numpy(target_array.astype(np.float32))
        self.seq_len = seq_len
        self.n_valid = len(grid_array) - seq_len

    def __len__(self) -> int:
        return max(0, self.n_valid)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx: idx + self.seq_len]       # (T, C, H, W)
        y = self.y[idx + self.seq_len]             # (H, W)
        return x, y


def build_grid_arrays(
    grid_csv: str | Path,
    target_csv: str | Path | None,
    feature_cols: list[str],
    target_col: str = "pm25",
    img_h: int = 30,
    img_w: int = 30,
    resolution: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Reshape the flat gridded feature CSV into 4-D NumPy arrays
    (T, C, H, W) for the CNN-LSTM.

    Parameters
    ----------
    grid_csv : str | Path
        data/processed/grid_daily_features.csv
    target_csv : str | Path | None
        If None, target is extracted from grid_csv itself (must contain PM2.5).
    feature_cols : list[str]
    target_col : str
    img_h, img_w : int
        Cropped grid size (centred on India's high-density region).
    resolution : float

    Returns
    -------
    (grid_array, target_array, dates)
        grid_array   shape (T, C, img_h, img_w)
        target_array shape (T, img_h, img_w)
        dates        list[str] of length T
    """
    import pandas as pd
    from src.data.grid_definition import grid_to_array

    df = pd.read_csv(grid_csv)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    # Determine bounding box for a square crop
    lat_vals = np.sort(df["lat"].unique())
    lon_vals = np.sort(df["lon"].unique())
    lat_center = (lat_vals.min() + lat_vals.max()) / 2
    lon_center = (lon_vals.min() + lon_vals.max()) / 2
    half_h = img_h * resolution / 2
    half_w = img_w * resolution / 2
    lat_min, lat_max = lat_center - half_h, lat_center + half_h
    lon_min, lon_max = lon_center - half_w, lon_center + half_w

    grid_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []

    for date in dates:
        day = df[df["date"] == date]
        day_crop = day[
            (day["lat"] >= lat_min) & (day["lat"] <= lat_max) &
            (day["lon"] >= lon_min) & (day["lon"] <= lon_max)
        ]

        channels: list[np.ndarray] = []
        for col in feature_cols:
            if col in day_crop.columns:
                arr, _, _ = grid_to_array(day_crop, col, resolution)
                # Pad / crop to fixed size
                arr = _pad_or_crop(arr, img_h, img_w)
                channels.append(arr)
            else:
                channels.append(np.zeros((img_h, img_w), dtype=np.float32))

        grid_arrays.append(np.stack(channels, axis=0))   # (C, H, W)

        if target_col in day_crop.columns:
            t_arr, _, _ = grid_to_array(day_crop, target_col, resolution)
            t_arr = _pad_or_crop(t_arr, img_h, img_w)
        else:
            t_arr = np.zeros((img_h, img_w), dtype=np.float32)
        target_arrays.append(t_arr)

    X = np.stack(grid_arrays, axis=0)    # (T, C, H, W)
    y = np.stack(target_arrays, axis=0)  # (T, H, W)
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    return X, y, date_strs


def _pad_or_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad with NaN or crop a 2-D array to (target_h, target_w)."""
    h, w = arr.shape
    out = np.full((target_h, target_w), np.nan, dtype=np.float32)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
    return np.nan_to_num(out, nan=0.0)


if __name__ == "__main__":
    model = CNNLSTM(in_channels=11, img_h=30, img_w=30)
    dummy = torch.randn(4, 7, 11, 30, 30)
    out = model(dummy)
    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
