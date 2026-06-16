"""
cnn_lstm_aqi.py
===============
V3 spatio-temporal models for PM2.5 / AQI prediction over India.

Models
------
CNNLSTM      — original V2 architecture (CNN encoder → LSTM → FC head)
ConvLSTMCell — single ConvLSTM cell (Shi et al., 2015)
ConvLSTM     — multi-layer ConvLSTM module
ConvLSTMModel — full model: ConvLSTM backbone → Conv head

Factory
-------
build_model(config)  — returns the correct model class based on
                        ``config["model"]["model_type"]``

Input tensor shape : (B, T, C, H, W)
Output tensor shape: (B, H, W) — predicted PM2.5 surface at day T+1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Shared building blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Single Conv2D + BN + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class SpatialEncoder(nn.Module):
    """
    Stacked CNN encoder mapping a single-day feature grid to a spatial feature map.

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


# ══════════════════════════════════════════════════════════════════════════════
# V2 Architecture — CNN-LSTM
# ══════════════════════════════════════════════════════════════════════════════

class CNNLSTM(nn.Module):
    """
    CNN-LSTM for gridded PM2.5 prediction.

    Pipeline
    --------
    1. Per-timestep SpatialEncoder (CNN) extracts (B, feat_dim, H, W)
    2. Flatten spatial dims → (B, T, feat_dim × H × W)
    3. LSTM over T timesteps
    4. FC head from last hidden state → (B, H × W)
    5. Reshape to (B, H, W)
    """

    def __init__(
        self,
        in_channels:  int = 11,
        img_h:        int = 30,
        img_w:        int = 30,
        cnn_filters:  tuple[int, ...] = (32, 64),
        lstm_hidden:  int = 128,
        lstm_layers:  int = 2,
        dropout:      float = 0.3,
        fc_hidden:    int = 64,
    ) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.spatial_encoder = SpatialEncoder(in_channels, list(cnn_filters))
        cnn_out_ch   = self.spatial_encoder.out_channels
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
        x : Tensor  (B, T, C, H, W)

        Returns
        -------
        Tensor  (B, H, W)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feat   = self.spatial_encoder(x_flat)     # (B*T, ch, H, W)
        feat   = feat.view(B, T, -1)              # (B, T, feat_dim)
        lstm_out, _ = self.lstm(feat)             # (B, T, hidden)
        last   = self.dropout(lstm_out[:, -1, :]) # (B, hidden)
        out    = self.head(last)                  # (B, H*W)
        return out.view(B, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# V3 Architecture — ConvLSTM
# ══════════════════════════════════════════════════════════════════════════════

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell (Shi et al., 2015).

    State tensors h and c have shape (B, hidden_channels, H, W).
    The cell applies spatial convolutions instead of fully-connected gates
    so spatial structure is preserved through the temporal recurrence.

    Parameters
    ----------
    in_channels    : channels of the input x_t
    hidden_channels: channels of the hidden/cell states
    kernel_size    : convolutional kernel size (default 3)
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # Gates: input, forget, cell candidate, output — packed into one conv
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            hidden_channels * 4,
            kernel_size,
            padding=pad,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x     : (B, in_channels, H, W)
        state : (h, c) each (B, hidden_channels, H, W), or None

        Returns
        -------
        (h_new, c_new)  each (B, hidden_channels, H, W)
        """
        B, _, H, W = x.shape
        if state is None:
            h = torch.zeros(B, self.hidden_channels, H, W, device=x.device, dtype=x.dtype)
            c = torch.zeros_like(h)
        else:
            h, c = state

        combined = torch.cat([x, h], dim=1)      # (B, in_ch + hidden_ch, H, W)
        gates    = self.gates(combined)           # (B, 4*hidden, H, W)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM module.

    Processes a (B, T, C, H, W) sequence and returns the final hidden state
    of the last layer: (B, hidden_channels[-1], H, W).

    Parameters
    ----------
    in_channels     : input channel count
    hidden_channels : list of hidden-channel counts per layer
    kernel_size     : kernel size used in all ConvLSTMCells
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: list[int] = (64, 128),
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.cells = nn.ModuleList()
        ch = in_channels
        for h_ch in hidden_channels:
            self.cells.append(ConvLSTMCell(ch, h_ch, kernel_size))
            ch = h_ch
        self.out_channels = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        Tensor  (B, hidden_channels[-1], H, W) — final hidden state of last layer
        """
        B, T, C, H, W = x.shape
        layer_input = x  # (B, T, ch, H, W)

        for cell in self.cells:
            state: tuple[torch.Tensor, torch.Tensor] | None = None
            outputs: list[torch.Tensor] = []
            for t in range(T):
                h, c = cell(layer_input[:, t], state)
                state = (h, c)
                outputs.append(h.unsqueeze(1))
            layer_input = torch.cat(outputs, dim=1)  # (B, T, hidden_ch, H, W)

        # Return final hidden state of last layer
        return layer_input[:, -1]  # (B, hidden_ch[-1], H, W)


class ConvLSTMModel(nn.Module):
    """
    Full model: ConvLSTM backbone → optional CNN refinement → prediction head.

    The ConvLSTM processes the full spatio-temporal sequence preserving 2-D
    spatial structure throughout, unlike CNNLSTM which flattens before LSTM.

    Parameters
    ----------
    in_channels     : feature channels per timestep (C)
    img_h, img_w    : spatial grid size
    hidden_channels : ConvLSTM layer sizes (e.g. [64, 128])
    kernel_size     : ConvLSTM kernel size
    dropout         : dropout in the prediction head
    refine_channels : additional CNN channels applied to the final hidden state
    """

    def __init__(
        self,
        in_channels:     int = 11,
        img_h:           int = 30,
        img_w:           int = 30,
        hidden_channels: tuple[int, ...] = (64, 128),
        kernel_size:     int = 3,
        dropout:         float = 0.3,
        refine_channels: int = 64,
    ) -> None:
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w

        self.convlstm = ConvLSTM(in_channels, list(hidden_channels), kernel_size)
        final_ch      = self.convlstm.out_channels

        # Spatial refinement + prediction
        self.refine = nn.Sequential(
            ConvBlock(final_ch, refine_channels),
            nn.Dropout2d(dropout),
        )
        self.pred_head = nn.Conv2d(refine_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C, H, W)

        Returns
        -------
        Tensor  (B, H, W)
        """
        h_final = self.convlstm(x)            # (B, hidden[-1], H, W)
        refined  = self.refine(h_final)       # (B, refine_ch, H, W)
        out      = self.pred_head(refined)    # (B, 1, H, W)
        return out.squeeze(1)                 # (B, H, W)


# ══════════════════════════════════════════════════════════════════════════════
# Model Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(config: dict) -> nn.Module:
    """
    Instantiate the correct model from a training config dict.

    The ``config["model"]["model_type"]`` key selects the architecture:

    ==================  ====================================
    ``"cnnlstm"``       V2 CNN-LSTM (default)
    ``"convlstm"``      V3 ConvLSTM (spatial-through-time)
    ==================  ====================================

    Parameters
    ----------
    config : dict
        Loaded from ``config/aqi_training.yaml``.  The function reads:

        - ``config["model"]["model_type"]``
        - ``config["model"]["in_channels"]``  (or len of feature lists)
        - ``config["model"]["img_h"]`` / ``"img_w"``
        - ``config["cnn_lstm"]["architecture"]``
        - ``config["convlstm"]["architecture"]`` (for ConvLSTM)

    Returns
    -------
    nn.Module  (CNNLSTM or ConvLSTMModel)

    Examples
    --------
    >>> import yaml
    >>> config = yaml.safe_load(open("config/aqi_training.yaml"))
    >>> model = build_model(config)
    """
    model_cfg = config.get("model", {})
    model_type = model_cfg.get("model_type", "cnnlstm").lower().replace("-", "")

    n_sat = len(config.get("features", {}).get("satellite", []))
    n_met = len(config.get("features", {}).get("meteorological", []))
    n_derived = len(config.get("features", {}).get("derived", []))
    in_channels = model_cfg.get("in_channels", max(1, n_sat + n_met + n_derived))
    img_h = model_cfg.get("img_h", 30)
    img_w = model_cfg.get("img_w", 30)

    if model_type == "cnnlstm":
        arch = config.get("cnn_lstm", {}).get("architecture", {})
        model = CNNLSTM(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            cnn_filters=tuple(arch.get("cnn_filters", [32, 64])),
            lstm_hidden=arch.get("lstm_hidden", 128),
            lstm_layers=arch.get("lstm_layers", 2),
            dropout=arch.get("dropout", 0.3),
            fc_hidden=arch.get("fc_hidden", 64),
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Built CNNLSTM: %s params", f"{n_params:,}")

    elif model_type == "convlstm":
        arch = config.get("convlstm", {}).get("architecture", {})
        model = ConvLSTMModel(
            in_channels=in_channels,
            img_h=img_h,
            img_w=img_w,
            hidden_channels=tuple(arch.get("hidden_channels", [64, 128])),
            kernel_size=arch.get("kernel_size", 3),
            dropout=arch.get("dropout", 0.3),
            refine_channels=arch.get("refine_channels", 64),
        )
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Built ConvLSTMModel: %s params", f"{n_params:,}")

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            "Supported: 'cnnlstm', 'convlstm'."
        )

    return model


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class AQIDataset(torch.utils.data.Dataset):
    """
    Sliding-window dataset for CNN-LSTM / ConvLSTM training.

    Parameters
    ----------
    grid_array   : np.ndarray (T_total, C, H, W) — feature time series
    target_array : np.ndarray (T_total, H, W)    — PM2.5 target grids
    seq_len      : int  input sequence length T
    """

    def __init__(
        self,
        grid_array:   np.ndarray,
        target_array: np.ndarray,
        seq_len:      int = 7,
    ) -> None:
        assert grid_array.shape[0] == target_array.shape[0]
        self.X       = torch.from_numpy(grid_array.astype(np.float32))
        self.y       = torch.from_numpy(target_array.astype(np.float32))
        self.seq_len = seq_len
        self.n_valid = len(grid_array) - seq_len

    def __len__(self) -> int:
        return max(0, self.n_valid)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx: idx + self.seq_len]  # (T, C, H, W)
        y = self.y[idx + self.seq_len]        # (H, W)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Grid array builder
# ══════════════════════════════════════════════════════════════════════════════

def build_grid_arrays(
    grid_csv:     str | Path,
    target_csv:   str | Path | None,
    feature_cols: list[str],
    target_col:   str = "pm25",
    img_h:        int = 30,
    img_w:        int = 30,
    resolution:   float = 0.1,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Reshape flat gridded CSV into (T, C, H, W) NumPy arrays.

    Returns
    -------
    (grid_array, target_array, date_strs)
        grid_array   shape (T, C, img_h, img_w)
        target_array shape (T, img_h, img_w)
        date_strs    list[str] of length T
    """
    import pandas as pd
    from src.data.grid_definition import grid_to_array

    df = pd.read_csv(grid_csv)
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    lat_vals = np.sort(df["lat"].unique())
    lon_vals = np.sort(df["lon"].unique())
    lat_center = (lat_vals.min() + lat_vals.max()) / 2
    lon_center = (lon_vals.min() + lon_vals.max()) / 2
    half_h = img_h * resolution / 2
    half_w = img_w * resolution / 2
    lat_min, lat_max = lat_center - half_h, lat_center + half_h
    lon_min, lon_max = lon_center - half_w, lon_center + half_w

    grid_arrays:  list[np.ndarray] = []
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
                arr = _pad_or_crop(arr, img_h, img_w)
                channels.append(arr)
            else:
                channels.append(np.zeros((img_h, img_w), dtype=np.float32))

        grid_arrays.append(np.stack(channels, axis=0))

        if target_col in day_crop.columns:
            t_arr, _, _ = grid_to_array(day_crop, target_col, resolution)
            t_arr = _pad_or_crop(t_arr, img_h, img_w)
        else:
            t_arr = np.zeros((img_h, img_w), dtype=np.float32)
        target_arrays.append(t_arr)

    X = np.stack(grid_arrays,  axis=0)   # (T, C, H, W)
    y = np.stack(target_arrays, axis=0)  # (T, H, W)
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    return X, y, date_strs


def _pad_or_crop(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad with zero or crop a 2-D array to (target_h, target_w)."""
    h, w = arr.shape
    out = np.full((target_h, target_w), np.nan, dtype=np.float32)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
    return np.nan_to_num(out, nan=0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Quick self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import yaml

    dummy_config = {
        "features": {
            "satellite":      ["no2_column", "so2_column", "co_column", "o3_column", "hcho_column", "insat_aod"],
            "meteorological": ["t2m", "rh2m", "u10", "v10", "tp", "sp", "blh"],
            "derived":        [],
        },
        "model": {
            "model_type": "cnnlstm",
            "img_h": 30,
            "img_w": 30,
        },
        "cnn_lstm": {
            "architecture": {
                "cnn_filters": [32, 64],
                "lstm_hidden": 128,
                "lstm_layers": 2,
                "dropout": 0.3,
                "fc_hidden": 64,
            }
        },
        "convlstm": {
            "architecture": {
                "hidden_channels": [64, 128],
                "kernel_size": 3,
                "dropout": 0.3,
                "refine_channels": 64,
            }
        },
    }

    dummy = torch.randn(2, 7, 13, 30, 30)

    print("=== CNNLSTM ===")
    dummy_config["model"]["model_type"] = "cnnlstm"
    m1 = build_model(dummy_config)
    out1 = m1(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out1.shape)}")
    print(f"  Params: {sum(p.numel() for p in m1.parameters() if p.requires_grad):,}")

    print("\n=== ConvLSTMModel ===")
    dummy_config["model"]["model_type"] = "convlstm"
    m2 = build_model(dummy_config)
    out2 = m2(dummy)
    print(f"  Input:  {tuple(dummy.shape)}")
    print(f"  Output: {tuple(out2.shape)}")
    print(f"  Params: {sum(p.numel() for p in m2.parameters() if p.requires_grad):,}")
