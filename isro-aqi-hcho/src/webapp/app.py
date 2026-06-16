"""
app.py  —  V3
=============
Streamlit dashboard for the ISRO AQI & HCHO Hotspot project.

Run from isro-aqi-hcho/:
    streamlit run src/webapp/app.py

Seven pages
-----------
Overview · AQI Maps · Satellite Features · HCHO Hotspots ·
Time Series · Model Performance · About
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# ── Ensure the project root is on sys.path ────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import io
import json
import logging
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.WARNING)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISRO AQI & HCHO Hotspot Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

AQI_COLORS = {
    "Good":         "#00e400",
    "Satisfactory": "#ffff00",
    "Moderate":     "#ff7e00",
    "Poor":         "#ff0000",
    "Very Poor":    "#8f3f97",
    "Severe":       "#7e0023",
}

SEASON_LABELS = {
    "winter":       "Winter (Dec–Feb)",
    "pre_monsoon":  "Pre-Monsoon (Mar–May)",
    "monsoon":      "Monsoon (Jun–Sep)",
    "post_monsoon": "Post-Monsoon (Oct–Nov)",
}

REGIONS = {
    "Indo-Gangetic Plain": {"lat": (23.0, 30.0), "lon": (75.0, 90.0)},
    "Punjab-Haryana":      {"lat": (28.0, 32.0), "lon": (73.0, 78.0)},
    "Northeast India":     {"lat": (23.0, 28.0), "lon": (90.0, 97.5)},
    "Central Forests":     {"lat": (18.0, 25.0), "lon": (77.0, 87.0)},
    "Peninsular India":    {"lat": (8.0,  18.0), "lon": (73.0, 83.0)},
}

# Feature display names and units
FEATURE_META = {
    "no2_column":  ("NO₂ column",   "10⁻⁵ mol m⁻²"),
    "so2_column":  ("SO₂ column",   "10⁻⁵ mol m⁻²"),
    "co_column":   ("CO column",    "10⁻² mol m⁻²"),
    "o3_column":   ("O₃ column",    "10⁻³ mol m⁻²"),
    "hcho_column": ("HCHO column",  "µmol m⁻²"),
    "insat_aod":   ("INSAT-3D AOD", "dimensionless"),
    "t2m":         ("2m Temperature", "K"),
    "rh2m":        ("2m Relative Humidity", "%"),
    "u10":         ("10m Zonal Wind (u)", "m s⁻¹"),
    "v10":         ("10m Meridional Wind (v)", "m s⁻¹"),
    "tp":          ("Total Precipitation", "m d⁻¹"),
    "sp":          ("Surface Pressure", "Pa"),
    "blh":         ("Boundary Layer Height", "m"),
    "fire_count":  ("Fire Count (FIRMS)", "count d⁻¹"),
}

SEASON_MAP = {
    1: "winter", 2: "winter",
    3: "pre_monsoon", 4: "pre_monsoon", 5: "pre_monsoon",
    6: "monsoon", 7: "monsoon", 8: "monsoon", 9: "monsoon",
    10: "post_monsoon", 11: "post_monsoon", 12: "winter",
}


# ══════════════════════════════════════════════════════════════════════════════
# V3 UI Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _help_box(title: str, content: str) -> None:
    """Render a collapsible help/info box."""
    with st.expander(f"ℹ️  {title}", expanded=False):
        st.markdown(content)


def _csv_download_btn(
    df: pd.DataFrame,
    filename: str,
    label: str = "⬇️ Download CSV",
    key: str | None = None,
) -> None:
    """Render a Streamlit download button for a DataFrame as CSV."""
    if df is None or df.empty:
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key or filename,
    )


def _fig_download_btn(fig: plt.Figure, filename: str, label: str = "⬇️ Download figure") -> None:
    """Render a download button for a matplotlib figure as PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button(label, buf, file_name=filename, mime="image/png", key=filename)


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_grid_features() -> pd.DataFrame | None:
    p = DATA_DIR / "grid_daily_features.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_cpcb_daily() -> pd.DataFrame | None:
    p = DATA_DIR / "cpcb_daily.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_aqi_training() -> pd.DataFrame | None:
    p = DATA_DIR / "aqi_training_dataset.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_hcho_fire() -> pd.DataFrame | None:
    p = DATA_DIR / "hcho_fire_daily_grid.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_hcho_hotspots() -> pd.DataFrame | None:
    p = DATA_DIR / "hcho_hotspot_features.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_baseline_metrics() -> dict | None:
    p = MODELS_DIR / "baseline" / "baseline_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def generate_demo_data() -> dict[str, pd.DataFrame]:
    """Generate all synthetic demo datasets on-the-fly if CSVs not found."""
    from src.data.build_dataset_aqi import _build_synthetic_features, generate_synthetic_cpcb
    from src.data.build_dataset_hcho import _build_synthetic_hcho_dataset
    from src.utils.aqi_calculator import compute_aqi_series

    grid = _build_synthetic_features(n_cells=200, start="2021-01-01", end="2022-12-31")
    grid["date"] = pd.to_datetime(grid["date"])
    grid["aqi"]  = compute_aqi_series(
        grid.rename(columns={
            "no2_column": "no2", "so2_column": "so2",
            "co_column": "co", "o3_column": "o3",
        })
    )

    rng = np.random.default_rng(1)
    cpcb_rows = []
    sample_cells = grid.drop_duplicates("cell_id").sample(20, random_state=0)
    for _, cell in sample_cells.iterrows():
        for d in pd.date_range("2021-01-01", "2022-12-31", freq="D"):
            pm25 = max(5, cell["insat_aod"] * 100 + rng.normal(0, 10))
            cpcb_rows.append({
                "station_id": f"ST_{cell['cell_id'][:8]}",
                "city": f"City_{cell['lat']:.0f}_{cell['lon']:.0f}",
                "lat": cell["lat"], "lon": cell["lon"],
                "cell_id": cell["cell_id"],
                "date": d,
                "pm25":  round(pm25, 1),
                "pm10":  round(pm25 * rng.uniform(1.5, 2.5), 1),
                "no2":   round(abs(rng.normal(60, 20)), 1),
                "so2":   round(abs(rng.normal(30, 10)), 1),
                "o3":    round(abs(rng.normal(80, 15)), 1),
                "co":    round(abs(rng.normal(1.5, 0.4)), 2),
            })
    cpcb = pd.DataFrame(cpcb_rows)
    cpcb["aqi_observed"] = compute_aqi_series(cpcb)

    hcho_path = DATA_DIR / "hcho_fire_daily_grid_demo.csv"
    hcho = _build_synthetic_hcho_dataset(hcho_path, n_cells=300, start="2021-01-01", end="2022-12-31")

    return {"grid": grid, "cpcb": cpcb, "hcho": hcho}


# ══════════════════════════════════════════════════════════════════════════════
# Map helper
# ══════════════════════════════════════════════════════════════════════════════

def _aqi_scatter_fig(
    df: pd.DataFrame, val_col: str, title: str,
    cmap: str = "RdYlGn_r",
    vmin: float | None = None,
    vmax: float | None = None,
    s: int = 8,
) -> plt.Figure:
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(10, 7))

    if val_col in ("aqi", "aqi_observed"):
        bounds = [0, 50, 100, 200, 300, 400, 500]
        colors = ["#00e400", "#ffff00", "#ff7e00", "#ff0000", "#8f3f97", "#7e0023"]
        cmap_use = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(bounds, ncolors=len(colors))
        sc   = ax.scatter(df["lon"], df["lat"], c=df[val_col],
                          cmap=cmap_use, norm=norm, s=s, marker="s", linewidths=0)
        cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_ticks([25, 75, 150, 250, 350, 450])
        cbar.set_ticklabels(["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"])
        cbar.set_label("CPCB AQI Category")
    else:
        _vmin = vmin if vmin is not None else df[val_col].quantile(0.02)
        _vmax = vmax if vmax is not None else df[val_col].quantile(0.98)
        sc    = ax.scatter(df["lon"], df["lat"], c=df[val_col], cmap=cmap,
                           vmin=_vmin, vmax=_vmax, s=s, marker="s", linewidths=0)
        unit  = FEATURE_META.get(val_col, ("", ""))[1]
        label = f"{val_col} ({unit})" if unit else val_col
        plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04, label=label)

    ax.set_xlim(68, 97.5); ax.set_ylim(8, 37.5)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🛰️ ISRO AQI & HCHO")
st.sidebar.caption("Surface AQI Estimation & HCHO Hotspot Identification over India")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "AQI Maps", "Satellite Features", "HCHO Hotspots",
     "Time Series", "Model Performance", "About"],
)

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data …"):
    grid_df        = load_grid_features()
    cpcb_df        = load_cpcb_daily()
    hcho_df        = load_hcho_fire()
    hotspot_df     = load_hcho_hotspots()
    baseline_metrics = load_baseline_metrics()

is_demo = grid_df is None
if is_demo:
    demo    = generate_demo_data()
    grid_df = demo["grid"]
    cpcb_df = demo["cpcb"]
    hcho_df = demo["hcho"]

if is_demo:
    st.info(
        "**Demo mode** — synthetic data is displayed. "
        "Run `python scripts/run_pipeline.py build_datasets --synthetic` "
        "then restart the dashboard to load real data.",
        icon="ℹ️",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("ISRO Hackathon: Surface AQI & HCHO Hotspot Detection")
    st.markdown("""
    **Problem statement:** *Development of Surface AQI & Identification of HCHO Hotspots
    over India using Satellite Data*

    This dashboard visualises the outputs of a two-objective ML + GIS pipeline:

    | Objective | Description |
    |---|---|
    | **AQI Estimation** | Predict surface PM2.5 and compute Indian AQI from Sentinel-5P TROPOMI, INSAT-3D AOD, and ERA5 reanalysis using a CNN-LSTM model |
    | **HCHO Hotspots** | Identify biomass burning hotspots from TROPOMI HCHO + FIRMS fire counts with seasonal/wind analysis |
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_days = grid_df["date"].nunique() if grid_df is not None else 0
        st.metric("Days of data", f"{n_days:,}")
    with col2:
        n_cells = grid_df["cell_id"].nunique() if grid_df is not None else 0
        st.metric("0.1° grid cells", f"{n_cells:,}")
    with col3:
        n_stations = cpcb_df["station_id"].nunique() if cpcb_df is not None and "station_id" in cpcb_df.columns else 0
        st.metric("CPCB stations", f"{n_stations}")
    with col4:
        if hcho_df is not None and "fire_count" in hcho_df.columns:
            n_fires = int(hcho_df["fire_count"].sum())
            st.metric("Total fire detections", f"{n_fires:,}")

    st.divider()

    _help_box("About the AQI Scale (CPCB, India)", """
    The Central Pollution Control Board (CPCB) Indian AQI scale is based on eight
    key pollutants: PM2.5, PM10, NO₂, SO₂, O₃, CO, NH₃, Pb. Each pollutant's
    sub-index is computed from breakpoint tables and the final AQI is the maximum
    sub-index value.

    | AQI Range | Category | Health Message |
    |---|---|---|
    | 0 – 50 | Good | Minimal impact |
    | 51 – 100 | Satisfactory | Minor breathing discomfort for sensitive people |
    | 101 – 200 | Moderate | Breathing discomfort for people with lung/heart disease |
    | 201 – 300 | Poor | Breathing discomfort for most on prolonged exposure |
    | 301 – 400 | Very Poor | Respiratory illness on prolonged exposure |
    | 401 – 500 | Severe | Affects healthy people; serious risk for sensitive |
    """)

    st.subheader("Data Pipeline Architecture")
    st.markdown("""
    ```
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           DATA SOURCES                                   │
    │  Sentinel-5P/TROPOMI   INSAT-3D AOD   ERA5 Reanalysis   CPCB Ground    │
    │  NO2, SO2, CO, O3, HCHO    AOD        T, RH, Wind, BLH   PM2.5, AQI   │
    └────────────┬──────────────┬───────────────┬────────────────┬─────────────┘
                 └──────────────┴───────────────┴────────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  build_dataset_aqi  │  0.1° grid · daily join
                              └─────────┬──────────┘
                                        │ Feature Engineering V3
                              ┌─────────▼─────────┐
                              │ make_features_aqi  │  rolling means · spatial ctx
                              └─────────┬──────────┘
                         ┌──────────────┼───────────────┐
                         ▼              ▼               ▼
                  ┌─────────────┐  ┌──────────┐  ┌──────────────┐
                  │  Random     │  │ Gradient │  │  CNN-LSTM /  │
                  │  Forest     │  │ Boosting │  │  ConvLSTM    │
                  └──────┬──────┘  └────┬─────┘  └──────┬───────┘
                         └──────────────┴────────────────┘
                                        │
                              ┌─────────▼──────────┐
                              │  Indian AQI Grids   │
                              │  PM2.5 Maps         │
                              └────────────────────┘

    ┌─────────────────────────────────────────────┐
    │  HCHO PIPELINE                               │
    │  TROPOMI HCHO + FIRMS fires + ERA5 wind      │
    │  → HCHO anomaly → Seasonal stats             │
    │  → Percentile hotspots → DBSCAN clustering   │
    │  → Persistence → GeoJSON export              │
    └─────────────────────────────────────────────┘
    ```
    """)

    st.subheader("AQI Category Reference")
    cols = st.columns(6)
    for idx, (cat, color) in enumerate(AQI_COLORS.items()):
        with cols[idx]:
            st.markdown(
                f'<div style="background:{color};padding:10px;border-radius:6px;'
                f'text-align:center;color:{"black" if cat in ("Good","Satisfactory") else "white"};'
                f'font-weight:bold;">{cat}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AQI MAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "AQI Maps":
    st.title("AQI Maps over India")

    _help_box("How to read this map", """
    - **Predicted AQI (gridded):** Satellite-derived AQI on the 0.1° India grid.
      PM2.5 is estimated from TROPOMI satellite columns + ERA5 meteorology, then
      converted to CPCB AQI using breakpoint tables.
    - **Observed AQI (stations):** CPCB CAAQM ground-station measurements.
      Dots are coloured by AQI category; larger dots = higher AQI.
    - **INSAT AOD:** INSAT-3D Aerosol Optical Depth (a proxy for PM2.5 / haze
      intensity). Higher AOD = more aerosol loading.
    - The map covers India (68–97.5°E, 8–37.5°N) at 0.1° (≈11 km) resolution.
    """)

    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        layer = st.selectbox(
            "Layer",
            ["Predicted AQI (gridded)", "Observed AQI (stations)", "INSAT AOD (gridded)"],
        )
        if grid_df is not None:
            min_date = grid_df["date"].min().date()
            max_date = grid_df["date"].max().date()
        else:
            min_date = pd.Timestamp("2021-01-01").date()
            max_date = pd.Timestamp("2022-12-31").date()

        selected_date = st.date_input(
            "Select date", min_value=min_date, max_value=max_date,
            value=min_date + (max_date - min_date) // 2,
        )

    day_df_download = None

    with col_ctrl2:
        if layer == "Predicted AQI (gridded)" and grid_df is not None:
            from src.utils.aqi_calculator import compute_aqi_series
            day_df = grid_df[grid_df["date"] == pd.Timestamp(selected_date)].copy()
            if day_df.empty:
                st.warning("No data for this date.")
            else:
                if "aqi" not in day_df.columns:
                    day_df["aqi"] = compute_aqi_series(
                        day_df.rename(columns={
                            "no2_column": "no2", "so2_column": "so2",
                            "co_column": "co", "o3_column": "o3",
                        })
                    )
                fig = _aqi_scatter_fig(day_df, "aqi", f"Predicted AQI — {selected_date}", s=10)
                st.pyplot(fig)
                _fig_download_btn(fig, f"aqi_map_{selected_date}.png")
                plt.close(fig)

                mean_aqi     = day_df["aqi"].mean()
                dominant_pct = (day_df["aqi"] > 200).mean() * 100
                c1, c2, c3   = st.columns(3)
                c1.metric("Mean AQI",                   f"{mean_aqi:.0f} / 500")
                c2.metric("Cells with Poor or worse",    f"{dominant_pct:.1f}%")
                c3.metric("Max AQI cell",                f"{day_df['aqi'].max():.0f}")
                day_df_download = day_df

        elif layer == "Observed AQI (stations)" and cpcb_df is not None:
            day_df = cpcb_df[cpcb_df["date"] == pd.Timestamp(selected_date)].copy()
            if day_df.empty:
                st.warning("No CPCB data for this date.")
            else:
                aqi_col = "aqi_observed" if "aqi_observed" in day_df.columns else "pm25"
                fig = _aqi_scatter_fig(day_df, aqi_col, f"CPCB Observed AQI — {selected_date}", s=50)
                st.pyplot(fig)
                _fig_download_btn(fig, f"cpcb_aqi_{selected_date}.png")
                plt.close(fig)
                day_df_download = day_df

        elif layer == "INSAT AOD (gridded)" and grid_df is not None:
            day_df = grid_df[grid_df["date"] == pd.Timestamp(selected_date)].copy()
            if day_df.empty:
                st.warning("No data for this date.")
            else:
                fig = _aqi_scatter_fig(day_df, "insat_aod",
                                       f"INSAT-3D AOD — {selected_date}", cmap="YlOrRd", s=10)
                st.pyplot(fig)
                _fig_download_btn(fig, f"insat_aod_{selected_date}.png")
                plt.close(fig)
                day_df_download = day_df

    if day_df_download is not None:
        _csv_download_btn(day_df_download, f"aqi_map_data_{selected_date}.csv",
                          "⬇️ Download map data (CSV)", key=f"dl_aqi_{selected_date}")

    st.divider()
    st.subheader("Seasonal Mean AQI")
    sel_season = st.selectbox(
        "Season", list(SEASON_LABELS.keys()), format_func=lambda x: SEASON_LABELS[x],
    )

    if grid_df is not None:
        from src.utils.aqi_calculator import compute_aqi_series
        df_s = grid_df.copy()
        df_s["season"] = df_s["date"].dt.month.map(SEASON_MAP)
        df_s["aqi"]    = compute_aqi_series(
            df_s.rename(columns={"no2_column": "no2", "so2_column": "so2",
                                  "co_column": "co", "o3_column": "o3"})
        )
        seasonal = (
            df_s[df_s["season"] == sel_season]
            .groupby(["lat", "lon"])["aqi"]
            .mean()
            .reset_index()
        )
        if not seasonal.empty:
            fig2 = _aqi_scatter_fig(seasonal, "aqi",
                                    f"Seasonal Mean AQI — {SEASON_LABELS[sel_season]}", s=10)
            st.pyplot(fig2)
            plt.close(fig2)
            _csv_download_btn(seasonal, f"seasonal_aqi_{sel_season}.csv",
                              "⬇️ Download seasonal map data", key=f"dl_seas_{sel_season}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SATELLITE FEATURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Satellite Features":
    st.title("Satellite & Meteorological Features")

    _help_box("Feature descriptions and units", "\n".join([
        f"- **{k}** — {v[0]}  `[{v[1]}]`"
        for k, v in FEATURE_META.items()
    ]))

    if grid_df is None:
        st.error("No grid data found.")
        st.stop()

    feature_options = [c for c in list(FEATURE_META.keys()) if c in grid_df.columns]

    col1, col2 = st.columns([1, 3])
    with col1:
        sel_feature  = st.selectbox("Feature", feature_options,
                                    format_func=lambda c: f"{c}  [{FEATURE_META.get(c,('',''))[1]}]")
        min_date = grid_df["date"].min().date()
        max_date = grid_df["date"].max().date()
        sel_date = st.date_input(
            "Date", min_value=min_date, max_value=max_date,
            value=min_date + (max_date - min_date) // 2, key="feat_date",
        )
        cmap_choice = st.selectbox("Colour map",
                                   ["viridis", "plasma", "YlOrRd", "RdYlGn", "Blues"])

    day = grid_df[grid_df["date"] == pd.Timestamp(sel_date)].copy()
    with col2:
        if day.empty:
            st.warning("No data for this date.")
        else:
            feat_name, unit = FEATURE_META.get(sel_feature, (sel_feature, ""))
            fig = _aqi_scatter_fig(
                day, sel_feature,
                f"{feat_name} — {sel_date}   [{unit}]",
                cmap=cmap_choice, s=10,
            )
            st.pyplot(fig)
            _fig_download_btn(fig, f"{sel_feature}_{sel_date}.png")
            plt.close(fig)

    st.divider()
    st.subheader("Feature Statistics")
    if not day.empty:
        stats = day[feature_options].describe().T[["mean", "std", "min", "max"]]
        stats.columns = ["Mean", "Std", "Min", "Max"]
        stats = stats.round(4)
        # Add units column
        stats["Unit"] = [FEATURE_META.get(c, ("", ""))[1] for c in stats.index]
        st.dataframe(stats, use_container_width=True)
        _csv_download_btn(stats.reset_index(), f"feature_stats_{sel_date}.csv",
                          "⬇️ Download statistics", key="dl_feat_stats")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HCHO HOTSPOTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "HCHO Hotspots":
    st.title("HCHO Hotspots & Biomass Burning")

    _help_box("How HCHO hotspots are detected", """
    **Formaldehyde (HCHO)** from TROPOMI is a proxy for volatile organic compound (VOC)
    emissions from biomass burning and vegetation stress.

    **Detection pipeline:**
    1. Compute seasonal mean HCHO per 0.1° cell over the full data record.
    2. Flag cells whose seasonal mean exceeds the **90th percentile** within the season
       as "hotspots."
    3. DBSCAN clustering (ε = 1.5°, min_samples = 4) groups spatially contiguous
       hotspot cells into labelled clusters.
    4. Cross-correlation with FIRMS fire counts identifies **2–3 day lag** between
       fire detection and peak HCHO enhancement.

    **V3 new:** HCHO anomaly (daily − seasonal mean) and persistence metrics
    (fraction of days each cell is a hotspot) are exported alongside the map.
    """)

    if hcho_df is None:
        st.error("HCHO data not found. Run:\n"
                 "`python scripts/run_pipeline.py build_datasets --synthetic`")
        st.stop()

    col_l, col_r = st.columns([1, 3])
    with col_l:
        layer_h      = st.selectbox("Layer", ["HCHO Column", "Fire Count", "Wind Transport"])
        sel_season_h = st.selectbox(
            "Season", list(SEASON_LABELS.keys()),
            format_func=lambda x: SEASON_LABELS[x], key="hcho_season",
        )

    if "season" not in hcho_df.columns:
        hcho_df = hcho_df.copy()
        hcho_df["season"] = hcho_df["date"].dt.month.map(SEASON_MAP)

    seasonal_hcho = (
        hcho_df[hcho_df["season"] == sel_season_h]
        .groupby(["lat", "lon"])
        .agg(
            hcho_column=("hcho_column", "mean"),
            fire_count=("fire_count", "mean"),
            u10=("u10", "mean") if "u10" in hcho_df.columns else ("hcho_column", "mean"),
            v10=("v10", "mean") if "v10" in hcho_df.columns else ("hcho_column", "mean"),
        )
        .reset_index()
    )

    with col_r:
        if layer_h == "HCHO Column":
            threshold = seasonal_hcho["hcho_column"].quantile(0.90)
            seasonal_hcho["is_hotspot"] = seasonal_hcho["hcho_column"] >= threshold

            fig, ax = plt.subplots(figsize=(10, 7))
            sc = ax.scatter(
                seasonal_hcho["lon"], seasonal_hcho["lat"],
                c=seasonal_hcho["hcho_column"], cmap="YlOrRd",
                vmin=seasonal_hcho["hcho_column"].quantile(0.05),
                vmax=seasonal_hcho["hcho_column"].quantile(0.98),
                s=10, marker="s", linewidths=0,
            )
            plt.colorbar(sc, ax=ax, label="HCHO column (µmol m⁻²)", fraction=0.03, pad=0.04)

            hs = seasonal_hcho[seasonal_hcho["is_hotspot"]]
            ax.scatter(hs["lon"], hs["lat"], s=25, facecolors="none",
                       edgecolors="blue", linewidths=0.8,
                       label=f"Hotspot (≥90th pct, n={len(hs)})")
            ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(68, 97.5); ax.set_ylim(8, 37.5)
            ax.set_title(f"HCHO Hotspots — {SEASON_LABELS[sel_season_h]}")
            ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
            plt.tight_layout()
            st.pyplot(fig)
            _fig_download_btn(fig, f"hcho_hotspot_{sel_season_h}.png")
            plt.close(fig)

        elif layer_h == "Fire Count":
            fire = seasonal_hcho[seasonal_hcho["fire_count"] > 0]
            fig, ax = plt.subplots(figsize=(10, 7))
            sc = ax.scatter(
                fire["lon"], fire["lat"], c=fire["fire_count"],
                cmap="hot_r", vmin=0, vmax=fire["fire_count"].quantile(0.99),
                s=12, marker="s", linewidths=0,
            )
            plt.colorbar(sc, ax=ax, label="Mean daily fire count", fraction=0.03, pad=0.04)
            ax.set_xlim(68, 97.5); ax.set_ylim(8, 37.5)
            ax.set_title(f"Fire Density — {SEASON_LABELS[sel_season_h]}")
            ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
            plt.tight_layout()
            st.pyplot(fig)
            _fig_download_btn(fig, f"fire_density_{sel_season_h}.png")
            plt.close(fig)

        else:  # Wind Transport
            fig, ax = plt.subplots(figsize=(10, 7))
            sc = ax.scatter(
                seasonal_hcho["lon"], seasonal_hcho["lat"],
                c=seasonal_hcho["hcho_column"], cmap="YlOrRd", s=8, marker="s", linewidths=0,
            )
            plt.colorbar(sc, ax=ax, label="HCHO (µmol m⁻²)", fraction=0.03, pad=0.04)

            sub = seasonal_hcho.iloc[::6]
            if "u10" in sub.columns and "v10" in sub.columns:
                ax.quiver(sub["lon"], sub["lat"], sub["u10"], sub["v10"],
                          scale=150, width=0.0025, color="navy", alpha=0.7, label="Wind 10m")
                ax.legend(loc="lower right", fontsize=8)
            ax.set_xlim(68, 97.5); ax.set_ylim(8, 37.5)
            ax.set_title(f"HCHO + Wind Transport — {SEASON_LABELS[sel_season_h]}")
            ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
            plt.tight_layout()
            st.pyplot(fig)
            _fig_download_btn(fig, f"hcho_wind_{sel_season_h}.png")
            plt.close(fig)

    # Download hotspot table
    _csv_download_btn(
        seasonal_hcho, f"hcho_seasonal_{sel_season_h}.csv",
        "⬇️ Download seasonal HCHO data", key=f"dl_hcho_{sel_season_h}",
    )

    # Top hotspot regions
    if hotspot_df is not None and not hotspot_df.empty:
        st.divider()
        st.subheader("Top Hotspot Regions")
        hs_season = hotspot_df[hotspot_df.get("season", pd.Series()).eq(sel_season_h)] \
            if "season" in hotspot_df.columns else hotspot_df
        if not hs_season.empty and "is_hotspot" in hs_season.columns:
            top = hs_season[hs_season["is_hotspot"]].nlargest(10, "mean_hcho")
            cols_show = [c for c in ["lat","lon","mean_hcho","hcho_percentile",
                                     "cluster","hotspot_fraction"] if c in top.columns]
            st.dataframe(top[cols_show].reset_index(drop=True), use_container_width=True)
            _csv_download_btn(top, f"top_hotspots_{sel_season_h}.csv",
                              "⬇️ Download top hotspot table", key=f"dl_top_{sel_season_h}")

    st.divider()
    st.subheader("HCHO vs Fire Count Correlation")
    sel_region = st.selectbox("Region", list(REGIONS.keys()))
    bbox = REGIONS[sel_region]
    mask = (
        (hcho_df["lat"] >= bbox["lat"][0]) & (hcho_df["lat"] <= bbox["lat"][1]) &
        (hcho_df["lon"] >= bbox["lon"][0]) & (hcho_df["lon"] <= bbox["lon"][1])
    )
    region_df = hcho_df[mask].copy()
    if len(region_df) > 0:
        daily_mean = region_df.groupby("date")[["hcho_column", "fire_count"]].mean().reset_index()

        fig2, ax1 = plt.subplots(figsize=(13, 4))
        ax1.plot(daily_mean["date"], daily_mean["hcho_column"],
                 color="steelblue", linewidth=1.5, label="HCHO (µmol m⁻²)")
        ax1.set_ylabel("HCHO (µmol m⁻²)", color="steelblue")
        ax2 = ax1.twinx()
        ax2.fill_between(daily_mean["date"], daily_mean["fire_count"],
                         alpha=0.35, color="tomato", label="Fire count")
        ax2.set_ylabel("Fire count", color="tomato")
        ax1.set_title(f"{sel_region} — HCHO vs Fire Count")

        from scipy.stats import pearsonr
        ann = []
        for lag in [0, 1, 2, 3]:
            sh    = daily_mean["fire_count"].shift(lag)
            valid = ~(sh.isna() | daily_mean["hcho_column"].isna())
            if valid.sum() >= 10:
                r, _ = pearsonr(daily_mean["hcho_column"][valid], sh[valid])
                ann.append(f"r(lag={lag}d) = {r:.3f}")
        if ann:
            ax1.annotate("\n".join(ann), xy=(0.02, 0.96), xycoords="axes fraction",
                         fontsize=8, va="top", ha="left",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        _csv_download_btn(daily_mean, f"hcho_fire_correlation_{sel_region.replace(' ','_')}.csv",
                          "⬇️ Download correlation data",
                          key=f"dl_corr_{sel_region}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Time Series":
    st.title("City / Region Time Series")

    _help_box("Reading the time series charts", """
    - **Gray line:** raw daily values.
    - **Coloured line:** 14-day centred rolling mean — smooths out short-term noise.
    - **CPCB stations:** data from official ground-monitoring stations.
    - **Satellite grid:** area-averaged satellite/reanalysis variables.
    - Use the region selector to focus on a specific part of India.
    - Download the full time series CSV to analyse trends in your own tools.
    """)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        source = st.selectbox("Data source", ["CPCB stations", "Satellite grid"])
    with col2:
        variable_opts = {
            "CPCB stations": [c for c in ["pm25", "pm10", "no2", "so2", "o3", "co", "aqi_observed"]
                              if cpcb_df is not None and c in cpcb_df.columns],
            "Satellite grid": [c for c in ["no2_column", "so2_column", "hcho_column",
                                            "insat_aod", "t2m", "rh2m"]
                               if grid_df is not None and c in grid_df.columns],
        }
        sel_var = st.selectbox("Variable", variable_opts.get(source, ["pm25"]))

    ts_df_download = None

    if source == "CPCB stations" and cpcb_df is not None:
        cities = sorted(cpcb_df["city"].dropna().unique()) if "city" in cpcb_df.columns else []
        with col3:
            sel_city = st.selectbox("City", cities or ["No cities found"])

        if cities:
            city_df = (
                cpcb_df[cpcb_df["city"] == sel_city]
                .groupby("date")[[sel_var]]
                .mean()
                .reset_index()
            )
            if not city_df.empty:
                fig, ax = plt.subplots(figsize=(13, 4))
                ax.plot(city_df["date"], city_df[sel_var],
                        color="gray", linewidth=0.7, alpha=0.6, label="Daily")
                rolled = city_df[sel_var].rolling(14, center=True, min_periods=1).mean()
                ax.plot(city_df["date"], rolled,
                        color="steelblue", linewidth=2.0, label="14-day mean")
                unit = FEATURE_META.get(sel_var, ("", ""))[1]
                ax.set_title(f"{sel_city} — {sel_var}")
                ax.set_ylabel(f"{sel_var}  [{unit}]" if unit else sel_var)
                ax.legend()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                c_a, c_b, c_c = st.columns(3)
                unit_label = f"  [{unit}]" if unit else ""
                c_a.metric(f"Mean{unit_label}", f"{city_df[sel_var].mean():.1f}")
                c_b.metric(f"Max{unit_label}",  f"{city_df[sel_var].max():.1f}")
                c_c.metric(f"Min (>0){unit_label}",
                           f"{city_df[sel_var][city_df[sel_var] > 0].min():.1f}")
                ts_df_download = city_df

    else:  # Satellite grid
        with col3:
            sel_region_ts = st.selectbox("Region", list(REGIONS.keys()), key="ts_region")

        bbox = REGIONS[sel_region_ts]
        mask = (
            (grid_df["lat"] >= bbox["lat"][0]) & (grid_df["lat"] <= bbox["lat"][1]) &
            (grid_df["lon"] >= bbox["lon"][0]) & (grid_df["lon"] <= bbox["lon"][1])
        )
        region_ts = grid_df[mask].groupby("date")[[sel_var]].mean().reset_index()

        if not region_ts.empty:
            fig, ax = plt.subplots(figsize=(13, 4))
            ax.plot(region_ts["date"], region_ts[sel_var],
                    color="gray", linewidth=0.7, alpha=0.6, label="Daily mean")
            rolled = region_ts[sel_var].rolling(14, center=True, min_periods=1).mean()
            ax.plot(region_ts["date"], rolled,
                    color="tomato", linewidth=2.0, label="14-day mean")
            unit = FEATURE_META.get(sel_var, ("", ""))[1]
            ax.set_title(f"{sel_region_ts} — {sel_var}")
            ax.set_ylabel(f"{sel_var}  [{unit}]" if unit else sel_var)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            ts_df_download = region_ts

    if ts_df_download is not None:
        _csv_download_btn(ts_df_download, f"timeseries_{sel_var}.csv",
                          "⬇️ Download time series (CSV)", key=f"dl_ts_{sel_var}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Evaluation Metrics")

    _help_box("Understanding the metrics", """
    | Metric | Meaning | Better when… |
    |---|---|---|
    | **RMSE** | Root Mean Square Error (in µg m⁻³ for PM2.5 or AQI units) | Lower |
    | **MAE** | Mean Absolute Error — average magnitude of errors | Lower |
    | **R²** | Coefficient of determination — fraction of variance explained (0–1) | Higher |
    | **Pearson r** | Linear correlation between predicted and observed | Higher (max 1) |

    **Baseline models:** Random Forest and Gradient Boosting trained on tabular
    satellite + meteorological features.

    **CNN-LSTM / ConvLSTM:** Spatio-temporal deep learning model processing a
    7-day sequence of feature grids (T × C × H × W tensors).

    Train period: 2019–2021 · Test period: 2022 (temporal hold-out).
    """)

    if baseline_metrics is None:
        st.info("No trained model metrics found. Train the models first:")
        st.code("python scripts/run_pipeline.py train_baseline")
        st.markdown("### Demo metrics (illustrative)")
        demo_metrics = {
            "RandomForest": {
                "pm25_target": {"rmse": 18.4, "mae": 12.1, "r2": 0.74, "pearson_r": 0.87},
                "aqi_target":  {"rmse": 22.7, "mae": 14.8, "r2": 0.71, "pearson_r": 0.85},
            },
            "GradientBoosting": {
                "pm25_target": {"rmse": 16.2, "mae": 10.9, "r2": 0.79, "pearson_r": 0.89},
                "aqi_target":  {"rmse": 20.1, "mae": 13.2, "r2": 0.76, "pearson_r": 0.87},
            },
            "CNN-LSTM": {
                "pm25_target": {"rmse": 14.1, "mae": 9.3, "r2": 0.83, "pearson_r": 0.91},
                "aqi_target":  {"rmse": 17.5, "mae": 11.4, "r2": 0.81, "pearson_r": 0.90},
            },
            "ConvLSTM": {
                "pm25_target": {"rmse": 13.2, "mae": 8.7, "r2": 0.85, "pearson_r": 0.92},
                "aqi_target":  {"rmse": 16.1, "mae": 10.5, "r2": 0.83, "pearson_r": 0.91},
            },
        }
        baseline_metrics = demo_metrics

    all_rows = []
    for model_name, targets in baseline_metrics.items():
        st.subheader(f"Model: {model_name}")
        rows = []
        for target, m in targets.items():
            row = {
                "Target": target.replace("_target", "").upper(),
                "RMSE": round(m.get("rmse", 0), 2),
                "MAE":  round(m.get("mae", 0), 2),
                "R²":   round(m.get("r2", 0), 3),
                "Pearson r": round(m.get("pearson_r", 0), 3),
            }
            rows.append(row)
            all_rows.append({"Model": model_name, **row})
        st.dataframe(pd.DataFrame(rows).set_index("Target"), use_container_width=True)

    st.divider()
    st.subheader("Model Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    model_names  = list(baseline_metrics.keys())
    targets_plot = ["pm25_target", "aqi_target"]
    metrics_plot = ["rmse", "r2"]
    titles       = ["RMSE — lower is better (µg m⁻³ or AQI units)",
                    "R² — higher is better (0–1)"]

    for idx, (metric, title) in enumerate(zip(metrics_plot, titles)):
        ax = axes[idx]
        for tgt_idx, tgt in enumerate(targets_plot):
            vals = [baseline_metrics[m].get(tgt, {}).get(metric, 0) for m in model_names]
            x    = np.arange(len(model_names)) + tgt_idx * 0.35
            ax.bar(x, vals, 0.35, label=tgt.replace("_target", "").upper(),
                   color=["steelblue", "tomato"][tgt_idx], alpha=0.8)
        ax.set_xticks(np.arange(len(model_names)) + 0.175)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    leaderboard = pd.DataFrame(all_rows)
    _csv_download_btn(leaderboard, "model_leaderboard.csv",
                      "⬇️ Download leaderboard (CSV)", key="dl_leaderboard")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ## ISRO Hackathon: Surface AQI & HCHO Hotspot Detection

    ### Objectives
    - **Objective 1:** Estimate surface AQI over India by fusing TROPOMI satellite
      columns, INSAT-3D AOD, and ERA5 reanalysis with CPCB ground truth using
      ML / deep-learning models.
    - **Objective 2:** Identify HCHO hotspots from biomass burning using TROPOMI
      HCHO + FIRMS fire detections, with wind transport and persistence analysis.

    ---

    ### Data Sources
    | Source | Variables | Portal |
    |---|---|---|
    | Sentinel-5P TROPOMI | NO₂, SO₂, CO, O₃, HCHO | GEE or DLR |
    | INSAT-3D | AOD | MOSDAC |
    | ERA5 Reanalysis | T, RH, Wind, BLH, Precip | Copernicus CDS |
    | CPCB CAAQM | PM2.5, PM10, NO₂, SO₂, O₃, CO | CPCB portal |
    | NASA FIRMS | Fire detections | firms.modaps.eosdis.nasa.gov |

    ---

    ### Models
    | Model | Architecture | Input |
    |---|---|---|
    | Random Forest | 500 trees, Gini impurity | Tabular features |
    | Gradient Boosting | 200 trees, learning_rate=0.05 | Tabular features |
    | CNN-LSTM | SpatialEncoder → LSTM → FC | T×C×H×W grids |
    | ConvLSTM | Multi-layer ConvLSTM → Conv1×1 | T×C×H×W grids |

    ---

    ### V3 Quick-Start (5 commands)
    ```bash
    cd isro-aqi-hcho

    # 1. Generate synthetic training data
    python scripts/run_pipeline.py build_datasets --synthetic

    # 2. Train baseline models (RF + GBM)
    python scripts/run_pipeline.py train_baseline

    # 3. Train CNN-LSTM deep model
    python scripts/run_pipeline.py train_deep --synthetic

    # 4. Prepare dashboard exports
    python scripts/run_pipeline.py export_for_dashboard

    # 5. Launch dashboard  (already running if you see this page!)
    streamlit run src/webapp/app.py
    ```

    Or run everything in one command:
    ```bash
    python scripts/run_pipeline.py run_all --synthetic
    ```

    ---

    ### Key References
    - CPCB AQI Technical Document (2014)
    - Sentinel-5P TROPOMI Algorithm Theoretical Basis (ESA)
    - Shi et al. (2015) — Convolutional LSTM Network
    - ERA5 reanalysis: Hersbach et al. (2020)
    - FIRMS: Giglio et al. (2016)
    """)

    st.divider()
    st.subheader("Version History")
    st.markdown("""
    | Version | Changes |
    |---|---|
    | **V1** | Initial dashboard + synthetic demo data |
    | **V2** | Improved baseline ML (GridSearchCV), CNN-LSTM, Jupyter notebooks |
    | **V3** | Rolling & spatial features · ConvLSTM · HCHO anomaly · GeoJSON export · download buttons · unit labels · `run_pipeline.py` CLI |
    """)
