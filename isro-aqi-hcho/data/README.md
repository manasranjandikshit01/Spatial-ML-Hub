# Data Directory

This directory holds all raw, intermediate, and processed data for the ISRO
AQI & HCHO Hotspot project.  **Nothing here is committed to version control**
(all sub-folders are listed in `.gitignore`) — you must download or generate
the data locally by following the recipe below.

---

## Folder Structure

```
data/
├── raw/                  # Unprocessed downloads — never modified
│   ├── cpcb/             # CPCB CAAQM station CSVs (one per station-year)
│   ├── tropomi/          # Sentinel-5P TROPOMI NetCDF files
│   ├── insat_aod/        # INSAT-3D AOD HDF5 / NetCDF files
│   ├── reanalysis/       # ERA5 NetCDF files (downloaded via CDS API)
│   └── firms/            # NASA FIRMS fire CSV / shapefiles
│
├── interim/              # Cleaned & grid-aligned intermediates
│   ├── cpcb_daily.csv         # Daily pollutant averages per station
│   ├── tropomi_no2_daily.csv  # Gridded TROPOMI NO2 column (mol m⁻²)
│   ├── tropomi_so2_daily.csv
│   ├── tropomi_co_daily.csv
│   ├── tropomi_o3_daily.csv
│   ├── tropomi_hcho_daily.csv
│   ├── insat_aod_daily.csv    # Gridded INSAT-3D AOD (550 nm)
│   ├── era5_daily.csv         # ERA5 met variables on 0.1° grid
│   └── firms_fire_daily.csv   # FIRMS fire pixel counts per grid cell
│
└── processed/            # Final datasets used by models & dashboard
    ├── aqi_training_dataset.csv    # Station-level AQI training data
    ├── grid_daily_features.csv     # Gridded satellite + met features
    ├── hcho_fire_daily_grid.csv    # HCHO + fire daily grid
    ├── aqi_maps/                   # Predicted AQI raster outputs
    └── cnn_lstm_predictions.csv    # Daily mean PM2.5 from CNN-LSTM
```

---

## Key File Schemas

### `interim/cpcb_daily.csv`
Produced by: `src/data/download_cpcb.py`
Used by: `build_dataset_aqi.py`, `notebooks/01_explore_cpcb.ipynb`

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Daily average date |
| `station_id` | str | Unique CPCB station code |
| `station_name` | str | Human-readable station name |
| `city` | str | City name |
| `state` | str | State name |
| `lat` | float | Latitude (°N) |
| `lon` | float | Longitude (°E) |
| `pm25` | float | PM2.5 24-h average (µg m⁻³) |
| `pm10` | float | PM10 24-h average (µg m⁻³) |
| `no2` | float | NO₂ 24-h average (µg m⁻³) |
| `so2` | float | SO₂ 24-h average (µg m⁻³) |
| `o3` | float | O₃ 8-h maximum (µg m⁻³) |
| `co` | float | CO 8-h average (mg m⁻³) |

### `processed/aqi_training_dataset.csv`
Produced by: `src/data/build_dataset_aqi.py`
Used by: `baseline_ml.py`, `notebooks/03_train_baseline_and_cnn_lstm.ipynb`

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Date |
| `cell_id` | str | Grid cell ID (`CELL_{lat:.2f}_{lon:.2f}`) |
| `lat`, `lon` | float | Cell centre coordinates (°) |
| `no2_column` | float | TROPOMI NO₂ column (mol m⁻²) |
| `so2_column` | float | TROPOMI SO₂ column (mol m⁻²) |
| `co_column` | float | TROPOMI CO column (mol m⁻²) |
| `o3_column` | float | TROPOMI O₃ column (mol m⁻²) |
| `hcho_column` | float | TROPOMI HCHO column (mol m⁻²) |
| `insat_aod` | float | INSAT-3D AOD 550 nm (dimensionless) |
| `t2m` | float | ERA5 2-m temperature (°C) |
| `rh2m` | float | ERA5 2-m relative humidity (%) |
| `u10`, `v10` | float | ERA5 10-m wind components (m s⁻¹) |
| `tp` | float | ERA5 total precipitation (mm day⁻¹) |
| `sp` | float | ERA5 surface pressure (hPa) |
| `blh` | float | ERA5 boundary layer height (m) |
| `pm25_target` | float | Observed PM2.5 interpolated from CPCB stations |
| `aqi_target` | float | Computed Indian AQI from CPCB measurements |

### `processed/grid_daily_features.csv`
Produced by: `src/data/build_dataset_aqi.py` (gridded output)
Used by: `cnn_lstm_aqi.py`, `train_aqi.py`, `notebooks/02_explore_satellite_reanalysis.ipynb`

Same columns as `aqi_training_dataset.csv`, but one row per **grid cell** per
day covering the entire 0.1° India grid (~295 000 cell-days per year).

### `processed/hcho_fire_daily_grid.csv`
Produced by: `src/data/build_dataset_hcho.py`
Used by: hotspot detection, `notebooks/04_hcho_hotspots_and_fire.ipynb`

| Column | Type | Description |
|--------|------|-------------|
| `date` | YYYY-MM-DD | Date |
| `lat`, `lon` | float | Grid cell centre (0.1°) |
| `hcho_column` | float | TROPOMI HCHO column (mol m⁻²) |
| `fire_count` | int | FIRMS fire pixels within grid cell |
| `u10`, `v10` | float | ERA5 wind components (m s⁻¹) |
| `is_hotspot` | bool | True if HCHO ≥ 90th seasonal percentile |
| `cluster_id` | int | DBSCAN cluster label (−1 = noise) |
| `season` | str | IMD season label |

---

## Prerequisites — API Credentials

| Source | Credential | Setup link |
|--------|------------|------------|
| ERA5 (CDS API) | `~/.cdsapirc` with UID & key | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/api-how-to) |
| TROPOMI (GEE) | `earthengine authenticate` | [GEE docs](https://developers.google.com/earth-engine/guides/auth) |
| INSAT-3D (MOSDAC) | `MOSDAC_USER` / `MOSDAC_PASS` env vars | [mosdac.gov.in](https://www.mosdac.gov.in) |
| FIRMS | `FIRMS_MAP_KEY` env var | [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov/api/) |

Copy `.env.example` to `.env` and fill in your keys before running any download script.

---

## Step-by-Step: Zero to Fully-Built Datasets

### Step 1 — Download CPCB data
```bash
cd isro-aqi-hcho
python -m src.data.download_cpcb \
    --start_date 2019-01-01 \
    --end_date   2022-12-31 \
    --output_dir data/raw/cpcb
```

### Step 2 — Download satellite data (TROPOMI + INSAT-3D)
```bash
python -m src.data.download_tropomi \
    --start_date 2019-01-01 \
    --end_date   2022-12-31 \
    --output_dir data/raw/tropomi

python -m src.data.download_insat_aod \
    --start_date 2019-01-01 \
    --end_date   2022-12-31 \
    --output_dir data/raw/insat_aod
```

### Step 3 — Download reanalysis and fire data
```bash
python -m src.data.download_reanalysis \
    --start_date 2019-01-01 \
    --end_date   2022-12-31 \
    --output_dir data/raw/reanalysis

python -m src.data.download_firms_fire \
    --start_date 2019-01-01 \
    --end_date   2022-12-31 \
    --output_dir data/raw/firms
```

### Step 4 — Build AQI training dataset
```bash
# Real data
python -m src.data.build_dataset_aqi

# Synthetic fallback (no downloads required — runs in seconds)
python -m src.data.build_dataset_aqi --synthetic
```
Outputs: `data/processed/aqi_training_dataset.csv` and `data/processed/grid_daily_features.csv`

### Step 5 — Build HCHO hotspot dataset
```bash
python -m src.data.build_dataset_hcho            # real data
python -m src.data.build_dataset_hcho --synthetic # synthetic fallback
```
Output: `data/processed/hcho_fire_daily_grid.csv`

### Step 6 — Verify outputs
```bash
python - <<'EOF'
import pandas as pd
for f in [
    "data/processed/aqi_training_dataset.csv",
    "data/processed/grid_daily_features.csv",
    "data/processed/hcho_fire_daily_grid.csv",
]:
    df = pd.read_csv(f)
    print(f"{f}: {df.shape}")
EOF
```

---

## Notes

- All gridded data uses a **0.1° regular lat/lon grid** over India
  (8–37.5 °N, 68–97.5 °E).  Cell IDs follow the format `CELL_{lat:.2f}_{lon:.2f}`.
- **Temporal split:** train 2019–2021, test 2022 (avoids data leakage).
- Missing values are represented as `NaN`; models impute with column median or 0.
- Synthetic demo data is generated in-memory and is suitable for notebook
  exploration and model smoke-tests — no API keys required.
