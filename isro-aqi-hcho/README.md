# ISRO Hackathon: Surface AQI & HCHO Hotspot Detection over India

**Problem statement:** Development of Surface AQI & Identification of HCHO Hotspots over India using Satellite Data

## Overview

This repository implements a two-objective ML/GIS pipeline:

| Objective | Description |
|---|---|
| **AQI Estimation** | Fuse TROPOMI satellite columns, INSAT-3D AOD, and ERA5 reanalysis with CPCB ground data to predict surface PM2.5 and compute Indian AQI using Random Forest, Gradient Boosting, and CNN-LSTM models |
| **HCHO Hotspots** | Identify biomass burning hotspots from TROPOMI HCHO + FIRMS fire counts; analyse seasonal patterns and wind transport over India |

---

## Repository Structure

```
isro-aqi-hcho/
├── README.md
├── requirements.txt
├── env_example.yml
├── config/
│   ├── paths.yaml           # Data directory paths and grid settings
│   ├── aqi_training.yaml    # Model architecture and training config
│   └── hcho_hotspot.yaml    # Hotspot detection parameters
├── data/
│   ├── README.md            # Data download instructions
│   ├── raw/                 # Downloaded source data (not committed)
│   ├── interim/             # Intermediate processed files
│   └── processed/           # Final datasets used by models/dashboard
├── notebooks/
│   ├── 01_explore_cpcb.ipynb
│   ├── 02_explore_satellite_reanalysis.ipynb
│   └── 03_model_results_and_plots.ipynb
├── src/
│   ├── data/
│   │   ├── download_cpcb.py
│   │   ├── download_tropomi.py
│   │   ├── download_insat_aod.py
│   │   ├── download_reanalysis.py
│   │   ├── download_firms_fire.py
│   │   ├── grid_definition.py
│   │   ├── build_dataset_aqi.py
│   │   └── build_dataset_hcho.py
│   ├── features/
│   │   ├── make_features_aqi.py
│   │   └── make_features_hcho.py
│   ├── models/
│   │   ├── baseline_ml.py
│   │   ├── cnn_lstm_aqi.py
│   │   ├── train_aqi.py
│   │   └── evaluate_aqi.py
│   ├── utils/
│   │   └── aqi_calculator.py
│   ├── visualization/
│   │   ├── plot_maps.py
│   │   ├── plot_time_series.py
│   │   └── plot_hotspots.py
│   └── webapp/
│       └── app.py            # Streamlit dashboard
└── scripts/
    ├── run_build_aqi_dataset.sh
    ├── run_train_aqi.sh
    └── run_hcho_hotspots.sh
```

---

## Step-by-Step Guide for Team Members

### 1. Create the Environment

**Option A – pip (recommended on Replit or plain Python):**
```bash
pip install -r requirements.txt
```

**Option B – conda:**
```bash
conda env create -f env_example.yml
conda activate isro-aqi-hcho
```

### 2. Configure Paths

Edit `config/paths.yaml` if you want to store data outside the default `data/` directory.

### 3. Set Up API Credentials

#### ERA5 (Copernicus CDS)
1. Register at https://cds.climate.copernicus.eu
2. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <UID>:<API-KEY>
   ```

#### Google Earth Engine (for TROPOMI via GEE)
1. Register at https://earthengine.google.com
2. Run `earthengine authenticate`

#### NASA FIRMS (fire data)
1. Get a MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/
2. Export as env variable: `export FIRMS_MAP_KEY=<your-key>`

### 4. Download Data

```bash
# Ground truth (download CSV from portal; see data/README.md for instructions)
# Place exported CSVs in data/raw/cpcb/
python -m src.data.download_cpcb \
    --input_dir data/raw/cpcb \
    --output data/processed/cpcb_daily.csv \
    --start 2019-01-01 --end 2022-12-31

# TROPOMI satellite (requires GEE auth)
python -m src.data.download_tropomi \
    --method gee \
    --pollutants NO2 SO2 CO O3 HCHO \
    --start 2019-01-01 --end 2022-12-31 \
    --output_dir data/raw/tropomi

# INSAT-3D AOD (manual download from MOSDAC; place files in data/raw/insat_aod/)
python -m src.data.download_insat_aod \
    --input_dir data/raw/insat_aod \
    --output_csv data/processed/insat_aod_daily.csv

# ERA5 Reanalysis (requires CDS API key)
python -m src.data.download_reanalysis \
    --start 2019-01-01 --end 2022-12-31 \
    --output_dir data/raw/reanalysis \
    --output_csv data/processed/era5_daily.csv

# FIRMS Fire Counts
python -m src.data.download_firms_fire \
    --map_key $FIRMS_MAP_KEY \
    --start 2019-01-01 --end 2022-12-31 \
    --output_dir data/raw/firms \
    --output_csv data/processed/firms_fire_daily.csv
```

**Don't have real data yet? Use synthetic demo data:**
```bash
python -m src.data.build_dataset_aqi --synthetic
python -m src.data.build_dataset_hcho --synthetic
```

### 5. Build Datasets

```bash
python -m src.data.build_dataset_aqi
python -m src.data.build_dataset_hcho
```

### 6. Engineer Features

```bash
python -m src.features.make_features_aqi \
    --input  data/processed/aqi_training_dataset.csv \
    --output data/processed/aqi_features.csv \
    --scaler models/baseline/scaler.pkl

python -m src.features.make_features_hcho \
    --input  data/processed/hcho_fire_daily_grid.csv \
    --output data/processed/hcho_hotspot_features.csv \
    --config config/hcho_hotspot.yaml
```

### 7. Train AQI Models

```bash
# Baseline (Random Forest + Gradient Boosting)
python -m src.models.baseline_ml \
    --input      data/processed/aqi_training_dataset.csv \
    --output_dir models/baseline

# CNN-LSTM (spatio-temporal deep learning)
python -m src.models.train_aqi \
    --config   config/aqi_training.yaml \
    --grid_csv data/processed/grid_daily_features.csv

# Quick smoke test (synthetic, 5 epochs, small grid)
python -m src.models.train_aqi --config config/aqi_training.yaml --synthetic
```

### 8. Evaluate Models

```bash
# Baseline
python -m src.models.evaluate_aqi \
    --model_type baseline \
    --model_dir  models/baseline \
    --test_csv   data/processed/aqi_training_dataset.csv

# CNN-LSTM
python -m src.models.evaluate_aqi \
    --model_type cnn_lstm \
    --model_dir  models/cnn_lstm \
    --grid_csv   data/processed/grid_daily_features.csv \
    --config     config/aqi_training.yaml
```

### 9. Generate Maps and Plots

```bash
# AQI raster maps
python -m src.visualization.plot_maps \
    --csv      data/processed/grid_daily_features.csv \
    --variable no2_column \
    --date     2022-06-15 \
    --output   outputs/maps/no2_2022-06-15.png

# HCHO hotspot maps
python -m src.visualization.plot_hotspots \
    --hcho_csv    data/processed/hcho_fire_daily_grid.csv \
    --hotspot_csv data/processed/hcho_hotspot_features.csv \
    --output_dir  outputs/hotspot_maps

# City time series
python -m src.visualization.plot_time_series \
    --csv      data/processed/cpcb_daily.csv \
    --city     Delhi \
    --variable pm25 \
    --output   outputs/delhi_pm25.png
```

### 10. Run the Streamlit Dashboard

```bash
streamlit run src/webapp/app.py
```

The dashboard will be available at http://localhost:5000

---

## Dataset Schemas

### `cpcb_daily.csv`
| Column | Type | Description |
|---|---|---|
| station_id | str | CPCB station identifier |
| station_name | str | Station name |
| city, state | str | Location |
| lat, lon | float | Coordinates |
| cell_id | str | Nearest 0.1° grid cell |
| date | date | |
| pm25, pm10, no2, so2, o3, co | float | Daily mean concentrations (µg/m³ or mg/m³ for CO) |
| aqi_observed | int | Indian AQI (CPCB formula) |

### `grid_daily_features.csv`
| Column | Description |
|---|---|
| cell_id, lat, lon, date | Grid cell identifier and coordinates |
| no2_column, so2_column, co_column, o3_column, hcho_column | TROPOMI column densities (µmol/m²) |
| insat_aod | INSAT-3D AOD (dimensionless) |
| t2m | 2m temperature (K) |
| rh2m | Relative humidity (%) |
| u10, v10 | 10m wind components (m/s) |
| tp | Total precipitation (m/day) |
| sp | Surface pressure (Pa) |
| blh | Boundary layer height (m) |
| fire_count | FIRMS fire detections per cell per day |

### `hcho_fire_daily_grid.csv`
| Column | Description |
|---|---|
| cell_id, lat, lon, date | Grid identifiers |
| hcho_column | TROPOMI HCHO (µmol/m²) |
| fire_count | Daily fire detections |
| u10, v10, blh, tp | Wind and met for transport analysis |
| season | winter / pre_monsoon / monsoon / post_monsoon |

---

## Indian AQI Breakpoints Reference

| AQI Range | Category | PM2.5 (µg/m³) |
|---|---|---|
| 0–50 | Good | 0–30 |
| 51–100 | Satisfactory | 30–60 |
| 101–200 | Moderate | 60–90 |
| 201–300 | Poor | 90–120 |
| 301–400 | Very Poor | 120–250 |
| 401–500 | Severe | >250 |

---

## Key References

1. CPCB (2014). *National Air Quality Index*. Central Pollution Control Board, India.
2. Veefkind et al. (2012). TROPOMI on Sentinel-5P. *Remote Sensing of Environment*, 120, 70–83.
3. Hersbach et al. (2020). ERA5 global reanalysis. *QJRMS*, 146, 1999–2049.
4. Giglio et al. (2016). Active fire detection and characterization with MODIS. *Remote Sensing of Environment*.
5. Dey et al. (2012). PM2.5 estimation from MODIS AOD over India. *Atmospheric Environment*.

---

## Acknowledgements

This project was developed for the ISRO hackathon. Data providers: CPCB, ESA/DLR, MOSDAC/ISRO, ECMWF, NASA FIRMS.
