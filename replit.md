# ISRO AQI & HCHO Hotspot Detection

A complete ML/GIS hackathon repository for ISRO's problem statement:
*"Development of Surface AQI & Identification of HCHO Hotspots over India using Satellite Data"*

## Run & Operate

- **Streamlit dashboard:** `cd isro-aqi-hcho && streamlit run src/webapp/app.py` (port 5000)
- **Generate demo data:** `cd isro-aqi-hcho && python -m src.data.build_dataset_aqi --synthetic && python -m src.data.build_dataset_hcho --synthetic`
- **Train baseline models:** `bash isro-aqi-hcho/scripts/run_train_aqi.sh`
- **HCHO hotspot pipeline:** `bash isro-aqi-hcho/scripts/run_hcho_hotspots.sh --synthetic`

## Stack

- Python 3.11, Streamlit 1.25+
- ML: scikit-learn (Random Forest, Gradient Boosting), PyTorch (CNN-LSTM)
- GIS/data: geopandas, xarray, rasterio, folium
- Data sources: Sentinel-5P TROPOMI, INSAT-3D AOD, ERA5, CPCB CAAQM, NASA FIRMS

## Where things live

- `isro-aqi-hcho/` — the entire Python project
- `isro-aqi-hcho/src/webapp/app.py` — Streamlit dashboard (7 pages)
- `isro-aqi-hcho/src/data/` — data downloaders + dataset builders
- `isro-aqi-hcho/src/models/` — baseline ML + CNN-LSTM training + evaluation
- `isro-aqi-hcho/src/utils/aqi_calculator.py` — official CPCB Indian AQI formula
- `isro-aqi-hcho/config/` — YAML configs for paths, training, hotspot detection

## Architecture decisions

- **0.1° regular grid** over India (68–97.5°E, 8–37.5°N) — all datasets snapped to this grid before joining
- **CNN-LSTM** input: (B, T, C, H, W) — per-timestep SpatialEncoder (CNN) feeds into LSTM → FC head
- **Temporal train/test split:** train 2019–2021, test 2022 (avoids data leakage)
- **Synthetic fallback:** all pipelines generate realistic synthetic demo data when real downloads are absent, enabling the dashboard to run immediately
- **Seasonal AQI analysis:** seasons defined as per IMD convention (winter/pre-monsoon/monsoon/post-monsoon)

## Product

- AQI estimation map over India from satellite data (no ground sensors needed at inference)
- Biomass burning HCHO hotspot map with fire correlation and wind transport overlay
- CPCB station time series comparison with predictions
- Seasonal trend analysis across all regions of India

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Always `cd isro-aqi-hcho` before running Python modules (config paths are relative to project root)
- `streamlit run` must be called from within `isro-aqi-hcho/` for relative config paths to resolve
- Real data requires: CDS API key (ERA5), GEE authentication (TROPOMI), MOSDAC login (INSAT), FIRMS MAP_KEY
- Synthetic demo data is generated in-memory by the Streamlit app if CSVs are not found; or pre-generate with `--synthetic` flags

## Pointers

- See `isro-aqi-hcho/README.md` for full step-by-step team guide
- See `isro-aqi-hcho/data/README.md` for data download instructions
