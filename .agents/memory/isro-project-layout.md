---
name: ISRO AQI project layout
description: Key architectural decisions for the ISRO AQI/HCHO project in isro-aqi-hcho/.
---

# ISRO Project Layout

**Why:** Project is a standalone Python ML project inside the pnpm monorepo workspace. All relative config/data paths resolve from `isro-aqi-hcho/` as working directory.

## Key rules

- Streamlit workflow: `cd isro-aqi-hcho && streamlit run src/webapp/app.py` (port 5000)
- Python modules: run as `python -m src.data.build_dataset_aqi` from inside `isro-aqi-hcho/`
- Synthetic demo: all pipelines have `--synthetic` flag; Streamlit auto-generates demo data if CSVs absent
- Grid: 0.1° regular lat/lon over India (68–97.5°E, 8–37.5°N), `cell_id = CELL_{lat:.2f}_{lon:.2f}`

## Config files
- `config/paths.yaml` — data dirs and grid bbox
- `config/aqi_training.yaml` — CNN-LSTM + baseline hyperparams
- `config/hcho_hotspot.yaml` — hotspot percentile, DBSCAN params, region bboxes

## CNN-LSTM
- Input: (B, T, C, H, W) — per-timestep SpatialEncoder (CNN) → LSTM → FC → (B, H, W)
- Feature cols: TROPOMI NO2/SO2/CO/O3/HCHO, INSAT AOD, ERA5 t2m/rh2m/u10/v10/tp/sp/blh

**How to apply:** When modifying models or data pipelines, ensure working dir is `isro-aqi-hcho/` and use `python -m src.<module>` syntax.
