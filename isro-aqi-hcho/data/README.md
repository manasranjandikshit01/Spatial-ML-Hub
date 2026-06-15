# Data Directory

This directory is **empty** in the repository. Populate it as follows:

## `raw/` — Downloaded source data

| Sub-folder | Contents | Script |
|---|---|---|
| `raw/cpcb/` | CPCB CAAQM hourly/daily CSV exports | Manual download from https://airquality.cpcb.gov.in |
| `raw/tropomi/` | TROPOMI L3 CSVs (one per pollutant) from GEE export or DLR | `src/data/download_tropomi.py` |
| `raw/insat_aod/` | INSAT-3D AOD HDF5/NetCDF files | Manual download from https://www.mosdac.gov.in |
| `raw/reanalysis/` | ERA5 monthly NetCDF files | `src/data/download_reanalysis.py` |
| `raw/firms/` | FIRMS fire CSV chunks | `src/data/download_firms_fire.py` |

## `interim/` — Intermediate processing outputs

Intermediate reprojected / regridded files before final join.

## `processed/` — Final datasets used by models and dashboard

| File | Description |
|---|---|
| `cpcb_daily.csv` | CPCB stations daily means (station_id, lat, lon, date, pm25, …, aqi_observed) |
| `grid_daily_features.csv` | Gridded satellite + met features (cell_id, lat, lon, date, no2_column, …) |
| `aqi_training_dataset.csv` | Joined CPCB + grid features (training dataset for AQI models) |
| `hcho_fire_daily_grid.csv` | HCHO + fire counts + met data (for HCHO hotspot pipeline) |
| `hcho_hotspot_features.csv` | Seasonal HCHO stats with hotspot flags and cluster IDs |

## Generating demo (synthetic) data

If you do not yet have real data, run:

```bash
python -m src.data.build_dataset_aqi --synthetic
python -m src.data.build_dataset_hcho --synthetic
```

This populates `processed/` with realistic synthetic data so the dashboard is fully functional.
