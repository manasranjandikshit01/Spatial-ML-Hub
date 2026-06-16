---
name: ISRO V3 architecture decisions
description: Key design decisions made during the V3 upgrade of the ISRO AQI & HCHO pipeline
---

## Model factory pattern
`build_model(config)` in `src/models/cnn_lstm_aqi.py` reads `config["model"]["model_type"]`:
- `"cnnlstm"` → `CNNLSTM` (V2 architecture, SpatialEncoder→LSTM→FC)
- `"convlstm"` → `ConvLSTMModel` (V3, ConvLSTMCell through time → Conv1×1 head)

**Why:** Single config toggle without import changes. All training code calls `build_model(config)`.

## Rolling features requirement
`add_rolling_features()` in `make_features_aqi.py` **must** sort by `(cell_id, date)` before calling
`.rolling()` — otherwise rolling windows bleed across cell boundaries, producing invalid features.

**Why:** The DataFrame has many cells interleaved. `groupby(cell_id).transform(rolling)` handles
this correctly but only if the sort order is right for `min_periods`.

## Spatial context via scipy.ndimage
`add_spatial_context()` uses `pivot_table → uniform_filter → melt → merge` pattern per date.
Using `apply()` per row is too slow (O(n²)). The pivot+filter approach is O(n log n).

**How to apply:** Only enable with `--spatial_context` flag on small grids (dev_mode).
On full India 0.1° grid (~90k cells × 730 days) it can take several minutes.

## HCHO anomaly definition
`compute_hcho_anomaly()` computes z-score within `(cell_id, season)` groups:
`hcho_anomaly = (hcho_column - group_mean) / (group_std + 1e-6)`
Raw anomaly in µmol/m² is also kept as `hcho_anomaly_raw`.

## GeoJSON export
`export_hotspot_geojson()` emits RFC 7946 FeatureCollection. DBSCAN clusters become
MultiPoint features; noise cells (cluster=-1) become individual Point features.
Output goes to `data/processed/hcho_hotspot_clusters.geojson`.

## CLI orchestrator
`scripts/run_pipeline.py` uses argparse subparsers. Must be run from `isro-aqi-hcho/`.
The `run_all --synthetic` subcommand runs the full pipeline end-to-end.
