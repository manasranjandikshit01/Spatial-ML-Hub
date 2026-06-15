#!/usr/bin/env bash
# run_build_aqi_dataset.sh
# ========================
# Build the AQI training dataset (uses synthetic data if raw files are absent).
#
# Usage:
#   bash scripts/run_build_aqi_dataset.sh            # real data
#   bash scripts/run_build_aqi_dataset.sh --synthetic  # demo / smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building AQI dataset ==="
python -m src.data.build_dataset_aqi "$@"

echo "=== Engineering AQI features ==="
python -m src.features.make_features_aqi \
    --input  data/processed/aqi_training_dataset.csv \
    --output data/processed/aqi_features.csv \
    --scaler models/baseline/scaler.pkl

echo "Done. Output files:"
ls -lh data/processed/aqi_training_dataset.csv data/processed/aqi_features.csv 2>/dev/null || true
