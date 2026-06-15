#!/usr/bin/env bash
# run_hcho_hotspots.sh
# ====================
# Build HCHO-fire dataset, run hotspot detection, and generate maps.
#
# Usage:
#   bash scripts/run_hcho_hotspots.sh            # real data
#   bash scripts/run_hcho_hotspots.sh --synthetic  # demo

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building HCHO-fire dataset ==="
python -m src.data.build_dataset_hcho "$@"

echo "=== Engineering HCHO features & detecting hotspots ==="
python -m src.features.make_features_hcho \
    --input  data/processed/hcho_fire_daily_grid.csv \
    --output data/processed/hcho_hotspot_features.csv \
    --config config/hcho_hotspot.yaml

echo "=== Generating hotspot plots ==="
python -m src.visualization.plot_hotspots \
    --hcho_csv    data/processed/hcho_fire_daily_grid.csv \
    --hotspot_csv data/processed/hcho_hotspot_features.csv \
    --output_dir  outputs/hotspot_maps

echo "=== Done. Maps saved to outputs/hotspot_maps/ ==="
ls -lh outputs/hotspot_maps/ 2>/dev/null || true
