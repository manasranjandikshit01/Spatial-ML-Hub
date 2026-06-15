#!/usr/bin/env bash
# run_train_aqi.sh
# ================
# Train baseline ML models and optionally the CNN-LSTM.
#
# Usage:
#   bash scripts/run_train_aqi.sh                   # baseline only
#   bash scripts/run_train_aqi.sh --cnn_lstm         # baseline + CNN-LSTM
#   bash scripts/run_train_aqi.sh --cnn_lstm --synthetic  # smoke test

set -euo pipefail
cd "$(dirname "$0")/.."

RUN_CNN_LSTM=false
SYNTHETIC=""

for arg in "$@"; do
    case $arg in
        --cnn_lstm) RUN_CNN_LSTM=true ;;
        --synthetic) SYNTHETIC="--synthetic" ;;
    esac
done

echo "=== Training baseline models ==="
python -m src.models.baseline_ml \
    --input      data/processed/aqi_training_dataset.csv \
    --output_dir models/baseline

echo "=== Evaluating baseline models ==="
python -m src.models.evaluate_aqi \
    --model_type baseline \
    --model_dir  models/baseline \
    --test_csv   data/processed/aqi_training_dataset.csv

if [ "$RUN_CNN_LSTM" = true ]; then
    echo "=== Training CNN-LSTM ==="
    python -m src.models.train_aqi \
        --config   config/aqi_training.yaml \
        --grid_csv data/processed/grid_daily_features.csv \
        $SYNTHETIC

    echo "=== Evaluating CNN-LSTM ==="
    python -m src.models.evaluate_aqi \
        --model_type cnn_lstm \
        --model_dir  models/cnn_lstm \
        --grid_csv   data/processed/grid_daily_features.csv \
        --config     config/aqi_training.yaml
fi

echo "=== Done ==="
