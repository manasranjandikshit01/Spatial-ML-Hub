# Model Registry

All trained model artifacts live here, organised by model family.
This file is the authoritative index — update it whenever a new model is trained.

---

## Naming Convention

### Baseline (scikit-learn) models
```
{ModelName}_{target}_{train_years}.joblib
```
Examples:
- `RandomForest_pm25_target.joblib`
- `GradientBoosting_aqi_target.joblib`

### Deep-learning (PyTorch) checkpoints
```
best_model_{run_id}.pt
cnn_lstm_{year_range}_{img_h}x{img_w}_{n_ch}ch.pt
```
Examples:
- `best_model_default.pt` — default training run
- `cnn_lstm_2019-2022_30x30_13ch.pt` — explicitly named

### ConvLSTM
```
convlstm_{year_range}_{img_size}_{run_id}.pt
```

---

## Metadata Files

Every training run automatically writes the following alongside the model file:

| File | Contents |
|------|----------|
| `baseline/baseline_metrics.json` | RMSE, MAE, R², Pearson-r per model × target |
| `baseline/baseline_results.csv` | Flat leaderboard CSV — all runs sortable |
| `baseline/feature_importance_*.csv` | Gini importances per model × target |
| `baseline/predictions_*.csv` | Test-set predicted vs observed |
| `baseline/hparam_sweep_results.csv` | GridSearchCV or manual sweep results |
| `cnn_lstm/training_history_*.json` | Per-epoch train/val loss |
| `cnn_lstm/hparam_sweep_results.csv` | LR × dropout sweep results |

---

## How to Load a Model

### Baseline (RF / GBM)
```python
import joblib
import pandas as pd

model = joblib.load("models/baseline/RandomForest_pm25_target.joblib")
X_test = pd.read_csv("data/processed/aqi_training_dataset.csv").iloc[-100:]
features = ["no2_column","so2_column","co_column","o3_column",
            "hcho_column","insat_aod","t2m","rh2m","u10","v10","tp","sp","blh"]
preds = model.predict(X_test[[f for f in features if f in X_test.columns]])
```

### CNN-LSTM
```python
import torch
from src.models.cnn_lstm_aqi import build_model
import yaml

with open("config/aqi_training.yaml") as f:
    config = yaml.safe_load(f)

model = build_model(config)          # uses model_type from config
model.load_state_dict(torch.load("models/cnn_lstm/best_model_default.pt",
                                  map_location="cpu"))
model.eval()
```

### ConvLSTM
```python
config["model"]["model_type"] = "convlstm"
model = build_model(config)
model.load_state_dict(torch.load("models/cnn_lstm/convlstm_best.pt", map_location="cpu"))
```

---

## Training Reproduceability

Config snapshots are embedded inside `training_history_*.json` under the key
`"config"`. To reproduce any run:

```bash
cd isro-aqi-hcho

# Baseline
python -m src.models.baseline_ml \
    --input data/processed/aqi_training_dataset.csv \
    --output_dir models/baseline \
    --train_end 2021-12-31 \
    --test_start 2022-01-01

# CNN-LSTM (default)
python -m src.models.train_aqi --config config/aqi_training.yaml

# ConvLSTM
# Set model_type: convlstm in config/aqi_training.yaml, then:
python -m src.models.train_aqi --config config/aqi_training.yaml
```

Or use the single pipeline CLI:

```bash
python scripts/run_pipeline.py train_baseline
python scripts/run_pipeline.py train_deep
```

---

## Dashboard Integration

The Streamlit dashboard reads `models/baseline/baseline_metrics.json` to render
the Model Performance page. Retrain → restart the dashboard to pick up new metrics.

The **Model Performance** page auto-detects available `.joblib` files and lists
them in the sidebar model selector.
