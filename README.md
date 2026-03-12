# NSW Electricity Demand Forecasting
### Boosted Hybrid: Ridge Regression + LightGBM

A modular, production-ready forecasting pipeline for half-hourly electricity demand in New South Wales (2018–2023).  

---

## Results

| Model | Valid RMSE | Test RMSE | Test R² |
|---|---|---|---|
| LR + XGBoost (baseline) | 381.99 | 534.75 | 0.819 |
| Ridge + LightGBM | 381.14 | 536.34 | 0.818 |
| **Ridge + LightGBM + Weekly features** | **331.70** | **479.17** | **0.855** |

SHAP analysis confirms `lag_1` (immediate past demand) and `rolling_336` (7-day rolling mean) as the dominant predictors for residual correction.

---

## Architecture

```
ŷ  =  M1(X_trend)  +  M2(X_interaction)
       │                   │
       Ridge               LightGBM
       (Fourier features)  (Lag / rolling / cyclical features)
       Captures: trend      Captures: nonlinear volatility
       + seasonality        + day-of-week patterns
```

---

## Project Structure

```
nsw-electricity-forecast/
├── src/
│   ├── data_loader.py   # CSV ingestion & validation
│   ├── features.py      # All feature engineering (lags, Fourier, cyclical)
│   ├── model.py         # BoostedHybrid class + evaluation helpers
│   ├── interpret.py     # SHAP plots & forecast visualisation
│   └── train.py         # End-to-end CLI training script
├── notebooks/
│   └── exploration.ipynb  # (original Kaggle notebook)
├── tests/
│   └── test_features.py   # pytest unit tests
├── data/                  # Place raw CSVs here (gitignored)
├── outputs/               # Saved predictions (gitignored)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Simarbir2112006/electricity-demand-forecasting-nsw.git
cd nsw-electricity-forecast

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
#    https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023
#    Place CSV files in data/

# 4. Train
cd src
python train.py --data_dir ../data --output_dir ../outputs
```

---

## Usage as a Library

```python
from src.data_loader import load_raw_data
from src.features    import build_features, get_lr_features, FEATURE_TREE, TARGET
from src.model       import build_default_model, evaluate_all_splits
from src.interpret   import shap_summary, plot_forecast

# Load & engineer
df_raw      = load_raw_data("data/")
df, dp      = build_features(df_raw)

# Split
n  = len(df)
i1, i2 = int(n*0.7), int(n*0.9)
train, valid, test = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]

LR_FEATS = get_lr_features(df)

# Train
model = build_default_model()
model.fit(train[LR_FEATS], train[FEATURE_TREE], train[TARGET],
          valid[LR_FEATS], valid[FEATURE_TREE], valid[TARGET])

# Evaluate
splits = {"test": (test[LR_FEATS], test[FEATURE_TREE], test[TARGET])}
print(evaluate_all_splits(model, splits))

# Interpret
shap_summary(model, test[FEATURE_TREE])
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Feature Engineering Summary

| Group | Features |
|---|---|
| Calendar | `hour`, `weekday`, `month`, `is_holiday` |
| Cyclical | `hour_sin/cos`, `month_sin/cos` |
| Lag | `lag_1`, `lag_2`, `lag_48` (1-day), `lag_336` (1-week) |
| Rolling mean | `rolling_48` (1-day), `rolling_336` (1-week) |
| Interaction | `hour_day_interaction` (label-encoded) |
| Fourier (dp_) | Daily (order 4) + Annual (order 2) via DeterministicProcess |

---

## Dataset

[NSW Australia Electricity Demand 2018–2023](https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023) — Kaggle  
248,592 raw rows · 30-minute settlement intervals · columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `statsmodels` · `SHAP` · `matplotlib`
