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
electricity-demand-forecasting-nsw/
├── src/
│   ├── data_loader.py   # CSV ingestion & validation
│   ├── features.py      # All feature engineering (lags, Fourier, cyclical)
│   ├── model.py         # BoostedHybrid class + evaluation helpers
│   ├── interpret.py     # SHAP plots & forecast visualisation
│   └── train.py         # End-to-end training script
├── tutorials/
│   ├── tutorial.py      # Script walkthrough
│   └── tutorial.ipynb   # Interactive notebook walkthrough
├── notebooks/
│   └── exploration.ipynb  # Original Kaggle notebook
├── docs/
│   └── abstract.pdf
├── data/                  # Place raw CSVs here (gitignored)
├── outputs/               # Generated predictions & plots (gitignored)
├── requirements.txt
└── README.md
```

---

## Quick Start
```bash
# 1. Clone
git clone https://github.com/Simarbir2112006/electricity-demand-forecasting-nsw.git
cd electricity-demand-forecasting-nsw

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset
#    https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023
#    Place all CSV files in the data/ folder before running.

# 4. Train
cd src
python train.py
```

---

## Tutorials

Two formats available:

**Notebook (recommended):**
```bash
jupyter notebook tutorials/tutorial.ipynb
```

**Script:**
```bash
python tutorials/tutorial.py
```

Both save predictions and plots to `outputs/` and print the path when done.

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

Download all CSV files from Kaggle and place them in the `data/` folder:
https://www.kaggle.com/datasets/joebeachcapital/nsw-australia-electricity-demand-2018-2023

248,592 raw rows · 30-minute settlement intervals · columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE

---

## Paper

A one-page methodology overview is available in [`docs/abstract.pdf`](docs/abstract.pdf).

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `statsmodels` · `SHAP` · `matplotlib`

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

**Simarbir Singh Sandhu**  
[GitHub](https://github.com/Simarbir2112006) · [Kaggle](https://www.kaggle.com/simarbirsinghsandhu) · [LinkedIn](https://www.linkedin.com/in/simarbir-singh-sandhu/) · [X](https://x.com/sandhusimarbir)