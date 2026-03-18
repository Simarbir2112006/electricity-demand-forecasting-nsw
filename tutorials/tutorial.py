import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from data_loader import load_raw_data
from features import build_features
from model import build_model, metrics
from interpret import plot_forecast, shap_summary

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load
print("Step 1: Loading data...")
df_raw = load_raw_data("../data")
print("")

# Features
print("Step 2: Building features...")
df, dp = build_features(df_raw)
print("")

# Split
print("Step 3: Splitting into train / valid / test...")
FEATURE_LR   = [c for c in df.columns if c.startswith("dp_")] + ["is_holiday"]
FEATURE_TREE = ["lag_1","lag_2","lag_48","lag_336","rolling_48","rolling_336",
                "hour_sin","hour_cos","month_sin","month_cos",
                "hour_day_interaction","is_holiday","RRP"]
TARGET = "TOTALDEMAND"

n = len(df)
train_end = int(n * 0.70)
valid_end  = int(n * 0.90)

train = df.iloc[:train_end]
valid = df.iloc[train_end:valid_end]
test  = df.iloc[valid_end:]

X1_train, y_train = train[FEATURE_LR],   train[TARGET]
X1_valid, y_valid = valid[FEATURE_LR],   valid[TARGET]
X1_test,  y_test  = test[FEATURE_LR],    test[TARGET]
X2_train = train[FEATURE_TREE]
X2_valid = valid[FEATURE_TREE]
X2_test  = test[FEATURE_TREE]
print("")

# Train
print("Step 4: Training model...")
model = build_model()
model.fit(X1_train, X2_train, y_train,
          X1_valid=X1_valid, X2_valid=X2_valid, y_valid=y_valid)
print("")

# Evaluate

print("Step 5: Evaluating...")
print("  Train:", metrics(y_train, model.predict(X1_train, X2_train)))
print("  Valid:", metrics(y_valid, model.predict(X1_valid, X2_valid)))
print("  Test: ", metrics(y_test,  model.predict(X1_test,  X2_test)))
print("")

# Save predictions
print("Step 6: Saving predictions...")
y_pred_test = model.predict(X1_test, X2_test)
preds_df = pd.DataFrame({"actual": y_test, "predicted": y_pred_test})
preds_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
preds_df.to_csv(preds_path)
print(f"  Predictions saved → {preds_path}")
print("")

# Forecast plot
print("Step 7: Saving forecast plot...")
forecast_path = os.path.join(OUTPUT_DIR, "forecast.png")
plot_forecast(y_test, y_pred_test, save_path=forecast_path)
print(f"  Forecast plot saved → {forecast_path}")
print("")

# SHAP 
print("Step 8: Saving SHAP summary...")
shap_path = os.path.join(OUTPUT_DIR, "shap_summary.png")
shap_summary(model, X2_test, save_path=shap_path)
print(f"  SHAP plot saved → {shap_path}")
print("")

print("\nDone! All outputs saved to outputs/")