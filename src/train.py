import os
import glob
import pandas as pd
from data_loader import load_raw_data
from features import build_features
from model import build_model, metrics
from interpret import plot_forecast, shap_summary

# Load 
df_raw = load_raw_data("../data")

# Features 
df, dp = build_features(df_raw)
print("Built rows:", len(df))

# Feature lists 
FEATURE_LR   = [c for c in df.columns if c.startswith("dp_")] + ["is_holiday"]
FEATURE_TREE = ["lag_1","lag_2","lag_48","lag_336","rolling_48","rolling_336",
                "hour_sin","hour_cos","month_sin","month_cos",
                "hour_day_interaction","is_holiday","RRP"]
TARGET = "TOTALDEMAND"

# Split 
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

# Train
model = build_model()
model.fit(X1_train, X2_train, y_train,
          X1_valid=X1_valid, X2_valid=X2_valid, y_valid=y_valid)

# Evaluate
print("Train:", metrics(y_train, model.predict(X1_train, X2_train)))
print("Valid:", metrics(y_valid, model.predict(X1_valid, X2_valid)))
print("Test: ", metrics(y_test,  model.predict(X1_test,  X2_test)))

# Plots & SHAP
plot_forecast(y_test, model.predict(X1_test, X2_test))
shap_summary(model, X2_test)