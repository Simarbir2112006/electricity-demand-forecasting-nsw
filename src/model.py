import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

class BoostedHybrid:
    def __init__(self, model_1, model_2):
        self.model_1 = model_1
        self.model_2 = model_2
        self.fitted = False

    def fit(self, X1, X2, y, X1_valid=None, X2_valid=None, y_valid=None):
        self.model_1.fit(X1, y)
        y1_train = pd.Series(self.model_1.predict(X1), index=X1.index)
        resid_train = y - y1_train

        y1_valid = pd.Series(self.model_1.predict(X1_valid), index=X1_valid.index)
        resid_valid = y_valid - y1_valid

        self.model_2.fit(
            X2, resid_train,
            eval_set=[(X2_valid, resid_valid)]
        )
        self.fitted = True
        return self

    def predict(self, X1, X2, beta=1.0):
        y1 = pd.Series(self.model_1.predict(X1), index=X1.index)
        y2 = pd.Series(self.model_2.predict(X2), index=X2.index)
        return y1 + beta * y2


def build_model():
    rr = Ridge(alpha=0.35, fit_intercept=False)
    lgb = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=50,
        reg_alpha=5.0,
        reg_lambda=10.0,
        metric="rmse",
        random_state=42,
        verbosity=-1,
    )
    return BoostedHybrid(rr, lgb)


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": round(rmse, 5), "MAE": round(mae, 5), "R2": round(r2, 5)}