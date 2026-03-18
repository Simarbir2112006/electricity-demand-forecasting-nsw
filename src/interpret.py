import pandas as pd
import matplotlib.pyplot as plt
import shap

def plot_forecast(y_true, y_pred, n_steps=336, save_path=None):
    results = pd.DataFrame(index=y_true.index)
    results["Actual (MW)"] = y_true
    results["Predicted (MW)"] = y_pred

    results.iloc[:n_steps].plot(
        y=["Actual (MW)", "Predicted (MW)"],
        figsize=(10, 8),
        title="First Week Forecast (Actual vs Predicted)",
        color=["black", "red"],
    )
    plt.ylabel("Total Demand (MW)")
    plt.xlabel("Date")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def shap_summary(model, X_test, n_samples=2000, save_path=None):
    shap.initjs()
    X_sample = X_test.sample(n_samples, random_state=42)
    explainer = shap.TreeExplainer(model.model_2)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="dot", show=save_path is None)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()