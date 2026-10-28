"""
SHAP explainability (IDEAS.md #3).

Permutation importance (already in error_analysis.py) answers "which
features matter globally, on average." SHAP answers a sharper question:
"for THIS specific customer, how much did each feature push the
prediction up or down" -- the standard tool for per-prediction
explanations, and a stronger interview visual than one global bar chart.

Uses shap.TreeExplainer directly on the Random Forest step of the
Pipeline (exact, not approximated, for tree models) on the already
preprocessed (one-hot encoded) feature matrix -- SHAP needs the same
numeric representation the tree actually splits on, not the raw
DataFrame the Pipeline's ColumnTransformer consumes.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap


def compute_shap_values(pipe, X: pd.DataFrame):
    """Returns (shap_values_for_churn_class, X_transformed_df, explainer)."""
    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    X_transformed = prep.transform(X)
    feature_names = prep.get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

    explainer = shap.TreeExplainer(model)
    raw_values = explainer.shap_values(X_transformed)
    # shap 0.5x returns (n_samples, n_features, n_classes) for a binary
    # RandomForestClassifier; take the "churn" (class 1) slice.
    if isinstance(raw_values, list):
        shap_values = raw_values[1]
    elif raw_values.ndim == 3:
        shap_values = raw_values[:, :, 1]
    else:
        shap_values = raw_values
    return shap_values, X_transformed_df, explainer


def global_importance_from_shap(shap_values: np.ndarray, feature_names) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def plot_shap_summary(shap_values: np.ndarray, X_transformed_df: pd.DataFrame, out_path, max_display: int = 15):
    fig = plt.figure(figsize=(8, 7))
    shap.summary_plot(shap_values, X_transformed_df, max_display=max_display, show=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_shap_waterfall(explainer, shap_values_row: np.ndarray, X_transformed_row: pd.Series,
                         base_value: float, out_path, max_display: int = 12):
    """Per-customer explanation: which features pushed this ONE person's
    predicted churn probability up or down, and by how much."""
    explanation = shap.Explanation(
        values=shap_values_row, base_values=base_value,
        data=X_transformed_row.values, feature_names=list(X_transformed_row.index),
    )
    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
