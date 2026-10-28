"""
IDEAS.md #3 -- SHAP explainability.

Computes SHAP values for the classical model on the full held-out test
set, saves a global summary plot + importance ranking, and generates
per-customer waterfall explanations for three illustrative cases: a
correctly-caught "classic" churner, a missed "surprise" churner (the
loyal-but-churning segment found in error_analysis.py's misclassification
breakdown), and a confidently-correct "stays" prediction.

Usage:
    python run_shap_analysis.py
"""
import logging

import joblib
import numpy as np

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import chronological_free_split
from src.shap_analysis import compute_shap_values, global_importance_from_shap, plot_shap_summary, plot_shap_waterfall

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    raw = clean(load_raw())
    engineered = engineer_features(raw)
    model_frame = prepare_model_frame(engineered)
    _, test = chronological_free_split(model_frame)

    pipe = joblib.load(config.REPORTS_DIR / "final_model.joblib")
    y_true = test[config.TARGET_COL].values
    y_proba = pipe.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    logger.info(f"Computing SHAP values for all {len(test)} test customers (TreeExplainer, exact)")
    shap_values, X_t, explainer = compute_shap_values(pipe, test[FEATURE_COLUMNS])

    imp = global_importance_from_shap(shap_values, X_t.columns)
    imp.to_csv(config.REPORTS_DIR / "shap_feature_importance.csv", index=False)
    logger.info(f"Top 10 SHAP features:\n{imp.head(10)}")

    plot_shap_summary(shap_values, X_t, config.FIGURES_DIR / "shap_summary.png")

    expected_value = explainer.expected_value
    base_value = float(expected_value[1]) if hasattr(expected_value, "__len__") else float(expected_value)

    # Pick three illustrative customers by prediction outcome.
    true_positive_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    false_negative_idx = np.where((y_true == 1) & (y_pred == 0))[0]
    true_negative_idx = np.where((y_true == 0) & (y_pred == 0) & (y_proba < 0.15))[0]

    examples = {
        "true_positive_classic_churner": true_positive_idx[0] if len(true_positive_idx) else None,
        "false_negative_surprise_churner": false_negative_idx[0] if len(false_negative_idx) else None,
        "true_negative_confident_stay": true_negative_idx[0] if len(true_negative_idx) else None,
    }

    for name, idx in examples.items():
        if idx is None:
            logger.warning(f"No example found for {name}, skipping")
            continue
        out_path = config.FIGURES_DIR / f"shap_waterfall_{name}.png"
        plot_shap_waterfall(explainer, shap_values[idx], X_t.iloc[idx], base_value, out_path)
        logger.info(f"{name}: customerID={test.iloc[idx][config.ID_COL]}, "
                    f"y_true={y_true[idx]}, predicted_proba={y_proba[idx]:.3f} -> {out_path.name}")

    logger.info(f"Done. See {config.REPORTS_DIR / 'shap_feature_importance.csv'} and reports/figures/shap_*.png")


if __name__ == "__main__":
    main()
