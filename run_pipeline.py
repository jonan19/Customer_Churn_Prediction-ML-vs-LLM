"""
End-to-end entry point: raw data -> cleaning -> features -> CV model
comparison -> held-out test evaluation -> error analysis -> reports/.

Usage:
    python run_pipeline.py
"""
import json
import logging

import joblib
import pandas as pd

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import (
    chronological_free_split, run_cv_comparison, final_fit_evaluate, naive_baseline_metrics,
)
from src.error_analysis import (
    permutation_feature_importance, plot_feature_importance, plot_confusion_matrix,
    plot_precision_recall_curve, lift_table, misclassification_breakdown,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Step 1/6: loading + cleaning raw data")
    df = clean(load_raw())

    logger.info("Step 2/6: feature engineering")
    df = engineer_features(df)
    df = prepare_model_frame(df)
    df.to_csv(config.PROCESSED_DIR / "churn_model_frame.csv", index=False)
    logger.info(f"{len(df)} customers, churn rate = {df[config.TARGET_COL].mean():.1%}")

    logger.info("Step 3/6: stratified train/test split")
    train, test = chronological_free_split(df)
    logger.info(f"train={len(train)} test={len(test)} "
                f"(train churn rate={train[config.TARGET_COL].mean():.1%}, "
                f"test churn rate={test[config.TARGET_COL].mean():.1%})")

    logger.info("Step 4/6: 5-fold CV model comparison on train")
    cv_results = run_cv_comparison(train, FEATURE_COLUMNS)
    cv_results.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)
    cv_means = cv_results.groupby("model")[["roc_auc", "average_precision", "f1", "precision", "recall"]].mean()
    logger.info(f"CV mean metrics by model:\n{cv_means}")

    best_model = cv_means["average_precision"].idxmax()
    logger.info(f"Selected final model (highest CV mean PR-AUC): {best_model}")

    logger.info("Step 5/6: final fit on train, single evaluation on held-out test")
    pipe, test_metrics, y_pred, y_proba = final_fit_evaluate(train, test, FEATURE_COLUMNS, best_model)
    naive_metrics = naive_baseline_metrics(train, test, FEATURE_COLUMNS)
    logger.info(f"TEST metrics ({best_model}): {test_metrics}")
    logger.info(f"TEST metrics (naive majority-class baseline): {naive_metrics}")

    joblib.dump(pipe, config.REPORTS_DIR / "final_model.joblib")

    logger.info("Step 6/6: error analysis")
    imp = permutation_feature_importance(pipe, test[FEATURE_COLUMNS], test[config.TARGET_COL])
    imp.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(imp, f"Permutation importance -- {best_model}",
                             config.FIGURES_DIR / "feature_importance.png")

    plot_confusion_matrix(test[config.TARGET_COL], y_pred, config.FIGURES_DIR / "confusion_matrix.png")
    plot_precision_recall_curve(test[config.TARGET_COL], y_proba, config.FIGURES_DIR / "precision_recall_curve.png")

    lift = lift_table(test[config.TARGET_COL].values, y_proba)
    lift.to_csv(config.REPORTS_DIR / "lift_table.csv", index=False)
    logger.info(f"Lift table:\n{lift}")

    misclass = misclassification_breakdown(test, y_pred)

    results = {
        "n_customers": len(df),
        "churn_rate": float(df[config.TARGET_COL].mean()),
        "n_train": len(train),
        "n_test": len(test),
        "best_model": best_model,
        "cv_mean_metrics": cv_means.to_dict(),
        "test_metrics": {k: v for k, v in test_metrics.items() if k != "confusion_matrix"},
        "confusion_matrix": test_metrics["confusion_matrix"],
        "naive_baseline_test_metrics": {k: v for k, v in naive_metrics.items() if k != "confusion_matrix"},
        "lift_table": lift.to_dict(orient="records"),
        "misclassification_breakdown": misclass,
    }
    with open(config.REPORTS_DIR / "metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Done. Artifacts written to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
