"""
IDEAS.md #9 -- survival analysis (Kaplan-Meier + Cox Proportional Hazards).

Usage:
    python run_survival_analysis.py
"""
import json
import logging

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame
from src.modeling import chronological_free_split
from src.survival_analysis import (
    build_cox_frame, fit_kaplan_meier, plot_km_overall, plot_km_by_group,
    fit_cox_model, evaluate_cox_concordance, hazard_ratio_table,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    raw = clean(load_raw())
    df = engineer_features(raw)  # raw category strings kept (needed for KM-by-group + dummying)

    logger.info("Fitting Kaplan-Meier (overall)")
    kmf = fit_kaplan_meier(df)
    plot_km_overall(kmf, config.FIGURES_DIR / "survival_km_overall.png")
    logger.info(f"Median survival time (tenure at which 50% have churned, if reached): "
                f"{kmf.median_survival_time_}")

    logger.info("Fitting Kaplan-Meier by contract type")
    plot_km_by_group(df, "Contract", config.FIGURES_DIR / "survival_km_by_contract.png",
                      title="Survival curves by contract type")

    logger.info("Fitting Kaplan-Meier by internet service")
    plot_km_by_group(df, "InternetService", config.FIGURES_DIR / "survival_km_by_internet.png",
                      title="Survival curves by internet service")

    logger.info("Train/test split (same chronological-free stratified split used elsewhere)")
    # build_cox_frame needs the binary Yes/No columns as 0/1 (prepare_model_frame
    # does exactly that, without touching the multi-category columns get_dummies needs).
    df_model = prepare_model_frame(df)
    train, test = chronological_free_split(df_model)
    train_cox = build_cox_frame(train)
    test_cox = build_cox_frame(test)

    logger.info("Fitting Cox Proportional Hazards model on TRAIN")
    cph = fit_cox_model(train_cox)

    c_index = evaluate_cox_concordance(cph, test_cox)
    logger.info(f"Cox model concordance index on held-out TEST: {c_index:.4f} "
                f"(0.5 = random, 1.0 = perfect ranking of who churns sooner)")

    hr_table = hazard_ratio_table(cph)
    hr_table.to_csv(config.REPORTS_DIR / "cox_hazard_ratios.csv", index=False)
    logger.info(f"Top hazard-increasing features (HR > 1, raises churn risk):\n"
                f"{hr_table.head(8)[['feature', 'hazard_ratio', 'p']]}")
    logger.info(f"Top hazard-decreasing features (HR < 1, lowers churn risk):\n"
                f"{hr_table.tail(8)[['feature', 'hazard_ratio', 'p']]}")

    summary = {
        "median_survival_time_months": float(kmf.median_survival_time_)
        if kmf.median_survival_time_ != float("inf") else None,
        "cox_concordance_index_test": float(c_index),
        "n_train": len(train), "n_test": len(test),
    }
    with open(config.REPORTS_DIR / "survival_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Done. See {config.REPORTS_DIR / 'cox_hazard_ratios.csv'}, "
                f"{config.REPORTS_DIR / 'survival_analysis_summary.json'}, and reports/figures/survival_*.png")


if __name__ == "__main__":
    main()
