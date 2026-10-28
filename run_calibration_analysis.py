"""
IDEAS.md #5 -- calibration study.

Compares Random Forest, LLM zero-shot, and LLM few-shot on calibration
(Brier score, Expected Calibration Error, reliability diagram), on the
exact same 200-customer sample used in run_llm_comparison.py. Reuses
predictions already saved to reports/ -- no API key, no retraining,
no new API cost.

Usage:
    python run_calibration_analysis.py
"""
import json
import logging

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import chronological_free_split
from src.calibration_analysis import calibration_metrics, plot_reliability_diagram

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Rebuilding the exact same 200-customer sample used in run_llm_comparison.py")
    raw = clean(load_raw())
    engineered = engineer_features(raw)
    model_frame = prepare_model_frame(engineered)
    _, test_model = chronological_free_split(model_frame)
    sample_model, _ = train_test_split(
        test_model, train_size=config.LLM_SAMPLE_SIZE,
        stratify=test_model[config.TARGET_COL], random_state=config.RANDOM_STATE,
    )

    logger.info("Scoring the classical model on this sample")
    pipe = joblib.load(config.REPORTS_DIR / "final_model.joblib")
    classical_proba = pipe.predict_proba(sample_model[FEATURE_COLUMNS])[:, 1]
    classical_y_true = sample_model[config.TARGET_COL].values

    logger.info("Loading already-collected LLM predictions")
    results = {"Random Forest (classical)": (classical_y_true, classical_proba)}
    for mode, label in [("zero_shot", "LLM (zero-shot)"), ("few_shot", "LLM (few-shot)")]:
        llm_df = pd.read_csv(config.REPORTS_DIR / f"llm_predictions_{mode}.csv")
        parsed = llm_df[llm_df["parse_ok"]]
        results[label] = (parsed["y_true"].values, parsed["churn_probability"].values)

    metrics = {}
    for label, (y_true, y_prob) in results.items():
        m = calibration_metrics(y_true, y_prob)
        logger.info(f"{label}: {m}")
        metrics[label] = m

    with open(config.REPORTS_DIR / "calibration_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_reliability_diagram(results, config.FIGURES_DIR / "calibration_reliability_diagram.png")
    logger.info(f"Done. See {config.REPORTS_DIR / 'calibration_metrics.json'} "
                f"and {config.FIGURES_DIR / 'calibration_reliability_diagram.png'}")


if __name__ == "__main__":
    main()
