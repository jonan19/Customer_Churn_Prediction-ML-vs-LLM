"""
Cost/accuracy comparison: the classical model already trained by
run_pipeline.py vs. an LLM (zero-shot and few-shot) on the exact same
held-out customers.

Requires a Groq API key as an environment variable:
    PowerShell:  $env:GROQ_API_KEY = "your-key-here"
    bash:        export GROQ_API_KEY="your-key-here"

Usage:
    python run_llm_comparison.py
"""
import json
import logging
import os
import time

import joblib
from sklearn.model_selection import train_test_split

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import chronological_free_split, evaluate
from src.llm_baseline import classify_sample, compute_cost
from src.error_analysis import plot_llm_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set. PowerShell: $env:GROQ_API_KEY = 'your-key-here', then rerun."
        )
    if not (config.REPORTS_DIR / "final_model.joblib").exists():
        raise SystemExit("reports/final_model.joblib not found -- run `python run_pipeline.py` first.")

    logger.info("Step 1/5: rebuilding the exact same data + split used by the classical model")
    raw = clean(load_raw())
    engineered = engineer_features(raw)             # raw Yes/No strings -- what the LLM reads
    model_frame = prepare_model_frame(engineered)    # 0/1-encoded -- what the classical model reads
    train_model, test_model = chronological_free_split(model_frame)

    engineered_by_id = engineered.set_index(config.ID_COL)
    train_engineered = engineered_by_id.loc[train_model[config.ID_COL]].reset_index()

    logger.info(f"Step 2/5: drawing a stratified {config.LLM_SAMPLE_SIZE}-row sample of the held-out test set")
    sample_model, _ = train_test_split(
        test_model, train_size=config.LLM_SAMPLE_SIZE,
        stratify=test_model[config.TARGET_COL], random_state=config.RANDOM_STATE,
    )
    sample_engineered = engineered_by_id.loc[sample_model[config.ID_COL]].reset_index()
    logger.info(f"Sample churn rate: {sample_model[config.TARGET_COL].mean():.1%} "
                f"(full test set churn rate: {test_model[config.TARGET_COL].mean():.1%})")

    logger.info("Step 3/5: scoring the already-trained classical model on this exact sample")
    pipe = joblib.load(config.REPORTS_DIR / "final_model.joblib")
    classical_proba = pipe.predict_proba(sample_model[FEATURE_COLUMNS])[:, 1]
    classical_pred = (classical_proba >= 0.5).astype(int)
    classical_metrics = evaluate(sample_model[config.TARGET_COL].values, classical_pred, classical_proba)
    classical_metrics.pop("confusion_matrix")

    # Local-inference latency microbenchmark, for a genuine apples-to-apples
    # comparison against the LLM's network round-trip latency.
    single_row = sample_model[FEATURE_COLUMNS].iloc[[0]]
    pipe.predict_proba(single_row)  # warm up
    n_bench = 100
    start = time.perf_counter()
    for _ in range(n_bench):
        pipe.predict_proba(single_row)
    classical_metrics["mean_latency_sec"] = (time.perf_counter() - start) / n_bench
    classical_metrics["cost_per_1000_predictions_usd"] = 0.0  # local CPU inference, no per-call billing
    logger.info(f"Classical model on LLM sample: {classical_metrics}")

    all_results = {
        "sample_size": config.LLM_SAMPLE_SIZE,
        "sample_churn_rate": float(sample_model[config.TARGET_COL].mean()),
        "classical_random_forest": classical_metrics,
    }

    for mode in ("zero_shot", "few_shot"):
        logger.info(f"Step 4/5: running LLM ({mode}) on {len(sample_engineered)} customers")
        llm_results = classify_sample(sample_engineered, api_key, mode=mode, train_df=train_engineered)
        llm_results.to_csv(config.REPORTS_DIR / f"llm_predictions_{mode}.csv", index=False)

        parsed = llm_results[llm_results["parse_ok"]]
        if len(parsed) < len(llm_results):
            logger.warning(f"[{mode}] {len(llm_results) - len(parsed)} responses failed to parse "
                            f"and are excluded from accuracy metrics (counted in parse_failure_rate)")

        y_true = parsed["y_true"].values
        y_proba = parsed["churn_probability"].values
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = evaluate(y_true, y_pred, y_proba)
        metrics.pop("confusion_matrix")
        metrics.update(compute_cost(llm_results))
        logger.info(f"LLM ({mode}) metrics: {metrics}")
        all_results[f"llm_{mode}"] = metrics

    logger.info("Step 5/5: saving comparison + plot")
    out_path = config.REPORTS_DIR / "llm_vs_classical_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    plot_llm_comparison(all_results, config.FIGURES_DIR / "llm_vs_classical_comparison.png")
    logger.info(f"Done. See {out_path}")


if __name__ == "__main__":
    main()
