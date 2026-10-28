"""
IDEAS.md #2 -- ROI simulator.

Runs the classical model on the FULL held-out test set (1,409 customers,
not just the 200-customer LLM-comparison sample) and sweeps the contact
rate to find the net-value-maximizing retention-call budget.

Usage:
    python run_roi_simulation.py
"""
import json
import logging

import joblib

from src import config
from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import chronological_free_split
from src.roi_simulator import simulate_roi, find_optimal_k, plot_roi_curve, simulate_roi_random_baseline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Documented assumptions -- not in the dataset, swap for real figures if you have them.
COST_PER_CALL = 15.0       # agent time + overhead per retention outreach attempt
SAVE_RATE = 0.30           # fraction of contacted true churners a call actually retains
RETENTION_MONTHS = 12      # months of revenue counted as "saved" per retained customer


def main():
    raw = clean(load_raw())
    engineered = engineer_features(raw)
    model_frame = prepare_model_frame(engineered)
    _, test = chronological_free_split(model_frame)

    pipe = joblib.load(config.REPORTS_DIR / "final_model.joblib")
    y_proba = pipe.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    y_true = test[config.TARGET_COL].values
    avg_monthly_charge = float(test["MonthlyCharges"].mean())

    logger.info(f"Test set: n={len(test)}, avg MonthlyCharges=${avg_monthly_charge:.2f}, "
                f"assumptions: cost/call=${COST_PER_CALL}, save_rate={SAVE_RATE:.0%}, "
                f"retention_months={RETENTION_MONTHS}")

    roi_df = simulate_roi(
        y_true, y_proba, avg_monthly_charge,
        cost_per_call=COST_PER_CALL, save_rate=SAVE_RATE, retention_months=RETENTION_MONTHS,
    )
    roi_df.to_csv(config.REPORTS_DIR / "roi_simulation.csv", index=False)
    random_df = simulate_roi_random_baseline(
        y_true, avg_monthly_charge,
        cost_per_call=COST_PER_CALL, save_rate=SAVE_RATE, retention_months=RETENTION_MONTHS,
    )

    optimal = find_optimal_k(roi_df)
    logger.info(f"Optimal (unconstrained) contact rate: {optimal['k_pct']:.0%} "
                f"({optimal['n_contacted']} customers) -> net value ${optimal['net_value']:,.0f}, "
                f"catching {optimal['churners_caught']} churners, ROI={optimal['roi']:.1%}")

    # The unconstrained optimum calls most customers (these assumptions make
    # almost any call profitable); the more realistic question is what the
    # model is worth at a fixed, constrained budget -- compare it to random
    # targeting at a couple of representative budgets.
    budget_comparison = {}
    for k in (0.10, 0.20, 0.30):
        model_row = roi_df.iloc[(roi_df["k_pct"] - k).abs().idxmin()]
        random_row = random_df.iloc[(random_df["k_pct"] - k).abs().idxmin()]
        budget_comparison[f"top_{int(k*100)}pct"] = {
            "model_net_value": float(model_row["net_value"]),
            "random_net_value": float(random_row["net_value"]),
            "model_value_add_over_random": float(model_row["net_value"] - random_row["net_value"]),
            "model_churners_caught": float(model_row["churners_caught"]),
            "random_churners_caught_expected": float(random_row["churners_caught"]),
        }
        logger.info(f"At top {k:.0%} budget: model net value ${model_row['net_value']:,.0f} vs. "
                    f"random ${random_row['net_value']:,.0f} "
                    f"(model adds ${model_row['net_value'] - random_row['net_value']:,.0f})")

    contact_all = roi_df.iloc[-1]
    summary = {
        "assumptions": {
            "cost_per_call_usd": COST_PER_CALL, "save_rate": SAVE_RATE,
            "retention_months": RETENTION_MONTHS, "avg_monthly_charge_usd": avg_monthly_charge,
        },
        "optimal_unconstrained": optimal.to_dict(),
        "contact_everyone": contact_all.to_dict(),
        "contact_nobody_net_value": 0.0,
        "budget_constrained_model_vs_random": budget_comparison,
    }
    with open(config.REPORTS_DIR / "roi_simulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    plot_roi_curve(roi_df, optimal, config.FIGURES_DIR / "roi_simulation.png", random_df=random_df)
    logger.info(f"Done. See {config.REPORTS_DIR / 'roi_simulation_summary.json'}")


if __name__ == "__main__":
    main()
