"""
ROI simulator (IDEAS.md #2).

The lift table already answers "if we call the riskiest K% of
customers, what fraction of churners do we catch." This module answers
the sharper question the README's own "Limitations" section flags as
missing: "...but is that actually worth doing, in dollars?"

Every dollar figure here is a documented ASSUMPTION, not something in
the dataset -- there is no real cost or revenue data in the Telco churn
dataset. The point is to show the mechanics of turning model output into
a business decision, not to claim these specific numbers are accurate
for a real telecom. Each assumption is named as a parameter so it can be
swapped for real figures.

Model (per customer contacted):
    cost        = cost_per_call
    if the customer is an actual churner AND the call succeeds:
        revenue_saved = save_rate * retained_revenue_per_customer
    net_value   = revenue_saved - cost  (summed over everyone contacted)

retained_revenue_per_customer defaults to `retention_months *
avg_monthly_charge` -- a standard simplifying assumption (N months of
revenue retained), not profit margin -- stated explicitly so it reads as
a documented assumption, not a hidden one.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_roi(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    avg_monthly_charge: float,
    cost_per_call: float = 15.0,
    save_rate: float = 0.30,
    retention_months: int = 12,
    k_grid=None,
) -> pd.DataFrame:
    """Sweep the contact fraction K from ~1% to 100% and compute the
    economics of contacting the top-K%-riskiest customers at each point."""
    if k_grid is None:
        k_grid = np.arange(0.01, 1.01, 0.01)

    retained_revenue_per_save = retention_months * avg_monthly_charge
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    n = len(y_true)

    rows = []
    for k in k_grid:
        contacted = max(1, int(round(n * k)))
        churners_in_slice = y_sorted[:contacted].sum()
        cost = contacted * cost_per_call
        revenue_saved = churners_in_slice * save_rate * retained_revenue_per_save
        net_value = revenue_saved - cost
        rows.append({
            "k_pct": k,
            "n_contacted": contacted,
            "churners_caught": int(churners_in_slice),
            "cost": cost,
            "revenue_saved": revenue_saved,
            "net_value": net_value,
            "roi": (revenue_saved - cost) / cost if cost > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def find_optimal_k(roi_df: pd.DataFrame) -> pd.Series:
    return roi_df.loc[roi_df["net_value"].idxmax()]


def simulate_roi_random_baseline(
    y_true: np.ndarray, avg_monthly_charge: float,
    cost_per_call: float = 15.0, save_rate: float = 0.30, retention_months: int = 12,
    k_grid=None,
) -> pd.DataFrame:
    """Same economics, but customers are contacted in RANDOM order instead
    of ranked by model risk -- i.e. what you'd get with no model at all.
    At unconstrained optimization this barely matters (eventually you'd
    call almost everyone either way, see run_roi_simulation.py's finding),
    but at a REALISTIC fixed budget (a team that can only make so many
    calls this month) the gap between this curve and the model-ranked one
    is exactly what the model is worth."""
    if k_grid is None:
        k_grid = np.arange(0.01, 1.01, 0.01)
    retained_revenue_per_save = retention_months * avg_monthly_charge
    base_rate = y_true.mean()
    n = len(y_true)

    rows = []
    for k in k_grid:
        contacted = max(1, int(round(n * k)))
        expected_churners_in_slice = contacted * base_rate  # no ranking signal -- just the base rate
        cost = contacted * cost_per_call
        revenue_saved = expected_churners_in_slice * save_rate * retained_revenue_per_save
        net_value = revenue_saved - cost
        rows.append({"k_pct": k, "n_contacted": contacted,
                      "churners_caught": expected_churners_in_slice,
                      "cost": cost, "revenue_saved": revenue_saved, "net_value": net_value})
    return pd.DataFrame(rows)


def plot_roi_curve(roi_df: pd.DataFrame, optimal_row: pd.Series, out_path, random_df: pd.DataFrame = None):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(roi_df["k_pct"] * 100, roi_df["net_value"], color="#2b6cb0", label="Model-ranked targeting")
    if random_df is not None:
        ax1.plot(random_df["k_pct"] * 100, random_df["net_value"], color="gray",
                  linestyle=":", label="Random targeting (no model)")
    ax1.axvline(optimal_row["k_pct"] * 100, color="#c05621", linestyle="--",
                label=f"Optimal: contact top {optimal_row['k_pct']:.0%}")
    ax1.axhline(0, color="gray", linewidth=0.8)
    ax1.set_xlabel("% of customers contacted")
    ax1.set_ylabel("Net value ($) -- revenue saved minus call cost")
    ax1.set_title("Retention-campaign net value vs. contact rate")
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
