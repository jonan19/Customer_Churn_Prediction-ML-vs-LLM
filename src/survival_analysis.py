"""
Survival analysis (IDEAS.md #9).

Reframes the problem: instead of "will this customer churn" (binary,
answered elsewhere in this project), ask "how long until this customer
churns" -- time-to-event. `tenure` is the natural duration and `Churn`
the event indicator: a churned customer's tenure is their observed
survival time, and a still-active customer's tenure is a right-censored
observation (we know they survived AT LEAST this long, not their true
total lifetime). This is the standard, textbook-correct way to model
churn in survival/reliability analysis, and a genuinely different
modeling paradigm from the binary classifiers used elsewhere here.

Two tools from `lifelines`:
  - Kaplan-Meier: a non-parametric estimate of the survival function
    (% of customers still active at each tenure month), overall and by
    group (e.g. contract type) -- purely descriptive, no covariates.
  - Cox Proportional Hazards: a semi-parametric regression that
    estimates each feature's HAZARD RATIO -- how much it multiplies a
    customer's instantaneous churn risk at any given moment, holding
    other features fixed. HR > 1 raises risk, HR < 1 lowers it. Evaluated
    with the concordance index (c-index), survival analysis's analog of
    ROC-AUC, on a train/test split exactly like the classifiers elsewhere
    in this project.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index

from src.features import CATEGORICAL_COLS, BINARY_YESNO_COLS

# Features that are either the duration itself or a direct function of it
# (tenure_bucket) or of another covariate already included as a dummy
# (is_month_to_month duplicates Contract, has_internet duplicates
# InternetService) -- excluded from the Cox model to avoid circularity /
# near-perfect collinearity.
COX_NUMERIC_COLS = ["SeniorCitizen", "MonthlyCharges", "TotalCharges", "num_services", "charge_increase_ratio"]
COX_CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c != "tenure_bucket"]


def build_cox_frame(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df[COX_CATEGORICAL_COLS], drop_first=True, dtype=float)
    numeric = df[COX_NUMERIC_COLS + BINARY_YESNO_COLS].astype(float).reset_index(drop=True)
    cox_df = pd.concat([numeric, dummies.reset_index(drop=True)], axis=1)
    cox_df["tenure"] = df["tenure"].astype(float).values
    cox_df["Churn"] = df["Churn"].astype(int).values
    # lifelines chokes on non-alphanumeric column names from one-hot encoding
    cox_df.columns = [str(c).replace(" ", "_").replace("-", "_") for c in cox_df.columns]
    return cox_df


def fit_kaplan_meier(df: pd.DataFrame) -> KaplanMeierFitter:
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df["tenure"], event_observed=df["Churn"], label="All customers")
    return kmf


def plot_km_overall(kmf: KaplanMeierFitter, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    kmf.plot_survival_function(ax=ax, color="#2b6cb0")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Estimated probability customer is still active")
    ax.set_title("Kaplan-Meier survival curve (all customers)")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_km_by_group(df: pd.DataFrame, group_col: str, out_path, title=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    for group_value, group_df in df.groupby(group_col):
        kmf = KaplanMeierFitter()
        kmf.fit(durations=group_df["tenure"], event_observed=group_df["Churn"], label=str(group_value))
        kmf.plot_survival_function(ax=ax)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Estimated probability customer is still active")
    ax.set_title(title or f"Kaplan-Meier survival curves by {group_col}")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_cox_model(train_cox_df: pd.DataFrame, penalizer: float = 0.1) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_cox_df, duration_col="tenure", event_col="Churn")
    return cph


def evaluate_cox_concordance(cph: CoxPHFitter, test_cox_df: pd.DataFrame) -> float:
    partial_hazard = cph.predict_partial_hazard(test_cox_df)
    # lifelines convention: concordance_index expects risk scores where
    # HIGHER = predicted to survive LONGER, so negate the hazard (higher
    # hazard = shorter survival).
    return concordance_index(test_cox_df["tenure"], -partial_hazard, test_cox_df["Churn"])


def hazard_ratio_table(cph: CoxPHFitter) -> pd.DataFrame:
    summary = cph.summary.copy()
    summary["hazard_ratio"] = np.exp(summary["coef"])
    return summary[["hazard_ratio", "coef", "p", "coef lower 95%", "coef upper 95%"]] \
        .sort_values("hazard_ratio", ascending=False).rename_axis("feature").reset_index()
