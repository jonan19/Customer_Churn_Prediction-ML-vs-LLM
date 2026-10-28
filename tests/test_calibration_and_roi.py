"""
Tests for the pure-arithmetic pieces of the calibration and ROI-simulator
add-ons (IDEAS.md #5 and #2) -- no model, no API key needed.
"""
import numpy as np
import pandas as pd

from src.calibration_analysis import expected_calibration_error, calibration_metrics
from src.roi_simulator import simulate_roi, simulate_roi_random_baseline, find_optimal_k


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # 10 customers per decile, predicted probability exactly matches the
    # observed rate in each decile -> perfect calibration.
    y_true = np.array([0] * 9 + [1] + [0] * 8 + [1] * 2 + [0] * 5 + [1] * 5)
    y_prob = np.array([0.1] * 10 + [0.2] * 10 + [0.5] * 10)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 1e-9


def test_ece_is_large_for_systematically_overconfident_predictions():
    # Model always predicts 0.9 churn probability, but nobody actually churns.
    y_true = np.zeros(50)
    y_prob = np.full(50, 0.9)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece > 0.85  # should be close to the full 0.9 gap


def test_calibration_metrics_brier_matches_manual_calc():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.6, 0.4])
    expected_brier = np.mean((y_prob - y_true) ** 2)
    m = calibration_metrics(y_true, y_prob)
    assert abs(m["brier_score"] - expected_brier) < 1e-9


def test_roi_simulation_matches_hand_computed_value_at_k_100pct():
    # 10 customers, 4 actual churners, perfect ranking (all churners first).
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    roi_df = simulate_roi(
        y_true, y_proba, avg_monthly_charge=100.0,
        cost_per_call=10.0, save_rate=0.5, retention_months=1,
        k_grid=np.array([1.0]),
    )
    row = roi_df.iloc[0]
    # contact all 10 -> cost = 100; catch all 4 churners -> revenue = 4*0.5*100 = 200
    assert row["n_contacted"] == 10
    assert row["churners_caught"] == 4
    assert abs(row["cost"] - 100.0) < 1e-9
    assert abs(row["revenue_saved"] - 200.0) < 1e-9
    assert abs(row["net_value"] - 100.0) < 1e-9


def test_roi_model_ranking_beats_random_at_fixed_budget_when_model_is_informative():
    # Model ranking puts all 4 churners first (perfect separation); random
    # baseline uses the base rate instead. At a 40% contact budget, the
    # model should catch strictly more (expected) churners than random.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    model_df = simulate_roi(y_true, y_proba, avg_monthly_charge=100.0, k_grid=np.array([0.4]))
    random_df = simulate_roi_random_baseline(y_true, avg_monthly_charge=100.0, k_grid=np.array([0.4]))
    assert model_df.iloc[0]["churners_caught"] > random_df.iloc[0]["churners_caught"]


def test_find_optimal_k_returns_the_max_net_value_row():
    roi_df = pd.DataFrame({"k_pct": [0.1, 0.2, 0.3], "net_value": [10, 50, 30]})
    optimal = find_optimal_k(roi_df)
    assert optimal["k_pct"] == 0.2
