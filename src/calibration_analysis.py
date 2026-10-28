"""
Calibration analysis (IDEAS.md #5).

Accuracy/ROC-AUC/PR-AUC answer "does the model rank customers correctly."
Calibration answers a different question: "when the model says 70%
churn risk, do roughly 70% of those customers actually churn?" A model
can rank well and still be badly calibrated (confidently wrong), which
matters a lot if the predicted probability itself is used downstream
(e.g. to prioritize or cost-weight a retention call, see roi_simulator.py).

Two standard metrics, computed from scratch where sklearn doesn't ship
them directly:
  - Brier score: mean squared error between predicted probability and
    the 0/1 outcome. Lower is better; a well-calibrated model on this
    ~26.5%-churn dataset should land well under the naive "always predict
    the base rate" Brier score of base_rate*(1-base_rate) ~= 0.195.
  - Expected Calibration Error (ECE): bin predictions into deciles,
    compare each bin's mean predicted probability to its actual observed
    churn rate, weight by bin size. Lower is better; 0 is perfect.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_confidence - bin_accuracy)
    return ece


def calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    base_rate = y_true.mean()
    naive_brier = base_rate * (1 - base_rate)  # Brier score of always predicting the base rate
    return {
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "naive_brier_score": float(naive_brier),
        "ece": float(expected_calibration_error(y_true, y_prob)),
        "n": int(len(y_true)),
    }


def plot_reliability_diagram(results: dict, out_path, n_bins: int = 10):
    """results: {label: (y_true, y_prob)}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfectly calibrated")

    colors = ["#2b6cb0", "#dd6b20", "#c05621"]
    for (label, (y_true, y_prob)), color in zip(results.items(), colors):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=label, color=color)

    ax.set_xlabel("Mean predicted churn probability (per bin)")
    ax.set_ylabel("Observed churn rate (per bin)")
    ax.set_title("Reliability diagram: predicted vs. actual churn rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
