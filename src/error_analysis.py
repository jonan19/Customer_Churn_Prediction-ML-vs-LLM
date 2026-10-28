"""
Error analysis -- the business-facing payoff of this project.

- Permutation feature importance: which signals actually drive the
  model's predictions (model-agnostic, works for logistic/RF/XGB alike).
- Confusion matrix + precision/recall curve.
- Lift / "precision at top-K%" chart: if a retention team can only call
  the highest-risk K% of customers, what fraction of all actual churners
  does that capture? This is the single easiest number to explain to a
  non-technical interviewer or hiring manager.
- Misclassified-case breakdown: what the false negatives (missed
  churners) and false positives (false alarms) look like on key raw
  features, so "where does it fail" has a concrete answer.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src import config


def permutation_feature_importance(pipe, X_test, y_test, n_repeats: int = 10):
    result = permutation_importance(
        pipe, X_test, y_test, scoring="average_precision",
        n_repeats=n_repeats, random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return imp


def plot_feature_importance(imp_df: pd.DataFrame, title: str, out_path):
    top = imp_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#2b6cb0")
    ax.set_title(title)
    ax.set_xlabel("Permutation importance (drop in PR-AUC when shuffled)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stay", "Churn"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Churn classifier: confusion matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


def plot_precision_recall_curve(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#2b6cb0")
    base_rate = y_true.mean()
    ax.axhline(base_rate, color="gray", linestyle="--", label=f"random classifier ({base_rate:.0%} churn rate)")
    ax.set_xlabel("Recall (share of actual churners caught)")
    ax.set_ylabel("Precision (share of flagged customers who actually churn)")
    ax.set_title("Precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def lift_table(y_true: np.ndarray, y_proba: np.ndarray, ks=(0.05, 0.10, 0.20, 0.30, 0.50)) -> pd.DataFrame:
    """For each top-K% of customers ranked by predicted churn risk, what
    fraction of ALL actual churners falls in that slice, and what's the
    precision within it? This is a cumulative gains / lift table -- the
    direct translation of model skill into "who does the retention team
    call."""
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    n = len(y_true)
    total_churners = y_true.sum()

    rows = []
    for k in ks:
        cutoff = max(1, int(n * k))
        caught = y_sorted[:cutoff].sum()
        rows.append({
            "top_k_pct": k,
            "n_customers_contacted": cutoff,
            "churners_caught": int(caught),
            "recall_at_k": caught / total_churners,
            "precision_at_k": caught / cutoff,
        })
    return pd.DataFrame(rows)


def plot_llm_comparison(results: dict, out_path):
    """Grouped bar chart: classical model vs. LLM (zero-shot / few-shot)
    across ranking metrics, plus a log-scale cost comparison."""
    approaches = ["classical_random_forest", "llm_zero_shot", "llm_few_shot"]
    labels = ["Random Forest\n(classical)", "LLM\n(zero-shot)", "LLM\n(few-shot)"]
    colors = ["#2b6cb0", "#dd6b20", "#c05621"]
    metrics = ["roc_auc", "pr_auc", "f1", "accuracy"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1", "Accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.25
    for i, approach in enumerate(approaches):
        values = [results[approach][m] for m in metrics]
        ax.bar(x + (i - 1) * width, values, width, label=labels[i], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy metrics (same 200-customer sample)")
    ax.legend(fontsize=8)

    ax = axes[1]
    costs = [max(results[a]["cost_per_1000_predictions_usd"], 0.0001) for a in approaches]
    bars = ax.bar(labels, costs, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Cost per 1,000 predictions (USD, log scale)")
    ax.set_title("Cost comparison")
    for bar, approach in zip(bars, approaches):
        actual = results[approach]["cost_per_1000_predictions_usd"]
        label = "$0 (local)" if actual == 0 else f"${actual:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def misclassification_breakdown(test_df: pd.DataFrame, y_pred: np.ndarray) -> dict:
    df = test_df.copy()
    df["y_true"] = df[config.TARGET_COL].values
    df["y_pred"] = y_pred

    false_negatives = df[(df.y_true == 1) & (df.y_pred == 0)]  # churners the model missed
    false_positives = df[(df.y_true == 0) & (df.y_pred == 1)]  # false alarms
    true_positives = df[(df.y_true == 1) & (df.y_pred == 1)]

    def profile(subset, cols=("Contract", "tenure", "MonthlyCharges", "num_services")):
        out = {}
        for c in cols:
            if pd.api.types.is_numeric_dtype(subset[c]):
                out[c] = float(subset[c].mean())
            else:
                out[c] = subset[c].value_counts(normalize=True).round(3).to_dict()
        return out

    return {
        "n_false_negatives": len(false_negatives),
        "n_false_positives": len(false_positives),
        "n_true_positives": len(true_positives),
        "false_negative_profile": profile(false_negatives),
        "false_positive_profile": profile(false_positives),
        "true_positive_profile": profile(true_positives),
    }
