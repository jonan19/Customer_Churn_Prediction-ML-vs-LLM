# Results & Interview Prep Notes

Generated from a real run of `python run_pipeline.py` against the committed
IBM Telco Customer Churn dataset (7,043 customers, 26.5% churn rate). Raw
numbers live in `reports/metrics_summary.json`, `reports/cv_results.csv`,
`reports/feature_importance.csv`, and `reports/lift_table.csv`.

## 1. One-paragraph summary (lead with this)

Built a churn-prediction pipeline on a real telecom dataset: given a
customer's contract, billing, and service details, predict whether they
are about to cancel. The chosen model (**Random Forest**, selected by
5-fold CV on PR-AUC) gets **84.4% ROC-AUC** and **65.5% PR-AUC** on a
held-out test set, catching **79% of actual churners** (recall) while
**53% of everyone it flags actually churns** (precision) — against a
naive "always predict stay" baseline that gets a deceptively high 73.5%
*accuracy* while catching **zero** churners. The headline business number:
if a retention team can only call the riskiest **20%** of customers, this
model lets them catch **50% of all churners** at **67% precision** — more
than 2.5x better than calling a random 20% of the customer base.

## 2. Why accuracy is the wrong metric here (say this unprompted)

- 73.5% of customers don't churn, so a model that never predicts churn at
  all already scores 73.5% accuracy while being completely useless.
- That's exactly what the naive baseline does in this project: 73.5%
  accuracy, 0% precision, 0% recall, 0% F1 — a concrete, measured
  demonstration of why accuracy alone is the wrong metric for an
  imbalanced classification problem.
- Used **PR-AUC** (average precision) as the primary model-selection
  metric instead, because it's sensitive to exactly the thing that
  matters here: can the model rank true churners above everyone else,
  even though they're the minority class.

## 3. How the pipeline avoids the usual tabular-ML mistakes

1. This is a **snapshot** dataset (one row per customer, no dates) — no
   time-series lookahead question to manage, unlike a forecasting
   problem. Stated explicitly so I don't over-claim leakage controls
   that don't apply here.
2. The leakage risk that *does* apply: one-hot encoding and scaling must
   be fit on TRAIN only. Every model is wrapped in an sklearn `Pipeline`,
   refit per CV fold and on the final train split — never fit on the
   full dataset. Tested directly in `tests/test_pipeline.py::test_pipeline_fit_on_train_only_does_not_require_test_rows`.
3. Stratified 80/20 train/test split (preserves the 26.5% churn rate in
   both), test rows verified to have zero ID overlap with train
   (`test_split_has_no_row_overlap_and_preserves_class_balance`).
4. A real data-quality bug found and fixed correctly: `TotalCharges` has
   11 blank values, every one of them a brand-new customer with
   `tenure == 0` — filled with 0 (no billing history yet exists), not a
   mean (which would fabricate history) or a dropped row. The code
   **asserts** this pattern holds rather than assuming it
   (`src/data_pipeline.py::clean`) — if a future data refresh broke that
   assumption, the pipeline would fail loudly instead of silently
   mis-imputing.

## 4. Models compared & why

| Model | CV mean ROC-AUC | CV mean PR-AUC | Why included |
|---|---|---|---|
| Logistic Regression | 0.846 | 0.663 | Interpretable linear baseline — coefficients read directly as "this feature raises/lowers churn odds" |
| **Random Forest (selected)** | **0.848** | **0.663** | Captures nonlinear interactions (e.g. contract type x tenure) without needing scaling |
| XGBoost | 0.843 | 0.655 | Gradient boosting comparison point |

All three models land within ~1 point of PR-AUC of each other — a sign
the signal here is strong and not especially sensitive to model choice;
Random Forest wins by the thinnest of margins, which I'd flag honestly
rather than oversell as a decisive win.

## 5. Final numbers (held-out test, n=1,409)

- **Confusion matrix** (rows=actual, cols=predicted): `[[774 stay-correct, 261 false alarm], [77 missed churner, 297 caught churner]]`
- Accuracy: **76.0%** | Precision: **53.2%** | Recall: **79.4%** | F1: **63.7%**
- ROC-AUC: **84.4%** | PR-AUC: **65.5%**
- Naive baseline: accuracy 73.5%, precision/recall/F1 **all 0%**, PR-AUC 26.5% (= the base churn rate, the PR-AUC floor for any useless model)

### Lift table (the number to actually lead with)

| Contact top X% of customers (ranked by risk) | # customers called | % of all churners caught | Precision among those called |
|---|---|---|---|
| 5% | 70 | 15.5% | 82.9% |
| 10% | 140 | 27.8% | 74.3% |
| **20%** | **281** | **50.3%** | **66.9%** |
| 30% | 422 | 66.3% | 58.8% |
| 50% | 704 | 86.4% | 45.9% |

Plain-English version: *"Call the riskiest 1 in 5 customers, and you'll
reach half of everyone who was about to leave — and two out of three of
those calls are to a real at-risk customer, not a false alarm."*

## 6. Error analysis takeaways

**Top permutation-importance features:** `Contract` (0.050), `tenure`
(0.046), `InternetService` (0.038), `is_month_to_month` (0.032),
`TotalCharges` (0.029), `OnlineSecurity` (0.023), `TechSupport` (0.019).
Contract type and tenure dominate, exactly matching the well-known
real-world churn story: month-to-month customers with no cancellation
penalty and low tenure (haven't built habit/switching cost yet) are the
highest-risk group.

**Misclassification breakdown is the most interview-worthy part:**
- **True positives (297, correctly caught):** average tenure 11.8
  months, 99.7% month-to-month, only 1.4 add-on services — the "classic"
  churn profile, and the model nails it.
- **False positives (261, false alarms):** average tenure 18.1 months,
  99.6% month-to-month — customers who *look* exactly like churners
  (same contract type, similarly short tenure) but didn't leave. The
  model can't distinguish them from true churners because, on the
  features available, they genuinely look the same — this is the
  ceiling of what contract/billing data alone can predict.
- **False negatives (77, missed churners):** average tenure **34.3
  months** (3x longer than true positives), **2.6 add-on services**
  (nearly double), and only 43% month-to-month (the rest on 1-2 year
  contracts). These are "surprise" churners — loyal-looking, multi-service,
  often-contracted customers who left anyway, almost certainly for
  reasons this dataset doesn't capture (a price increase, a competitor
  offer, a life event, a bad support experience). That's a concrete,
  honest answer to "where does the model fail and why."

## 7. Honest limitations (say these before being asked)

- Single snapshot, not a time series — no sense of trend, seasonality,
  or how risk changes after an intervention (e.g. a retention call).
- No usage-volume, customer-support-interaction, or satisfaction/NPS
  data, which a real telco would have and which likely explains a good
  chunk of the "surprise churner" false negatives above.
- The lift table shows what the model *could* catch, not realized
  campaign ROI — that needs the cost of a retention call and the
  probability it actually changes the customer's decision, neither of
  which is in this dataset.
- Random Forest's win over Logistic Regression and XGBoost is small
  (within ~1 point of PR-AUC) — reported as a close call, not a decisive
  win.

## 8. Likely interview questions, pre-answered

**"Why not just maximize accuracy?"**
Walk through section 2 — the naive "always predict stay" baseline already
gets 73.5% accuracy while catching zero churners; accuracy can't tell a
useful model from a useless one here.

**"How did you handle class imbalance?"**
`class_weight="balanced"` for Logistic Regression and Random Forest,
`scale_pos_weight` (ratio of negative to positive class counts) for
XGBoost; model selection driven by PR-AUC, not accuracy or even plain
ROC-AUC, since PR-AUC is more sensitive to minority-class performance.

**"Walk me through a real decision you had to make with the data."**
The `TotalCharges` blank-value fix — found 11 blanks, checked they were
all `tenure == 0` (new customers), filled with 0 rather than a mean or a
dropped row, and wrote an assertion so the pipeline fails loudly if that
assumption ever breaks on new data.

**"What would you do with more time/data?"**
Usage-volume and support-interaction features, a time-series version
(churn risk trajectory over a customer's lifetime, not a single
snapshot), and a real cost-benefit model for the lift table (cost per
retention call x estimated save-rate) to turn "catches 50% of churners"
into an actual ROI number.
