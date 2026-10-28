# Customer Churn Prediction

A classical-ML project on a real, public dataset: given a telecom
customer's account details (contract type, billing, tenure, which
services they use), predict whether they're about to cancel ("churn").

Built to be **easy to explain end-to-end** -- no domain jargon, just a
business problem everyone immediately understands: "who is about to
leave, and can we tell early enough to do something about it?"

## The business problem, in one sentence

Acquiring a new customer costs far more than retaining an existing one,
so a retention team wants a ranked list: "call these customers first,
they're the most likely to cancel this month."

## Data

- **Source:** IBM's publicly released "Telco Customer Churn" sample
  dataset (7,043 customers, 21 columns) -- a standard, widely-used
  teaching dataset for exactly this problem.
- Committed locally at `data/raw/telco_churn_raw.csv` so the pipeline
  reproduces identically without depending on an external URL staying up;
  `src/data_pipeline.py` re-downloads it automatically if that file is
  ever missing.
- One row per customer, no dates -- this is a **snapshot**, not a time
  series, so there's no lookahead-leakage question to manage (unlike a
  forecasting problem). The leakage risk here is the ordinary tabular-ML
  kind: encoders/scalers must be fit on TRAIN only, never on the full
  dataset -- handled via sklearn `Pipeline`s throughout.

## A real data-quality issue, found and fixed (good interview material)

`TotalCharges` loads as text because 11 rows hold a blank string instead
of a number. Checked *why*: every one of those 11 rows has `tenure == 0`
-- brand-new customers who haven't been billed yet. So the fix is to
treat blank as `0`, not drop the rows or impute a mean (which would
fabricate billing history that doesn't exist). This is asserted in code
(`src/data_pipeline.py::clean`), not just assumed -- if a future data
pull had a blank `TotalCharges` row with nonzero tenure, the pipeline
would fail loudly rather than silently mis-impute it.

## Features (`src/features.py`)

| Feature | Why it's useful |
|---|---|
| Raw account fields (contract type, payment method, internet service, add-ons, tenure, monthly/total charges) | The base signal -- directly observable account attributes |
| `num_services` (count of 6 optional add-ons subscribed) | Proxy for how embedded/sticky the customer is -- more add-ons, more switching cost |
| `tenure_bucket` (0-12mo / 13-24mo / ... / 61mo+) | Churn risk is famously front-loaded in the first year; this nonlinear relationship is easier for a linear model to pick up from buckets than from raw tenure |
| `is_month_to_month` | Contract type is the single strongest known churn driver here -- a month-to-month customer can leave with zero penalty, pulled out explicitly so its effect reads directly off a model coefficient |
| `charge_increase_ratio` (current monthly rate / implied historical average) | Flags "bill shock" -- a customer paying noticeably more than their own historical average (e.g. a promo just expired) is a different risk profile than one on a stable rate |

## Modeling & evaluation (`src/modeling.py`)

- **Split:** stratified 80/20 train/test (stratified on churn, since this
  is a snapshot with no time order to preserve). Test is evaluated
  exactly once, after model selection is already finalized from CV.
- **Class imbalance:** ~26.5% of customers churn. Every model uses
  balanced class weighting (`class_weight="balanced"` / XGBoost's
  `scale_pos_weight`) and is judged primarily on **PR-AUC** (average
  precision), not raw accuracy -- a model that always predicts "stays"
  gets 73.5% accuracy while catching zero churners, which is exactly why
  accuracy alone is the wrong metric for this problem.
- **Models compared:** Logistic Regression (interpretable baseline),
  Random Forest, XGBoost -- selected by 5-fold stratified CV mean PR-AUC.
- **Metrics:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, plus a
  naive majority-class baseline for direct comparison.

## Error analysis (`src/error_analysis.py`)

- Permutation feature importance (model-agnostic).
- Confusion matrix + precision-recall curve.
- **Lift / "precision at top-K%" table** -- the most business-relevant
  output of the whole project: if a retention team can only call the
  riskiest 10%/20%/30% of customers, what fraction of *all* actual
  churners does that catch? Directly answers "is this model worth using"
  in a sentence anyone can follow.
- Misclassified-case breakdown: what false negatives (missed churners)
  and false positives (false alarms) look like on contract type, tenure,
  monthly charges, and service count.

## Project structure

```
customer-churn-prediction/
  src/
    config.py          # constants: paths, split fractions, CV folds
    data_pipeline.py    # load + clean (the TotalCharges fix lives here)
    features.py          # feature engineering + encoding prep
    modeling.py          # model definitions, CV comparison, final eval
    error_analysis.py   # importance, confusion matrix, lift table
  tests/
    test_pipeline.py     # data-quality + split correctness tests
  reports/               # generated: metrics, CV table, lift table, figures
  run_pipeline.py        # single entry point, end to end
  RESULTS.md             # numeric results + interview talking points
  requirements.txt       # pinned versions
```

## Setup & reproduction

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
python run_pipeline.py          # full run: data -> features -> models -> reports/
pytest                            # data-quality + split correctness tests
```

## Limitations (stated honestly)

- Single historical snapshot (not time-series) -- no sense of trend,
  seasonality, or how churn risk evolves after an intervention.
- No customer-support-interaction, NPS/satisfaction, or usage-volume
  data, which a real telco would have and which would likely add
  meaningful signal beyond billing/contract fields.
- The lift table shows what the model *could* catch, not actual
  retention-campaign ROI -- that would need the cost of an intervention
  and the probability it actually prevents a churn, neither of which is
  in this dataset.
