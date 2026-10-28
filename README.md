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
    error_analysis.py   # importance, confusion matrix, lift table, LLM comparison plot
    llm_baseline.py      # LLM add-on: prompts, Groq API client, cost tracking
    calibration_analysis.py  # Brier score, ECE, reliability diagrams
    shap_analysis.py          # per-customer SHAP explanations
    roi_simulator.py          # lift table -> real (assumption-labeled) dollars
    survival_analysis.py      # Kaplan-Meier + Cox Proportional Hazards
  tests/
    test_pipeline.py     # data-quality + split correctness tests
    test_llm_baseline.py # prompt/parsing/cost-math tests (no API key needed)
    test_calibration_and_roi.py  # ECE/Brier/ROI arithmetic tests
  reports/               # generated: metrics, CV table, lift table, figures
  run_pipeline.py        # single entry point, end to end
  run_llm_comparison.py  # add-on: LLM vs. classical cost/accuracy comparison
  run_calibration_analysis.py  # add-on: calibration study
  run_shap_analysis.py          # add-on: SHAP explainability
  run_roi_simulation.py         # add-on: ROI simulator
  run_survival_analysis.py      # add-on: survival analysis
  app.py                  # interactive Streamlit demo
  RESULTS.md             # numeric results + interview talking points
  IDEAS.md               # brainstormed future extensions (built ones marked)
  requirements.txt       # pinned versions
```

## Add-on: do you even need an LLM for this?

`run_llm_comparison.py` answers a question worth asking honestly rather
than assuming: given the classical model above already works, is it
worth replacing with an LLM? It serializes the same held-out test
customers into plain-English profiles, asks an LLM (`openai/gpt-oss-20b`
via Groq) for a churn probability under zero-shot and few-shot prompting,
and compares accuracy, latency, and real API cost against the classical
model on the *exact same* customers. Short answer: no -- the classical
model wins on every ranking metric, is ~11x faster, and costs ~$0 vs.
~$38-97 per million predictions. The more interesting part is *why* (a
few-shot numeric-anchoring effect, and a reasoning model's hidden
chain-of-thought token cost) -- full writeup in `RESULTS.md` section 9.

```bash
$env:GROQ_API_KEY = "your-key-here"   # PowerShell; needs a Groq API key
python run_llm_comparison.py
```

## Four further analyses + an interactive demo

- **Calibration** (`run_calibration_analysis.py`) -- is a predicted 70%
  churn probability actually ~70% in reality? Compares Brier score and
  Expected Calibration Error across classical/zero-shot/few-shot.
- **SHAP explainability** (`run_shap_analysis.py`) -- per-customer
  "why did the model flag *this* person" waterfall plots, on top of the
  global feature importance already in `error_analysis.py`.
- **ROI simulator** (`run_roi_simulation.py`) -- turns the lift table
  into real (assumption-labeled) dollars, and compares the model's
  ranking against random targeting at a realistic fixed call budget.
- **Survival analysis** (`run_survival_analysis.py`) -- reframes "will
  they churn" as "when will they churn" (Kaplan-Meier + Cox Proportional
  Hazards), a genuinely different modeling paradigm from everything else
  in this repo.
- **Interactive demo** (`app.py`, Streamlit) -- type in any customer
  profile, see the classical model's live prediction + SHAP explanation
  next to a live LLM call, with real measured latency/cost for each; a
  second tab browses every figure from this whole project.

Full numbers and the two most interesting findings (calibration beats
ranking in an unexpected direction; the Cox model's concordance index
edges out the classifier's ROC-AUC) are in `RESULTS.md` section 10.

```bash
streamlit run app.py   # needs GROQ_API_KEY for the live LLM side (sidebar prompt if unset)
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
