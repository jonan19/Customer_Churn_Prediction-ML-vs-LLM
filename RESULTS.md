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

---

## 9. Add-on: do you even need an LLM for this? (cost/accuracy tradeoff)

A natural question in 2026 is "why not just ask an LLM?" So I measured
it, on the exact same 200-customer stratified sample of the held-out test
set, rather than assuming an answer either way. Code: `src/llm_baseline.py`,
`run_llm_comparison.py`. Model: `openai/gpt-oss-20b` via Groq (chosen
because `llama-3.1-8b-instant` moved to enterprise/contact-sales pricing
in late 2026; gpt-oss-20b is Groq's current cheapest self-serve,
production model). Each customer was serialized into a plain-English
profile (no engineered features, just the raw account facts) and the LLM
was asked for a single churn-probability number, under two prompting
strategies: zero-shot, and few-shot (6 labeled examples from TRAIN only).

### The numbers (same 200 customers, every approach)

| Approach | Accuracy | F1 | ROC-AUC | PR-AUC | Latency/prediction | Cost/1,000 predictions |
|---|---|---|---|---|---|---|
| **Random Forest (classical)** | **73.0%** | 59.7% | **80.8%** | **60.9%** | **47.5ms** | **$0 (local)** |
| LLM, zero-shot | 71.0% | 45.3% | 71.2% | 41.2% | 506ms | $0.038 (~$38/1M) |
| LLM, few-shot | 63.0% | 53.2% | 73.2% | 42.2% | 520ms | $0.097 (~$97/1M) |

See `reports/figures/llm_vs_classical_comparison.png` and
`reports/llm_vs_classical_comparison.json` for the full numbers.

### The verdict

**No, an LLM is not worth it for this task.** The classical model, which
already existed and runs for ~$0 marginal cost, beats both LLM
configurations on every ranking metric (ROC-AUC, PR-AUC) and on raw
accuracy, while being roughly **11x faster per prediction** and
**1,000x+ cheaper at any real volume** (~$38-97 per million predictions
via API vs. effectively free locally). This isn't a surprising result in
hindsight — churn prediction from 19 structured fields is exactly the
regime classical ML was built for, and there was no reason to expect an
LLM to out-model a purpose-built Random Forest here. The value of doing
this measurement wasn't the direction of the answer, it was replacing an
assumption with a number.

### The more interesting finding: *why*, not just *that*

- **Few-shot made ranking better but accuracy worse.** Going from
  zero-shot to few-shot raised ROC-AUC (71.2% -> 73.2%) and F1
  (45.3% -> 53.2%), but *dropped* accuracy (71.0% -> 63.0%). Looking at
  the per-customer outputs explains it: recall jumped from 45.3% to
  79.2% while precision fell (45.3% -> 40.0%) -- few-shot made the model
  predict "churn" far more aggressively, catching more true churners but
  at the cost of many more false alarms. A single metric would have
  hidden this trade-off entirely.
- **The few-shot examples appear to have anchored the model's outputs,
  not calibrated them.** My 6 few-shot examples used representative
  probabilities of 85 (churn) and 10 (stay) as labels. Inspecting raw
  responses (`reports/llm_predictions_few_shot.csv`), a surprising number
  of *different* customer profiles got probabilities suspiciously close
  to those exact anchors (e.g. five consecutive real outputs: 5, 80, 90,
  80, 80) rather than smoothly varying with each profile's actual risk
  factors. That's a concrete, measured instance of a known LLM few-shot
  failure mode -- numeric anchoring -- not a vague claim about it.
- **`gpt-oss-20b` is a reasoning model, and that's an invisible cost
  driver.** At Groq's default ("medium") reasoning effort, a test call
  spent 89 of its 106 completion tokens (84%) on hidden chain-of-thought
  before the visible one-line answer. I switched to `reasoning_effort=
  "low"` for the full run to keep cost/latency sane -- and separately
  found that `"high"` effort burned an entire 300-token budget on
  reasoning alone without ever emitting a parseable answer. For a task
  this simple, a reasoning model's cost (and even its ability to answer
  at all) depends on a configuration knob that has nothing to do with
  the actual decision logic -- a real operational risk a classical model
  simply doesn't have.
- **The LLM run hit real rate limits.** The run log shows frequent `429`
  responses from Groq's free tier, handled by the retry/backoff already
  built into `call_groq()` (0% final parse failures). That's a genuine
  extra layer of operational fragility -- external dependency, retries,
  backoff -- that a model running in-process simply does not have.

### Honest caveats on this add-on

- n=200, one stratified sample, one model, one provider, two prompt
  styles -- not a general "LLMs vs. classical ML" claim, a measurement
  of *this* task under *these* conditions.
- A larger/more capable model (e.g. a 70B+ model, or a frontier model
  like GPT-4o or Claude) might narrow or reverse the accuracy gap; it
  would also cost more per call, likely making the cost gap worse, not
  better -- the core cost argument would still hold.
- Prompt engineering wasn't deeply iterated (one zero-shot, one few-shot
  variant) -- a more carefully tuned prompt, better-chosen few-shot
  examples, or structured-output mode could plausibly close some of the
  accuracy gap. I'd treat this as a reasonable first measurement, not
  the ceiling of what's possible.

### Pre-answered: "why does this matter for an AI CoE role?"

Because the job isn't "use an LLM for everything" -- it's knowing when
not to. This add-on is a small, honest demonstration of that judgment:
I didn't assume the fancier tool was better or worse, I measured
accuracy, cost, and latency on the same data and let the numbers make
the call. The answer here (classical wins, LLMs lose on every axis) will
not always be the answer for every task -- and showing I can run this

---

## 10. Four further analyses (calibration, SHAP, ROI, survival)

Built after the core project + LLM comparison, from `IDEAS.md`. Code:
`src/calibration_analysis.py`, `src/shap_analysis.py`,
`src/roi_simulator.py`, `src/survival_analysis.py`, each with its own
`run_*.py` entry point and real output in `reports/`.

### 10.1 Calibration: is "70% risk" actually 70%?

Ranking metrics (ROC-AUC, PR-AUC) say nothing about whether a predicted
probability is meaningful on its own. Checked it, on the same
200-customer sample as the LLM comparison:

| Approach | Brier score (lower better) | ECE (lower better) |
|---|---|---|
| Random Forest (classical) | **0.177** (vs. 0.195 naive floor) | 0.147 |
| LLM, zero-shot | 0.188 | **0.103** |
| LLM, few-shot | 0.263 (worse than the naive floor) | 0.242 |

The genuinely interesting, non-obvious result: **the classical model
has the best Brier score, but the LLM (zero-shot) is actually
better-calibrated by ECE**, despite ranking worse overall (lower
ROC-AUC/PR-AUC, see section 9). Discrimination (can it rank churners
above non-churners) and calibration (does "70%" mean 70%) are different
properties, and a model can win on one while losing on the other --
RF's advantage here is entirely in ranking, not in probability
trustworthiness. Few-shot's calibration is badly broken (ECE 0.242,
worse Brier than just guessing the base rate every time), consistent
with the numeric-anchoring effect from section 9: its outputs cluster
near the few-shot examples' anchor values rather than reflecting each
customer's actual risk. Caveat: n=200 with 10 bins is a small sample for
a reliability diagram (a few bins have very few points, see
`reports/figures/calibration_reliability_diagram.png`) -- treat the
direction of these results as solid, the exact ECE values as approximate.

### 10.2 SHAP: per-customer, not just global, explanations

Permutation importance (section 6) says what matters on average; SHAP
(`shap.TreeExplainer`, exact for tree models) says what mattered for
*this one customer's* prediction. Global ranking matches the earlier
permutation importance closely (top feature: `is_month_to_month`, mean
|SHAP| 0.059, followed by `Contract_Month-to-month` 0.059 and
`InternetService_Fiber_optic` 0.040) -- a good cross-check that the two
independent methods agree. The new part is per-customer waterfall plots
(`reports/figures/shap_waterfall_*.png`): for the "surprise churner"
case from section 6's misclassification breakdown (31-month tenure, DSL,
moderate charges), SHAP shows every one of their individual factors
pushing the prediction *down* from the 0.499 base rate to 0.267 --
concretely showing the model had good reasons to predict "stays," and
that reasoning was simply wrong for this specific person, not a model bug.

### 10.3 ROI simulator: closing the "is this worth it" gap

Section 7 flagged that the lift table shows what the model *could*
catch, not real dollars. With documented assumptions ($15/call,
30% of contacted churners actually retained, 12 months of their
MonthlyCharges counted as revenue saved -- swap these for real figures,
they're not in the dataset):

- **Unconstrained optimum:** contact the riskiest 76% of customers ->
  net value **$69,301** (catches 370/374 churners). At these assumed
  economics (expected value per call is positive down to a churn
  probability of ~6.5%), almost any contact is profitable, which makes
  the unconstrained optimum a less interesting number than...
- **...the realistic, budget-constrained case.** A retention team
  rarely has an unlimited calling budget. Compared the model's ranking
  against calling customers in random order, at three fixed budgets:

| Budget | Model net value | Random net value | Model's value-add |
|---|---|---|---|
| Top 10% | $22,111 | $6,520 | **+$15,591** |
| Top 20% | $39,145 | $13,040 | **+$26,105** |
| Top 30% | $50,874 | $19,560 | **+$31,313** |

This is the number that actually answers "is the model worth it": at a
realistic fixed budget, good ranking is worth tens of thousands of
dollars over no model at all -- a sharper, more honest framing than the
unconstrained optimum, which mostly says "the assumed economics favor
calling almost everyone" rather than anything about the model's skill.

### 10.4 Survival analysis: a different modeling paradigm entirely

Reframed the problem from "will this customer churn" (binary) to "how
long until they churn" (time-to-event), treating still-active customers
as right-censored. Kaplan-Meier survival curves by contract type
(`reports/figures/survival_km_by_contract.png`) show the starkest
visual in the whole project: two-year contract customers have ~94%
estimated survival probability even out at 72 months, one-year sits
around ~65%, and month-to-month customers fall to barely ~13% -- the
same "contract type dominates" finding as everywhere else in this
project, but visualized as an actual survival curve rather than a
feature-importance bar.

A **Cox Proportional Hazards model** (fit on TRAIN, evaluated on the
same held-out TEST split used everywhere else) gets a **concordance
index of 0.890** -- survival analysis's analog of ROC-AUC, and notably
higher than the classical binary model's 0.844 ROC-AUC on the same
split (not a strictly fair comparison -- different target framing,
different information used -- but a sign the time-to-event framing
captures real structure). Hazard ratios are directly interpretable:
a two-year contract multiplies instantaneous churn hazard by **0.35**
(65% lower risk) relative to the reference category, one-year by 0.51,
having online security by 0.67, having a partner by 0.74 -- all with
p < 0.0001 (`reports/cox_hazard_ratios.csv` has the full table).
Fiber-optic internet raises hazard by 1.60x, consistent with every
other analysis in this project pointing at the same feature.

### 10.5 Interactive demo (`app.py`, Streamlit)

Type in any customer profile and see, side by side, in real time: the
classical model's prediction + its top SHAP factors (local inference,
~tens of milliseconds, $0), and a live `openai/gpt-oss-20b` call via
Groq (zero-shot) with its actual measured latency and cost for that one
call. A second tab browses every static figure from this whole project
(calibration, SHAP, ROI, survival, the LLM comparison) without rerunning
anything. Run locally with `streamlit run app.py`; needs `GROQ_API_KEY`
as an environment variable, a Streamlit secret, or typed into the
sidebar for that session only (never written to disk by the app).

### Pre-answered: new interview questions this section opens up

**"What's the difference between a model being accurate and being
calibrated?"** Walk through 10.1 directly -- RF ranks better (higher
ROC-AUC/PR-AUC) but is *less* calibrated by ECE than the zero-shot LLM.
A model can nail one property and not the other; which one matters
depends on whether you use the output to rank (accuracy/ranking is
what matters) or to compute something with the raw probability, like
expected cost (calibration is what matters).

**"Why Cox regression instead of just more binary classification
metrics?"** Because `tenure` + `Churn` is a textbook right-censored
survival setup already sitting in this dataset, and framing it that way
answers a question binary classification can't: not just *whether* a
customer is likely to churn, but *when* -- useful for prioritizing
outreach by urgency, not just risk.

**"The ROI number changes a lot based on your assumptions -- so what's
the point?"** The point isn't the specific dollar figure (clearly
labeled as assumptions, not data), it's the *mechanism*: showing the
model's ranking is worth something concrete under a realistic budget
constraint (section 10.3's model-vs-random comparison), and that the
mechanism generalizes to real figures if a real retention team's costs
and success rates were plugged in.
comparison, not just cite a hot take, is the actual point.
