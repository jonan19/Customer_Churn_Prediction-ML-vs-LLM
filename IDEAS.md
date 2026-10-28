# Future Ideas

Brainstormed while deciding how to extend this project further. Pick
one, say so, and it gets built as its own staged commit (or a few), same
as everything else in this repo.

**Built (2026-10-28):** #1 (demo, `app.py`), #2 (ROI simulator,
`run_roi_simulation.py`), #3 (SHAP, `run_shap_analysis.py`), #5
(calibration, `run_calibration_analysis.py`), #9 (survival analysis,
`run_survival_analysis.py`). Real results and figures are in `reports/`;
not yet reflected in README.md/RESULTS.md's prose or pushed to GitHub.

## Polish / "make it look alive" ideas

1. **Interactive demo (Streamlit, free-tier deploy).** Type in a
   customer profile, see the classical model's prediction + top drivers
   and the LLM's prediction + its reasoning, side by side. Highest
   "show, don't tell" value — an interviewer can click it in 10 seconds
   instead of reading a README. Tradeoff: a few hours of UI work, plus a
   hosting dependency (uptime, free-tier limits) that doesn't deepen the
   actual ML rigor.

2. **ROI simulator for the lift table.** Turn "top-K% capture rate" into
   real dollars: cost per retention call x estimated save-rate -> net $
   saved. Closes a gap the README already flags as a limitation
   ("lift table shows what the model *could* catch, not actual
   retention-campaign ROI"). Low effort, makes the business pitch land
   harder with real numbers instead of precision/recall alone.

3. **SHAP instead of (or alongside) permutation importance.** Same
   information, but SHAP is the industry-standard explainability tool,
   and per-customer SHAP plots ("why did the model flag *this* person")
   are a stronger interview visual than one global importance bar chart.

4. **Widen the LLM comparison to a second model size** (e.g. a 70B-class
   model alongside gpt-oss-20b) to plot an actual cost-accuracy Pareto
   curve instead of a single data point. Strengthens the "is it ever
   worth it" argument since right now it's one model's answer, not a
   trend across model scale.

## Research-flavored ideas (hypothesis-driven, not just "I built X and got Y")

5. **Calibration study (recommended first research add-on — free,
   uses data already collected).** Right now the project compares
   accuracy/ROC-AUC but never asks "when the model says 70% churn risk,
   do ~70% of those customers actually churn?" Plot reliability diagrams
   + compute Expected Calibration Error (ECE) and Brier score for Random
   Forest vs. the LLM, reusing the predictions already saved in
   `reports/llm_predictions_*.csv` and the classical model's
   `predict_proba` output. Zero extra API cost, a few hours of analysis
   code. Calibration is a real, citable ML subfield — this is the
   highest rigor-per-effort option on this whole list.

6. **Few-shot numeric-anchoring ablation.** The project already
   *observed* that few-shot examples using probabilities ~85/10 seem to
   pull the LLM's outputs toward those exact numbers regardless of the
   actual customer profile (see `RESULTS.md` section 9). Turn that
   anecdote into a controlled experiment: systematically vary the
   few-shot anchor values (e.g. 10/90, 30/70, 50/50) and measure how
   much the LLM's output distribution shifts toward whichever anchor
   it's fed. Costs a few hundred more (still-cheap) Groq API calls and
   more design time than #5, but produces a genuinely novel mini-result
   instead of citing an existing one -- the strongest "I did original
   research" story on this list.

7. **Reasoning-effort ablation as a formal study.** The project already
   found, informally, that `reasoning_effort="high"` burned an entire
   300-token budget without ever answering. Formalize this: sweep
   low/medium/high (and maybe a few `max_tokens` budgets) across the
   full 200-customer sample, plot accuracy/cost/latency as a proper
   3-way tradeoff surface instead of a single anecdote from one test
   call.

8. **Prompt-phrasing sensitivity analysis.** Rewrite the system prompt
   3-5 different ways (same intent, different wording) and measure how
   much the LLM's predictions vary across phrasings on the same
   customers. A prompt-robustness/reliability angle — relevant to
   "can you trust this system in production" conversations.

## More ideas (round 2)

9. **Survival analysis / time-to-churn (Kaplan-Meier + Cox Proportional
   Hazards, via the `lifelines` library).** Reframes the whole problem:
   instead of "will this customer churn" (binary), model "how long until
   this customer churns" (time-to-event), treating still-active customers
   as right-censored observations. This is arguably the textbook-correct
   way to model churn in the stats/actuarial world, and it's a genuinely
   different modeling paradigm from everything else in this project —
   strong research flavor, moderate effort (new library, new framing of
   the target, but same features/data).

10. **Fairness/bias audit.** Break out recall, precision, and false
    negative rate by `gender` and `SeniorCitizen` and check for disparate
    impact (does the model systematically miss churners in one subgroup
    more than another?). Reuses predictions already on disk — low
    effort, but a real, current, "responsible AI" angle that most
    portfolio churn projects skip entirely, so it stands out.

11. **Cost-sensitive threshold optimization.** The model currently uses
    a default 0.5 cutoff. With a real cost matrix (e.g. $ cost of a
    missed churner vs. $ cost of an unnecessary retention call), find
    the threshold that minimizes *expected business cost*, not just
    maximizes F1, and plot cost vs. threshold. A natural, more rigorous
    follow-up to the ROI simulator idea (#2) — same inputs, sharper math.

12. **Counterfactual explanations.** For a flagged at-risk customer,
    answer "what's the smallest change that would move them from
    high-risk to low-risk?" (e.g. via the `dice-ml` library or a manual
    perturbation search). Concrete and actionable in a way SHAP isn't —
    SHAP explains *why*, counterfactuals explain *what to do about it*.

13. **LLM as explanation/retention-script generator, not classifier.**
    Flip the LLM's role: let the classical model do the ranking/decision
    (it already wins on that), and use the LLM to *write* a short,
    personalized retention-call talking-point script for each flagged
    customer, grounded in their actual risk factors. This is a more
    realistic production pairing than pitting the two against each
    other — "classical model decides, LLM narrates" is a common real
    pattern and a good answer to "so is the LLM work useless, then?"

14. **Knowledge distillation from LLM to classical model.** Use the
    LLM's churn-probability outputs as an extra soft-label signal (or
    input feature) when training the classical model, and test whether
    it actually improves anything. Directly answers "does the LLM know
    something the engineered features don't," with a number instead of
    a guess — likely "no," which is itself a clean, citable result given
    everything else already found here.

15. **LLM self-consistency / repeated sampling.** Call the LLM N times
    per customer at temperature > 0 and average the probabilities,
    instead of one greedy call. Check whether accuracy/calibration
    improves — and explicitly report the Nx cost increase against
    whatever gain shows up. Extends the reasoning-effort ablation (#7)
    into "is paying more for more samples worth it" territory.

16. **Segment-level vs. global modeling.** Train separate models per
    contract type (or per tenure bucket) instead of one global model,
    and compare. Tests whether "one model fits all customers" is even
    the right assumption, or whether month-to-month and two-year
    customers churn for different enough reasons to need different
    models.

## How these were ranked in discussion

- Free + fast + still "real research": **#5 (calibration)**.
- Best original-research story if willing to spend another session: **#6 (anchoring ablation)**.
- Best "looks impressive in 10 seconds" if time allows: **#1 (demo)**.
- Everything else is a solid but smaller add-on, roughly in effort order within its section.

Round-2 standouts: **#9 (survival analysis)** if you want a genuinely
different modeling paradigm on the resume, not just more metrics on the
same classifier; **#10 (fairness audit)** is nearly free and a current,
differentiating angle almost no competing portfolio project will have;
**#13 (LLM-as-narrator)** is the best answer if an interviewer pushes
back with "okay so was the LLM work pointless then?" — it reframes the
whole add-on as "I found the right division of labor," not "I proved
LLMs lose."
