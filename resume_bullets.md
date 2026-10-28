# Resume Bullets

## Final pick (balances classical ML + AI/LLM evaluation skill)

**Customer Churn Prediction & LLM Cost-Accuracy Benchmark** | Python, scikit-learn, XGBoost, Groq LLM API

- Engineered a churn-prediction classifier (Random Forest, XGBoost) on 7,000+ telecom accounts with leak-proof preprocessing pipelines and class-imbalance handling, achieving 84% ROC-AUC and catching 79% of at-risk customers.
- Benchmarked zero-shot and few-shot LLM prompting (Groq gpt-oss-20b) against the classical model on identical data, uncovering a numeric-anchoring bias in few-shot outputs and a hidden reasoning-token cost unique to reasoning models.
- Quantified the cost-accuracy-latency tradeoff between classical ML and LLMs for structured prediction, proving the classical model wins on every ranking metric at 1/1000th the cost and 11x lower latency — turning an AI-hype question into an evidence-based recommendation.

---

Matched to your existing bullet style (action verb + what was built +
technique + outcome/rigor). Pick 2-3, or mix/trim to fit your layout.

1. **Built a customer churn classifier on 7,000+ telecom accounts using engineered behavioral features and Random Forest, achieving 84% ROC-AUC and catching 79% of churners (vs. 0% for a naive baseline).**

2. **Engineered a lift-analysis framework quantifying that targeting the riskiest 20% of customers captures 50% of all churners at 67% precision, translating model output into a directly actionable retention strategy.**

3. **Benchmarked Logistic Regression, Random Forest, and XGBoost under 5-fold cross-validation with class-imbalance handling, and diagnosed failure modes via misclassification analysis to identify a distinct "loyal-but-churning" customer segment.**

4. **Quantified the cost-accuracy tradeoff of LLMs vs. classical ML for structured prediction by benchmarking an LLM (Llama/gpt-oss via Groq) against the churn model on identical data, finding classical ML wins on every ranking metric at 1/1000th the cost and 11x lower latency.**

## Alternates (swap in if you want a different emphasis)

- **Diagnosed a real data-quality bug (ambiguous null billing values) in a production-style dataset, root-caused it to a specific customer segment, and fixed it with an assertion-backed, auditable transformation rather than a blind imputation.**
- **Designed and validated an end-to-end churn-prediction pipeline (Python, scikit-learn, XGBoost) with leak-proof preprocessing pipelines and automated correctness tests, improving PR-AUC 2.5x over the majority-class baseline.**
- **Diagnosed a numeric-anchoring failure mode in few-shot LLM prompting and a hidden reasoning-token cost driver in gpt-oss models, translating both into a concrete engineering recommendation on when to use classical ML over an LLM.**
- **Extended model evaluation beyond accuracy with calibration analysis (Brier score, ECE), SHAP-based per-customer explainability, a survival-analysis (Cox Proportional Hazards, 0.89 concordance) reframing of the problem, and an ROI simulator quantifying the model's dollar value over random targeting under a fixed budget.**
