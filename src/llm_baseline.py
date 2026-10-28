"""
LLM baseline: can a general-purpose LLM do this classification task, and
at what cost relative to the classical model already built in
modeling.py?

Design, kept deliberately small (this is a bolt-on comparison, not a
second project):

- Each customer is serialized into one natural-language paragraph
  (`serialize_customer`) -- the LLM never sees engineered/encoded
  features, only the same raw facts a human analyst would read off an
  account screen.
- Two prompting strategies are compared: zero-shot (no examples) and
  few-shot (a handful of labeled example profiles drawn from TRAIN only,
  never from the test rows being scored -- the same train/test boundary
  the classical model respects).
- The model is asked for a single number -- a 0-100 churn probability --
  rather than a label, so its output is directly comparable to the
  classical model's `predict_proba` on ROC-AUC/PR-AUC, not just accuracy.
- Run on a fixed-size stratified SAMPLE of the test set (not all 1,409
  rows), to keep API cost and wall-clock time bounded for a 2-day
  project. The classical model's metrics are recomputed on the exact
  same sample for an apples-to-apples comparison.
- Every call's actual token usage (from the API response, not estimated)
  is used to compute real cost against Groq's published per-token price.
"""
import json
import logging
import re
import time

import numpy as np
import pandas as pd
import requests

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIELD_LABELS = {
    "gender": "gender", "SeniorCitizen": "senior citizen", "Partner": "has a partner",
    "Dependents": "has dependents", "PhoneService": "has phone service",
    "MultipleLines": "multiple phone lines", "InternetService": "internet service",
    "OnlineSecurity": "online security add-on", "OnlineBackup": "online backup add-on",
    "DeviceProtection": "device protection add-on", "TechSupport": "tech support add-on",
    "StreamingTV": "streaming TV add-on", "StreamingMovies": "streaming movies add-on",
    "Contract": "contract type", "PaperlessBilling": "paperless billing",
    "PaymentMethod": "payment method",
}


def serialize_customer(row: pd.Series) -> str:
    """Turn one customer's raw fields into a plain-English profile -- the
    same facts a retention analyst would read off an account screen, no
    engineered/encoded features."""
    senior = "a senior citizen" if row["SeniorCitizen"] == 1 else "not a senior citizen"
    parts = [
        f"This customer is {row['gender']} and {senior}.",
        f"Partner: {row['Partner']}. Dependents: {row['Dependents']}.",
        f"They have been a customer for {int(row['tenure'])} months, on a '{row['Contract']}' contract.",
        f"Phone service: {row['PhoneService']} (multiple lines: {row['MultipleLines']}).",
        f"Internet service: {row['InternetService']}.",
        f"Add-ons subscribed -- online security: {row['OnlineSecurity']}, online backup: {row['OnlineBackup']}, "
        f"device protection: {row['DeviceProtection']}, tech support: {row['TechSupport']}, "
        f"streaming TV: {row['StreamingTV']}, streaming movies: {row['StreamingMovies']}.",
        f"Paperless billing: {row['PaperlessBilling']}. Payment method: {row['PaymentMethod']}.",
        f"Monthly charge: ${row['MonthlyCharges']:.2f}. Total charges to date: ${row['TotalCharges']:.2f}.",
    ]
    return " ".join(parts)


SYSTEM_PROMPT = (
    "You are a churn-risk analyst for a telecom company. You will be given a customer's "
    "account profile. Estimate the probability (0-100) that this customer will cancel "
    "('churn') their service. Respond with EXACTLY one line, this exact format, no other "
    "text and no explanation:\nCHURN_PROBABILITY: <integer 0-100>"
)

RESPONSE_PATTERN = re.compile(r"CHURN_PROBABILITY:\s*(\d{1,3})", re.IGNORECASE)


def build_fewshot_messages(train_df: pd.DataFrame, n_per_class: int) -> list:
    """Sample labeled examples from TRAIN only -- the same boundary the
    classical model's Pipeline respects, so the LLM arm doesn't get an
    unfair look at anything from the held-out test rows."""
    examples = []
    for label_value in (0, 1):
        subset = train_df[train_df[config.TARGET_COL] == label_value].sample(
            n=n_per_class, random_state=config.RANDOM_STATE
        )
        examples.append(subset)
    examples_df = pd.concat(examples).sample(frac=1, random_state=config.RANDOM_STATE)  # shuffle order

    messages = []
    for _, row in examples_df.iterrows():
        prob = 85 if row[config.TARGET_COL] == 1 else 10  # a representative, not row-specific, target
        messages.append({"role": "user", "content": serialize_customer(row) + "\n\nWill this customer churn?"})
        messages.append({"role": "assistant", "content": f"CHURN_PROBABILITY: {prob}"})
    return messages


def call_groq(messages: list, api_key: str, max_attempts: int = 5) -> dict:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.LLM_MODEL, "messages": messages, "temperature": 0,
        "max_tokens": config.LLM_MAX_TOKENS, "reasoning_effort": config.LLM_REASONING_EFFORT,
    }

    for attempt in range(1, max_attempts + 1):
        start = time.time()
        try:
            resp = requests.post(config.GROQ_API_URL, headers=headers, json=payload, timeout=30)
        except requests.RequestException as exc:
            logger.warning(f"attempt {attempt}/{max_attempts} request error: {exc!r}")
            time.sleep(2 ** attempt)
            continue
        latency = time.time() - start

        if resp.status_code == 200:
            data = resp.json()
            return {
                "content": data["choices"][0]["message"]["content"],
                "prompt_tokens": data["usage"]["prompt_tokens"],
                "completion_tokens": data["usage"]["completion_tokens"],
                "latency_sec": latency,
            }
        if resp.status_code == 429:
            wait = 2 ** attempt
            logger.warning(f"rate limited (attempt {attempt}/{max_attempts}), retrying in {wait}s")
            time.sleep(wait)
            continue
        raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError(f"Groq API failed after {max_attempts} attempts")


def classify_sample(
    sample_df: pd.DataFrame, api_key: str, mode: str, train_df: pd.DataFrame = None,
) -> pd.DataFrame:
    assert mode in ("zero_shot", "few_shot")
    fewshot_messages = build_fewshot_messages(train_df, config.LLM_FEWSHOT_PER_CLASS) if mode == "few_shot" else []

    rows = []
    for i, (_, row) in enumerate(sample_df.iterrows()):
        messages = (
            [{"role": "system", "content": SYSTEM_PROMPT}]
            + fewshot_messages
            + [{"role": "user", "content": serialize_customer(row) + "\n\nWill this customer churn?"}]
        )
        result = call_groq(messages, api_key)
        match = RESPONSE_PATTERN.search(result["content"])
        parse_ok = match is not None
        prob = int(match.group(1)) / 100.0 if parse_ok else None

        rows.append({
            "customerID": row[config.ID_COL],
            "y_true": row[config.TARGET_COL],
            "parse_ok": parse_ok,
            "raw_response": result["content"].strip(),
            "churn_probability": prob,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "latency_sec": result["latency_sec"],
        })
        if (i + 1) % 25 == 0:
            logger.info(f"[{mode}] {i + 1}/{len(sample_df)} done")
        time.sleep(0.3)  # polite pacing, well under Groq's rate limit

    return pd.DataFrame(rows)


def compute_cost(results_df: pd.DataFrame) -> dict:
    total_input = results_df["prompt_tokens"].sum()
    total_output = results_df["completion_tokens"].sum()
    cost = (
        total_input / 1_000_000 * config.LLM_PRICE_PER_1M_INPUT_TOKENS
        + total_output / 1_000_000 * config.LLM_PRICE_PER_1M_OUTPUT_TOKENS
    )
    return {
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "total_cost_usd": cost,
        "cost_per_1000_predictions_usd": cost / len(results_df) * 1000,
        "mean_latency_sec": float(results_df["latency_sec"].mean()),
        "parse_failure_rate": float((~results_df["parse_ok"]).mean()),
    }
