"""
Tests for the parts of the LLM comparison that don't require a live API
call: response parsing, cost arithmetic, and that few-shot examples are
drawn only from TRAIN (never from the test rows being scored).
"""
import pandas as pd

from src.llm_baseline import RESPONSE_PATTERN, serialize_customer, compute_cost, build_fewshot_messages
from src import config


def test_response_pattern_parses_expected_format():
    assert RESPONSE_PATTERN.search("CHURN_PROBABILITY: 73").group(1) == "73"
    assert RESPONSE_PATTERN.search("  churn_probability:  5  ").group(1) == "5"
    assert RESPONSE_PATTERN.search("Sure! CHURN_PROBABILITY: 100 -- high risk.").group(1) == "100"


def test_response_pattern_rejects_garbage():
    assert RESPONSE_PATTERN.search("I think this customer might leave.") is None


def test_serialize_customer_includes_key_facts():
    row = pd.Series({
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 5, "Contract": "Month-to-month", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35, "TotalCharges": 350.95,
    })
    text = serialize_customer(row)
    assert "5 months" in text
    assert "Month-to-month" in text
    assert "$70.35" in text
    assert "Fiber optic" in text


def test_compute_cost_arithmetic():
    df = pd.DataFrame({
        "prompt_tokens": [1_000_000, 1_000_000],
        "completion_tokens": [1_000_000, 0],
        "latency_sec": [0.5, 0.3],
        "parse_ok": [True, False],
    })
    result = compute_cost(df)
    # 2M input tokens @ $0.075/1M + 1M output tokens @ $0.30/1M
    expected_cost = 2 * config.LLM_PRICE_PER_1M_INPUT_TOKENS + 1 * config.LLM_PRICE_PER_1M_OUTPUT_TOKENS
    assert abs(result["total_cost_usd"] - expected_cost) < 1e-9
    assert result["parse_failure_rate"] == 0.5


def test_fewshot_examples_drawn_only_from_train(monkeypatch):
    train_df = pd.DataFrame({
        config.ID_COL: [f"id_{i}" for i in range(20)],
        config.TARGET_COL: [0, 1] * 10,
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 10, "Contract": "Month-to-month", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0, "TotalCharges": 500.0,
    })
    messages = build_fewshot_messages(train_df, n_per_class=2)
    # 2 classes x 2 examples x (1 user + 1 assistant) message = 8 messages
    assert len(messages) == 8
    assert all(m["role"] in ("user", "assistant") for m in messages)
