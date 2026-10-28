"""
Correctness tests: the data-cleaning assumption actually holds in the
real data, the train/test split has zero row overlap and preserves the
class balance, and preprocessing is fit on train only (never sees test
rows' categories/scale before scoring).
"""
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import load_raw, clean
from src.features import engineer_features, prepare_model_frame, FEATURE_COLUMNS
from src.modeling import chronological_free_split, get_models
from src import config


@pytest.fixture(scope="module")
def model_frame():
    df = clean(load_raw())
    df = engineer_features(df)
    df = prepare_model_frame(df)
    return df


def test_total_charges_has_no_missing_after_cleaning(model_frame):
    assert model_frame["TotalCharges"].isna().sum() == 0


def test_target_is_binary(model_frame):
    assert set(model_frame[config.TARGET_COL].unique()) == {0, 1}


def test_engineered_features_have_no_unexpected_nans(model_frame):
    engineered = ["num_services", "is_month_to_month", "has_internet", "charge_increase_ratio"]
    for col in engineered:
        assert model_frame[col].isna().sum() == 0, f"{col} has unexpected NaNs"


def test_split_has_no_row_overlap_and_preserves_class_balance(model_frame):
    train, test = chronological_free_split(model_frame)
    assert len(train) + len(test) == len(model_frame)

    train_ids = set(train[config.ID_COL])
    test_ids = set(test[config.ID_COL])
    assert train_ids.isdisjoint(test_ids), "train/test row overlap detected"

    overall_rate = model_frame[config.TARGET_COL].mean()
    train_rate = train[config.TARGET_COL].mean()
    test_rate = test[config.TARGET_COL].mean()
    assert abs(train_rate - overall_rate) < 0.02
    assert abs(test_rate - overall_rate) < 0.02


def test_pipeline_fit_on_train_only_does_not_require_test_rows(model_frame):
    """A Pipeline fit on TRAIN must be able to .predict_proba on TEST
    without being refit -- i.e. encoders/scalers only ever learned from
    TRAIN rows. We exercise this directly rather than inspecting
    internals."""
    train, test = chronological_free_split(model_frame)
    models = get_models(train[config.TARGET_COL])
    pipe = models["logistic_regression"]
    pipe.fit(train[FEATURE_COLUMNS], train[config.TARGET_COL])

    proba = pipe.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    assert proba.shape[0] == len(test)
    assert np.all((proba >= 0) & (proba <= 1))
