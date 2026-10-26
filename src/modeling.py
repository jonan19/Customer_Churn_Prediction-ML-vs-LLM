"""
Model definitions, cross-validated comparison, and final held-out
evaluation.

Preprocessing (one-hot encoding, scaling) is always inside an sklearn
`Pipeline`, refit on each CV fold's TRAIN rows and on the final TRAIN
split -- never fit on the full dataset up front. That is the leakage
control for a non-time-series tabular problem: the thing that must never
happen is the encoder/scaler "seeing" the test rows' category frequencies
or value ranges before those rows are scored.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, BINARY_YESNO_COLS


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("scaler", StandardScaler())] if scale_numeric else [("passthrough", "passthrough")]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL_COLS),
            ("num", Pipeline(numeric_steps) if scale_numeric else "passthrough",
             NUMERIC_COLS + BINARY_YESNO_COLS),
        ]
    )


def get_models(y_train: pd.Series):
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos  # XGBoost's native class-imbalance handle

    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=config.RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=10,
                class_weight="balanced", n_jobs=-1, random_state=config.RANDOM_STATE,
            )),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
                eval_metric="logloss", n_jobs=-1, random_state=config.RANDOM_STATE,
            )),
        ]),
    }


CV_SCORING = ["roc_auc", "average_precision", "f1", "precision", "recall"]


def chronological_free_split(df: pd.DataFrame):
    """Stratified random split -- this dataset is a snapshot, not a time
    series, so there is no calendar order to respect; stratifying on the
    target keeps the ~26.5% churn rate consistent across train/test."""
    train, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET_COL], random_state=config.RANDOM_STATE
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def run_cv_comparison(train: pd.DataFrame, feature_cols) -> pd.DataFrame:
    X, y = train[feature_cols], train[config.TARGET_COL]
    models = get_models(y)
    cv = StratifiedKFold(n_splits=config.N_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    rows = []
    for name, pipe in models.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=CV_SCORING, n_jobs=-1)
        for fold in range(config.N_CV_FOLDS):
            rows.append({"model": name, "fold": fold, **{m: scores[f"test_{m}"][fold] for m in CV_SCORING}})
    return pd.DataFrame(rows)


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def final_fit_evaluate(train: pd.DataFrame, test: pd.DataFrame, feature_cols, model_name: str):
    models = get_models(train[config.TARGET_COL])
    pipe = models[model_name]

    X_train, y_train = train[feature_cols], train[config.TARGET_COL]
    X_test, y_test = test[feature_cols], test[config.TARGET_COL]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test.values, y_pred, y_proba)
    return pipe, metrics, y_pred, y_proba


def naive_baseline_metrics(train: pd.DataFrame, test: pd.DataFrame, feature_cols) -> dict:
    """Majority-class baseline: what you'd get by never using the model at all."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train[feature_cols], train[config.TARGET_COL])
    y_pred = dummy.predict(test[feature_cols])
    y_proba = dummy.predict_proba(test[feature_cols])[:, 1]
    return evaluate(test[config.TARGET_COL].values, y_pred, y_proba)
