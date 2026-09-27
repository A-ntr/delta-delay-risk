"""
Model training + evaluation utilities for the night-before delay-risk model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_curve


def chronological_split(
    df: pd.DataFrame, date_col: str, train_end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date, not randomly — rows on or before train_end go to train,
    everything after goes to test. Leakage rule applied at the dataset level:
    the model must never be evaluated on a date range that overlaps or 
    precedes what it was trained on.
    """
    train_end_ts = pd.Timestamp(train_end)
    train_df = df[df[date_col] <= train_end_ts].copy()
    test_df = df[df[date_col] > train_end_ts].copy()
    return train_df, test_df


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Baseline gradient-boosted tree classifier. Modest depth/learning rate
    to avoid overfitting on a small dataset; random_state fixed for
    reproducibility.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def best_fbeta_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0
) -> dict:
    """
    Sweep candidate decision thresholds and return the one that maximizes
    the F-beta score — recall weighted `beta` times as important as precision
    (missing a real delay costs more than a false alarm, so beta=2 by default).

    Return a dict with keys: "threshold", "f_beta", "precision", "recall".
    """
    
