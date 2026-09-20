"""
Model training + evaluation utilities for the night-before delay-risk model.

chronological_split() and train_baseline_model() are fully implemented —
use them as the pattern/style to follow.

best_fbeta_threshold() is YOUR exercise. Run
`pytest tests/test_model.py -v` as you go.
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
    everything after goes to test. This is the Stage 2 leakage rule applied
    at the dataset level: the model must never be evaluated on a date range
    that overlaps or precedes what it was trained on.
    """
    train_end_ts = pd.Timestamp(train_end)
    train_df = df[df[date_col] <= train_end_ts].copy()
    test_df = df[df[date_col] > train_end_ts].copy()
    return train_df, test_df


def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Baseline gradient-boosted tree classifier. Modest depth/learning rate
    to avoid overfitting on a small dataset; random_state fixed for
    reproducibility, which matters when you're presenting results.
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
    the F-beta score — recall weighted `beta` times as important as
    precision, matching the Stage 5 reasoning (missing a real delay costs
    more than a false alarm, so beta=2 by default).

    Return a dict with keys: "threshold", "f_beta", "precision", "recall".

    Steps:
      1. Get candidate (precision, recall, threshold) triples from
         sklearn.metrics.precision_recall_curve(y_true, y_proba). Note:
         it returns one MORE precision/recall pair than thresholds (the
         last point has no associated threshold — you can't act on it,
         so skip it).
      2. For each candidate threshold, compute the F-beta score:
             F_beta = (1 + beta^2) * precision * recall
                      / (beta^2 * precision + recall)
         Watch for the case where precision and recall are both 0 (empty
         denominator) — that candidate can't be the winner, so it's safe
         to skip it rather than divide by zero.
      3. Track and return the threshold with the highest F-beta, along with
         its precision, recall, and F-beta value.

    Why not just use sklearn's fbeta_score() directly in a loop over your
    own arbitrary threshold grid? You could — but precision_recall_curve
    already gives you every threshold where the prediction set actually
    changes, so you're not wasting compute checking thresholds that
    produce identical predictions to a neighboring one.
    """
    raise NotImplementedError("Your turn — see the steps in the docstring above.")
