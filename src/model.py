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
    @param df: The input DataFrame containing the date column and features.
    @param date_col: The name of the column containing dates.
    @param train_end: The end date for the training set (inclusive).
    @return: A tuple containing the training and test DataFrames.
    """
    train_end_ts = pd.Timestamp(train_end)
    train_df = df[df[date_col] <= train_end_ts].copy()
    test_df = df[df[date_col] > train_end_ts].copy()
    return train_df, test_df

def chronological_split_with_validation_set(
    df: pd.DataFrame, date_col: str, train_end: str, validation_end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training, validation, and test sets based on chronological order.
    Rows on or before train_end go to the training set, rows between train_end and validation_end
    go to the validation set, and rows after validation_end go to the test set.
    @param df: The input DataFrame containing the date column and features.
    @param date_col: The name of the column containing dates.
    @param train_end: The end date for the training set (inclusive).
    @param validation_end: The end date for the validation set (inclusive).
    @return: A tuple containing the training, validation, and test DataFrames.
    """
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    train_df = df[df[date_col] <= train_end_ts].copy()
    test_df = df[df[date_col] > validation_end_ts].copy()
    validation_df = df[(df[date_col] > train_end_ts) & (df[date_col] <= validation_end_ts)].copy()
    test_df = test_df[test_df[date_col] > validation_end_ts]
    return train_df, validation_df, test_df

# Consider the use of eval_set for early stopping in XGBClassifier
def train_model_with_early_stopping(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> XGBClassifier:
    """
    Train an XGBClassifier with early stopping using the eval_set parameter.
    Look into n_estimators and early_stopping_rounds for tuning the model.
    The early stopping is based on the evaluation set performance.
    The model will stop training early if the evaluation set performance does not improve for a specified number of rounds.
    The early_stopping_rounds parameter controls how many rounds without improvement are allowed before training is stopped.
    Max_depth and learning_rate can also be tuned to control the model complexity and training speed.
    @param X_train: The training features DataFrame.
    @param y_train: The training target Series.
    @param X_eval: The evaluation features DataFrame.
    @param y_eval: The evaluation target Series.
    @return: The trained XGBClassifier model with early stopping.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        early_stopping_rounds=20,
        random_state=42,
        tree_method="hist"
    )

    model.fit(
        X_train, y_train, 
        eval_set=[(X_eval, y_eval)], 
        verbose=50
    )
    
    return model

def train_baseline_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """
    Baseline gradient-boosted tree classifier. Modest depth/learning rate
    to avoid overfitting on a small dataset; random_state fixed for
    reproducibility.
    Look into a higher n_estimators if the model is underfitting.
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
    @param y_true: The true binary labels.
    @param y_proba: The predicted probabilities for the positive class.
    @param beta: The beta value for the F-beta score (default is 2.0).
    @return: A dict with keys: "threshold", "f_beta", "precision", "recall".
    """
    # Computing precision-recall pairs for different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Dropping the last precision-recall pair to match the number of thresholds
    precisions = precisions[:-1]
    recalls = recalls[:-1]

    # Calculating the F-beta score for each threshold
    f_beta_scores = ((1 + beta**2) * (precisions * recalls)) / ((beta**2 * precisions) + recalls + 1e-10) # Avoid division by zero by adding tiny epsilon

    # Locating maximum f-beta score, threshold, corresponding precision, and recall
    best_idx = np.argmax(f_beta_scores)
    best_threshold = thresholds[best_idx]
    best_f_beta = f_beta_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    return {
        "threshold": best_threshold,
        "f_beta": best_f_beta,
        "precision": best_precision,
        "recall": best_recall
    }