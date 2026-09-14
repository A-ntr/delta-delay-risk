"""
Feature engineering for the night-before delay-risk model.

Two difficulty levels here on purpose:
  - add_calendar_features() is fully implemented — use it as the pattern/style
    to follow for the rest.
  - expanding_group_rate() and add_historical_delay_features() are YOUR
    exercise. Run `pytest tests/test_features.py` as you go — the tests
    encode the expected behavior we designed together, so they'll tell you
    if your logic is leakage-safe without me handing you the answer.
"""
from __future__ import annotations
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str = "FlightDate") -> pd.DataFrame:
    """Fully implemented example — night-before-knowable calendar features."""
    out = df.copy()
    out["month"] = out[date_col].dt.month
    out["day_of_week"] = out[date_col].dt.dayofweek  # 0=Mon
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_winter"] = out["month"].isin([12, 1, 2]).astype(int)
    return out


def expanding_group_rate(
    df: pd.DataFrame,
    group_cols: list[str],
    date_col: str,
    target_col: str,
    min_history: int = 20,
) -> pd.DataFrame:
    """
    For each row, compute two things using ONLY rows from strictly earlier
    dates within the same group (never same-day or future rows — that's the
    leakage rule from Stage 2):

      - f"{'_'.join(group_cols)}_hist_rate": the group's mean of target_col
        over all prior dates (NaN if there's no prior history at all for
        that group).
      - f"{'_'.join(group_cols)}_hist_count": how many prior rows existed
        for that group (used later to decide if there's ENOUGH history to
        trust this rate, per min_history).

    Think about *why* a naive `df.groupby(group_cols)[target_col].mean()`
    over the whole dataframe would be wrong here — that's exactly the leakage
    bug from Stage 2. You want each row's value computed as of that row's
    own date.

    Hints (not a solution):
    - Sort by date_col first.
    - pandas' `.groupby(group_cols)[target_col].expanding().mean()` computes
      a running mean INCLUDING the current row. You need it to exclude the
      current row — what shifts the window back by one to fix that?
    - `.expanding().count()` gives you the count needed for hist_count,
      with the same shift-by-one consideration.
    - Watch your index alignment when assigning the result back onto `out`
      after a groupby — reset_index or align on the original df's index.

    Returns a new DataFrame with the two new columns added (don't mutate df).
    """
    out = df.copy()
    out.sort_values(by=date_col, inplace=True)
    
    out[f"{'_'.join(group_cols)}_hist_rate"] = out.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).expanding().mean())
    out[f"{'_'.join(group_cols)}_hist_count"] = out.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).expanding().count())
    
    return out
    # raise NotImplementedError("Your turn — implement using the hints above.")
    


def add_historical_delay_features(
    df: pd.DataFrame,
    date_col: str = "FlightDate",
    target_col: str = "delayed",
    min_history: int = 20,
) -> pd.DataFrame:
    """
    Build the full fallback-hierarchy feature:
        tail -> route -> carrier -> time_of_day -> global

    For each row, walk the hierarchy from most specific to least specific and
    use the FIRST level that has at least `min_history` prior rows for that
    group. The global average (computed the same leakage-safe, prior-dates-
    only way) is the guaranteed last resort.

    Steps to implement:
      1. Call expanding_group_rate() once per grouping level: ["Tail_Number"],
         ["Origin", "Dest"] (route), ["Reporting_Airline"] (carrier), and
         whatever column represents "time of day" in this df.
      2. Also compute a global expanding rate with an empty group (i.e. the
         whole dataset's running average, prior-dates-only).
      3. For each row, pick the rate from the most specific level whose
         hist_count >= min_history; if none qualify, use global.
      4. Return df with one new column: "historical_delay_rate".

    This is the function whose design we walked through step by step in
    Stage 2 — the logic should already be clear in your head; this is about
    translating that reasoning into pandas operations.
    """
    raise NotImplementedError("Your turn — combine expanding_group_rate() calls per the hierarchy.")
