"""
Feature engineering for the night-before delay-risk model.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_calendar_features(df: pd.DataFrame, date_col: str = "FlightDate") -> pd.DataFrame:
    """
    @func: Adds calendar features to the DataFrame.
    @param df: The input DataFrame containing the data.
    @param date_col: Column name for the date values.
    @return: DataFrame with new columns for month, day of week, is weekend,"""
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
    @func: Calculates a cumulative, running statistic across a timeline, calculated independently for each entity or category.
    @param df: The input DataFrame containing the data.
    @param group_cols: List of columns to group by.
    @param date_col: Column name for the date values.
    @param target_col: Column name for the target values.
    @param min_history: Minimum number of historical observations required.
    @return: DataFrame with new columns for the historical rate and count of the target variable, calculated for each group.
    """
    out = df.copy()
    out.sort_values(by=date_col, inplace=True)
    
    out[f"{'_'.join(group_cols)}_hist_rate"] = out.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).expanding().mean())
    out[f"{'_'.join(group_cols)}_hist_count"] = out.groupby(group_cols)[target_col].transform(lambda s: s.shift(1).expanding().count())

    out = out.sort_index()  # Restore original order
    return out
    

def add_historical_delay_features(
    df: pd.DataFrame,
    date_col: str = "FlightDate",
    target_col: str = "delayed",
    min_history: int = 20,
) -> pd.DataFrame:
    """
    @func: Adds historical delay features to the DataFrame.
    @param df: The raw flight dataset containing identifiers, scheduled times, dates, and historical delay outcomes.
    @param date_col: The chronological anchor (default "FlightDate"). Ensures we never calculate historical stats using future or same-day data.
    @param target_col: The binary label (default "delayed", where 1 = delayed >= 15 min, 0 = on time).
    @param min_history: Minimum number of historical observations required.
    @return: DataFrame with historical delay features added.
    """
    # output df is copy of df to avoid mutating original df
    out = df.copy()
    # Adding delay column for flights delayed over 15 minutes
    if "ArrDelayMinutes" in out.columns:
      out["delayed"] = (out["ArrDelayMinutes"] >= 15).astype(int)
    # Adding time of day feature based on scheduled departure time
    if "CRSDepTime" in out.columns:
      # Changing out["CRSDepTime"] to time object
      out["CRSDepTime"] = pd.to_datetime(out["CRSDepTime"], format="%H%M").dt.time
      # Categorizing time_of_day to morning or evening by using a categorizing function and df.apply(func)
      out["time_of_day"] = out["CRSDepTime"].apply(lambda time: "morning" if time.hour < 12 else "evening")
    # Generating rate and frequency for each feature
      # Tail_Number
    out = expanding_group_rate(out, ["Tail_Number"], date_col, target_col)
      # Route ["Origin", "Dest"]
    out = expanding_group_rate(out, ["Origin", "Dest"], date_col, target_col)
      # Carrier
    out = expanding_group_rate(out, ["Reporting_Airline"], date_col, target_col)
      # Time of day
    out = expanding_group_rate(out, ["time_of_day"], date_col, target_col)
    # Global expanding rate
    global_rate = out[target_col].shift(1).expanding().mean()

    # Numpy.select to implement hierarchy
      # Conditions hierarchy, Tail -> Route -> Carrier -> Time of Day
      # Each frequency has to be greater than min_history
    cond_list = [
        out["Tail_Number_hist_count"] >= min_history,
        out["Origin_Dest_hist_count"] >= min_history,
        out["Reporting_Airline_hist_count"] >= min_history,
        out["time_of_day_hist_count"] >= min_history
    ]
    choice_list = [
        out["Tail_Number_hist_rate"],
        out["Origin_Dest_hist_rate"],
        out["Reporting_Airline_hist_rate"],
        out["time_of_day_hist_rate"]
    ]
      # Fall back to global rate (default) if none of the conditions above are True
    default_choice = global_rate
    out["historical_delay_rate"] = np.select(cond_list, choice_list, default_choice)

    # Returning relevant columns only
    relevant_cols = list(df.columns) + ["historical_delay_rate"]

    return out[relevant_cols]
    # raise NotImplementedError("Your turn — combine expanding_group_rate() calls per the hierarchy.")
