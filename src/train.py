"""
End-to-end: load data -> engineer features -> chronological split ->
train baseline XGBoost -> pick F2-optimal threshold -> report.

Won't run until implemented best_fbeta_threshold() in src/model.py
"""
import os
import pandas as pd
from src.make_sample_data import make_sample
from src.features import add_calendar_features, add_historical_delay_features
from src.model import chronological_split, train_baseline_model, best_fbeta_threshold

DATA_PATH = "data/raw/sample_flights.csv"
FEATURE_COLS = ["month", "day_of_week", "is_weekend", "is_winter", "Distance", "historical_delay_rate"]


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("No sample data found — generating it...")
        df = make_sample()
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    df = pd.read_csv(DATA_PATH, parse_dates=["FlightDate"])
    return df


def main():
    df = load_data()
    df["delayed"] = (df["ArrDelayMinutes"] >= 15).astype(int)

    df = add_calendar_features(df)
    df = add_historical_delay_features(df, target_col="delayed")

    # Train on year 1, test on year 2 — full seasonal cycles in both,
    # per the Stage 2 chronological-split reasoning.
    train_df, test_df = chronological_split(df, "FlightDate", train_end="2023-12-31")
    print(f"Train: {len(train_df):,} rows ({train_df['FlightDate'].min().date()} to {train_df['FlightDate'].max().date()})")
    print(f"Test:  {len(test_df):,} rows ({test_df['FlightDate'].min().date()} to {test_df['FlightDate'].max().date()})")

    X_train, y_train = train_df[FEATURE_COLS], train_df["delayed"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["delayed"]

    model = train_baseline_model(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = best_fbeta_threshold(y_test.values, y_proba, beta=2.0)
    print("\n--- F2-optimal operating point ---")
    print(f"threshold={result['threshold']:.3f}  precision={result['precision']:.3f}  "
          f"recall={result['recall']:.3f}  f2={result['f_beta']:.3f}")

    n_high = (y_proba >= result["threshold"]).sum()
    print(f"\n{n_high:,} of {len(test_df):,} test-set flights ({n_high/len(test_df)*100:.1f}%) "
          f"would be flagged 'high risk' at this threshold.")


if __name__ == "__main__":
    main()
