"""
Run with: pytest tests/test_features.py -v

These tests encode the leakage rules we designed in Stage 2. If your
implementation passes all of these, it's leakage-safe and the fallback
hierarchy is working as intended. If a test fails, the assertion message
tells you which rule was violated — use that to debug your own logic rather
than guessing.
"""
import pandas as pd
import pytest
from src.features import expanding_group_rate, add_historical_delay_features


@pytest.fixture
def toy_df():
    return pd.DataFrame({
        "FlightDate": pd.to_datetime([
            "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
        ]),
        "Tail_Number": ["A", "A", "B", "C"],
        "Origin": ["JFK", "JFK", "JFK", "LAX"],
        "Dest": ["ATL", "ATL", "ATL", "SEA"],
        "Reporting_Airline": ["DL", "DL", "DL", "UA"],
        "time_of_day": ["morning", "morning", "morning", "evening"],
        "delayed": [1, 0, 1, 0],
    })


class TestExpandingGroupRate:
    def test_first_ever_row_has_no_history(self, toy_df):
        """The very first row chronologically has zero prior rows for ANY
        group — hist_count must be 0 and hist_rate must be NaN."""
        result = expanding_group_rate(toy_df, ["Tail_Number"], "FlightDate", "delayed")
        row0 = result.iloc[0]
        assert row0["Tail_Number_hist_count"] == 0
        assert pd.isna(row0["Tail_Number_hist_rate"])

    def test_excludes_same_day_row_not_just_future(self, toy_df):
        """Row 2 (tail A, day 2) must use ONLY day 1's outcome (delayed=1),
        never its own day's outcome. If your implementation accidentally
        includes the current row, this will fail — that's the leakage bug
        from Stage 2 showing up mechanically."""
        result = expanding_group_rate(toy_df, ["Tail_Number"], "FlightDate", "delayed")
        row1 = result.iloc[1]  # tail A, day 2
        assert row1["Tail_Number_hist_count"] == 1
        assert row1["Tail_Number_hist_rate"] == pytest.approx(1.0)

    def test_never_uses_future_rows(self, toy_df):
        """Row 0 (tail A, day 1) must NOT be influenced by day 2's outcome
        even though it's the same tail. Prior-only, no exceptions."""
        result = expanding_group_rate(toy_df, ["Tail_Number"], "FlightDate", "delayed")
        row0 = result.iloc[0]
        assert row0["Tail_Number_hist_count"] == 0  # not 1 — day 2 hasn't happened yet


class TestFallbackHierarchy:
    def test_thin_tail_falls_back_to_route(self, toy_df):
        """Tail 'B' (day 3) has zero prior history of its own, but its route
        (JFK-ATL) has 2 prior rows (both from tail A) — with min_history=2,
        this should use the ROUTE rate (mean of [1, 0] = 0.5), not tail."""
        result = add_historical_delay_features(toy_df, min_history=2)
        row2 = result.iloc[2]
        assert row2["historical_delay_rate"] == pytest.approx(0.5)

    def test_everything_thin_falls_back_to_global(self, toy_df):
        """Tail 'C' (day 4) is brand new on a brand new route with a brand
        new carrier at a never-seen time-of-day. Only the global average
        (3 prior rows: [1, 0, 1] -> 0.667) has any history at all."""
        result = add_historical_delay_features(toy_df, min_history=2)
        row3 = result.iloc[3]
        assert row3["historical_delay_rate"] == pytest.approx(2 / 3, rel=1e-2)

    def test_global_is_last_resort_even_below_min_history(self, toy_df):
        """Row 1 (tail A, day 2): every level (tail/route/carrier/tod/global)
        only has 1 prior row — below min_history=2. But global is the
        guaranteed last resort, so it should still be used (rate=1.0) rather
        than returning NaN, since ops needs a number for every flight."""
        result = add_historical_delay_features(toy_df, min_history=2)
        row1 = result.iloc[1]
        assert row1["historical_delay_rate"] == pytest.approx(1.0)

    def test_true_cold_start_can_be_nan(self, toy_df):
        """Row 0 is the very first row in the entire dataset — there is
        NO prior data at any level, including global. NaN is the honest
        answer here; a real pipeline would impute this with a fixed prior
        (e.g. the industry-wide historical average) downstream."""
        result = add_historical_delay_features(toy_df, min_history=2)
        row0 = result.iloc[0]
        assert pd.isna(row0["historical_delay_rate"])
