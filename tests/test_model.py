"""
Run with: pytest tests/test_model.py -v

The toy dataset in TestBestFbetaThreshold was brute-force verified (not hand
calculated) to have DIFFERENT optimal thresholds for beta=1 vs beta=2 — this
is deliberate. If implementation just reimplements plain F1 by mistake
(e.g. hardcoding beta somewhere, or a formula bug), the beta=2 test will
catch it because the two expected thresholds are genuinely different.
"""
import numpy as np
import pandas as pd
import pytest
from src.model import chronological_split, best_fbeta_threshold


class TestChronologicalSplit:
    def test_splits_on_date_not_randomly(self):
        df = pd.DataFrame({
            "FlightDate": pd.to_datetime(
                ["2023-06-01", "2023-12-31", "2024-01-01", "2024-06-01"]
            ),
            "x": [1, 2, 3, 4],
        })
        train, test = chronological_split(df, "FlightDate", train_end="2023-12-31")
        assert list(train["x"]) == [1, 2]
        assert list(test["x"]) == [3, 4]

    def test_no_overlap_between_train_and_test(self):
        df = pd.DataFrame({
            "FlightDate": pd.to_datetime(pd.date_range("2023-01-01", periods=100)),
            "x": range(100),
        })
        train, test = chronological_split(df, "FlightDate", train_end="2023-03-01")
        assert train["FlightDate"].max() <= test["FlightDate"].min()
        assert len(train) + len(test) == len(df)


class TestBestFbetaThreshold:
    # 5 positives / 15 negatives. Verified by brute-force sweep (not hand
    # math) that beta=1 and beta=2 land on genuinely different thresholds.
    Y_TRUE = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Y_PROBA = np.array([
        0.92, 0.81, 0.65, 0.55, 0.30,
        0.60, 0.52, 0.48, 0.40, 0.35,
        0.25, 0.20, 0.18, 0.15, 0.12,
        0.10, 0.08, 0.06, 0.04, 0.02,
    ])

    def test_f1_prefers_the_higher_precision_threshold(self):
        """At beta=1 (precision and recall weighted equally), the optimal
        cutoff is 0.55: 4 true positives, 1 false positive, 1 false
        negative -> precision=recall=0.8."""
        result = best_fbeta_threshold(self.Y_TRUE, self.Y_PROBA, beta=1.0)
        assert result["threshold"] == pytest.approx(0.55, abs=1e-6)
        assert result["precision"] == pytest.approx(0.8, abs=1e-6)
        assert result["recall"] == pytest.approx(0.8, abs=1e-6)
        assert result["f_beta"] == pytest.approx(0.8, abs=1e-6)

    def test_f2_prefers_the_higher_recall_threshold(self):
        """At beta=2 (recall weighted 2x precision — our actual business
        preference from Stage 5), the optimal cutoff drops to 0.30: it
        catches every real delay (recall=1.0) at the cost of precision
        dropping to 0.5. This is a DIFFERENT threshold than beta=1 picked —
        that's the point of this test."""
        result = best_fbeta_threshold(self.Y_TRUE, self.Y_PROBA, beta=2.0)
        assert result["threshold"] == pytest.approx(0.30, abs=1e-6)
        assert result["recall"] == pytest.approx(1.0, abs=1e-6)
        assert result["precision"] == pytest.approx(0.5, abs=1e-6)
        assert result["f_beta"] == pytest.approx(0.8333, abs=1e-3)

    def test_beta1_and_beta2_pick_different_thresholds(self):
        """Sanity check on the property itself, independent of the exact
        numbers above: weighting recall higher should never pick a MORE
        conservative (higher) threshold than weighting it equally."""
        r1 = best_fbeta_threshold(self.Y_TRUE, self.Y_PROBA, beta=1.0)
        r2 = best_fbeta_threshold(self.Y_TRUE, self.Y_PROBA, beta=2.0)
        assert r2["threshold"] <= r1["threshold"]
        assert r2["recall"] >= r1["recall"]
