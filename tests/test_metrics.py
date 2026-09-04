"""Tests for the metric suite.

Two kinds. **Analytic**: a constant predictor's scores can be computed by hand,
so they are, and the implementation must reproduce them. **Differential**:
`average_precision` is implemented here rather than imported, so it is checked
against scikit-learn on random inputs — if the two agree over a few hundred
random problems, the hand-rolled version is the definition and not a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src.models.metrics import (
    alert_budget,
    average_precision,
    brier_score,
    confusion_at,
    evaluate,
    evaluate_by,
    reliability_curve,
    threshold_for_budget,
)


def test_average_precision_matches_sklearn_on_random_problems():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(5, 200))
        truth = rng.integers(0, 2, size=n)
        if truth.sum() == 0:
            continue
        score = rng.random(n)
        assert average_precision(truth, score) == pytest.approx(
            average_precision_score(truth, score)
        )


def test_average_precision_handles_ties_the_same_way_sklearn_does():
    """Tied scores are the case a constant predictor is made of, and the case a
    naive implementation gets wrong by letting row order decide."""
    rng = np.random.default_rng(1)
    for _ in range(100):
        n = int(rng.integers(10, 100))
        truth = rng.integers(0, 2, size=n)
        if truth.sum() == 0:
            continue
        score = rng.integers(0, 3, size=n).astype(float)  # heavy ties
        assert average_precision(truth, score) == pytest.approx(
            average_precision_score(truth, score)
        )


def test_a_constant_predictor_scores_exactly_the_base_rate():
    truth = np.array([1] * 7 + [0] * 93)
    for constant in (0.0, 0.01, 0.5, 0.99):
        assert average_precision(truth, np.full(100, constant)) == pytest.approx(0.07)


def test_a_perfect_ranking_scores_one():
    truth = np.array([1, 1, 0, 0, 0])
    assert average_precision(truth, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)


def test_average_precision_is_undefined_with_no_positives():
    """Undefined, and saying so beats returning zero — a zero would read as a
    real score and quietly rank a model below a coin flip."""
    assert np.isnan(average_precision(np.zeros(10), np.random.default_rng(0).random(10)))


def test_brier_of_predicting_the_base_rate_is_p_times_one_minus_p():
    for base_rate in (0.01, 0.1, 0.5):
        n = 10_000
        truth = np.zeros(n)
        truth[: int(base_rate * n)] = 1
        assert brier_score(truth, np.full(n, base_rate)) == pytest.approx(
            base_rate * (1 - base_rate)
        )


def test_the_budget_threshold_flags_exactly_the_budget():
    rng = np.random.default_rng(2)
    score = rng.random(500)
    for budget in (1, 20, 137, 500):
        threshold = threshold_for_budget(score, budget)
        assert int((score >= threshold).sum()) == budget


def test_the_budget_scales_with_the_days_in_the_block_and_is_capped_by_the_rows():
    assert alert_budget(n_rows=1000, n_days=3, budget_per_day=20) == 60
    assert alert_budget(n_rows=40, n_days=3, budget_per_day=20) == 40


def test_confusion_counts_are_the_four_cells_and_they_sum_to_n():
    truth = np.array([1, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1])
    result = confusion_at(truth, score, threshold=0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1.0, 1.0, 1.0, 3.0)
    assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == len(truth)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_accuracy_is_never_reported():
    """At a 1.2% base rate, always predicting 'stays open' scores 98.8%. There is
    no threshold at which that number carries information, so it must not be
    available as a convenience."""
    truth = np.array([1] * 5 + [0] * 95)
    reported = evaluate(truth, np.full(100, 0.05))
    assert "accuracy" not in reported
    assert not any("accur" in key.lower() for key in reported)


def test_evaluate_reproduces_the_analytic_constant_predictor_values():
    truth = np.array([1] * 12 + [0] * 988)
    reported = evaluate(truth, np.full(1000, 0.012), n_days=1)
    assert reported["base_rate"] == pytest.approx(0.012)
    assert reported["pr_auc"] == pytest.approx(0.012)
    assert np.isnan(reported["roc_auc"]) or reported["roc_auc"] == pytest.approx(0.5)
    assert reported["brier"] == pytest.approx(0.012 * 0.988)


def test_reliability_curve_bins_sum_to_the_rows_and_recovers_a_calibrated_model():
    rng = np.random.default_rng(3)
    predicted = rng.random(20_000)
    truth = (rng.random(20_000) < predicted).astype(int)  # perfectly calibrated by construction
    curve = reliability_curve(truth, predicted, n_bins=10)
    assert curve["n"].sum() == 20_000
    populated = curve[curve["n"] > 100]
    assert np.allclose(populated["mean_predicted"], populated["observed_rate"], atol=0.02)


def test_evaluate_by_reports_one_row_per_group():
    frame = pd.DataFrame(
        {
            "y": [1, 0, 0, 1, 0, 0],
            "source": ["a", "a", "a", "b", "b", "b"],
            "t": pd.to_datetime(["2026-09-01"] * 6, utc=True),
        }
    )
    result = evaluate_by(frame, np.array([0.9, 0.1, 0.2, 0.7, 0.3, 0.1]), group="source")
    assert list(result["source"]) == ["a", "b"]
    assert list(result["n"]) == [3.0, 3.0]
    assert all(result["pr_auc"] == 1.0)  # the positive ranks first within each group
