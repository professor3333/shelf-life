"""Tests for model comparison, threshold choice and calibration.

Two kinds the component asks for specifically.

**Hand-computed metrics on a tiny fixed array.** Every expected value below was
worked out on paper and the arithmetic is written into the test, so a metric
wired to the wrong argument fails here rather than surviving into a report where
nothing can be checked against anything.

**The test-set discipline check.** `test_the_test_block_is_read_in_exactly_one_place`
parses `src/` and fails if the test split is read anywhere but the property that
defines it. It uses the AST rather than a grep so that a docstring *mentioning*
the test block is not mistaken for code reading it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from panels import DAY, WAVE0, make_panel

from src.data.split import Cuts, temporal_split
from src.models.evaluate import (
    calibration_summary,
    compare_models,
    cross_validate,
    paired_fold_difference,
    select,
    summarise_folds,
    threshold_sweep,
    wave_forward_folds,
    write_report,
)
from src.models.metrics import (
    average_precision,
    brier_score,
    confusion_at,
    evaluate,
    expected_calibration_error,
    threshold_for_budget,
)

# A five-row problem small enough to do by hand.
#
#   score  0.9   0.8   0.4   0.3   0.1
#   truth   1     0     1     0     0
#
# Sorted descending, two positives among five rows:
#   precision at each cut  1/1  1/2  2/3  2/4  2/5
#   recall at each cut     0.5  0.5  1.0  1.0  1.0
#   AP = (0.5 - 0)*1 + (0.5 - 0.5)*0.5 + (1.0 - 0.5)*(2/3) = 0.5 + 1/3 = 5/6
TRUTH = np.array([1, 0, 1, 0, 0])
SCORE = np.array([0.9, 0.8, 0.4, 0.3, 0.1])


# --- metrics, by hand ------------------------------------------------------


def test_average_precision_on_the_hand_computed_array():
    assert average_precision(TRUTH, SCORE) == pytest.approx(5 / 6)


def test_brier_on_the_hand_computed_array():
    # 0.01 + 0.64 + 0.36 + 0.09 + 0.01 = 1.11, over five rows.
    assert brier_score(TRUTH, SCORE) == pytest.approx(1.11 / 5)


def test_roc_auc_on_the_hand_computed_array():
    # Six positive/negative pairs; the positive ranks higher in five of them
    # (0.4 loses to 0.8). 5/6 — the same value as AP here, by coincidence.
    assert evaluate(TRUTH, SCORE)["roc_auc"] == pytest.approx(5 / 6)


def test_confusion_on_the_hand_computed_array():
    # At 0.5, two rows are flagged: 0.9 (a hit) and 0.8 (a false alarm).
    result = confusion_at(TRUTH, SCORE, threshold=0.5)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1.0, 1.0, 1.0, 2.0)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def test_threshold_for_budget_on_the_hand_computed_array():
    assert threshold_for_budget(SCORE, budget=2) == pytest.approx(0.8)
    assert int((SCORE >= threshold_for_budget(SCORE, 2)).sum()) == 2


def test_expected_calibration_error_on_the_hand_computed_array():
    # Ten bins of width 0.1 put each row in a bin of its own, so each bin's gap
    # is |score - truth|: 0.1, 0.8, 0.6, 0.3, 0.1. Mean 1.9/5.
    assert expected_calibration_error(TRUTH, SCORE, n_bins=10) == pytest.approx(1.9 / 5)


def test_a_constant_predictor_is_perfectly_calibrated_and_useless():
    """The reason ECE is never reported alone: predicting the base rate for
    every row scores 0.0, which is perfect and worthless."""
    truth = np.array([1] * 20 + [0] * 80)
    assert expected_calibration_error(truth, np.full(100, 0.2), n_bins=10) == pytest.approx(0.0)
    assert average_precision(truth, np.full(100, 0.2)) == pytest.approx(0.2)


def test_calibration_summary_reports_predicted_against_observed():
    summary = calibration_summary(TRUTH, SCORE)
    assert summary["mean_predicted"] == pytest.approx(0.5)
    assert summary["observed_rate"] == pytest.approx(0.4)
    assert summary["brier"] == pytest.approx(1.11 / 5)


# --- the test-set discipline check -----------------------------------------

SPLIT_MODULE = Path("src/data/split.py")


def _reads_the_test_block(path: Path) -> bool:
    """Does this module *read* the test split, as code rather than in prose?"""
    tree = ast.parse(path.read_text())
    docstrings = {
        ast.get_docstring(node, clean=False)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "test":
            return True
        if isinstance(node, ast.Constant) and node.value == "test" and node.value not in docstrings:
            return True
    return False


def test_the_test_block_is_read_in_exactly_one_place():
    """`CLAUDE`-independent statement of the rule: a test set looked at twice is
    a validation set. The single evaluation happens once the pipeline is frozen,
    which is not this component and not any module under `src/models/`."""
    readers = [path for path in sorted(Path("src").rglob("*.py")) if _reads_the_test_block(path)]
    assert readers == [SPLIT_MODULE], f"the test block is read in {readers}"


def test_the_discipline_check_would_notice_a_violation(tmp_path):
    """A guard that cannot fail is not a guard."""
    offender = tmp_path / "sneaky.py"
    offender.write_text("def cheat(split):\n    return split.test\n")
    assert _reads_the_test_block(offender)

    innocent = tmp_path / "honest.py"
    innocent.write_text(
        '"""We never read split.test here."""\n\n\ndef fine(split):\n    return split.val\n'
    )
    assert not _reads_the_test_block(innocent)


# --- rolling-origin folds ---------------------------------------------------


def _deep_split(**kwargs):
    frame = make_panel(n_waves=20, per_wave=60, positives_per_wave=6, random_labels=True, **kwargs)
    return temporal_split(frame, Cuts(WAVE0 + 9 * DAY, WAVE0 + 14 * DAY))


def test_folds_expand_and_never_train_on_the_future():
    split = _deep_split()
    folds = wave_forward_folds(split.train, split.embargo)
    assert folds, "the fixture must be deep enough to cut folds from"

    previous = -1
    for fold in folds:
        train_block = split.train.iloc[fold.train_index]
        val_block = split.train.iloc[fold.val_index]
        assert train_block["t"].max() < val_block["t"].min()
        assert val_block["t"].min() - train_block["t"].max() > split.embargo
        assert fold.n_train > previous  # expanding window, not sliding
        previous = fold.n_train


def test_folds_are_cut_on_waves_not_row_positions():
    """`TimeSeriesSplit` cuts on row index, which on a panel can divide a single
    crawl wave between train and validation. Every fold boundary here must fall
    between waves."""
    split = _deep_split()
    for fold in wave_forward_folds(split.train, split.embargo):
        train_block = split.train.iloc[fold.train_index]
        val_block = split.train.iloc[fold.val_index]
        shared = set(train_block["t"]) & set(val_block["t"])
        assert not shared


def test_folds_are_deterministic():
    split = _deep_split()
    first = wave_forward_folds(split.train, split.embargo)
    second = wave_forward_folds(split.train, split.embargo)
    assert [(f.n_train, f.n_val) for f in first] == [(f.n_train, f.n_val) for f in second]


def test_a_wider_embargo_yields_fewer_folds():
    split = _deep_split()
    narrow = wave_forward_folds(split.train, split.embargo)
    wide = wave_forward_folds(split.train, split.embargo * 3)
    assert len(wide) < len(narrow)


# --- cross-validation and error bars ---------------------------------------


def test_cross_validate_reports_one_row_per_fold():
    from sklearn.dummy import DummyClassifier

    from src.features.preprocessing import build_pipeline

    split = _deep_split()
    folds = wave_forward_folds(split.train, split.embargo)
    scored = cross_validate(
        lambda: build_pipeline(DummyClassifier(strategy="prior"), min_category_frequency=1),
        split.train,
        folds,
    )
    assert len(scored) == len(folds)
    assert list(scored["fold"]) == list(range(len(folds)))
    assert (scored["n_train"] > 0).all() and (scored["n_val"] > 0).all()


def test_a_fold_with_no_positives_is_nan_not_zero():
    """Zero would be averaged in as a bad score; NaN says the fold could not be
    scored, which is what actually happened."""
    per_fold = pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.4, float("nan"), 0.6]})
    summary = summarise_folds(per_fold)
    assert summary["folds"] == 3.0
    assert summary["folds_scored"] == 2.0
    assert summary["cv_pr_auc_mean"] == pytest.approx(0.5)


def test_a_single_fold_reports_no_standard_deviation():
    """One fold has no spread. Reporting 0.0 would read as 'no variance' rather
    than 'no information'."""
    summary = summarise_folds(pd.DataFrame({"fold": [0], "pr_auc": [0.4]}))
    assert np.isnan(summary["cv_pr_auc_sd"])


def test_the_paired_difference_is_paired():
    """Model A wins every fold by 0.05 while fold-to-fold scores swing by 0.4.
    Differencing within the fold must recover 0.05; subtracting two means would
    too, but the spread is what separates them — paired sd is 0, unpaired is
    large."""
    a = pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.20, 0.60, 0.40]})
    b = pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.15, 0.55, 0.35]})
    result = paired_fold_difference(a, b)
    assert result["mean_difference"] == pytest.approx(0.05)
    assert result["sd"] == pytest.approx(0.0)
    assert result["wins"] == 3.0
    assert a["pr_auc"].std(ddof=1) > 0.15  # the swing the pairing removed


def test_the_paired_difference_ignores_folds_either_model_could_not_score():
    a = pd.DataFrame({"fold": [0, 1], "pr_auc": [0.4, float("nan")]})
    b = pd.DataFrame({"fold": [0, 1], "pr_auc": [0.3, 0.9]})
    assert paired_fold_difference(a, b)["folds"] == 1.0


# --- selection --------------------------------------------------------------


def test_a_gap_inside_one_standard_deviation_is_reported_as_a_tie():
    summary = pd.DataFrame({"model": ["forest", "logistic"], "cv_pr_auc_mean": [0.44, 0.43]})
    per_fold = {
        "forest": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.60, 0.20, 0.52]}),
        "logistic": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.20, 0.60, 0.49]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] == "forest"
    assert verdict["separated"] is False
    assert "treat them as tied" in verdict["reason"]


def test_a_consistent_lead_is_reported_as_separated():
    summary = pd.DataFrame({"model": ["a", "b"], "cv_pr_auc_mean": [0.50, 0.30]})
    per_fold = {
        "a": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.50, 0.51, 0.49]}),
        "b": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.30, 0.31, 0.29]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] == "a"
    assert verdict["separated"] is True
    assert verdict["vs_runner_up_wins"] == 3.0


def test_selection_says_so_when_nothing_could_be_scored():
    summary = pd.DataFrame({"model": ["a"], "cv_pr_auc_mean": [float("nan")]})
    assert select(summary, {})["chosen"] is None


# --- threshold --------------------------------------------------------------


def test_the_threshold_sweep_flags_more_as_the_budget_grows():
    split = _deep_split()
    truth = split.val["y"].astype(int)
    rng = np.random.default_rng(0)
    scores = rng.random(len(truth))
    table = threshold_sweep(truth, scores, n_days=1, budgets=(5, 10, 20, 50))
    assert table["alerts"].is_monotonic_increasing
    assert table["threshold"].is_monotonic_decreasing
    assert (table["tp"] + table["fn"] == truth.sum()).all()


def test_the_sweep_covers_the_chosen_budget_so_the_choice_is_defended():
    from src.models.evaluate import BUDGET_SWEEP
    from src.models.metrics import DEFAULT_ALERT_BUDGET

    assert DEFAULT_ALERT_BUDGET in BUDGET_SWEEP


# --- end to end -------------------------------------------------------------


def test_compare_models_reports_cv_and_validation_for_every_candidate():
    split = _deep_split()
    summary, per_fold, val_scores = compare_models(split)
    assert set(summary["model"]) == set(per_fold) == set(val_scores)
    assert {"cv_pr_auc_mean", "cv_pr_auc_sd", "val_pr_auc", "val_ece"} <= set(summary.columns)
    for scores in val_scores.values():
        assert len(scores) == len(split.val)


def test_the_report_records_the_blocker_when_nothing_could_run(tmp_path):
    path = tmp_path / "model_comparison.md"
    write_report(
        path,
        make_panel(n_waves=4),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        blocker="No split yet.",
    )
    text = path.read_text()
    assert "No split yet." in text
    assert "ROC-AUC is reported but not decisive" in text
