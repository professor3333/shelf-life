"""Tests for the baseline ladder.

The one the component asks for is
`test_the_fitted_dummy_reproduces_the_analytic_values`: a model whose answer can
be computed by hand, so that a wrongly wired metric is caught here rather than
in the evaluation component where nothing can be checked against arithmetic.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from panels import DAY, WAVE0, make_panel

from src.data.split import Cuts, temporal_split
from src.features.preprocessing import features_and_target, fit_on_training_fold
from src.models.baselines import BoardHazardBaseline
from src.models.metrics import evaluate
from src.models.train_baseline import (
    LADDER,
    analytic_reference,
    prediction_days,
    run_ladder,
    score_rung,
    write_report,
)


def _split(frame: pd.DataFrame | None = None):
    frame = make_panel() if frame is None else frame
    return temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


# --- the assertion the component asks for ---------------------------------


def test_the_fitted_dummy_reproduces_the_analytic_values():
    """For a constant predictor: average precision **is** the base rate, ROC-AUC
    is exactly 0.5 (undefined here, since every score ties), and the Brier score
    of predicting `p` is `p(1-p)`. If the fitted dummy disagrees with the
    arithmetic, the metric code is wrong — and this is the last place it can be
    checked against a number computed by hand."""
    split = _split()
    rung = next(rung for rung in LADDER if rung.name == "prior")
    summary, scores = score_rung(rung, split, budget_per_day=20)

    validation = split.val
    base_rate = float((validation["y"] == 1).mean())
    train_rate = float((split.train["y"] == 1).mean())

    # The dummy predicts the *training* prior, which is what it learned.
    assert scores.nunique() == 1
    assert scores.iloc[0] == pytest.approx(train_rate)

    # And the suite reproduces the arithmetic on the validation block.
    assert summary["base_rate"] == pytest.approx(base_rate)
    assert summary["pr_auc"] == pytest.approx(base_rate)
    assert summary["brier"] == pytest.approx(
        base_rate * (1 - train_rate) ** 2 + (1 - base_rate) * train_rate**2
    )


def test_the_analytic_reference_matches_a_hand_count():
    frame = make_panel(n_waves=4, per_wave=40)  # 2 positives per wave by construction
    reference = analytic_reference(frame)
    assert reference["n"] == 160
    assert reference["positives"] == 8
    assert reference["base_rate"] == pytest.approx(0.05)
    assert reference["pr_auc"] == pytest.approx(0.05)
    assert reference["brier"] == pytest.approx(0.05 * 0.95)
    assert reference["roc_auc"] == 0.5


# --- the ladder ------------------------------------------------------------


def test_every_rung_runs_and_is_reported_once():
    results, scores = run_ladder(_split())
    assert list(results["model"]) == [rung.name for rung in LADDER]
    assert set(scores) == {rung.name for rung in LADDER}
    for name, rung_scores in scores.items():
        assert rung_scores.notna().all(), name
        assert ((rung_scores >= 0) & (rung_scores <= 1)).all(), name


def test_no_rung_beats_the_base_rate_when_the_label_is_noise():
    """On a panel whose label is drawn independently of every feature, no rung
    can do much better than the base rate. A rung that does is finding structure
    that is not in the data, which would mean the plumbing is leaking the target.

    The default fixture will not do for this: its categoricals encode `index` by
    the Chinese remainder theorem and its label is a function of `index`, so a
    forest reaches PR-AUC 1.0 on it honestly."""
    frame = make_panel(per_wave=200, positives_per_wave=10, random_labels=True)
    split = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    results, _ = run_ladder(split)
    base_rate = float((split.val["y"] == 1).mean())
    assert (results["pr_auc"] < base_rate + 0.20).all(), results[["model", "pr_auc"]]


def test_the_ladder_is_deterministic():
    first, _ = run_ladder(_split())
    second, _ = run_ladder(_split())
    pd.testing.assert_frame_equal(first, second)


def test_the_test_block_is_never_read():
    """Opened once, at the end of the build. Corrupting every label in the test
    block must leave every reported number identical."""
    split = _split()
    before, _ = run_ladder(split)

    corrupted = split.frame.copy()
    is_test = corrupted["split"] == "test"
    corrupted.loc[is_test, "y"] = 1 - corrupted.loc[is_test, "y"].astype(int)
    after, _ = run_ladder(dataclasses.replace(split, frame=corrupted))

    pd.testing.assert_frame_equal(before, after)


def test_no_rung_is_fitted_on_anything_but_the_training_fold():
    """Checked on the one rung whose learned state is a number you can compare:
    the board hazard is a per-source mean, so it either equals the training
    fold's or it does not."""
    split = _split()
    model = fit_on_training_fold(BoardHazardBaseline(), split)

    train_rates = split.train.groupby("source")["y"].mean().to_dict()
    everything = split.frame.groupby("source")["y"].mean().to_dict()
    for source, rate in model.rates_.items():
        assert rate == pytest.approx(train_rates[source])
    assert model.pooled_rate_ == pytest.approx(float(split.train["y"].mean()))
    assert model.pooled_rate_ != pytest.approx(float(split.frame["y"].mean())) or (
        train_rates == everything
    )


def test_prediction_days_counts_calendar_days_not_rows():
    split = _split()
    assert prediction_days(split.val) == split.val["t"].dt.date.nunique()


# --- the board-hazard baseline --------------------------------------------


def test_board_hazard_predicts_each_boards_training_rate():
    frame = make_panel(n_waves=9, per_wave=40)
    frame.loc[frame["source_id"].isin(["p0", "p1"]), "source"] = "greenhouse:other"
    split = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))

    model = fit_on_training_fold(BoardHazardBaseline(), split)
    features, _ = features_and_target(split.val)
    predicted = pd.Series(model.predict_proba(features)[:, 1], index=split.val.index)

    for source, block in split.val.groupby("source"):
        assert predicted.loc[block.index].nunique() == 1
        assert predicted.loc[block.index].iloc[0] == pytest.approx(model.rates_[source])


def test_board_hazard_falls_back_to_the_pooled_rate_for_an_unseen_board():
    split = _split()
    model = fit_on_training_fold(BoardHazardBaseline(), split)
    unseen = split.val.head(3).copy()
    unseen["source"] = "greenhouse:never-seen"
    assert np.allclose(model.predict_proba(unseen)[:, 1], model.pooled_rate_)


# --- the report ------------------------------------------------------------


def test_the_report_records_the_blocker_when_the_ladder_cannot_run(tmp_path):
    """On a shallow panel this is the normal outcome, and the report has to say
    so rather than quietly omit the ladder."""
    frame = make_panel(n_waves=4)
    path = tmp_path / "baseline_results.md"
    write_report(path, frame, None, None, None, blocker="No honest split exists yet.")

    text = path.read_text()
    assert "The number every model must beat" in text
    assert "No honest split exists yet." in text
    assert "0.05" in text  # the base rate, computed from the frame


def test_the_report_carries_the_ladder_and_both_breakdowns(tmp_path):
    split = _split()
    results, scores = run_ladder(split)
    path = tmp_path / "baseline_results.md"
    write_report(path, make_panel(), split, results, scores, blocker=None)

    text = path.read_text()
    for rung in LADDER:
        assert rung.name in text
    assert "Per source" in text
    assert "Carried-over postings against unseen ones" in text
    assert "Calibration" in text
    assert "test" not in text.split("## The ladder")[1].split("## Per source")[0]


def test_evaluate_is_reported_on_validation_not_train():
    split = _split()
    rung = next(rung for rung in LADDER if rung.name == "prior")
    summary, _ = score_rung(rung, split, budget_per_day=20)
    assert summary["n"] == len(split.val)
    assert summary["n"] != len(split.train)
    assert summary["positives"] == float((split.val["y"] == 1).sum())


def test_the_suite_reported_for_a_rung_carries_no_accuracy():
    split = _split()
    summary, _ = score_rung(LADDER[0], split, budget_per_day=20)
    assert not any("accur" in key.lower() for key in summary)


def test_evaluate_on_an_empty_positive_block_is_undefined_not_zero():
    truth = np.zeros(50)
    assert np.isnan(evaluate(truth, np.full(50, 0.1))["pr_auc"])
