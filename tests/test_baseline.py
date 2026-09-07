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


# --- rules a person could follow ---------------------------------------------


def _aged(ages: list[float], labels: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.DataFrame({"age_days": ages, "y": labels})
    return frame, frame["y"]


def test_a_rule_predicts_each_groups_training_rate():
    """A person holding a rule holds a rate with it. Two groups, two rates,
    computed on the fold the model was fitted on and nowhere else."""
    from src.models.baselines import RuleBaseline, older_than_thirty_days

    frame, target = _aged([10, 20, 40, 50, 60, 70], [0, 0, 1, 1, 1, 0])
    model = RuleBaseline(rule=older_than_thirty_days).fit(frame, target)
    scores = model.predict_proba(frame)[:, 1]

    assert scores[0] == pytest.approx(0.0)  # age <= 30: 0 of 2 closed
    assert scores[2] == pytest.approx(0.75)  # age > 30:  3 of 4 closed


def test_a_group_unseen_at_fit_time_falls_back_to_the_pooled_rate():
    """The honest answer to "what do you predict for a bucket you have never
    seen" — and the case that would otherwise be NaN and poison the metric."""
    from src.models.baselines import RuleBaseline, older_than_thirty_days

    frame, target = _aged([10, 20, 25], [1, 0, 0])
    model = RuleBaseline(rule=older_than_thirty_days).fit(frame, target)
    unseen, _ = _aged([90], [0])
    assert model.predict_proba(unseen)[0, 1] == pytest.approx(1 / 3)


def test_the_ceiling_sees_a_shape_a_logistic_fit_cannot():
    """The claim that justifies `age_ceiling` being in the ladder at all.

    `age_only` is a logistic fit on `age_days`, so it can express "closure rises
    with age" or "closure falls with age" and nothing else. The observed
    relationship on the real panel is neither — 1.13%, 1.59%, 0.86%, 1.49%,
    1.77%, 0.82% across age buckets — so a monotone fit can report "no signal"
    for a column whose signal it has no way to represent.

    Here closure is high at both ends of the age range and low in the middle. A
    logistic fit is at chance on that by construction; the ceiling should not be.
    """
    from sklearn.linear_model import LogisticRegression

    from src.models.baselines import SingleFeatureCeiling
    from src.models.metrics import average_precision

    rng = np.random.default_rng(0)
    ages = np.concatenate(
        [rng.uniform(0, 10, 200), rng.uniform(10, 40, 400), rng.uniform(40, 50, 200)]
    )
    labels = np.concatenate(
        [
            rng.binomial(1, 0.5, 200),  # young: closes often
            rng.binomial(1, 0.05, 400),  # middle: rarely
            rng.binomial(1, 0.5, 200),  # old: closes often
        ]
    )
    frame = pd.DataFrame({"age_days": ages})

    ceiling = SingleFeatureCeiling(column="age_days", bins=10).fit(frame, labels)
    monotone = LogisticRegression(max_iter=1000).fit(frame[["age_days"]], labels)

    ceiling_ap = average_precision(labels, ceiling.predict_proba(frame)[:, 1])
    monotone_ap = average_precision(labels, monotone.predict_proba(frame[["age_days"]])[:, 1])
    base = labels.mean()

    # A monotone fit is not at *chance* on a U — it captures whichever arm its
    # slope points at, and here that is worth about a tenth over the base rate.
    # The claim is not that it sees nothing; it is that it can only ever see
    # half the shape, and the margin is what that costs.
    assert monotone_ap > base, "sanity: the logistic should still capture one arm"
    assert ceiling_ap > monotone_ap + 0.10, "the ceiling should see the arm the logistic cannot"
    assert ceiling_ap > base + 0.15


def test_the_ceiling_reuses_training_bin_edges_on_unseen_values():
    """Quantiles recomputed on the evaluation block would be a statistic of the
    future. A validation value beyond the training range must land in the nearest
    bin rather than becoming NaN and silently taking the pooled fallback."""
    from src.models.baselines import SingleFeatureCeiling

    train = pd.DataFrame({"age_days": list(range(10, 110, 10))})
    model = SingleFeatureCeiling(column="age_days", bins=4).fit(train, [0] * 5 + [1] * 5)
    edges = list(model.edges_)

    beyond = pd.DataFrame({"age_days": [-500.0, 5000.0]})
    scores = model.predict_proba(beyond)[:, 1]
    assert not np.isnan(scores).any()
    assert edges[0] == -np.inf and edges[-1] == np.inf

    # and the edges do not move when a different frame is scored
    model.predict_proba(pd.DataFrame({"age_days": [1.0, 2.0, 3.0]}))
    assert list(model.edges_) == edges


def test_every_rule_rung_is_named_as_a_heuristic():
    """A rung that is a rule but is counted as a model would make the verdict
    compare the modelling against itself."""
    from src.models.train_baseline import HEURISTIC_RUNGS

    rule_rungs = {rung.name for rung in LADDER if rung.name.startswith("rule_")}
    assert rule_rungs <= set(HEURISTIC_RUNGS)
    assert set(HEURISTIC_RUNGS) <= {rung.name for rung in LADDER}


# --- the verdict --------------------------------------------------------------


def _ladder_table(rule_precision: float, model_precision: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "prior", "precision": 0.0, "pr_auc": 0.0},
            {"model": "rule_older_than_30d", "precision": rule_precision, "pr_auc": 0.1},
            {"model": "logistic", "precision": model_precision, "pr_auc": 0.2},
        ]
    )


def test_the_verdict_says_when_a_rule_wins():
    """The finding this whole section exists to be able to report. A model that
    loses to one sentence is a result, not a bug to tune away."""
    from src.models.train_baseline import heuristic_verdict

    text = "\n".join(heuristic_verdict(_ladder_table(0.30, 0.10), 20))
    assert "best rule wins" in text
    assert "0.2000" in text


def test_the_verdict_says_when_the_modelling_earns_its_keep():
    from src.models.train_baseline import heuristic_verdict

    text = "\n".join(heuristic_verdict(_ladder_table(0.10, 0.30), 20))
    assert "buys **+0.2000**" in text


def test_the_verdict_names_a_tie_as_a_tie():
    from src.models.train_baseline import heuristic_verdict

    text = "\n".join(heuristic_verdict(_ladder_table(0.20, 0.20), 20))
    assert "exactly level" in text


def test_the_verdict_warns_that_one_block_is_one_draw():
    from src.models.train_baseline import heuristic_verdict

    text = "\n".join(heuristic_verdict(_ladder_table(0.10, 0.30), 20))
    assert "fold spread" in text


def test_the_verdict_is_silent_without_both_families():
    from src.models.train_baseline import heuristic_verdict

    only_rules = pd.DataFrame([{"model": "prior", "precision": 0.1, "pr_auc": 0.1}])
    assert heuristic_verdict(only_rules, 20) == []
    assert heuristic_verdict(pd.DataFrame(), 20) == []
