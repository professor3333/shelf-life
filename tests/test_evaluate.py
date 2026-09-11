"""Tests for model comparison, threshold choice and calibration.

Two kinds the component asks for specifically.

**Hand-computed metrics on a tiny fixed array.** Every expected value below was
worked out on paper and the arithmetic is written into the test, so a metric
wired to the wrong argument fails here rather than surviving into a report where
nothing can be checked against anything.

**The test-set discipline check.** `test_the_test_block_is_read_only_where_it_should_be`
parses `src/` and fails if the test split is read anywhere but the two places
entitled to: the property that defines it, and the freeze step that opens it
once. It uses the AST rather than a grep so that a docstring *mentioning* the
test block is not mistaken for code reading it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from panels import DAY, WAVE0, make_panel

from src.data.split import Cuts, rolling_origin_folds, temporal_split
from src.models.evaluate import (
    FOLD_COLUMNS,
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

#: The only modules allowed to read `split.test`, and why each one is.
#:
#: `src/data/split.py` defines the property. `src/models/freeze.py` is the single
#: evaluation — it fits the pipeline, fixes the threshold on validation, and only
#: then reads the block, once, at the end of the experimental phase.
#:
#: Adding a third name here is not a maintenance chore, it is a decision to look
#: at the test set again, and the list is short so that decision cannot be made
#: by accident.
TEST_BLOCK_READERS = [Path("src/data/split.py"), Path("src/models/freeze.py")]


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


def test_projected_folds_match_the_real_splitter():
    """The pin between `split.rolling_origin_folds` and this module.

    `minimum_waves` has to say how many folds a panel *will* yield at a depth the
    scraper has not reached, so it carries a projection of the fold rule rather
    than the rule itself — it cannot import this module, because this module
    imports it. That projection is only trustworthy while it agrees with the
    real splitter, which is what this checks. If `wave_forward_folds` ever
    changes its window, expanding to sliding or its embargo handling, this fails
    and `rolling_origin_folds` has to be brought along.
    """
    embargo = 2 * DAY
    burnt = 3  # floor(2 days / 1 day) + 1, the same arithmetic minimum_waves does
    for n_waves in range(1, 11):
        block = make_panel(n_waves=n_waves, per_wave=6)
        assert len(wave_forward_folds(block, embargo)) == rolling_origin_folds(n_waves, burnt), (
            f"{n_waves} waves"
        )


def test_the_test_block_is_read_only_where_it_should_be():
    """A test set looked at twice is a validation set.

    So the block is read where it is defined, and in the freeze step that opens
    it once — and nowhere else. Not in the comparison, not in the ablation, not
    in the tuning: every one of those chooses something, and a choice made
    against test is a choice that cannot be un-made.
    """
    readers = [path for path in sorted(Path("src").rglob("*.py")) if _reads_the_test_block(path)]
    assert readers == sorted(TEST_BLOCK_READERS), f"the test block is read in {readers}"


def test_the_shell_scripts_do_not_read_the_test_block_either():
    """The AST guard above walks `src/` only, so a script that reaches into the
    held-out block satisfies it by living somewhere else.

    That is not hypothetical: `scripts/watch_depth.sh` is a scheduled job whose
    whole purpose is deciding *when the test set is worth spending*, and the
    first draft counted its positives to answer that. A daily peek is still a
    peek, and one the guard could not see is worse than one it could.

    Text rather than AST because the Python here lives inside heredocs, which no
    parser will reach. Crude, and it only has to catch the obvious reach — the
    subtle ones are not what a scheduled job drifts into.
    """
    offenders = []
    for path in sorted(Path("scripts").glob("*.sh")):
        text = path.read_text()
        for line in text.splitlines():
            code = line.split("#", 1)[0]
            if ".test" in code or '"test"' in code or "'test'" in code:
                offenders.append(f"{path}: {line.strip()}")
    assert not offenders, "a script reads the held-out block:\n" + "\n".join(offenders)


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


def test_a_panel_too_shallow_to_cut_folds_scores_nothing_rather_than_raising():
    """Zero folds is the ordinary state of a panel still accruing depth, so it
    has to be a reportable answer rather than an exception. Before this, an
    empty result carried no columns at all and `summarise_folds` raised
    `KeyError: 'pr_auc'` — a message about pandas indexing, two modules from the
    cause, on the first run against the real panel."""
    from sklearn.dummy import DummyClassifier

    from src.features.preprocessing import build_pipeline

    split = _deep_split()
    scored = cross_validate(
        lambda: build_pipeline(DummyClassifier(strategy="prior"), min_category_frequency=1),
        split.train,
        folds=[],
    )
    assert scored.empty
    assert list(scored.columns) == list(FOLD_COLUMNS)

    summary = summarise_folds(scored)
    assert summary["folds"] == 0.0
    assert summary["folds_scored"] == 0.0
    assert np.isnan(summary["cv_pr_auc_mean"])
    assert np.isnan(summary["cv_pr_auc_sd"])

    # And the comparison built on top of it says "no folds", not "a tie".
    difference = paired_fold_difference(scored, scored)
    assert difference["folds"] == 0.0
    assert np.isnan(difference["mean_difference"])


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


def test_a_gap_inside_one_standard_deviation_hands_the_pick_to_the_simpler_model():
    """The rule the old implementation documented and did not apply.

    `forest` has the higher mean and loses anyway, because the lead is smaller
    than its own fold-to-fold spread and `logistic` is earlier in the ladder.
    Complexity has to be *bought* with a lead that survives fold variance;
    CLAUDE.md §4.4 is explicit that reaching the top rung is not the goal.
    """
    summary = pd.DataFrame({"model": ["forest", "logistic"], "cv_pr_auc_mean": [0.44, 0.43]})
    per_fold = {
        "forest": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.60, 0.20, 0.52]}),
        "logistic": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.20, 0.60, 0.49]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] == "logistic"
    assert verdict["leader"] == "forest"
    assert verdict["chosen_on"] == "parsimony"
    assert verdict["separated"] is False
    assert "ties with" in verdict["reason"]


def test_a_fitted_model_that_cannot_separate_from_a_rule_loses_to_the_rule():
    """CLAUDE.md §4.4 #12, enforced by the ladder's ordering rather than beside it.

    `age_ceiling` is a heuristic rung — the best any age-only rule could do — and
    it sits early in the ladder. An XGBoost that leads it on the mean but not
    past the spread finds it in its own tied set and loses on parsimony. A model
    that does not beat the baseline is not a model, it is a slower baseline.
    """
    summary = pd.DataFrame({"model": ["xgboost", "age_ceiling"], "cv_pr_auc_mean": [0.31, 0.30]})
    per_fold = {
        "xgboost": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.45, 0.16, 0.32]}),
        "age_ceiling": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.20, 0.42, 0.28]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] == "age_ceiling"
    assert verdict["chosen_is_heuristic"] is True
    assert verdict["chosen_on"] == "parsimony"


def test_a_lead_that_survives_fold_variance_does_buy_complexity():
    """The rule is parsimony among *equals*, not a preference for simplicity.

    Without this the rule would be unfalsifiable in the other direction: it would
    always return the simplest candidate and the ladder would be decoration.
    """
    summary = pd.DataFrame({"model": ["xgboost", "logistic"], "cv_pr_auc_mean": [0.60, 0.30]})
    per_fold = {
        "xgboost": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.60, 0.61, 0.59]}),
        "logistic": pd.DataFrame({"fold": [0, 1, 2], "pr_auc": [0.30, 0.31, 0.29]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] == "xgboost"
    assert verdict["chosen_on"] == "separation"
    assert verdict["separated"] is True
    assert verdict["tied_with"] == []


def test_selection_refuses_a_candidate_with_too_few_folds():
    """Two folds is a number without a spread, and the verdict says so.

    The failure this prevents is the quiet one: a leader crowned on two folds
    reads in the report exactly like a leader crowned on ten.
    """
    summary = pd.DataFrame({"model": ["xgboost", "logistic"], "cv_pr_auc_mean": [0.6, 0.3]})
    per_fold = {
        "xgboost": pd.DataFrame({"fold": [0, 1], "pr_auc": [0.6, 0.6]}),
        "logistic": pd.DataFrame({"fold": [0, 1], "pr_auc": [0.3, 0.3]}),
    }
    verdict = select(summary, per_fold)
    assert verdict["chosen"] is None
    assert "2" in verdict["reason"]


def test_the_ladder_order_is_complexity_and_an_unknown_name_sorts_last():
    """Parsimony must never be an argument *for* something the ladder never saw."""
    from src.models.evaluate import _complexity, ladder_order

    order = ladder_order()
    assert order[0] == "prior", "the constant is no longer the simplest rung"
    assert order[-1] == "xgboost", "the boosted rung is no longer the most complex"
    assert _complexity("logistic") < _complexity("random_forest")
    assert _complexity("a name the ladder never had") == len(order)


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


def test_a_comparison_that_chose_nothing_reports_the_table_and_stops(tmp_path):
    """Zero folds is not a blocker — the models were still scored on validation,
    and that table is worth writing. It is also not a selection, so everything
    that describes a single chosen model has to stop rather than run on a default
    nobody picked. Before this, `main` stringified the absent choice and died on
    `KeyError: 'None'` the first time the real panel produced no folds."""
    path = tmp_path / "model_comparison.md"
    summary = pd.DataFrame(
        {
            "model": ["prior", "xgboost"],
            "folds_scored": [0.0, 0.0],
            "cv_pr_auc_mean": [float("nan")] * 2,
            "cv_pr_auc_sd": [float("nan")] * 2,
            "val_pr_auc": [0.019, 0.023],
            "val_brier": [0.019, 0.021],
            "val_ece": [0.01, 0.02],
            "val_roc_auc": [0.5, 0.58],
        }
    )
    verdict = select(summary, {})
    assert verdict["chosen"] is None

    write_report(
        path,
        make_panel(n_waves=4),
        summary,
        {name: pd.DataFrame(columns=list(FOLD_COLUMNS)) for name in summary["model"]},
        verdict,
        None,  # thresholds
        None,  # calibration
        None,  # by_source
        None,  # by_carryover
        blocker=None,
    )
    text = path.read_text()
    assert "Models, with fold variance" in text
    assert "xgboost" in text
    assert "No threshold, calibration or breakdown yet" in text
    # The sections that need a chosen model must be absent, not empty.
    assert "## Threshold" not in text
    assert "## Calibration" not in text
    assert "## Per source" not in text


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


def test_no_fitted_model_wins_on_a_label_that_is_pure_noise():
    """The end-to-end guarantee, on data whose correct answer is known.

    The panel's label is drawn independently of every feature, so nothing can
    legitimately beat the base rate. Individual means still scatter above it —
    that is what makes this the right test — and a rule that reads the highest
    of them as a result would crown a random forest at 0.2078 against a base rate
    of 0.1000.

    It is pinned here rather than left as prose in the report because this is the
    failure that would look like success: a selection rule that quietly starts
    preferring the top of the ladder produces a *better-looking* comparison, and
    nothing else in the suite would notice.
    """
    import sys

    sys.path.insert(0, "tests")
    from panels import make_panel

    from src.data.split import temporal_split
    from src.models.evaluate import compare_models, select
    from src.models.experiments import default_cuts
    from src.models.train_baseline import HEURISTIC_RUNGS

    panel = make_panel(n_waves=20, random_labels=True, positives_per_wave=4)
    split = temporal_split(panel, default_cuts(panel))
    summary, per_fold, _ = compare_models(split)
    verdict = select(summary, per_fold)

    best_mean = summary["cv_pr_auc_mean"].max()
    assert best_mean > 0.15, (
        "the panel stopped producing a tempting leader, so this test no longer "
        f"exercises the thing it exists for (best mean {best_mean:.4f})"
    )
    assert verdict["chosen"] in HEURISTIC_RUNGS, (
        f"a fitted model ({verdict['chosen']}) was selected on a label that is pure "
        f"noise, by: {verdict['reason']}"
    )


# --- showing the refusal rather than asserting it ----------------------------


def test_the_fold_census_enumerates_every_legal_cut():
    """ "No fold is available" invites the retort *try a different cut*.

    The census answers it exhaustively, and pins the contract `best_cuts`
    documents: once any cut reaches the target the chosen one must too, and while
    no cut does, the chosen one must be the deepest available. `best_cuts` does
    *not* simply maximise folds — past the target it prefers proportional blocks,
    because maximising folds starves validation and test — so the check has to be
    the contract rather than the maximum.
    """
    import sys

    sys.path.insert(0, "tests")
    from panels import make_panel

    from src.data.split import DEFAULT_TARGET_FOLDS, best_cuts, temporal_split
    from src.models.evaluate import fold_census, wave_forward_folds

    panel = make_panel(n_waves=20)
    census = fold_census(panel)
    assert not census.empty, "a twenty-wave panel admits at least one legal cut"
    assert set(census.columns) >= {"train_end", "val_end", "folds", "val_positives"}

    chosen = temporal_split(panel, best_cuts(panel))
    chosen_folds = len(wave_forward_folds(chosen.train, chosen.embargo))
    available = int(census["folds"].max())

    if available >= DEFAULT_TARGET_FOLDS:
        assert chosen_folds >= DEFAULT_TARGET_FOLDS, (
            f"a legal cut yields {available} fold(s) but the chosen cut yields "
            f"{chosen_folds}, below the target of {DEFAULT_TARGET_FOLDS}"
        )
    else:
        assert chosen_folds == available, (
            f"no cut reaches the target, so the chosen cut must be the deepest available "
            f"({available}); it yields {chosen_folds}"
        )


def test_the_census_is_empty_when_no_cut_is_legal():
    """A panel too shallow to cut has nothing to enumerate, and must not raise."""
    import sys

    sys.path.insert(0, "tests")
    from panels import make_panel

    from src.models.evaluate import fold_census

    assert fold_census(make_panel(n_waves=2)).empty


# --- the reading that must not become a verdict ------------------------------


def _summary(fitted: float, heuristic: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["xgboost", "age_ceiling"],
            "val_pr_auc": [fitted, heuristic],
        }
    )


def test_the_complexity_reading_never_reads_as_a_selection():
    """It reports a direction. A direction is not evidence, and it has to say so.

    The failure this guards is a reader taking the sentence "xgboost is ahead" out
    of a report whose verdict is `Chosen: None` — which is selection on the
    validation block arriving by paraphrase.
    """
    import sys

    from src.models.evaluate import _complexity_reading

    sys.path.insert(0, "tests")
    from panels import make_panel

    text = "\n".join(_complexity_reading(_summary(0.4, 0.1), make_panel(n_waves=6)))
    assert "not evidence" in text
    assert "selection on the validation block" in text
    assert "Chosen" not in text


def test_the_reading_branches_on_which_side_actually_leads():
    """The consequence must follow the direction, not be pasted under both.

    An earlier draft closed with "step 4 returns the heuristic" regardless, so a
    report where the fitted model led contradicted itself two lines later.
    """
    import sys

    from src.models.evaluate import _complexity_reading

    sys.path.insert(0, "tests")
    from panels import make_panel

    panel = make_panel(n_waves=6)
    ahead = "\n".join(_complexity_reading(_summary(0.4, 0.1), panel))
    assert "step 3 of the rule buys the complexity" in ahead

    behind = "\n".join(_complexity_reading(_summary(0.05, 0.2), panel))
    assert "no fitted rung beats a rule" in behind
    assert "step 4 of the rule returns the heuristic" in behind


def test_a_target_dominated_by_short_lived_postings_is_flagged_in_the_reading():
    """The score that looks most like a working model is the one to distrust here.

    `label_validity.md` carries the measurement, but this is the table where
    somebody decides whether a model works — and the two files being separate is
    how a reader ends up believing the optimistic one.
    """
    import sys

    sys.path.insert(0, "tests")
    from panels import make_panel

    from src.models.evaluate import _lifespan_caveat

    panel = make_panel(n_waves=8).copy()
    # Half the postings are held for one wave only and every one of them closes;
    # the rest are held throughout and none do. That is the shape the real panel
    # has, exaggerated so the threshold is unambiguous.
    key = panel["source"].astype(str) + "|" + panel["source_id"].astype(str)
    brief_ids = set(pd.unique(key)[: len(pd.unique(key)) // 2])
    is_brief = key.isin(brief_ids)
    panel = panel[~is_brief | (panel["t"] == panel["t"].min())].copy()
    key = panel["source"].astype(str) + "|" + panel["source_id"].astype(str)
    panel["y"] = key.isin(brief_ids).astype("Int8")
    panel["label_observable"] = True

    text = "\n".join(_lifespan_caveat(panel))
    assert "seen in fewer than 6 complete crawls as of their own `t` close at" in text
    # The caveat must carry the verification, not the superseded reading of it:
    # the concentration is real, and `label_check.md` showed the removals behind
    # it are genuine, so "this is measuring the crawl" is the wrong conclusion.
    assert "label_check.md" in text
    assert "removed is not filled" in text


def test_a_target_spread_across_lifespans_is_not_flagged():
    """A caveat that fires on every panel is a caveat nobody reads."""
    import sys

    sys.path.insert(0, "tests")
    from panels import make_panel

    from src.models.evaluate import _lifespan_caveat

    panel = make_panel(n_waves=8).copy()
    panel["label_observable"] = True
    assert _lifespan_caveat(panel) == []
