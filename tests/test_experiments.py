"""Tests for the replayable experiment history.

Three things are worth asserting here, and only one of them is about MLflow.

**That the leak is real and measurable.** A demonstration of leakage that does
not actually leak is worse than none — it teaches the shape of the lesson while
proving nothing. So the tests check that run 06 beats the honest runs by a wide
margin *and* that run 07 puts the score back exactly, which together say the
gap is attributable to the admitted columns and to nothing else.

**That a run can be reproduced from what was logged.** The standard a tracking
setup has to meet: pick any run and reproduce its metric from its logged params
and dataset version, and if you cannot, you are logging too little. That is
`test_a_logged_run_can_be_refitted_from_its_params`, and it is the one that
would fail if someone added a hyperparameter to a spec and forgot to log it.

**That the honest runs stay honest.** The fixture's label is drawn independently
of every feature, so the correct validation score for every non-leaky run is the
base rate. A run that beats it is reading something it should not — which makes
this file able to catch a *future* leak, not merely to display the planted one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.data.split import Cuts, crawl_waves, temporal_split
from src.features.leaky import LEAKY_COLUMNS, add_leaky_features
from src.features.preprocessing import feature_columns, known_columns
from src.models import provenance
from src.models.experiments import (
    RUNS,
    default_cuts,
    execute,
    leak_delta,
    replay,
    reproduce,
    spec_by_name,
)
from tests.panels import make_closing_panel

mlflow = pytest.importorskip("mlflow", reason="experiment tracking is an optional extra")


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return make_closing_panel()


@pytest.fixture(scope="module")
def split(panel):
    prepared = add_leaky_features(panel)
    waves = crawl_waves(prepared[prepared["label_observable"]])
    return temporal_split(prepared, default_cuts(waves))


@pytest.fixture(scope="module")
def history(panel, tmp_path_factory):
    """Replay the whole history once, into a throwaway tracking store."""
    store = tmp_path_factory.mktemp("mlruns") / "mlflow.db"
    table, run_ids = replay(
        panel,
        Path("tests/panels.py"),
        dataset=provenance.SYNTHETIC,
        experiment="shelf-life-tests",
        tracking_uri=f"sqlite:///{store}",
    )
    return table, run_ids, f"sqlite:///{store}"


# --- the fixture has to be able to show a leak ------------------------------


def test_the_closing_panel_has_attrition_and_censoring(panel):
    """Without postings that actually leave, the leak is unmeasurable."""
    per_posting = panel.groupby("source_id").size()
    assert per_posting.min() < per_posting.max(), "no attrition: every posting has the same rows"
    assert not panel["label_observable"].all(), "no censored tail: every row is labelled"


def test_the_leaky_columns_separate_the_classes(panel):
    """The planted leak must actually carry the outcome, or run 06 proves nothing."""
    prepared = add_leaky_features(panel)
    labelled = prepared[prepared["label_observable"]]
    for column in LEAKY_COLUMNS:
        closed = labelled.loc[labelled["y"] == 1, column].mean()
        open_ = labelled.loc[labelled["y"] == 0, column].mean()
        assert closed < open_, f"{column} does not distinguish closed postings from open ones"


def test_leaky_columns_are_computed_over_the_whole_panel(panel):
    """The defect is the open right-hand window, and it must survive slicing.

    Computing the columns on a subset gives different values, which is precisely
    why they cannot be computed inside the pipeline: a per-fold count would be a
    milder, different bug.
    """
    whole = add_leaky_features(panel)
    early = add_leaky_features(panel[panel["t"] < panel["t"].max()])
    joined = whole.merge(early, on=["source_id", "t"], suffixes=("_whole", "_early"), how="inner")
    assert (joined["n_observations_total_whole"] != joined["n_observations_total_early"]).any(), (
        "the leaky count does not depend on the frame it was computed over"
    )


# --- the leak, measured -----------------------------------------------------


def test_the_leak_inflates_the_score_substantially(history):
    table, _, _ = history
    delta = leak_delta(table)
    assert delta["inflation"] > 0.3, (
        f"the planted leak was only worth {delta['inflation']:.4f} PR-AUC; a demonstration "
        f"of leakage that barely leaks teaches the shape of the lesson and proves nothing"
    )


def test_removing_the_leak_restores_the_honest_score_exactly(history):
    """Runs 05 and 07 differ in nothing, so they must agree to the bit.

    This is what makes run 06 a controlled comparison. If these two ever
    diverge, something other than the leaky columns changed between them and the
    inflation figure is measuring more than one thing.
    """
    table, _, _ = history
    delta = leak_delta(table)
    assert delta["restored"] == pytest.approx(0.0, abs=1e-12)


def test_the_honest_runs_do_not_beat_the_base_rate_by_much(history):
    """The fixture's label is independent of every feature, so nothing can learn it.

    Stated as a loose bound rather than an equality: PR-AUC on a few hundred
    validation rows is noisy, and the point is to catch a *new* leak of the size
    of the planted one, not to pin the third decimal place.
    """
    table, _, _ = history
    honest = table[~table["run"].str.contains("leaky")]
    base_rate = float(table["val_base_rate"].iloc[0])
    assert honest["val_pr_auc"].max() < base_rate + 0.25, (
        f"an honest run scored {honest['val_pr_auc'].max():.4f} against a base rate of "
        f"{base_rate:.4f} on a label drawn independently of every feature"
    )


# --- reproducibility, which is the component's own test ---------------------


def test_a_logged_run_can_be_refitted_from_its_params(history, panel):
    """Pick any run, reproduce its metric from what was logged.

    Every run, not one — a spec whose hyperparameters are only half-logged would
    otherwise hide behind the ones that are fully logged.
    """
    _, run_ids, uri = history
    for run_id in run_ids:
        report = reproduce(run_id, panel, tracking_uri=uri)
        assert report["replayed_val_pr_auc"] == pytest.approx(
            report["logged_val_pr_auc"], abs=1e-9
        ), f"{report['run_name']} did not reproduce: {report}"


def test_reproduction_reports_whether_code_and_data_still_match(history, panel):
    _, run_ids, uri = history
    report = reproduce(run_ids[0], panel, tracking_uri=uri)
    assert report["same_code"] is True
    assert report["same_data"] is True


def test_tuned_run_is_reproduced_from_logged_params_not_a_fresh_search(history, panel):
    """Run 08 searches. Replaying it must read the answer, not search again.

    A reproduction that re-runs the search would pass even if the chosen
    hyperparameters were never logged, which is exactly the gap this component's
    test is meant to close.
    """
    _, run_ids, uri = history
    mlflow.set_tracking_uri(uri)
    tuned = [
        run_id
        for run_id in run_ids
        if mlflow.get_run(run_id).data.tags["mlflow.runName"] == "08-xgboost_tuned"
    ]
    assert tuned, "no tuned run in the history"
    logged = mlflow.get_run(tuned[0]).data.params
    for key in ("model__max_depth", "model__learning_rate", "model__min_child_weight"):
        assert key in logged, f"{key} was searched for but never logged"


# --- what every run must carry ---------------------------------------------


def test_every_run_logs_provenance_and_the_metrics_that_matter(history):
    _, run_ids, uri = history
    mlflow.set_tracking_uri(uri)
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        for tag in ("git_sha", "git_dirty", "panel_sha256", "panel_rows", "dataset"):
            assert tag in run.data.tags, f"{tag} missing from {run.info.run_name}"
        for metric in ("val_pr_auc", "train_pr_auc", "gap", "val_brier", "val_ece"):
            assert metric in run.data.metrics, f"{metric} missing from {run.info.run_name}"
        assert "accuracy" not in run.data.metrics, "accuracy is banned as a reported metric"


def test_the_history_is_ordered_and_named_uniquely():
    numbers = [spec.number for spec in RUNS]
    assert numbers == sorted(numbers) == list(range(1, len(RUNS) + 1))
    assert len({spec.run_name for spec in RUNS}) == len(RUNS)


def test_specs_are_addressable_by_name():
    for spec in RUNS:
        assert spec_by_name(spec.run_name) is spec
    with pytest.raises(KeyError):
        spec_by_name("09-does-not-exist")


# --- the leaky switch stays off everywhere else -----------------------------


def test_leaky_columns_are_never_features_by_default():
    default = {column.name for column in feature_columns()}
    assert default.isdisjoint(LEAKY_COLUMNS)

    admitted = {column.name for column in feature_columns(include_leaky=True)}
    assert set(LEAKY_COLUMNS) <= admitted


def test_leaky_columns_have_a_verdict_so_a_prepared_panel_is_accepted(panel):
    """`assert_known_columns` must not reject a panel that has been through
    `add_leaky_features`, or the leak could never be measured at all."""
    assert set(LEAKY_COLUMNS) <= known_columns()
    assert set(add_leaky_features(panel).columns) <= known_columns()


def test_only_the_experiment_driver_turns_the_leak_on():
    """In the spirit of the test-set grep in `tests/test_evaluate.py`.

    A second place in `src/` that switches the leak on would mean a leaky model
    had escaped the one run that exists to measure it.

    Parsed rather than grepped. A substring search matches the switch's own
    documentation — `src/features/preprocessing.py` explains the flag in prose —
    so the check looks for an actual keyword argument set to `True` in the
    syntax tree, which is the thing that would do damage.
    """
    offenders = []
    for path in Path("src").rglob("*.py"):
        if path.name == "experiments.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "include_leaky"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    offenders.append(f"{path}:{node.lineno}")
    assert not offenders, f"the leak switch is turned on outside the driver: {offenders}"


# --- splitting ---------------------------------------------------------------


def test_default_cuts_leave_a_training_window_deep_enough_for_folds(split):
    """The bug this replaced: `waves[0], waves[len//2]` gives a one-wave training
    block on a deep panel, so cross-validation silently produces no folds and
    every run loses its error bars."""
    from src.models.evaluate import wave_forward_folds

    assert len(wave_forward_folds(split.train, split.embargo)) >= 3


def test_default_cuts_refuses_a_panel_too_shallow_to_cut():
    from src.data.split import SplitTooShallow

    with pytest.raises(SplitTooShallow):
        default_cuts(pd.Series([pd.Timestamp("2026-08-31T03:45:00Z")]))


def test_execute_never_touches_the_test_block(split):
    """Selection is a validation question. The test block is opened once, later."""
    spec = spec_by_name("01-prior")
    result = execute(spec, split, cross_validate_folds=False)
    assert "test" not in " ".join(result.metrics)
    assert result.metrics["n_val"] == float(len(split.val))


def test_cross_validation_survives_a_fold_with_no_positives_to_learn_from(split):
    """An expanding window's earliest folds can be all-negative. That is a fact
    about the fold, not a crash: it yields NaN and is excluded from the mean."""
    from sklearn.dummy import DummyClassifier

    from src.features.preprocessing import build_pipeline
    from src.models.evaluate import cross_validate, wave_forward_folds

    folds = wave_forward_folds(split.train, split.embargo)
    table = cross_validate(
        lambda: build_pipeline(DummyClassifier(strategy="prior")), split.train, folds
    )
    assert len(table) == len(folds)
    assert table["pr_auc"].notna().any(), "no fold scored at all"


def test_cuts_can_be_passed_explicitly(panel):
    prepared = add_leaky_features(panel)
    waves = crawl_waves(prepared[prepared["label_observable"]])
    cuts = Cuts(waves.iloc[9], waves.iloc[13])
    assert temporal_split(prepared, cuts).cuts == cuts
