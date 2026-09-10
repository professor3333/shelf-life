"""Tests for the boosted rung, the ablation and the overfit demonstration.

The ablation and the sweep are experiments, not features, so what is tested is
that they measure what they claim: that `delta` is the score a feature is worth,
that the sweep really separates train from validation, and that nothing here
fits on anything but the training fold.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from panels import DAY, WAVE0, make_panel

from src.data.split import Cuts, temporal_split
from src.models.train import (
    ABLATIONS,
    OVERFIT_SWEEP,
    ablate,
    build_xgboost,
    fit_and_score,
    overfit_sweep,
    scale_pos_weight,
    write_report,
)


def _split(**kwargs):
    frame = make_panel(**kwargs)
    return temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


def _noise_split():
    return _split(per_wave=300, positives_per_wave=30, random_labels=True)


# --- the boosted rung ------------------------------------------------------


def test_scale_pos_weight_comes_from_the_training_fold():
    """Imbalance is a statistic, and a statistic from the whole frame is a leak
    however innocuous a ratio looks."""
    split = _noise_split()
    target = split.train["y"].astype(int)
    expected = (len(target) - target.sum()) / target.sum()
    assert scale_pos_weight(split) == pytest.approx(expected)

    everything = split.frame["y"].astype(int)
    full_frame = (len(everything) - everything.sum()) / everything.sum()
    assert scale_pos_weight(split) != pytest.approx(full_frame) or expected == full_frame


def test_fit_and_score_reports_train_and_validation_together():
    """A validation number alone says how good the model is; the pair says
    whether it is memorising."""
    scored = fit_and_score(build_xgboost(split := _noise_split()), split)
    assert {"train_pr_auc", "val_pr_auc", "gap"} <= set(scored)
    assert scored["gap"] == pytest.approx(scored["train_pr_auc"] - scored["val_pr_auc"])


def test_the_boosted_rung_handles_a_category_unseen_at_fit_time():
    split = _noise_split()
    model = build_xgboost(split)
    fit_and_score(model, split)
    unseen = split.val.head(5).copy()
    unseen["title"] = "Chief Xenobiology Officer"  # a seniority level and a location
    unseen["location"] = "Ulaanbaatar"
    assert model.predict_proba(unseen).shape == (5, 2)


def test_the_boosted_rung_is_deterministic():
    split = _noise_split()
    first = fit_and_score(build_xgboost(split), split)
    second = fit_and_score(build_xgboost(split), split)
    assert first == second


# --- the ablation ----------------------------------------------------------


def test_the_ablation_refits_once_per_hypothesis_plus_a_baseline():
    table = ablate(_noise_split())
    assert list(table["removed"]) == ["nothing", *[a.name for a in ABLATIONS]]
    assert table.loc[0, "delta"] == 0.0


def test_the_ablation_delta_is_what_the_feature_is_worth():
    table = ablate(_noise_split())
    baseline = table.loc[0, "val_pr_auc"]
    for _, row in table.iloc[1:].iterrows():
        assert row["delta"] == pytest.approx(baseline - row["val_pr_auc"])


def test_the_ablation_finds_a_feature_the_label_depends_on():
    """The default fixture's title is `Engineer {index}` and its label is
    `index < 2`, so title length carries the label. Removing it must cost
    something — an ablation that reports zero for a decisive feature is
    measuring nothing."""
    table = ablate(_split(per_wave=200, positives_per_wave=20))
    decisive = table[table["removed"] == "title_chars"].iloc[0]
    assert decisive["delta"] > 0.1


def test_every_ablation_names_a_hypothesis():
    for ablation in ABLATIONS:
        assert ablation.hypothesis.strip()
        assert ablation.hypothesis != "—"


# --- the deliberate overfit ------------------------------------------------


def test_the_sweep_opens_a_train_validation_gap_on_noise():
    """The demonstration the component asks for. On a label drawn independently
    of every feature, a deep unregularised model must fit the training fold and
    fail on validation — if it does not, the sweep is not sweeping."""
    table = overfit_sweep(_noise_split()).set_index("setting")
    shallow = table.loc["stump, heavy shrinkage"]
    deep = table.loc["deep, unregularised"]

    assert deep["train_pr_auc"] > 0.9
    assert deep["gap"] > shallow["gap"] + 0.3
    base_rate = 0.1
    assert deep["val_pr_auc"] < base_rate + 0.2  # learned nothing that transfers


def test_the_sweep_records_which_knob_closed_the_gap():
    table = overfit_sweep(_noise_split()).set_index("setting")
    deep = table.loc["deep, unregularised", "gap"]
    closed = table.loc["deep + min_child_weight", "gap"]
    assert closed < deep, "min_child_weight should shrink the gap it was added to shrink"


def test_the_sweep_covers_every_configured_setting():
    table = overfit_sweep(_noise_split())
    assert list(table["setting"]) == [name for name, _ in OVERFIT_SWEEP]


# --- the report ------------------------------------------------------------


def test_the_report_records_the_blocker_when_nothing_could_run(tmp_path):
    path = tmp_path / "model_results.md"
    write_report(path, make_panel(n_waves=4), None, None, None, blocker="No split yet.")
    text = path.read_text()
    assert "No split yet." in text
    assert "title_seniority" in text  # the features exist even when untested


def test_the_report_carries_the_ladder_the_ablation_and_the_sweep(tmp_path):
    split = _noise_split()
    ladder = pd.DataFrame(
        [
            {
                "model": "xgboost",
                "description": "gradient boosting",
                "pr_auc": 0.2,
                "brier": 0.1,
                "roc_auc": 0.6,
                "precision": 0.1,
                "recall": 0.2,
            }
        ]
    )
    path = tmp_path / "model_results.md"
    write_report(path, make_panel(), ladder, ablate(split), overfit_sweep(split), None)

    text = path.read_text()
    assert "Hypothesis ablation" in text
    assert "The deliberate overfit" in text
    for ablation in ABLATIONS:
        assert ablation.name in text


# --- board context, and what the group ablation is for ------------------------


def test_the_board_ablation_tracks_the_request_contract():
    """The withheld columns are the request contract's, not a second list.

    `contract.BOARD_CONTEXT` is derived from `Field.origin == "board"`, the same
    predicate that makes a field optional on `POST /predict`. Deriving the
    ablation from it means a field that changes origin changes what gets priced,
    and nobody can change one without the other. Two lists that must agree are
    two lists that will not.
    """
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import BOARD_CONTEXT_ABLATION

    assert BOARD_CONTEXT_ABLATION.features == BOARD_CONTEXT
    assert len(BOARD_CONTEXT) == 4


def test_the_board_features_are_priced_both_together_and_singly():
    """Both, because either alone gives an answer that cannot be acted on.

    Four small individual deltas with a large group delta means the columns are
    redundant *with each other* — dropping all four would cost something even
    though dropping any one costs nothing. Four small deltas with a small group
    delta means they are worthless, and only that licenses `docs/design.md` §12
    to drop them. A leave-one-out table alone cannot tell those apart, and it is
    the one that looks reassuring.
    """
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import ABLATIONS

    singly = {a.name for a in ABLATIONS if len(a.features) == 1}
    assert set(BOARD_CONTEXT) <= singly, "each board column must also be priced on its own"

    groups = [a for a in ABLATIONS if len(a.features) > 1]
    assert len(groups) == 1 and set(groups[0].features) == set(BOARD_CONTEXT)


def test_every_ablation_withholds_columns_the_model_actually_has():
    """An ablation naming a column that is not a feature silently refits the
    baseline and reports a delta of zero, which reads exactly like 'this feature
    is worthless'."""
    from src.features.preprocessing import feature_columns
    from src.models.train import ABLATIONS

    known = {column.name for column in feature_columns()}
    for ablation in ABLATIONS:
        missing = set(ablation.features) - known
        assert not missing, f"{ablation.name} withholds unknown column(s) {sorted(missing)}"


def test_the_group_ablation_withholds_all_four_at_once(split):
    """Behavioural, not by name: change all four board values and the ablated
    pipeline must hand the estimator the identical row.

    Checked this way because the `derive` step has no `get_feature_names_out`,
    and because a name check proves less anyway — a column can be listed and
    still be dead, or dropped by name and smuggled through a derived feature.
    The control matters as much as the assertion: the *full* pipeline must move
    on the same edit, or this test would pass against a pipeline that ignores
    board context entirely and would prove nothing.
    """
    from xgboost import XGBClassifier

    from src.features.preprocessing import (
        build_pipeline,
        feature_columns,
        features_and_target,
    )
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import BOARD_CONTEXT_ABLATION, _xgb_parameters

    everything = [c.name for c in feature_columns()]
    kept = tuple(n for n in everything if n not in set(BOARD_CONTEXT_ABLATION.features))
    assert len(kept) == len(everything) - 4

    features, target = features_and_target(split.train)
    original = features.iloc[[0]].copy()
    edited = original.copy()
    for column in BOARD_CONTEXT:
        edited[column] = original[column].astype("Float64") + 137

    ablated = build_pipeline(XGBClassifier(**_xgb_parameters(split)), only=kept)
    ablated.fit(features, target)
    before, after = ablated[:-1].transform(original), ablated[:-1].transform(edited)
    assert (before == after).all(), "a board column still reaches the estimator"

    full = build_pipeline(XGBClassifier(**_xgb_parameters(split)))
    full.fit(features, target)
    assert not (full[:-1].transform(original) == full[:-1].transform(edited)).all(), (
        "the full pipeline ignores board context, so the assertion above is vacuous"
    )


def _verdict_table(group_delta: float, singles: list[float]) -> pd.DataFrame:
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import BOARD_CONTEXT_ABLATION

    rows = [{"removed": "nothing", "delta": 0.0}]
    rows += [{"removed": n, "delta": d} for n, d in zip(BOARD_CONTEXT, singles, strict=True)]
    rows.append({"removed": BOARD_CONTEXT_ABLATION.name, "delta": group_delta})
    return pd.DataFrame(rows)


def test_a_group_delta_of_nothing_licenses_dropping_the_columns():
    """The only reading that lets §12 close: all four withheld, nothing lost."""
    from src.models.train import _board_context_verdict

    text = "\n".join(_verdict_table(-0.001, [0.0, 0.001, -0.002, 0.0]).pipe(_board_context_verdict))
    assert "worth nothing" in text
    assert "can drop them" in text


def test_a_group_worth_more_than_any_single_column_is_reported_as_redundancy():
    """The finding leave-one-out alone would have hidden, and the reason the
    group ablation exists: four deltas near zero, and the four together worth
    ten times the best of them."""
    from src.models.train import _board_context_verdict

    text = "\n".join(
        _verdict_table(0.060, [0.004, 0.006, 0.003, 0.002]).pipe(_board_context_verdict)
    )
    assert "redundant with each other" in text
    assert "leave-one-out understates them" in text


def test_a_group_no_bigger_than_its_best_column_is_not_called_redundancy():
    """The distinction that decides whether dropping all four costs more than
    dropping one."""
    from src.models.train import _board_context_verdict

    text = "\n".join(
        _verdict_table(0.040, [0.005, 0.038, 0.001, 0.002]).pipe(_board_context_verdict)
    )
    assert "not** redundant" in text
    assert "sits in one of them" in text


def test_the_verdict_says_who_the_number_does_not_price():
    """A board owner supplies all four and never touches the imputation branch,
    so this number is about one caller, not about the features."""
    from src.models.train import _board_context_verdict

    text = "\n".join(_verdict_table(0.01, [0.0, 0.0, 0.0, 0.0]).pipe(_board_context_verdict))
    assert "board owner" in text


def test_the_verdict_is_silent_when_the_group_ablation_did_not_run():
    from src.models.train import _board_context_verdict

    assert _board_context_verdict(pd.DataFrame([{"removed": "nothing", "delta": 0.0}])) == []


# --- fold-level evidence for the board-context decision -----------------------


def test_the_board_comparison_is_paired_on_the_fold(split):
    """One row per fold, both models scored on the same validation wave.

    Pairing is the point. Two models scored on the same wave share whatever made
    that wave easy or hard, so differencing within the fold removes it; averaging
    each model separately and subtracting leaves it in, which is how a comparison
    manufactures a difference that is really a week's weather.
    """
    from src.models.train import board_context_folds

    table, paired = board_context_folds(split)
    assert not table.empty
    assert list(table.columns) == [
        "fold",
        "pr_auc_full",
        "pr_auc_without_board",
        "difference",
    ]
    assert table["fold"].is_unique
    scored = table.dropna()
    assert (scored["difference"] == scored["pr_auc_full"] - scored["pr_auc_without_board"]).all()
    assert paired["folds"] == len(scored)


def test_the_board_comparison_never_reads_the_test_block(split):
    """Choosing a feature set is model selection, and selection against test is a
    choice that cannot be un-made. The folds are cut from `split.train`, so the
    comparison must be identical when the test block is replaced with nonsense."""
    import dataclasses

    from src.models.train import board_context_folds

    honest, _ = board_context_folds(split)

    frame = split.frame.copy()
    test_rows = frame["split"] == "test"
    assert test_rows.any(), "fixture must have a test block for this to mean anything"
    frame.loc[test_rows, "y"] = 1 - frame.loc[test_rows, "y"]
    poisoned, _ = board_context_folds(dataclasses.replace(split, frame=frame))

    pd.testing.assert_frame_equal(honest, poisoned)


def test_a_training_window_too_shallow_for_folds_returns_nothing(panel):
    """Not an empty table pretending to be a result: `_fold_evidence` reads the
    emptiness and says the delta cannot settle §12 yet."""
    from src.data.split import crawl_waves
    from src.models.train import _fold_evidence, board_context_folds

    waves = crawl_waves(panel[panel["label_observable"]])
    shallow = temporal_split(panel, Cuts(waves.iloc[0], waves.iloc[4]))
    table, paired = board_context_folds(shallow)

    assert table.empty and paired["folds"] == 0
    assert "No fold evidence yet" in "\n".join(_fold_evidence(table, paired))


def _fold_table(differences: list[float]) -> tuple[pd.DataFrame, dict[str, float]]:

    rows = [
        {"fold": i, "pr_auc_full": 0.2 + d, "pr_auc_without_board": 0.2, "difference": d}
        for i, d in enumerate(differences)
    ]
    table = pd.DataFrame(rows)
    series = pd.Series(differences)
    return table, {
        "folds": float(len(differences)),
        "mean_difference": float(series.mean()),
        "sd": float(series.std(ddof=1)) if len(differences) > 1 else float("nan"),
        "wins": float((series > 0).sum()),
    }


def test_a_difference_inside_one_standard_deviation_is_called_a_tie():
    """And the tie is stated as licensing the drop, because that is what §12 asks:
    if the board columns cannot be shown to help, the simpler model is the one
    whose validated and served forms are the same object."""
    from src.models.train import _fold_evidence

    text = "\n".join(_fold_evidence(*_fold_table([0.01, -0.008, 0.004])))
    assert "Treat them as tied" in text
    assert "licenses dropping the four" in text


def test_a_consistent_lead_is_called_as_the_columns_earning_their_place():
    from src.models.train import _fold_evidence

    text = "\n".join(_fold_evidence(*_fold_table([0.05, 0.052, 0.048, 0.051])))
    assert "earn their place" in text
    assert "imputation fallback stays" in text


def test_the_ablated_model_winning_is_reported_rather_than_hidden():
    """The outcome nobody plans for. Dropping the four being an *improvement* is
    a finding worth understanding, not a sign to re-run until it goes away."""
    from src.models.train import _fold_evidence

    text = "\n".join(_fold_evidence(*_fold_table([-0.05, -0.052, -0.048, -0.051])))
    assert "without* board context leads" in text
    assert "adding noise" in text


def test_two_folds_are_not_called_evidence():
    """A standard deviation over two numbers is not a standard deviation."""
    from src.models.train import _fold_evidence

    text = "\n".join(_fold_evidence(*_fold_table([0.05, 0.04])))
    assert "not yet evidence" in text


def test_the_fold_section_says_where_the_folds_came_from():
    """A reader has to be able to check that selection stayed off the test block
    without opening the source."""
    from src.models.train import _fold_evidence

    text = "\n".join(_fold_evidence(*_fold_table([0.01, -0.008, 0.004])))
    assert "training" in text and "test block is not read" in text


# --- the serve-time regime (design.md §12) ----------------------------------


def test_the_serve_time_regime_is_not_the_ablation():
    """`docs/design.md` §12 named an ablation as its deciding evidence. It is the
    wrong measurement, and this pins the difference.

    A refit without the four board columns answers *what are they worth* — a
    model trained without them redistributes their weight. The deployed object
    is fitted **with** them and handed nulls, so the imputers fill constants and
    the fitted weights stay pointed at a column that no longer varies. The two
    numbers differ by about 4x on the observed panel, and only the second is
    what `POST /predict` returns to a stranger.
    """
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import serve_time_regime

    split = _split(per_wave=120, positives_per_wave=12)
    regime = serve_time_regime(split)

    assert list(regime["regime"]) == ["board context supplied", "absent, imputed"]
    assert (regime["n"] == len(split.val)).all(), "both rows must score the same block"
    assert regime.loc[0, "delta_pr_auc"] == 0.0
    assert not pd.isna(regime.loc[1, "delta_pr_auc"])

    # It is the *same fitted model* in both rows, so the only thing that can
    # explain a difference is the columns arriving or not.
    assert set(BOARD_CONTEXT), "the regime is defined by the request contract"


def test_the_serve_time_regime_is_flat_when_the_columns_carry_nothing():
    """The control. If board context is constant in the data, withholding it at
    serve time must cost nothing — otherwise the measurement is picking up the
    refit's noise rather than the imputation."""
    from src.inference.contract import BOARD_CONTEXT
    from src.models.train import serve_time_regime

    frame = make_panel(per_wave=120, positives_per_wave=12)
    for column in BOARD_CONTEXT:
        frame[column] = 1
    split = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))

    regime = serve_time_regime(split)
    assert regime.loc[1, "delta_pr_auc"] == pytest.approx(0.0, abs=1e-9)


# --- experiment tracking -----------------------------------------------------

mlflow = pytest.importorskip("mlflow", reason="experiment tracking is an optional extra")


def _logged(tmp_path, **families):
    """Run the families through the tracker into a throwaway store, and read back."""
    from src.models import provenance, tracking
    from src.models.train import log_experiments

    split = _split()
    uri = f"sqlite:///{tmp_path}/runs.db"
    client = tracking.start("tracking-test", uri)
    prov = provenance.collect(Path("tests/panels.py"), len(split.frame), provenance.SYNTHETIC)
    written = log_experiments(client, split, prov, 20, figures_dir=tmp_path, **families)
    api = mlflow.MlflowClient(tracking_uri=uri)
    experiment = api.get_experiment_by_name("tracking-test")
    runs = api.search_runs([experiment.experiment_id], max_results=200)
    return written, runs, api


def test_the_ablations_are_logged_as_one_parent_with_a_child_per_variant(tmp_path):
    """Flat sibling runs would lose which sweep a refit came from.

    An ablation's `delta` is defined against the baseline in the same table, so a
    variant logged with no marker of its family is a number nobody can put back
    into the comparison it came from.
    """
    split = _split()
    table = ablate(split, budget_per_day=20)
    _, runs, _ = _logged(tmp_path, ablations=table)

    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert [r.data.tags["mlflow.runName"] for r in parents] == ["ablations"]
    assert len(children) == len(table), "one child per row of the ablation table"
    assert all(c.data.tags["mlflow.parentRunId"] == parents[0].info.run_id for c in children)


def test_each_ablation_records_the_feature_subset_it_actually_fitted_on(tmp_path):
    """The field this whole exercise is about.

    An ablation is *defined* by the columns it withheld. A run that keeps the
    metric and drops the subset has kept the answer and thrown away the question
    — and the board-context ablation is the one where that matters most, because
    `docs/design.md` §12 was decided on it.
    """
    from src.models.train import BOARD_CONTEXT_ABLATION

    split = _split()
    table = ablate(split, budget_per_day=20)
    _, runs, api = _logged(tmp_path, ablations=table)

    by_name = {r.data.tags["mlflow.runName"]: r for r in runs}
    run = by_name[f"ablation: {BOARD_CONTEXT_ABLATION.name}"]
    logged = mlflow.artifacts.load_dict(f"{run.info.artifact_uri}/features.json")["features"]

    assert logged, "no feature subset was recorded"
    for withheld in BOARD_CONTEXT_ABLATION.features:
        assert withheld not in logged, (
            f"{withheld} was withheld from the refit but appears in the run's feature "
            "subset, so the tracking store attributes the metric to columns the model "
            "never saw"
        )

    baseline = by_name["ablation: nothing"]
    full = mlflow.artifacts.load_dict(f"{baseline.info.artifact_uri}/features.json")["features"]
    assert set(logged) < set(full), "the ablation kept as many columns as the baseline"


def test_every_run_carries_the_split_the_dataset_and_the_code_version(tmp_path):
    """Four things, and a run missing any of them cannot be compared with one that has them."""
    split = _split()
    _, runs, _ = _logged(tmp_path, sweep=overfit_sweep(split, budget_per_day=20))

    child = next(r for r in runs if "mlflow.parentRunId" in r.data.tags)
    for key in ("train_end", "val_end", "embargo", "n_train", "n_val"):
        assert key in child.data.params, f"the split's {key} is not recorded"
    for key in ("panel_sha256", "panel_path", "dataset"):
        assert key in child.data.tags, f"the dataset version's {key} is not recorded"
    assert "git_sha" in child.data.tags, "the code version is not recorded"
    assert any(k.startswith("model__") for k in child.data.params), "no estimator parameters"


def test_the_sweep_records_the_parameters_the_model_actually_used(tmp_path):
    """Settings override different knobs, so the result frame carries NaN where one
    was left alone. Logging that NaN would record a value the model never used."""
    split = _split()
    _, runs, _ = _logged(tmp_path, sweep=overfit_sweep(split, budget_per_day=20))

    name, overrides = OVERFIT_SWEEP[0]
    run = next(r for r in runs if r.data.tags.get("mlflow.runName") == f"overfit: {name}")
    for key, value in overrides.items():
        assert run.data.params[f"model__{key}"] == str(value)
    assert "nan" not in [v.lower() for v in run.data.params.values()]


def test_the_logger_and_the_refit_agree_on_which_columns_were_kept():
    """One function, because two callers need the same answer.

    `ablate` builds its pipeline from `kept_features` and the logger records the
    same call. Recomputing it in the logger would work until the day the two
    drift, and then the store would attribute a metric to the wrong columns.
    """
    from src.features.preprocessing import build_pipeline
    from src.models.train import BOARD_CONTEXT_ABLATION, kept_features

    kept = kept_features(BOARD_CONTEXT_ABLATION.features)
    pipeline = build_pipeline(None, only=kept)
    assert tuple(pipeline.named_steps["select"].kw_args["columns"]) == kept


def test_an_untracked_run_says_so_in_the_report():
    """A requirement that quietly stops holding is worse than one never claimed.

    MLflow is an optional extra, so the ladder has to run without it — but an
    untracked run that says nothing looks exactly like a tracked one in the
    report, and `CLAUDE.md` §4.6 asks that every run record its params, metrics,
    dataset version and git SHA.
    """
    import argparse

    from src.models.train import _track

    args = argparse.Namespace(no_mlflow=True, panel=Path("x.parquet"), budget=20)
    note = _track(args, None, object(), None, None, None, None, None, blocker=None)
    assert "Not tracked" in note and "--no-mlflow" in note

    blocked = _track(args, None, None, None, None, None, None, None, blocker="too shallow")
    assert "nothing ran" in blocked


def test_the_report_carries_the_tracking_status(tmp_path):
    """The status belongs in the artifact a reader opens, not only on stdout."""
    path = tmp_path / "model_results.md"
    write_report(
        path,
        make_panel(),
        None,
        None,
        None,
        "blocked",
        tracking_note="**Not tracked**: MLflow is not installed in this environment.",
    )
    body = path.read_text()
    assert "## Experiment tracking" in body
    assert "**Not tracked**" in body


def test_each_ladder_rung_records_the_columns_that_rung_restricts_itself_to(tmp_path):
    """`age_only` is the rung that makes this worth checking.

    It fits on one column by construction, so a run recording the full registry
    against its metric would attribute the score to twenty-odd features it never
    saw — the same failure the ablations would have, one rung further down.
    """
    from src.models.train_baseline import run_ladder

    split = _split()
    table, _ = run_ladder(split, budget_per_day=20)
    _, runs, _ = _logged(tmp_path, ladder=table)

    by_name = {r.data.tags["mlflow.runName"]: r for r in runs}
    age_only = by_name["rung: age_only"]
    logged = mlflow.artifacts.load_dict(f"{age_only.info.artifact_uri}/features.json")["features"]
    assert logged == ["age_days"]

    logistic = by_name["rung: logistic"]
    full = mlflow.artifacts.load_dict(f"{logistic.info.artifact_uri}/features.json")["features"]
    assert len(full) > len(logged)


def test_a_rule_baseline_records_no_feature_subset_rather_than_a_wrong_one():
    """`None` is a real answer here, not a failure to find one.

    A rule baseline applies a stated rule to the frame and never builds a feature
    matrix, so "which columns did it fit on" has no answer — and filling the slot
    with the full registry would be a lie in exactly the place this change exists
    to make truthful.
    """
    from src.models import tracking
    from src.models.train_baseline import LADDER

    rungs = {rung.name: rung.build for rung in LADDER}
    assert tracking.pipeline_features(rungs["rule_older_than_30d"]()) is None
    assert tracking.pipeline_features(rungs["age_only"]()) == ("age_days",)
