"""Tests for the boosted rung, the ablation and the overfit demonstration.

The ablation and the sweep are experiments, not features, so what is tested is
that they measure what they claim: that `delta` is the score a feature is worth,
that the sweep really separates train from validation, and that nothing here
fits on anything but the training fold.
"""

from __future__ import annotations

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
    assert list(table["removed"]) == ["nothing", *[a.feature for a in ABLATIONS]]
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
        assert ablation.feature in text
