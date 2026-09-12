"""The recalibration rule: fixed before the numbers, monotone, and recorded.

`src/models/calibration.py` decides whether the freeze wraps an isotonic map
around the estimator. What these tests hold: the rule fires on the numbers it
says it fires on and not otherwise; the positives floor is respected; applying
it lowers validation ECE; it cannot reorder postings or drop one from the alert
list; the artifact stays a full pipeline and still serves; and the decision is
on the artifact whichever way it went.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.artifact import assert_is_full_pipeline
from src.models import calibration
from src.models.freeze import _score, build_metadata, freeze
from src.models.metrics import alert_budget, expected_calibration_error, threshold_for_budget
from src.models.train_baseline import prediction_days

# --- the rule -----------------------------------------------------------------


def _block(n: int, positives: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    y[:positives] = 1
    rng.shuffle(y)
    return y


def test_the_rule_fires_on_ece_above_a_quarter_of_the_base_rate():
    y = _block(1000, 100)  # base rate 0.10, bound 0.025
    honest = np.where(y == 1, 0.10, 0.10)  # constant at the base rate: ECE 0
    assert not calibration.decide(y, honest).recalibrate
    overconfident = np.where(y == 1, 0.9, 0.3)  # far from observed rates in every bin
    decision = calibration.decide(y, overconfident)
    assert decision.recalibrate
    assert decision.ece > decision.bound == pytest.approx(0.025)
    assert "exceeds" in decision.reason


def test_the_rule_is_relative_to_the_base_rate_not_absolute():
    """The same absolute ECE is decisive at a 1% rate and noise at 50%."""
    rare = _block(2000, 20)
    common = _block(2000, 1000)
    shifted_rare = np.clip(np.where(rare == 1, 0.05, 0.05), 0, 1)  # ECE ≈ 0.04 at 1%
    shifted_common = np.where(common == 1, 0.54, 0.54)  # ECE ≈ 0.04 at 50%
    assert calibration.decide(rare, shifted_rare).ece == pytest.approx(
        calibration.decide(common, shifted_common).ece, abs=0.01
    )
    assert not calibration.decide(rare, shifted_rare).recalibrate, "under the positives floor"
    assert not calibration.decide(common, shifted_common).recalibrate


def test_too_few_positives_means_no_curve_whatever_the_ece_says():
    y = _block(500, calibration.MIN_POSITIVES - 1)
    terrible = np.where(y == 1, 0.99, 0.01)
    decision = calibration.decide(y, terrible)
    assert not decision.recalibrate
    assert "under the" in decision.reason and str(calibration.MIN_POSITIVES) in decision.reason


def test_the_decision_names_its_rule_and_its_numbers():
    y = _block(400, 60)
    decision = calibration.decide(y, np.full(400, 0.15)).as_dict()
    assert decision["rule"] == "ece > 0.25 * base_rate and positives >= 30"
    assert set(decision) >= {
        "recalibrate",
        "method",
        "validation_ece",
        "validation_base_rate",
        "bound",
        "validation_positives",
        "reason",
    }
    assert decision["method"] is None


# --- applying it ------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_and_recalibrated(split):
    """The frozen-run pipeline, before and after the isotonic wrap, on the same block."""
    from src.features.preprocessing import fit_on_frame
    from src.models.experiments import spec_by_name
    from tests.conftest import FROZEN_RUN

    spec = spec_by_name(FROZEN_RUN)
    raw = fit_on_frame(spec.build(split, spec.resolve(split)), split.train)
    wrapped = calibration.recalibrate(raw, split.val, split.val["y"])
    return raw, wrapped


def test_recalibration_keeps_the_pipeline_a_pipeline(raw_and_recalibrated):
    raw, wrapped = raw_and_recalibrated
    assert_is_full_pipeline(wrapped)
    assert [name for name, _ in wrapped.steps] == [name for name, _ in raw.steps]


def test_recalibration_is_monotone_so_ranks_survive_it(raw_and_recalibrated, split):
    """Descending order of the raw scores is descending order of the recalibrated
    ones, up to ties — isotonic regression pools neighbours into steps."""
    raw, wrapped = raw_and_recalibrated
    before = _score(raw, split.val).to_numpy()
    after = _score(wrapped, split.val).to_numpy()
    order = np.argsort(-before, kind="stable")
    assert np.all(np.diff(after[order]) <= 1e-12), "a higher raw score never gets a lower one"


def test_recalibration_cannot_drop_a_posting_from_the_alert_list(raw_and_recalibrated, split):
    raw, wrapped = raw_and_recalibrated
    before = _score(raw, split.val).to_numpy()
    after = _score(wrapped, split.val).to_numpy()
    budget = alert_budget(len(split.val), prediction_days(split.val), 20)
    flagged_before = before >= threshold_for_budget(before, budget)
    flagged_after = after >= threshold_for_budget(after, budget)
    assert not np.any(flagged_before & ~flagged_after), "recalibration only ever adds, at ties"


def test_recalibration_lowers_validation_ece_where_it_is_fitted(raw_and_recalibrated, split):
    raw, wrapped = raw_and_recalibrated
    y = split.val["y"].astype(int)
    before = expected_calibration_error(y, _score(raw, split.val))
    after = expected_calibration_error(y, _score(wrapped, split.val))
    assert after <= before + 1e-9


# --- through the freeze -------------------------------------------------------------


def test_the_freeze_records_the_decision_either_way(split, panel, tmp_path):
    from src.inference import artifact as artifact_module
    from tests.conftest import FROZEN_RUN

    frozen = freeze(split, FROZEN_RUN)
    assert frozen.recalibration["rule"].startswith("ece > 0.25")
    assert isinstance(frozen.recalibration["recalibrate"], bool)
    assert set(frozen.validation_intervals) == set(frozen.test_intervals)
    assert set(frozen.by_cohort["cohort"]) <= {"incident", "incumbent"}

    metadata = build_metadata(frozen, FROZEN_RUN, panel, tmp_path / "p.parquet", "synthetic", 20)
    path = tmp_path / "shelf_life.joblib"
    artifact_module.save(frozen.pipeline, metadata, path)
    loaded = artifact_module.load(path)
    assert loaded.metadata.recalibration == frozen.recalibration


def test_a_recalibrated_pipeline_freezes_and_serves(split, monkeypatch):
    """Force the rule to fire on the synthetic block and take the whole path."""
    from tests.conftest import FROZEN_RUN

    monkeypatch.setattr(calibration, "RECALIBRATION_ECE_RATIO", 0.0)
    monkeypatch.setattr(calibration, "MIN_POSITIVES", 1)
    frozen = freeze(split, FROZEN_RUN)
    assert frozen.recalibration["recalibrate"] is True
    assert frozen.recalibration["method"] == "isotonic"
    assert_is_full_pipeline(frozen.pipeline)
    scores = _score(frozen.pipeline, split.test)
    assert scores.between(0.0, 1.0).all()
    assert 0.0 <= frozen.threshold <= 1.0
