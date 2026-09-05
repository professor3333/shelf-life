"""The serving path, and the one test that catches a silent refactor.

`test_a_fixed_posting_scores_the_same_number_forever` is the highest-value test
in the repository, and it is worth being precise about what it protects. Every
other test here asserts a *property* — that a bad payload is refused, that an
unseen category does not crash. Properties survive a change in the feature
logic. A pinned probability does not: change how `age_days` rounds, reorder the
one-hot levels, swap an imputer's strategy, and this number moves. That is the
failure mode nobody notices otherwise, because the model still returns a
confident-looking float afterwards.

**The artifact under test is frozen from the synthetic panel**, not the real one.
`tests/panels.py` explains why the fixtures are synthetic in general — a test
that reads today's scrape changes its mind overnight — and here there is a second
reason: the real panel is still too shallow to split three ways, so there is no
real artifact to pin against yet. When there is, the pinned number moves to it
and this file's structure does not change.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.features.assemble import row_local_features
from src.inference import artifact as artifact_module
from src.inference.contract import (
    FIELDS,
    InvalidPayload,
    board_context_supplied,
    build_row,
    validate,
)
from src.inference.predict import Predictor
from src.models.freeze import freeze
from src.models.metrics import DEFAULT_ALERT_BUDGET, alert_budget, threshold_for_budget
from src.models.train_baseline import prediction_days
from tests.conftest import FROZEN_RUN

RUN = FROZEN_RUN

#: The payload the pinned probability belongs to. Every field a caller would
#: plausibly have, and none they would not: no board context, which is the
#: ordinary serve-time case.
FIXED_POSTING = {
    "title": "Senior Data Engineer",
    "location": "Berlin",
    "salary_raw": "120000 - 160000 USD",
    "departments": "Eng",
    "offices": "HQ",
    "n_offices": 1,
    "n_metadata": 3,
    "content_chars": 1400,
    "first_published": "2026-08-20T00:00:00Z",
    "updated_at": "2026-09-01T00:00:00Z",
}

#: The instant the posting is scored at. Pinned, because `age_days` is measured
#: from it: a test that let `t` default to now would re-score a different, older
#: posting every day and could not assert a constant.
FIXED_T = "2026-09-14T03:45:00Z"

#: What the pipeline in this repository returns for that payload at that instant.
#: Recomputed only when a deliberate change to the feature logic or the model
#: makes it wrong — and when that happens, the diff is the record of what moved.
EXPECTED_PROBABILITY = 0.04348672926425934


@pytest.fixture(scope="module")
def predictor(synthetic_artifact):
    return Predictor.load(synthetic_artifact)


# --- the regression pin -----------------------------------------------------


def test_a_fixed_posting_scores_the_same_number_forever(predictor):
    """A fixed input produces the expected probability. See the module docstring."""
    prediction = predictor.predict(FIXED_POSTING, t=FIXED_T)
    assert prediction.probability == pytest.approx(EXPECTED_PROBABILITY, abs=1e-6)


def test_the_same_payload_scores_the_same_twice(predictor):
    first = predictor.predict(FIXED_POSTING, t=FIXED_T).probability
    assert predictor.predict(FIXED_POSTING, t=FIXED_T).probability == first


# --- the artifact is the whole pipeline -------------------------------------


def test_the_artifact_holds_the_feature_logic_not_just_the_estimator(predictor):
    """The roadmap's REVIEW step, as an assertion rather than an instruction."""
    steps = predictor.artifact.pipeline.named_steps
    assert artifact_module.REQUIRED_STEPS == tuple(
        name for name in steps if name in artifact_module.REQUIRED_STEPS
    )
    assert steps["model"].__class__.__name__ == "XGBClassifier"


def test_saving_a_bare_estimator_is_refused(tmp_path):
    with pytest.raises(artifact_module.ArtifactError, match="not a Pipeline"):
        artifact_module.assert_is_full_pipeline(LogisticRegression())


def test_loading_a_missing_artifact_says_how_to_build_one(tmp_path):
    with pytest.raises(artifact_module.ArtifactError, match="src.models.freeze"):
        artifact_module.load(tmp_path / "absent.joblib")


def test_metadata_carries_the_threshold_and_the_provenance(predictor):
    metadata = predictor.metadata
    assert metadata.run_name == RUN
    assert 0.0 <= metadata.threshold <= 1.0
    assert metadata.provenance["dataset"] == "synthetic"
    assert "test_pr_auc" in metadata.metrics and "val_pr_auc" in metadata.metrics


# --- the input contract -----------------------------------------------------


def test_a_missing_optional_field_does_not_crash(predictor):
    """`title` alone is a valid request. Everything else routes to an imputer."""
    assert 0.0 <= predictor.predict({"title": "Engineer"}, t=FIXED_T).probability <= 1.0


def test_a_category_unseen_at_fit_time_does_not_crash(predictor):
    """§4.3.10: serve-time categories absent from the training fold must not raise."""
    payload = {**FIXED_POSTING, "location": "Ulaanbaatar", "departments": "Falconry"}
    assert 0.0 <= predictor.predict(payload, t=FIXED_T).probability <= 1.0


def test_a_missing_title_is_the_callers_fault():
    with pytest.raises(InvalidPayload, match="missing required"):
        validate({"location": "Berlin"})


def test_an_unknown_field_is_refused_not_ignored():
    """Dropping it silently would compute a confident number without it."""
    with pytest.raises(InvalidPayload, match="unknown field"):
        validate({"title": "Engineer", "salary": "lots"})


def test_a_non_numeric_number_is_the_callers_fault():
    with pytest.raises(InvalidPayload, match="content_chars"):
        validate({"title": "Engineer", "content_chars": "quite long"})


def test_an_unparseable_timestamp_is_the_callers_fault():
    with pytest.raises(InvalidPayload, match="first_published"):
        validate({"title": "Engineer", "first_published": "last Tuesday-ish"})


def test_board_context_is_reported_because_it_changes_what_the_model_saw(predictor):
    without = predictor.predict(FIXED_POSTING, t=FIXED_T)
    with_context = predictor.predict({**FIXED_POSTING, "board_size_at_t": 300}, t=FIXED_T)
    assert not without.board_context_supplied
    assert with_context.board_context_supplied


def test_every_contract_field_reaches_the_row():
    row = build_row({field.name: None for field in FIELDS} | {"title": "Engineer"}, FIXED_T)
    assert all(field.name in row.columns for field in FIELDS)
    assert len(row) == 1


# --- training/serving skew --------------------------------------------------


def test_serve_time_row_local_features_match_the_training_frame():
    """The same posting, through the panel builder and through the contract.

    This is the property that makes a second implementation unnecessary, checked
    rather than assumed: `contract.build_row` calls
    `assemble.row_local_features`, so a change to `age_days` in the panel builder
    changes it identically at serve time. If this test ever has to be relaxed,
    the two paths have diverged and the model is being served features it was not
    fitted on.
    """
    t = pd.Timestamp(FIXED_T)
    training_shaped = row_local_features(
        pd.DataFrame(
            [
                {
                    "t": t,
                    "first_published": pd.Timestamp(FIXED_POSTING["first_published"]),
                    "updated_at": pd.Timestamp(FIXED_POSTING["updated_at"]),
                    "salary_raw": FIXED_POSTING["salary_raw"],
                }
            ]
        )
    )
    served = build_row(FIXED_POSTING, t)

    for column in (
        "age_days",
        "days_since_update",
        "posted_dow",
        "salary_min_clean",
        "salary_max_clean",
        "salary_stated",
        "salary_currency_clean",
    ):
        assert served.loc[0, column] == training_shaped.loc[0, column], column


# --- the discipline the freeze step is supposed to keep ---------------------


def test_the_shipped_threshold_was_chosen_on_validation(frozen, split):
    """Not on test. Recomputed here from the validation block alone and compared."""
    from src.models.freeze import _score

    scores = _score(frozen.pipeline, split.val)
    expected = threshold_for_budget(
        scores,
        alert_budget(len(split.val), prediction_days(split.val), DEFAULT_ALERT_BUDGET),
    )
    assert frozen.threshold == pytest.approx(expected)


def test_a_leaky_run_cannot_be_frozen(split):
    """`06-xgboost_leaky` exists to measure a lie. It must never be servable."""
    with pytest.raises(ValueError, match="deliberately leaky"):
        freeze(split, "06-xgboost_leaky")


def test_board_context_helper_reads_any_not_all():
    assert board_context_supplied({"board_growth": -3})
    assert not board_context_supplied({"title": "Engineer"})
