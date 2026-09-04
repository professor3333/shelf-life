"""Tests for the preprocessing pipeline.

The three the component specification demands are
`test_a_category_unseen_at_fit_time_does_not_raise`,
`test_imputer_learns_the_training_folds_median_not_the_full_frames` and
`test_transforming_a_later_block_produces_no_nans`. The rest defend the property
that makes the first three worth having: that the set of columns reaching a
model is exactly the set `docs/leakage_audit.md` cleared, and that nothing in
front of the transformer has any state to leak through.

Fixtures are built by hand and nothing reads `data/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from panels import DAY, WAVE0, make_panel
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.split import Cuts, temporal_split
from src.features.derive import DERIVED_COLUMNS, derive_features
from src.features.preprocessing import (
    BOARD_IDENTITY,
    COMPANY_VOLUME,
    DERIVED,
    EXCLUDED,
    FEATURES,
    MISSING_CATEGORY,
    TEXT_FEATURES,
    CompanyVolumeEncoder,
    assert_known_columns,
    build_pipeline,
    build_preprocessor,
    feature_columns,
    features_and_target,
    fit_on_training_fold,
    known_columns,
    learned_statistics,
    select_columns,
)

NUMERIC = [column.name for column in FEATURES if column.kind == "numeric"]
CATEGORICAL = [column.name for column in FEATURES if column.kind == "categorical"]


_frame = make_panel


def _split(frame: pd.DataFrame):
    return temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


def _pipeline(**kwargs) -> Pipeline:
    return build_pipeline(LogisticRegression(max_iter=1000), min_category_frequency=1, **kwargs)


# --- the three the component asks for --------------------------------------


def test_a_category_unseen_at_fit_time_does_not_raise():
    """Serve time will show the model a department that did not exist when it
    was fitted. That must encode, not crash."""
    frame = _frame()
    fitted = _pipeline().fit(*features_and_target(frame))

    unseen = frame.head(5).copy()
    unseen["departments"] = "Xenobiology"
    unseen["location"] = "Ulaanbaatar"
    unseen["salary_currency_clean"] = "MNT"

    probabilities = fitted.predict_proba(unseen)[:, 1]
    assert probabilities.shape == (5,)
    assert np.isfinite(probabilities).all()


def test_imputer_learns_the_training_folds_median_not_the_full_frames():
    """The single most likely place for AI-written leakage: a transformer fitted
    before the split. Both medians are computed here and asserted to differ, so
    that matching the training fold is evidence rather than coincidence."""
    frame = _frame()
    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)
    learned = learned_statistics(fitted)

    train_median = split.train["age_days"].median()
    full_median = split.frame["age_days"].median()
    assert train_median != full_median, "fixture is degenerate; the test proves nothing"

    assert learned["age_days"] == pytest.approx(train_median)
    assert learned["age_days"] != pytest.approx(full_median)

    for column in ("content_chars", "salary_min_clean", "days_since_update"):
        assert learned[column] == pytest.approx(split.train[column].median())


def test_transforming_a_later_block_produces_no_nans():
    frame = _frame()
    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)

    matrix = fitted[:-1].transform(split.val)
    assert not np.isnan(np.asarray(matrix, dtype=float)).any()
    assert len(matrix) == len(split.val)


# --- missing values, per column and deliberate -----------------------------


def test_each_policy_fills_with_the_value_its_reason_states():
    frame = _frame()
    frame.loc[frame.index[:10], "board_growth"] = None
    frame.loc[frame.index[:10], "n_same_req_on_board"] = None
    frame.loc[frame.index[:10], "salary_currency_clean"] = None

    preprocessor = build_preprocessor(min_category_frequency=1)
    columns = tuple(column.name for column in feature_columns())
    preprocessor.fit(select_columns(derive_features(frame), columns))

    fills = {}
    for name, transformer, names in preprocessor.transformers_:
        if name != "category_fill":
            imputer = transformer.named_steps["impute"]
            fills.update(dict(zip(names, imputer.statistics_, strict=True)))
    assert fills["board_growth"] == 0.0  # the panel edge means "no observed change"
    assert fills["n_same_req_on_board"] == 1.0  # no requisition id: the posting stands alone

    encoded = preprocessor.get_feature_names_out()
    assert any(MISSING_CATEGORY in name for name in encoded)


def test_there_is_no_blanket_missing_indicator():
    """An indicator over the archive-derived columns reconstructs board identity,
    which design.md §4 has not decided to allow. Missingness is encoded only for
    categoricals, where it is a real level."""
    frame = _frame()
    preprocessor = build_preprocessor(min_category_frequency=1)
    columns = tuple(column.name for column in feature_columns())
    preprocessor.fit(select_columns(derive_features(frame), columns))

    indicators = [
        name for name in preprocessor.get_feature_names_out() if "missingindicator" in name
    ]
    assert indicators == []
    for numeric in NUMERIC:
        assert not any(
            name.startswith(f"{numeric}_") for name in preprocessor.get_feature_names_out()
        )


# --- the audit, made executable --------------------------------------------


def test_every_panel_column_has_exactly_one_verdict():
    frame = _frame()
    verdict_sets = [
        {column.name for column in FEATURES},
        {column.name for column in BOARD_IDENTITY},
        {column.name for column in DERIVED},
        set(TEXT_FEATURES),
        set(EXCLUDED),
    ]
    for i, first in enumerate(verdict_sets):
        for second in verdict_sets[i + 1 :]:
            assert not (first & second), f"column appears in two verdict sets: {first & second}"
    assert set(frame.columns) <= known_columns()


def test_a_column_with_no_verdict_is_refused():
    """A new column arriving from the assembly step must stop the build rather
    than become a silent feature or a silent omission."""
    frame = _frame()
    frame["days_until_deadline"] = 3.0
    with pytest.raises(ValueError, match="no leakage verdict"):
        assert_known_columns(frame)


def test_excluded_columns_never_reach_the_matrix():
    frame = _frame()
    columns = tuple(column.name for column in feature_columns())
    selected = select_columns(derive_features(frame), columns)
    assert set(selected.columns).isdisjoint(EXCLUDED)
    assert set(selected.columns).isdisjoint(TEXT_FEATURES)


def test_every_feature_carries_a_stated_reason_for_its_fill():
    for column in FEATURES + BOARD_IDENTITY:
        assert column.why_fill.strip(), f"{column.name} has no documented missing-value policy"


# --- board identity is a switch --------------------------------------------


def test_board_identity_is_off_by_default_and_moves_as_a_pair():
    default = {column.name for column in feature_columns()}
    assert "source" not in default and "company" not in default

    enabled = {column.name for column in feature_columns(include_board_identity=True)}
    assert {"source", "company"} <= enabled


def test_board_identity_columns_reach_the_matrix_only_when_enabled():
    frame = _frame()
    columns = tuple(column.name for column in feature_columns())
    assert "source" not in select_columns(derive_features(frame), columns).columns

    columns = tuple(column.name for column in feature_columns(include_board_identity=True))
    prepared = CompanyVolumeEncoder().fit(frame).transform(derive_features(frame))
    selected = select_columns(prepared, columns)
    assert "source" in selected.columns
    assert COMPANY_VOLUME in selected.columns


# --- nothing in front of the transformer has state -------------------------


def test_selection_is_stateless():
    """`select_columns` runs before the fitted transformer, so if it learned
    anything it would learn it from whatever frame it was handed — including a
    validation frame at transform time."""
    frame = _frame()
    columns = tuple(column.name for column in feature_columns())

    # Fitting on one frame cannot change how a different frame is transformed.
    derived = derive_features(frame)
    selector = FunctionTransformer(select_columns, kw_args={"columns": columns}, validate=False)
    before = selector.transform(derived)
    selector.fit(derive_features(_frame(seed=99, per_wave=7)))
    pd.testing.assert_frame_equal(before, selector.transform(derived))

    # And it does not depend on row order either.
    first = select_columns(derived, columns)
    second = select_columns(derived.iloc[::-1], columns)
    pd.testing.assert_frame_equal(first, second.iloc[::-1])


def test_fit_on_training_fold_cannot_see_validation_or_test():
    frame = _frame()
    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)
    learned = learned_statistics(fitted)

    for block in ("val", "test"):
        rows = split.frame[split.frame["split"] == block]
        assert learned["age_days"] != pytest.approx(rows["age_days"].median())


def test_unlabelled_rows_are_refused_before_fitting():
    frame = _frame()
    frame.loc[frame.index[0], "y"] = pd.NA
    with pytest.raises(ValueError, match="unlabelled"):
        features_and_target(frame)


def test_fitting_twice_learns_the_same_thing():
    frame = _frame()
    split = _split(frame)
    first = learned_statistics(fit_on_training_fold(_pipeline(), split))
    second = learned_statistics(fit_on_training_fold(_pipeline(), split))
    assert first == second


# --- one object, raw frame in --------------------------------------------


def test_the_pipeline_takes_a_raw_frame_with_no_manual_step():
    """`Done when`: hand it a raw frame and it handles everything. The estimator
    is deliberately a Dummy — this asserts the plumbing, not a model."""
    frame = _frame()
    split = _split(frame)
    pipeline = build_pipeline(DummyClassifier(strategy="prior"), min_category_frequency=1)
    fit_on_training_fold(pipeline, split)
    probabilities = pipeline.predict_proba(split.val)[:, 1]
    assert probabilities.shape == (len(split.val),)


# --- engineered features live inside the pipeline --------------------------


def test_the_derived_column_list_and_the_feature_records_agree():
    """Two lists naming the same columns drift. If they do, `select_columns`
    raises at fit time — but a test is a better place to find out."""
    assert {column.name for column in DERIVED} == set(DERIVED_COLUMNS)


def test_engineered_features_are_computed_inside_the_pipeline():
    """The component's real test: hand the pipeline a raw panel with no derived
    columns on it, and the derived features must still reach the matrix. A
    feature that needed code outside the pipeline would break at serve time."""
    frame = _frame()
    assert not set(DERIVED_COLUMNS) & set(frame.columns)

    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)
    encoded = fitted.named_steps["preprocess"].get_feature_names_out()

    for name in DERIVED_COLUMNS:
        assert any(str(column).startswith(name) for column in encoded), name


def test_the_pipeline_scores_a_single_raw_posting():
    """Serve time is one row with no board context computed for it. If the
    derived features needed a frame-wide pass, this is where it shows."""
    frame = _frame()
    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)
    one = split.val.head(1)
    assert fitted.predict_proba(one).shape == (1, 2)


def test_company_volume_counts_the_training_fold_and_nothing_else():
    """The subtlest leak the roadmap names: a per-company aggregate computed over
    the full frame lets each row's encoding depend on rows in validation and
    test. It raises nothing and inflates the score, so the counts are checked
    against both windows and asserted to match the training one."""
    frame = make_panel(per_wave=40)
    frame["company"] = np.where(frame["source_id"].isin(["p0", "p1"]), "Small", "Large")
    split = _split(frame)

    encoder = CompanyVolumeEncoder().fit(split.train)
    train_counts = (
        split.train.drop_duplicates(subset=["source", "source_id"])["company"]
        .value_counts()
        .to_dict()
    )
    everything = (
        split.frame.drop_duplicates(subset=["source", "source_id"])["company"]
        .value_counts()
        .to_dict()
    )
    assert encoder.volumes_ == train_counts
    assert encoder.volumes_ != everything or train_counts == everything


def test_company_volume_counts_postings_not_job_days():
    """A posting seen in five waves is one posting. Counting rows would make the
    feature a proxy for how long we have been watching, which is the skew verdict
    that excluded `n_complete_runs_observed`."""
    frame = make_panel(n_waves=9, per_wave=10)
    encoder = CompanyVolumeEncoder().fit(frame)
    assert encoder.volumes_ == {"Acme": 10}  # ten postings, ninety job-days


def test_an_unseen_company_maps_to_zero_volume():
    frame = make_panel(per_wave=40)
    split = _split(frame)
    encoder = CompanyVolumeEncoder().fit(split.train)
    newcomer = split.val.head(3).copy()
    newcomer["company"] = "Never Heard Of"
    assert (encoder.transform(newcomer)[COMPANY_VOLUME] == 0.0).all()


def test_company_volume_is_absent_unless_board_identity_is_admitted():
    frame = make_panel()
    split = _split(frame)
    fitted = fit_on_training_fold(_pipeline(), split)
    assert "company_volume" not in fitted.named_steps

    with_board = build_pipeline(
        LogisticRegression(max_iter=1000), include_board_identity=True, min_category_frequency=1
    )
    fit_on_training_fold(with_board, split)
    assert "company_volume" in with_board.named_steps
