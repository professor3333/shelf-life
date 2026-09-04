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
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data.split import Cuts, temporal_split
from src.features.preprocessing import (
    BOARD_IDENTITY,
    EXCLUDED,
    FEATURES,
    MISSING_CATEGORY,
    TEXT_FEATURES,
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

DAY = pd.Timedelta(days=1)
WAVE0 = pd.Timestamp("2026-08-31T03:45:00Z")

NUMERIC = [column.name for column in FEATURES if column.kind == "numeric"]
CATEGORICAL = [column.name for column in FEATURES if column.kind == "categorical"]


def _frame(n_waves: int = 9, per_wave: int = 40, seed: int = 0) -> pd.DataFrame:
    """A panel carrying every column the real one does.

    Numeric values drift upward wave by wave, so that a statistic learned on an
    early block is measurably different from one learned on the whole frame —
    which is the only way to tell the two apart in a test.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for wave in range(n_waves):
        for index in range(per_wave):
            rows.append(
                {
                    "source": "greenhouse:acme",
                    "source_id": f"p{index}",
                    "title": f"Engineer {index}",
                    "company": "Acme",
                    "location": ["Remote", "London", "Berlin"][index % 3],
                    "remote": None,
                    "salary_min": None,
                    "salary_max": None,
                    "currency": None,
                    "salary_raw": None,
                    "posted_at": "2026-01-01",
                    "url": f"https://acme.example/{index}",
                    "run_id": wave,
                    "t": WAVE0 + wave * DAY,
                    "run_index": wave,
                    "y": int(index < 2),  # ~5% positive, present in every wave
                    "label_observable": True,
                    "first_published": WAVE0 - 30 * DAY,
                    "updated_at": WAVE0 - DAY,
                    "departments": ["Eng", "Sales"][index % 2],
                    "n_departments": 1.0,
                    "offices": ["HQ", "Remote"][index % 2],
                    "n_offices": 1.0,
                    "requisition_id": f"R{index}",
                    "n_metadata": 3.0,
                    "content_chars": 1000.0 + 100 * wave + rng.integers(0, 10),
                    "board_size_at_t": float(per_wave),
                    "n_same_title_on_board": 1.0,
                    "n_same_req_on_board": 1.0,
                    "board_growth": 0.0,
                    "n_complete_runs_observed": wave + 1,
                    "age_days": 30.0 + wave + rng.integers(0, 3),
                    "days_since_update": 1.0 + wave,
                    "t_dow": (wave + 6) % 7,
                    "posted_dow": float(index % 7),
                    "posted_month": 1.0,
                    "salary_stated": True,
                    "salary_parsed": True,
                    "salary_min_clean": 50_000.0 + 5_000 * wave,
                    "salary_max_clean": 70_000.0 + 5_000 * wave,
                    "salary_period": "unstated",
                    "salary_currency_clean": "USD",
                    "horizon_days": 1,
                    "horizon_basis": "calendar",
                }
            )
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("Int8")
    return frame


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
    preprocessor.fit(select_columns(frame, columns))

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
    preprocessor.fit(select_columns(frame, columns))

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
    selected = select_columns(frame, columns)
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
    assert "source" not in select_columns(frame, columns).columns

    columns = tuple(column.name for column in feature_columns(include_board_identity=True))
    assert "source" in select_columns(frame, columns).columns


# --- nothing in front of the transformer has state -------------------------


def test_selection_is_stateless():
    """`select_columns` runs before the fitted transformer, so if it learned
    anything it would learn it from whatever frame it was handed — including a
    validation frame at transform time."""
    frame = _frame()
    columns = tuple(column.name for column in feature_columns())

    # Fitting on one frame cannot change how a different frame is transformed.
    selector = FunctionTransformer(select_columns, kw_args={"columns": columns}, validate=False)
    before = selector.transform(frame)
    selector.fit(_frame(seed=99, per_wave=7))
    pd.testing.assert_frame_equal(before, selector.transform(frame))

    # And it does not depend on row order either.
    first = select_columns(frame, columns)
    second = select_columns(frame.iloc[::-1], columns)
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
