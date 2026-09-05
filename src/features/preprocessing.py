"""The preprocessing pipeline: raw job-day frame in, model-ready matrix out.

Everything learned from data lives inside one `sklearn` object, so that "fitted
on the training fold only" is a property of *where the object is fitted* rather
than a discipline applied by hand in four different notebooks. There is exactly
one place in this module that learns anything — the `ColumnTransformer` — and
exactly one supported way to fit it, `fit_on_training_fold`, which takes a
`SplitResult` and can only see `.train`.

**The audit, made executable.** `docs/leakage_audit.md` gives a verdict for all
44 panel columns, and prose drifts from code. `FEATURES`, `BOARD_IDENTITY`,
`TEXT_FEATURES` and `EXCLUDED` below are that document as data: every column of
the panel appears in exactly one of them, and `assert_known_columns` refuses a
frame containing a column that appears in none. A column added to the assembly
step is therefore a test failure, not a silent new feature.

**Missingness is per column and deliberate.** No generic
"add-a-missing-indicator-to-everything" step, because that would be a leak of a
specific kind identified in the audit: `first_published`, `updated_at`,
`departments`, `n_metadata` and `content_chars` are null on exactly the 127
python_org rows and present on every Greenhouse row, so an indicator over any of
them reconstructs board identity for free — which is the thing `design.md` §4
has not yet decided to allow. Each column's fill and the reason for it are
carried on the `Column` record.

**Board identity is a switch, not a default.** `design.md` §4 is open, and the
audit widened it: `source` and `company` are the same information (all 31
companies map to exactly one source), so they are gated together by
`include_board_identity` and default to off. Turning one on without the other
would be incoherent.

**`include_leaky` is a switch that should always be off.** It admits `LEAKY`,
whose columns are wrong by construction (`src/features/leaky.py`). It exists
because the experiment history has to contain a measured leak to be worth
keeping — a run that says *this feature was worth 0.4 PR-AUC and all of it was
a lie* is the one artifact of this build that cannot be reconstructed after the
fact. One caller passes it: `src/models/experiments.py`, for run 06.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data.split import SplitResult
from src.features.derive import derive_features
from src.features.leaky import LEAKY_COLUMNS

#: The fill value standing in for "this categorical was not stated". A real
#: level, not a NaN, because "not stated" is information here — `remote` being
#: null was a fact about the board, not an accident.
MISSING_CATEGORY = "__missing__"

#: One-hot levels rarer than this in the *training fold* are folded into an
#: "infrequent" level. Learned from training data only, like every other
#: statistic in the transformer.
MIN_CATEGORY_FREQUENCY = 5

Kind = Literal["numeric", "categorical", "boolean"]
Fill = Literal["median", "zero", "one", "category"]


@dataclass(frozen=True)
class Column:
    """One feature, its type, and what happens to its missing values.

    `why_fill` is not decoration. `docs/leakage_audit.md` requires a per-column
    decision on whether missing means *noise to impute* or *signal to encode*,
    and this is where that decision is recorded next to the code that acts on
    it.
    """

    name: str
    kind: Kind
    fill: Fill
    why_fill: str


FEATURES: tuple[Column, ...] = (
    # --- as-of-t facts about the posting -----------------------------------
    Column(
        "age_days",
        "numeric",
        "median",
        "null only for python_org, which has no archive. Median-imputed and "
        "deliberately *not* indicated: the indicator is board identity",
    ),
    Column("days_since_update", "numeric", "median", "as age_days"),
    Column("content_chars", "numeric", "median", "as age_days"),
    Column("n_offices", "numeric", "median", "as age_days"),
    Column("n_metadata", "numeric", "median", "as age_days"),
    Column(
        "salary_min_clean",
        "numeric",
        "median",
        "null means no pay stated. Not indicated here because salary_stated "
        "already carries exactly that fact as its own feature",
    ),
    Column("salary_max_clean", "numeric", "median", "as salary_min_clean"),
    # --- board context, each windowed to end at t --------------------------
    Column("board_size_at_t", "numeric", "median", "never null; median is a formality"),
    Column("n_same_title_on_board", "numeric", "median", "never null"),
    Column(
        "n_same_req_on_board",
        "numeric",
        "one",
        "null when the posting has no requisition id, where the count is "
        "undefined rather than unknown. One is the honest reading: the "
        "posting stands alone",
    ),
    Column(
        "board_growth",
        "numeric",
        "zero",
        "null on each source's first observed wave, where no previous run "
        "exists to difference against. Zero says 'no observed change', which "
        "is what the panel edge actually means",
    ),
    # --- categoricals ------------------------------------------------------
    Column("departments", "categorical", "category", "null only for python_org"),
    Column("offices", "categorical", "category", "null for python_org and for postings with none"),
    Column("location", "categorical", "category", "free text, but bounded; never null"),
    Column(
        "posted_dow",
        "categorical",
        "category",
        "day of week of publication. Categorical rather than numeric: "
        "Monday is not less than Friday",
    ),
    Column("salary_currency_clean", "categorical", "category", "null when no pay stated"),
    # --- booleans ----------------------------------------------------------
    Column(
        "salary_stated",
        "boolean",
        "zero",
        "did the posting mention pay at all. Never null by construction",
    ),
)

#: Added by `src.features.derive`, which is stateless, so these are computed
#: inside the pipeline and exist identically at serve time. Their hypotheses are
#: stated on the functions that build them and tracked in
#: `reports/feature_hypotheses.md`.
DERIVED: tuple[Column, ...] = (
    Column(
        "title_seniority",
        "categorical",
        "category",
        "never null; 'unspecified' is a level, not a gap",
    ),
    Column("title_is_manager", "numeric", "zero", "a flag; absent means not a manager role"),
    Column("title_words", "numeric", "median", "never null"),
    Column("title_chars", "numeric", "median", "never null"),
    Column("location_is_remote", "numeric", "zero", "a flag; absent means not stated as remote"),
    Column("n_locations", "numeric", "zero", "zero locations means the field was empty"),
    Column(
        "salary_band",
        "categorical",
        "category",
        "'unstated' is already its own level, so nothing is left to impute",
    ),
)

#: Fitted, not derived — see `CompanyVolumeEncoder`. Gated with board identity
#: because on this data it *is* board identity: six of the seven sources have
#: exactly one company each, so a per-company count reproduces `source` with a
#: numeric face and carries no within-board information at all.
COMPANY_VOLUME = "company_posting_volume"


#: Gated by `include_board_identity`. `design.md` §4, widened by the audit:
#: these two are the same information, so they move together.
BOARD_IDENTITY: tuple[Column, ...] = (
    Column("source", "categorical", "category", "never null"),
    Column("company", "categorical", "category", "never null; determines source exactly"),
    Column(
        COMPANY_VOLUME,
        "numeric",
        "zero",
        "fitted on the training fold; zero means an employer we have no record of",
    ),
)

#: Gated by `include_leaky`, and **wrong on purpose**. `src/features/leaky.py`
#: says why each one is a leak; they are registered here rather than left
#: unknown so that a panel carrying them still satisfies `assert_known_columns`,
#: and so that admitting them is a visible argument at a call site instead of a
#: column that quietly appears in the matrix. Nothing in `src/` passes
#: `include_leaky=True` except `src/models/experiments.py`, which does it once,
#: for one run, to measure the size of the lie.
LEAKY: tuple[Column, ...] = (
    Column(
        "n_observations_total",
        "numeric",
        "median",
        "never null once computed — but the value itself is unknowable at `t`, "
        "which is the defect, not the missingness",
    ),
    Column("days_on_board_total", "numeric", "zero", "as n_observations_total"),
)


#: Allowed by the audit, not yet wired. Free text needs derived features, which
#: is the next component's work, not this one's.
TEXT_FEATURES: tuple[str, ...] = ("title",)

#: Every remaining panel column, with the audit's verdict. Present so that a
#: column which is neither a feature nor knowingly excluded cannot exist.
EXCLUDED: dict[str, str] = {
    "t": "axis — it is the prediction point and the thing the split cuts on",
    "run_id": "axis — names the crawl",
    "run_index": "axis — per-source, and not comparable across sources",
    "source_id": "axis — identity, and monotonic in creation order (rho 0.917 vs first_published)",
    "url": "axis — identity, and its domain re-admits board identity",
    "requisition_id": "axis — 1,120 distinct over 1,240 postings; only the windowed count is used",
    "horizon_days": "axis — label-construction metadata",
    "horizon_basis": "axis — label-construction metadata",
    "y": "the target",
    "label_observable": "the censoring flag",
    "split": "assigned by src/data/split.py; describes the experiment",
    "seen_in_train": "assigned by src/data/split.py; a reporting axis, not an input",
    "first_published": "consumed — age_days and posted_dow are derived from it",
    "updated_at": "consumed — days_since_update is derived from it",
    "n_complete_runs_observed": "skew — constant at serve time, and encodes left truncation",
    "t_dow": "axis proxy — with five crawl waves it nearly identifies the wave",
    "posted_month": "skew — collinear with age_days here, and cannot survive a year boundary",
    "remote": "dead — 0 non-null rows once arbeitnow is excluded",
    "n_departments": "dead — constant 1.0",
    "salary_period": "dead — constant 'unstated'",
    "salary_parsed": "dead — identical to salary_stated on every row",
    "salary_min": "dead — superseded by salary_min_clean, worse coverage, one value 1e6 wrong",
    "salary_max": "dead — superseded by salary_max_clean",
    "currency": "dead — superseded by salary_currency_clean",
    "posted_at": "dead — a string date truncated to midnight; first_published is precise",
    "salary_raw": "excluded as text — high cardinality, and its shape fingerprints the board",
}


def feature_columns(
    include_board_identity: bool = False,
    only: tuple[str, ...] | None = None,
    include_leaky: bool = False,
) -> tuple[Column, ...]:
    """The columns the model is allowed to see.

    `only` restricts to a named subset, for baselines that deliberately use one
    feature — `docs/design.md` §7's second baseline is `age_days` alone, and it
    has to travel through the same imputation and scaling as everything else or
    the comparison is not a comparison.

    `include_leaky` admits columns that are known to be wrong. It defaults to
    off and there is no path that turns it on implicitly: it is threaded down
    from `build_pipeline` so that a leaky model is leaky at the call site, in
    an argument a reader can see, rather than three layers away.
    """
    allowed = (
        FEATURES
        + DERIVED
        + (BOARD_IDENTITY if include_board_identity else ())
        + (LEAKY if include_leaky else ())
    )
    if only is None:
        return allowed
    by_name = {column.name: column for column in allowed}
    unknown = sorted(set(only) - set(by_name))
    if unknown:
        raise ValueError(f"not allowed features: {unknown}")
    return tuple(by_name[name] for name in only)


def known_columns() -> set[str]:
    """Every column this module has an opinion about."""
    return (
        {column.name for column in FEATURES}
        | {column.name for column in DERIVED}
        | {column.name for column in BOARD_IDENTITY}
        | set(LEAKY_COLUMNS)
        | set(TEXT_FEATURES)
        | set(EXCLUDED)
    )


def assert_known_columns(frame: pd.DataFrame) -> None:
    """Refuse a frame carrying a column no verdict has been written for.

    This is the mechanism `docs/leakage_audit.md` asks for in its closing
    section. A new column arriving from the assembly step should stop the build
    and force a verdict, rather than either being silently modelled or silently
    dropped — both of which are how an audit becomes fiction.
    """
    unknown = sorted(set(frame.columns) - known_columns())
    if unknown:
        raise ValueError(
            f"no leakage verdict for {unknown}. Add each to FEATURES, BOARD_IDENTITY, "
            f"TEXT_FEATURES or EXCLUDED in src/features/preprocessing.py, and to "
            f"docs/leakage_audit.md. A column with no verdict must not reach a model."
        )


def select_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Pick the feature columns and coerce them to dtypes sklearn is happy with.

    **Stateless on purpose.** It has no `fit`, learns nothing, and stores
    nothing, so it cannot carry information from one fold to another however it
    is called. That is why the selection sits in front of the transformer rather
    than inside it.

    The panel uses pandas' nullable extension dtypes (`Int64`, `Float64`,
    `boolean`, `string`), whose NA sentinel is not `numpy.nan`. Casting here
    keeps every downstream imputer looking at one kind of missing value — and
    the categorical case has teeth: `SimpleImputer` looks for `np.nan` by
    default, so a `None` left in an object column is *not* treated as missing.
    It would be one-hot encoded as its own level, silently bypassing the
    per-column fill policy, and then arrive at serve time as an unknown
    category. Missing categoricals are therefore normalised to `np.nan` here.
    """
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required feature columns: {missing}")

    out = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        if values.dtype == "boolean" or pd.api.types.is_bool_dtype(values):
            out[column] = values.astype("Float64").astype("float64")
        elif pd.api.types.is_numeric_dtype(values):
            out[column] = values.astype("float64")
        else:
            out[column] = values.astype("object").where(values.notna(), np.nan)
    return out


class CompanyVolumeEncoder(BaseEstimator, TransformerMixin):
    """How many postings a company had in the training window. **Fitted.**

    This is the trap the roadmap warns about, and it is worth stating precisely
    because it does not announce itself. Computing a per-company count over the
    whole frame would let each row's encoding depend on rows in the validation
    and test blocks — target leakage through an aggregate, which produces a
    number that looks like an ordinary feature, raises nothing, and inflates the
    score. So the lookup is learned in `fit`, from the training fold only, and
    `transform` applies it as a lookup and nothing more.

    A company unseen at fit time maps to 0: "we have no record of this employer
    posting", which is the honest extrapolation and the situation that will
    dominate at serve time.

    **On this data the feature is board identity.** Six of the seven sources
    have exactly one company each (Anthropic 629 postings, GitLab 249, Figma
    167, Duolingo 92, Discord 54, Airtable 16), so the count reproduces `source`
    almost exactly; the seventh, python_org, has 25 companies with one to three
    postings apiece and therefore almost no variance. That is why it is gated
    with `BOARD_IDENTITY` rather than shipped as an ordinary feature — see
    `docs/leakage_audit.md`.
    """

    def __init__(
        self,
        company_column: str = "company",
        posting_key: tuple[str, str] = ("source", "source_id"),
    ):
        self.company_column = company_column
        self.posting_key = posting_key

    def fit(self, X: pd.DataFrame, y=None) -> CompanyVolumeEncoder:
        key = list(self.posting_key)
        postings = X[[*key, self.company_column]].drop_duplicates(subset=key)
        self.volumes_ = postings[self.company_column].value_counts().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[COMPANY_VOLUME] = (
            X[self.company_column].map(self.volumes_).astype("float64").fillna(0.0)
        )
        return out


def _branch(fill: Fill, min_category_frequency: int = MIN_CATEGORY_FREQUENCY) -> Pipeline:
    """The transformer chain for one missing-value policy."""
    if fill == "category":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
                (
                    "encode",
                    # handle_unknown="ignore" is the requirement that a category
                    # unseen at fit time must not crash at serve time: it encodes
                    # as all-zeros rather than raising. min_frequency folds the
                    # training fold's rare tail into one level, which is learned
                    # from the training fold like everything else here.
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=min_category_frequency,
                        sparse_output=False,
                    ),
                ),
            ]
        )
    strategies = {
        "median": SimpleImputer(strategy="median"),
        "zero": SimpleImputer(strategy="constant", fill_value=0.0),
        "one": SimpleImputer(strategy="constant", fill_value=1.0),
    }
    return Pipeline([("impute", strategies[fill]), ("scale", StandardScaler())])


def build_preprocessor(
    include_board_identity: bool = False,
    min_category_frequency: int = MIN_CATEGORY_FREQUENCY,
    only: tuple[str, ...] | None = None,
    include_leaky: bool = False,
) -> ColumnTransformer:
    """The learned half: imputation, encoding and scaling, grouped by policy.

    Everything with state lives in here. Fitting this object on anything wider
    than a training fold is the leak this component exists to prevent, which is
    why the only supported entry point is `fit_on_training_fold`.
    """
    columns = feature_columns(include_board_identity, only, include_leaky)
    branches = []
    for fill in ("median", "zero", "one", "category"):
        names = [column.name for column in columns if column.fill == fill]
        if names:
            branches.append((f"{fill}_fill", _branch(fill, min_category_frequency), names))
    return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(
    estimator,
    include_board_identity: bool = False,
    min_category_frequency: int = MIN_CATEGORY_FREQUENCY,
    only: tuple[str, ...] | None = None,
    include_leaky: bool = False,
) -> Pipeline:
    """Selection, preprocessing and the estimator as one object.

    One object is the point. The same fitted `Pipeline` is what gets scored,
    persisted and served, so the serving path cannot drift from the training
    path by re-implementing the feature logic — the production form of leakage.
    """
    columns = tuple(
        column.name for column in feature_columns(include_board_identity, only, include_leaky)
    )
    return Pipeline(
        [
            # Stateless, so it cannot leak across the split however it is
            # called, and it produces identical values for a single posting at
            # serve time.
            ("derive", FunctionTransformer(derive_features, validate=False)),
            # Fitted, unlike everything else in front of the transformer, and
            # therefore only ever fitted on the training fold. Included only
            # when board identity is admitted, because on this data it is board
            # identity — see the class docstring.
            *([("company_volume", CompanyVolumeEncoder())] if include_board_identity else []),
            (
                "select",
                FunctionTransformer(select_columns, kw_args={"columns": columns}, validate=False),
            ),
            (
                "preprocess",
                build_preprocessor(
                    include_board_identity, min_category_frequency, only, include_leaky
                ),
            ),
            ("model", estimator),
        ]
    )


def features_and_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a labelled block into inputs and target."""
    assert_known_columns(frame)
    if frame["y"].isna().any():
        raise ValueError("frame contains unlabelled rows; split them out before fitting")
    return frame, frame["y"].astype(int)


def fit_on_frame(pipeline: Pipeline, block: pd.DataFrame) -> Pipeline:
    """Fit on an explicit block. **For cross-validation folds only.**

    `fit_on_training_fold` is the entry point for fitting a model; this one
    exists because a rolling-origin fold is a slice *inside* the training
    window, and there is no `SplitResult` describing it. Passing a frame is the
    mistake this module was built to prevent, so the two callers are named here
    and nowhere else: `fit_on_training_fold` below, and
    `src/models/evaluate.py:cross_validate`, which only ever hands it a slice of
    `split.train`.
    """
    features, target = features_and_target(block)
    return pipeline.fit(features, target)


def fit_on_training_fold(pipeline: Pipeline, split: SplitResult) -> Pipeline:
    """Fit on `split.train` and nothing else.

    The one supported way to fit. It takes a `SplitResult` rather than a frame
    so that the training fold is selected by this module rather than by the
    caller — passing the wrong frame is the leak, and the way to prevent it is
    to stop asking the caller to pass a frame at all.
    """
    return fit_on_frame(pipeline, split.train)


def learned_statistics(pipeline: Pipeline) -> dict[str, float]:
    """The numeric imputers' learned fill values, by column name.

    Exposed so that "was this fitted on the training fold?" is an assertion
    rather than an assurance: compare these against the training fold's medians
    and against the full frame's, and check which one they match.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    statistics: dict[str, float] = {}
    for name, transformer, columns in preprocessor.transformers_:
        if name.endswith("_fill") and not name.startswith("category"):
            imputer = transformer.named_steps["impute"]
            statistics.update(dict(zip(columns, imputer.statistics_, strict=True)))
    return statistics
