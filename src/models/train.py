"""XGBoost, the feature ablation, and the deliberate overfit.

Three things live here, and only the first is a model.

**1. The boosted rung.** It sits on top of the ladder in
`src/models/train_baseline.py` and is compared against it rather than reported
alone. The question is not "what does XGBoost score" but *whether increasing
complexity buys anything on this problem* — and if the forest ties it, that is a
finding about the data, to be written down rather than tuned past.

**2. The ablation.** Every engineered feature was a hypothesis before it was a
column (`src/features/derive.py` states each one). `ablate` refits with one
feature removed at a time and reports what the score does without it, which is
the only way a hypothesis gets tested rather than assumed. A feature whose
removal costs nothing is a feature to drop, and a feature whose removal costs a
great deal is a feature to look at hard — **a jump you cannot explain is leakage
until proven otherwise**, and the ablation is where an unexplained jump becomes
visible.

**3. The deliberate overfit.** `overfit_sweep` pushes depth up and regularisation
off until train and validation separate visibly, then closes the gap again,
recording which knob did what. The point is to see the separation happen on your
own data rather than read about it.

Every fit goes through `fit_on_training_fold`, so no experiment here can see the
validation block while learning, and none of them touches test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.data.split import SplitResult, SplitTooShallow, best_cuts, temporal_split
from src.features.assemble import horizon_banner
from src.features.derive import DERIVED_COLUMNS
from src.features.preprocessing import (
    DERIVED,
    FEATURES,
    build_pipeline,
    feature_columns,
    features_and_target,
    fit_on_training_fold,
)
from src.inference.contract import BOARD_CONTEXT
from src.models import provenance, tracking
from src.models.metrics import DEFAULT_ALERT_BUDGET, evaluate
from src.models.train_baseline import (
    DEFAULT_PANEL,
    LADDER,
    RANDOM_STATE,
    _table,
    analytic_reference,
    prediction_days,
    run_ladder,
)

DEFAULT_REPORT = Path("reports/model_results.md")


def scale_pos_weight(split: SplitResult) -> float:
    """`negatives / positives` in the training fold.

    XGBoost's handle on imbalance. Computed from the training fold rather than
    the frame, for the same reason every other statistic is: it is a quantity
    learned from data, and a quantity learned from the whole frame is a leak
    however innocuous it looks.
    """
    target = split.train["y"].astype(int)
    positives = int(target.sum())
    return float((len(target) - positives) / positives) if positives else 1.0


def _xgb_parameters(split: SplitResult, **overrides) -> dict:
    parameters = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight(split),
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    parameters.update(overrides)
    return parameters


def build_xgboost(split: SplitResult, **overrides) -> object:
    """The boosted rung, wrapped in the same pipeline as everything else."""
    return build_pipeline(XGBClassifier(**_xgb_parameters(split, **overrides)))


def _score(model, block: pd.DataFrame, budget_per_day: int) -> dict[str, float]:
    features, target = features_and_target(block)
    scores = model.predict_proba(features)[:, 1]
    return evaluate(target, scores, n_days=prediction_days(block), budget_per_day=budget_per_day)


def fit_and_score(
    model, split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> dict[str, float]:
    """Fit on the training fold, report train and validation together.

    Both, always. A validation number on its own says how good the model is; the
    pair says whether it is memorising, which is the question this component
    exists to make visible.
    """
    fit_on_training_fold(model, split)
    train = _score(model, split.train, budget_per_day)
    validation = _score(model, split.val, budget_per_day)
    return {
        "train_pr_auc": train["pr_auc"],
        "val_pr_auc": validation["pr_auc"],
        "gap": train["pr_auc"] - validation["pr_auc"],
        "val_brier": validation["brier"],
        "val_roc_auc": validation["roc_auc"],
        "val_precision": validation["precision"],
        "val_recall": validation["recall"],
    }


@dataclass(frozen=True)
class Ablation:
    """One refit, with `features` withheld. `name` is what the report calls it.

    `features` is a tuple rather than a string because the question that matters
    most here cannot be asked one column at a time — see `BOARD_CONTEXT_ABLATION`.
    A single-feature ablation is the one-element case, so there is one code path.
    """

    name: str
    features: tuple[str, ...]
    hypothesis: str


#: The four columns describing the board rather than the posting, taken from the
#: request contract rather than re-listed here. `contract.BOARD_CONTEXT` is
#: derived from `Field.origin == "board"`, which is the same predicate that
#: decides whether `POST /predict` treats a field as optional — so a field that
#: changes origin changes what gets ablated, and cannot change one without the
#: other. `test_the_board_ablation_tracks_the_request_contract` pins that.
BOARD_CONTEXT_ABLATION = Ablation(
    name="board context (all four)",
    features=BOARD_CONTEXT,
    hypothesis=(
        "board context is worth having at all — what `docs/design.md` §12 was decided "
        "on, and the closest a refit gets to what a caller actually loses"
    ),
)

ABLATIONS: tuple[Ablation, ...] = (
    *(
        Ablation(name, (name,), hypothesis)
        for name, hypothesis in (
            ("title_seniority", "how senior a role is relates to how long it takes to fill"),
            ("title_is_manager", "management roles have longer hiring processes"),
            ("title_words", "terse titles are boilerplate on high-volume reqs and churn faster"),
            ("title_chars", "as title_words, by another measure"),
            ("location_is_remote", "remote roles draw a larger pool and close faster"),
            ("n_locations", "a posting open in several places is a wider net"),
            ("salary_band", "pay level relates to fill speed, non-linearly"),
            # The four board columns, individually. Kept alongside the group
            # because the pair distinguishes two findings a single test conflates:
            # four small deltas with a large group delta means "redundant with
            # each other", four small deltas with a small group delta means
            # "worthless". Only the second licenses dropping them.
            ("board_size_at_t", "a posting on a large board competes with more of them"),
            ("board_growth", "a board that is growing is hiring, and hiring boards close reqs"),
            (
                "n_same_title_on_board",
                "a title duplicated across the board is a high-volume req and churns",
            ),
            (
                "n_same_req_on_board",
                "one requisition posted in several places closes when any of them does",
            ),
        )
    ),
    BOARD_CONTEXT_ABLATION,
)


def serve_time_regime(
    split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> pd.DataFrame:
    """What a caller who supplies no board context actually gets.

    **This is not the group ablation, and the difference is the point.** The
    ablation *refits* without the four columns, which produces a different model
    — one whose remaining features have absorbed whatever weight the board
    columns held. The deployed object is the opposite: fitted *with* them, then
    handed rows where they are null, so the training fold's imputers fill them
    with constants and the fitted weights stay pointed at a column that no longer
    varies. A refit measures what the features are worth; this measures what the
    shipped model does when they are missing, and only the second is what
    `POST /predict` returns to a stranger.

    Same model, same split, same threshold — the only difference is whether the
    four columns arrive. Anything else would confound the regime with a refit.

    `docs/design.md` §12 was written as though the ablation answered this. It
    does not, and until 2026-09-09 nothing did: `board_context_supplied` told a
    caller which regime they were in without anyone having measured what the
    other regime costs.
    """
    model = build_xgboost(split)
    fit_on_training_fold(model, split)

    withheld = split.val.copy()
    for column in BOARD_CONTEXT:
        # `float64` NaN rather than the column's own dtype: these arrive as
        # `Int64` from the panel but a plain `int64` from any frame that never
        # had a null, and a non-nullable integer column cannot hold the absence
        # this is trying to represent. Every board column is numeric, so
        # `select_columns` routes it to the same branch either way and the
        # imputer sees exactly what a caller who omitted the field would send.
        withheld[column] = pd.Series(np.nan, index=withheld.index, dtype="float64")

    rows = []
    for name, block in (("board context supplied", split.val), ("absent, imputed", withheld)):
        scored = _score(model, block, budget_per_day)
        rows.append(
            {
                "regime": name,
                "n": float(len(block)),
                "pr_auc": scored["pr_auc"],
                "brier": scored["brier"],
                "precision": scored["precision"],
                "recall": scored["recall"],
            }
        )
    supplied, absent = rows
    absent["delta_pr_auc"] = supplied["pr_auc"] - absent["pr_auc"]
    supplied["delta_pr_auc"] = 0.0
    return pd.DataFrame(rows)


def kept_features(withheld: tuple[str, ...]) -> tuple[str, ...]:
    """The feature subset an ablation actually fits on.

    One function because two callers need the same answer: `ablate` builds the
    pipeline from it, and the MLflow logging records it as the run's feature
    subset. Recomputing it in the logger would work until the day the two drift,
    and then the tracking store would attribute a metric to a set of columns the
    model never saw — which is worse than not recording the columns at all.
    """
    excluded = set(withheld)
    return tuple(column.name for column in feature_columns() if column.name not in excluded)


def ablate(split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET) -> pd.DataFrame:
    """Refit without each ablation's features in turn.

    Leave-one-out rather than add-one-in, because the question a hypothesis
    poses is "does the model need this?", and a feature can be redundant with
    another without being useless on its own. `delta` is the validation PR-AUC
    the withheld columns are worth: positive means removing them hurt, so they
    earned their place.

    **One ablation withholds four columns at once, and it is the important
    one.** `docs/design.md` §12 is open on whether the board-context features
    should be in the served model at all: they are honestly as-of-`t` and are
    not a leak, but a job seeker holding one posting cannot supply any of them,
    so at serve time they arrive as nulls and the training fold's imputers fill
    them with constants. Leave-one-out cannot price that. `board_size_at_t` and
    `board_growth` correlate at 0.42 on the observed panel, so dropping either
    leaves its partner carrying the signal and both deltas read near zero while
    the pair is worth something. And the serving question is not "what does
    dropping one cost" — it is *what does a caller who supplies none of them
    lose*, which is one refit with all four withheld.

    Every ablation shares the baseline's split, seed and preprocessing, so a
    delta is attributable to the withheld columns and nothing else. That is the
    same discipline runs 05/06/07 use in `src/models/experiments.py`, and it is
    what makes a difference evidence rather than an observation.
    """
    baseline = fit_and_score(build_xgboost(split), split, budget_per_day)

    rows = [{"removed": "nothing", "n_removed": 0, "hypothesis": "—", **baseline, "delta": 0.0}]
    for ablation in ABLATIONS:
        kept = kept_features(ablation.features)
        model = build_pipeline(XGBClassifier(**_xgb_parameters(split)), only=kept)
        scored = fit_and_score(model, split, budget_per_day)
        rows.append(
            {
                "removed": ablation.name,
                "n_removed": len(ablation.features),
                "hypothesis": ablation.hypothesis,
                **scored,
                "delta": baseline["val_pr_auc"] - scored["val_pr_auc"],
            }
        )
    return pd.DataFrame(rows)


#: Depth and regularisation, walked from "cannot overfit" to "overfits badly"
#: and back. The first three rungs open the gap; the last three close it, one
#: knob at a time, so that each knob's effect is attributable.
OVERFIT_SWEEP: tuple[tuple[str, dict], ...] = (
    ("stump, heavy shrinkage", {"max_depth": 1, "n_estimators": 50, "reg_lambda": 10.0}),
    ("moderate", {"max_depth": 4, "n_estimators": 400, "reg_lambda": 1.0}),
    (
        "deep, unregularised",
        {
            "max_depth": 12,
            "n_estimators": 1200,
            "reg_lambda": 0.0,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    ),
    (
        "deep + min_child_weight",
        {
            "max_depth": 12,
            "n_estimators": 1200,
            "reg_lambda": 0.0,
            "min_child_weight": 30,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    ),
    (
        "deep + lambda",
        {
            "max_depth": 12,
            "n_estimators": 1200,
            "reg_lambda": 50.0,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    ),
    (
        "deep + subsampling",
        {
            "max_depth": 12,
            "n_estimators": 1200,
            "reg_lambda": 0.0,
            "min_child_weight": 1,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
        },
    ),
)


def overfit_sweep(split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET) -> pd.DataFrame:
    """Open the train/validation gap on purpose, then close it one knob at a time."""
    rows = []
    for name, overrides in OVERFIT_SWEEP:
        model = build_pipeline(XGBClassifier(**_xgb_parameters(split, **overrides)))
        rows.append({"setting": name, **overrides, **fit_and_score(model, split, budget_per_day)})
    return pd.DataFrame(rows)


def board_context_folds(
    split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The board-context question, answered fold by fold rather than once.

    `ablate` refits on the training window and scores on validation, which gives
    one number per ablation. For most of the seven engineered features that is
    enough — they are cheap hypotheses and a single delta is proportionate. It is
    **not** enough for the board-context group, because that delta is what
    `docs/design.md` §12 will close on, and one number on one validation block
    cannot say whether a difference is real.

    So this refits the full model and the board-ablated model across the
    rolling-origin folds *inside the training window* and pairs them fold by
    fold. Paired, because both models see the same validation wave in each fold
    and therefore share whatever made that wave easy or hard; differencing within
    a fold removes that shared component, where scoring each model separately and
    subtracting the means leaves it in.

    **The folds are cut from `split.train` and the test block is never read.**
    Choosing a feature set is model selection, and selection against test is a
    choice that cannot be un-made.

    Returns the per-fold table and the paired summary. `wins` in that summary is
    the number to read first: on a handful of folds, a mean ± sd invites more
    confidence than the sample size supports, and "won 2 of 3" is a statement a
    reader can check.
    """
    # Imported here rather than at module scope: `src.models.evaluate` imports
    # `build_xgboost` from this module, so a top-level import either way round is
    # a cycle. The dependency is real and points that direction — evaluate is
    # built on top of the estimators defined here — so the deferral is on this
    # side, where it is one function rather than a module-wide constraint.
    from src.models.evaluate import (
        cross_validate,
        paired_fold_difference,
        wave_forward_folds,
    )

    folds = wave_forward_folds(split.train, split.embargo)
    if not folds:
        return pd.DataFrame(), {"folds": 0.0}

    kept = kept_features(BOARD_CONTEXT_ABLATION.features)
    parameters = _xgb_parameters(split)

    full = cross_validate(
        lambda: build_pipeline(XGBClassifier(**parameters)),
        split.train,
        folds,
        budget_per_day,
    )
    ablated = cross_validate(
        lambda: build_pipeline(XGBClassifier(**parameters), only=kept),
        split.train,
        folds,
        budget_per_day,
    )

    paired = paired_fold_difference(full, ablated)
    table = full[["fold", "pr_auc"]].merge(
        ablated[["fold", "pr_auc"]], on="fold", suffixes=("_full", "_without_board")
    )
    table["difference"] = table["pr_auc_full"] - table["pr_auc_without_board"]
    return table, paired


#: Which columns of each family's table are measurements. Everything else in a
#: row identifies the variant and becomes a param or a tag, because a param is
#: what you filter runs by and a metric is what you sort them on — putting a
#: name in the metric slot loses the first and putting a number in the param
#: slot loses the second.
_ABLATION_METRICS = (
    "train_pr_auc",
    "val_pr_auc",
    "gap",
    "val_brier",
    "val_roc_auc",
    "val_precision",
    "val_recall",
    "delta",
)
_REGIME_METRICS = ("n", "pr_auc", "brier", "precision", "recall", "delta_pr_auc")


def _sweep_figures(sweep: pd.DataFrame, figures_dir: Path | None) -> list[Path]:
    """The train-against-validation figure, or nothing if it cannot be drawn.

    `figures_dir` is a parameter rather than a module constant read at call time
    because a test that exercises this must not write into `reports/`, and
    `tests/conftest.py` fails the ones that try. Binding the destination at the
    call site is the version of that which cannot be forgotten.
    """
    from src import plots

    if figures_dir is None or not plots.available():
        return []
    return [plots.complexity_gap(sweep, figures_dir / "complexity_gap.png")]


def log_experiments(
    mlflow,
    split: SplitResult,
    prov,
    budget_per_day: int,
    *,
    ladder: pd.DataFrame | None = None,
    ablations: pd.DataFrame | None = None,
    sweep: pd.DataFrame | None = None,
    fold_evidence: tuple[pd.DataFrame, dict[str, float]] | None = None,
    serve_time: pd.DataFrame | None = None,
    figures_dir: Path | None = None,
) -> int:
    """Write every experiment this module runs to the tracking store.

    **Nested, not flat.** Each of these is a *family* of refits that only means
    anything read beside its siblings: an ablation's `delta` is defined against
    the baseline in the same table, and the overfit sweep's whole point is the
    shape of the gap across settings. Logged as a dozen unrelated sibling runs
    they would be arithmetic anyone could redo and nobody could find. So each
    family gets a parent run carrying what the variants share — the split, the
    dataset, the code, the alert budget — and each variant a child carrying only
    what makes it different.

    **Every child records its own feature subset**, from `kept_features`, which
    is the same function `ablate` builds its pipeline from. That is the field
    this whole exercise is about: an ablation is *defined* by the columns it
    withheld, and a run that records the metric without the subset has kept the
    answer and thrown away the question.

    Returns the number of runs written, parents included, so the caller can say
    what happened rather than assuming it worked.
    """
    from src.models import tracking

    shared = {"alert_budget_per_day": budget_per_day, **tracking.split_params(split)}
    # The estimator's parameters go on the parent for the families that hold them
    # constant, and on the child for the sweep, which exists precisely to vary
    # them. Logging them in both places would invite a reader to compare a
    # child's value against a parent's and find them disagreeing by design.
    base_model_params = {f"model__{k}": v for k, v in _xgb_parameters(split).items()}
    written = 0

    if ladder is not None and not ladder.empty:
        # Each rung's subset comes from building the rung again and reading the
        # column list off it. No fit is needed — `build_pipeline` fixes the list
        # at construction — so this costs nothing and cannot disagree with what
        # was scored, which recomputing it from the registry could.
        builders = {rung.name: rung.build for rung in LADDER}
        with tracking.family(mlflow, "ladder", prov=prov, params=shared):
            written += 1
            for _, row in ladder.iterrows():
                name = str(row["model"])
                build = builders.get(name)
                features = tracking.pipeline_features(build()) if build else kept_features(())
                written += 1
                tracking.log_variant(
                    mlflow,
                    run_name=f"rung: {name}",
                    prov=prov,
                    nested=True,
                    params={"model": name, **shared},
                    metrics={
                        key: value
                        for key, value in row.items()
                        if isinstance(value, int | float) and not isinstance(value, bool)
                    },
                    tags={"family": "ladder", "description": row.get("description", "")},
                    features=features,
                )

    if ablations is not None and not ablations.empty:
        withheld_by_name = {"nothing": ()}
        withheld_by_name.update({a.name: a.features for a in ABLATIONS})
        with tracking.family(
            mlflow, "ablations", prov=prov, params={**shared, **base_model_params}
        ):
            written += 1
            for _, row in ablations.iterrows():
                removed = str(row["removed"])
                written += 1
                tracking.log_variant(
                    mlflow,
                    run_name=f"ablation: {removed}",
                    prov=prov,
                    nested=True,
                    params={
                        "removed": removed,
                        "n_removed": int(row["n_removed"]),
                        **shared,
                    },
                    metrics={key: row[key] for key in _ABLATION_METRICS if key in row},
                    tags={"family": "ablations", "hypothesis": row.get("hypothesis", "")},
                    features=kept_features(withheld_by_name.get(removed, ())),
                )

    if sweep is not None and not sweep.empty:
        # The gap plot describes all six settings at once and belongs to none of
        # them, so it hangs on the family parent rather than on a child.
        sweep_figures = _sweep_figures(sweep, figures_dir)
        # Read from the sweep's own definition rather than from the result frame.
        # Settings override different knobs, so the frame carries a NaN wherever a
        # setting left one alone, and logging that NaN as a parameter would record
        # a value the model never used.
        overrides_by_setting = dict(OVERFIT_SWEEP)
        with tracking.family(mlflow, "overfit sweep", prov=prov, params=shared):
            written += 1
            tracking.log_figures(mlflow, sweep_figures)
            for _, row in sweep.iterrows():
                setting = str(row["setting"])
                effective = _xgb_parameters(split, **overrides_by_setting.get(setting, {}))
                written += 1
                tracking.log_variant(
                    mlflow,
                    run_name=f"overfit: {setting}",
                    prov=prov,
                    nested=True,
                    params={
                        **{f"model__{k}": v for k, v in effective.items()},
                        "setting": setting,
                        **shared,
                    },
                    metrics={key: row[key] for key in _ABLATION_METRICS if key in row},
                    tags={"family": "overfit sweep"},
                    features=kept_features(()),
                )

    if fold_evidence is not None:
        table, paired = fold_evidence
        if not table.empty:
            with tracking.family(
                mlflow, "board context, per fold", prov=prov, params={**shared, **base_model_params}
            ):
                written += 1
                mlflow.log_metrics({k: float(v) for k, v in paired.items() if pd.notna(v)})
                mlflow.log_text(table.to_csv(index=False), "per_fold.csv")
                # The table is one row per fold with a column per arm, which is
                # the shape the paired comparison needs. Each child takes its own
                # column: reading the frame's numeric mean would hand both arms
                # the same figures and quietly make the comparison vacuous.
                for variant, column, withheld in (
                    ("with board context", "pr_auc_full", ()),
                    (
                        "without board context",
                        "pr_auc_without_board",
                        BOARD_CONTEXT_ABLATION.features,
                    ),
                ):
                    scores = table[column]
                    written += 1
                    tracking.log_variant(
                        mlflow,
                        run_name=f"board context: {variant}",
                        prov=prov,
                        nested=True,
                        params={"variant": variant, "folds": int(len(table)), **shared},
                        metrics={
                            "cv_pr_auc_mean": scores.mean(),
                            "cv_pr_auc_sd": scores.std(ddof=1) if len(scores) > 1 else float("nan"),
                        },
                        tags={"family": "board context, per fold"},
                        features=kept_features(withheld),
                        tables={"folds": table[["fold", column]]},
                    )

    if serve_time is not None and not serve_time.empty:
        with tracking.family(
            mlflow, "serve-time regime", prov=prov, params={**shared, **base_model_params}
        ):
            written += 1
            for _, row in serve_time.iterrows():
                regime = str(row["regime"])
                written += 1
                tracking.log_variant(
                    mlflow,
                    run_name=f"regime: {regime}",
                    prov=prov,
                    nested=True,
                    params={"regime": regime, **shared},
                    metrics={key: row[key] for key in _REGIME_METRICS if key in row},
                    # One fitted model, scored twice. The feature subset is the
                    # full one in both arms: this measures what the *deployed*
                    # object does when columns arrive null, not what a refit
                    # without them would score.
                    tags={"family": "serve-time regime", "refit": False},
                    features=kept_features(()),
                )

    return written


def _fold_evidence(table: pd.DataFrame, paired: dict[str, float]) -> list[str]:
    """The fold table and what it does or does not license."""
    if table.empty or not paired.get("folds"):
        return [
            "**No fold evidence yet.** The training window is not deep enough to cut",
            "rolling-origin folds from, so the delta above is a single draw on one",
            "validation block and cannot settle §12. `python -m src.data.split` reports",
            "how much longer.",
            "",
        ]

    folds = int(paired["folds"])
    mean, sd, wins = paired["mean_difference"], paired["sd"], int(paired["wins"])
    spread = "—" if pd.isna(sd) else f"{sd:.4f}"

    if folds < 3:
        reading = (
            f"Only {folds} fold(s) scored. Two numbers have no spread worth the name, "
            "so this is not yet evidence either way."
        )
    elif pd.isna(sd) or abs(mean) <= sd:
        reading = (
            f"The full model leads by {mean:+.4f} PR-AUC on average, winning {wins} of "
            f"{folds} folds, which is inside one standard deviation ({spread}). **Treat "
            "them as tied.** A tie is the result that licenses dropping the four: if the "
            "board columns cannot be shown to help, the simpler model is the one whose "
            "validated and served forms are the same object for every caller."
        )
    elif mean > 0:
        reading = (
            f"The full model leads by {mean:+.4f} PR-AUC, winning {wins} of {folds} folds, "
            f"which clears one standard deviation ({spread}). **The board columns earn "
            "their place**, and the imputation fallback stays — with the cost §12 names, "
            "that a caller supplying nothing gets a model whose board columns are constant."
        )
    else:
        reading = (
            f"The model *without* board context leads by {-mean:+.4f} PR-AUC, winning "
            f"{folds - wins} of {folds} folds. Dropping the four is not merely free, it is "
            "an improvement on this evidence — which would be worth understanding before "
            "acting on, since it suggests the columns are adding noise rather than signal."
        )

    return [
        "**Fold-level evidence.** The delta above is one draw; these are several, paired",
        "on the fold so that both models see the same validation wave. Folds are cut",
        "inside the **training** window — the test block is not read here, because",
        "choosing a feature set is model selection.",
        "",
        _table(table, ["fold", "pr_auc_full", "pr_auc_without_board", "difference"]),
        "",
        reading,
        "",
    ]


def _board_context_verdict(
    ablations: pd.DataFrame, serve_time: pd.DataFrame | None = None
) -> list[str]:
    """Read the two board rows against each other and say what they settle.

    `docs/design.md` §12 is open on whether the four board-context columns
    belong in the served model. A job seeker holding one posting cannot supply
    any of them, so at serve time they arrive as nulls and the training fold's
    imputers fill them with constants — the served model is then not quite the
    model that was validated. §12 names the deciding evidence as an ablation,
    and this is the paragraph that reports it rather than leaving a reader to
    subtract two rows in a table.

    The group and the singles answer different halves and neither alone is
    actionable. Small singles with a large group means the four are redundant
    *with each other*, and dropping all four would cost something even though
    dropping any one costs nothing. Small singles with a small group means they
    are worthless — and only that licenses dropping them.
    """
    group = ablations[ablations["removed"] == BOARD_CONTEXT_ABLATION.name]
    if group.empty:
        return []
    together = float(group["delta"].iloc[0])
    singles = ablations[ablations["removed"].isin(BOARD_CONTEXT)]
    largest = float(singles["delta"].abs().max()) if not singles.empty else 0.0

    if together <= 0:
        reading = (
            "Withholding all four did not hurt. On this split the board columns are "
            "worth nothing, and **§12 can drop them**: the validated model and the "
            "served model become the same object for every caller."
        )
    elif largest > 0 and together <= largest * 1.5:
        reading = (
            "The group is worth about as much as its best single column, so the four "
            "are **not** redundant with each other — most of the value sits in one of "
            "them. Dropping all four costs roughly what dropping that one does."
        )
    else:
        reading = (
            "The group is worth more than any single column, so the four are "
            "**redundant with each other**: leave-one-out understates them and a "
            "per-feature table alone would have licensed dropping columns that "
            "jointly carry signal."
        )

    return [
        "### What this settles about board context (`docs/design.md` §12)",
        "",
        f"Withholding all four board columns costs **{together:+.4f}** validation "
        f"PR-AUC. The largest single board column is worth {largest:.4f}.",
        "",
        reading,
        "",
        "**But it is a refit, and the deployed model is not.** The number above "
        "says what the four columns are *worth*: a model trained without them "
        "redistributes their weight to whatever else it has. The shipped object "
        "does the opposite — it was fitted with them, and a caller who supplies "
        "none sends nulls that the training fold's imputers fill with constants, "
        "leaving fitted weights pointed at a column that no longer varies. The "
        "table below is that regime, and it is what `POST /predict` returns to a "
        "stranger.",
        "",
        *_serve_time_section(serve_time),
        "It does not price a caller who *can* supply them — a board owner scoring "
        "their own requisitions has all four, and for them the imputation branch "
        "never runs. Both rows are reported because the service serves both.",
        "",
    ]


def _serve_time_section(regime: pd.DataFrame | None) -> list[str]:
    if regime is None:
        return [
            "The serve-time regime could not be measured on this panel — it needs a "
            "split, like everything else here.",
            "",
        ]
    return [
        _table(regime, ["regime", "n", "pr_auc", "brier", "precision", "recall", "delta_pr_auc"]),
        "",
    ]


def write_report(
    path: Path,
    frame: pd.DataFrame,
    ladder: pd.DataFrame | None,
    ablations: pd.DataFrame | None,
    sweep: pd.DataFrame | None,
    blocker: str | None,
    fold_evidence: tuple[pd.DataFrame, dict[str, float]] | None = None,
    serve_time: pd.DataFrame | None = None,
    tracking_note: str | None = None,
) -> None:
    reference = analytic_reference(frame)
    derived = ", ".join(f"`{name}`" for name in DERIVED_COLUMNS)
    lines = [
        "# Model results — engineered features and XGBoost",
        "",
        f"Generated by `python -m src.models.train` on {horizon_banner(frame)}. "
        "Regenerate rather than edit.",
        "",
        "## Features",
        "",
        f"{len(FEATURES)} panel-native features and {len(DERIVED)} engineered ones: {derived}.",
        "Every engineered feature is computed inside the pipeline by",
        "`src.features.derive`, which is stateless — so each one exists identically",
        "for a single posting at serve time, and none of them can leak across the",
        "split however they are called.",
        "",
        f"The constant-predictor reference remains PR-AUC **{reference['pr_auc']:.4f}**",
        f"on {reference['n']:,.0f} labelled rows.",
        "",
    ]

    if blocker is not None:
        lines += ["## Nothing below has run", "", blocker, ""]
    else:
        assert ladder is not None and ablations is not None and sweep is not None
        lines += [
            "## The ladder, with the boosted rung",
            "",
            _table(
                ladder,
                ["model", "description", "pr_auc", "brier", "roc_auc", "precision", "recall"],
            ),
            "",
            "## Hypothesis ablation",
            "",
            "Each row refits without the named column(s). `delta` is the validation",
            "PR-AUC they are worth: positive means removing them hurt. Every row",
            "shares the baseline's split, seed and preprocessing, so a delta is",
            "attributable to the withheld columns and nothing else.",
            "",
            _table(
                ablations,
                [
                    "removed",
                    "n_removed",
                    "hypothesis",
                    "val_pr_auc",
                    "delta",
                    "train_pr_auc",
                    "gap",
                ],
            ),
            "",
            *_board_context_verdict(ablations, serve_time),
            *_fold_evidence(*(fold_evidence or (pd.DataFrame(), {}))),
            "## The deliberate overfit",
            "",
            "Depth up and regularisation off until train and validation separate, then",
            "one knob at a time to close the gap again.",
            "",
            _table(
                sweep,
                [
                    "setting",
                    "max_depth",
                    "n_estimators",
                    "reg_lambda",
                    "min_child_weight",
                    "subsample",
                    "train_pr_auc",
                    "val_pr_auc",
                    "gap",
                ],
            ),
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    lines += [
        "",
        "## Experiment tracking",
        "",
        "Every family above is logged to MLflow as a parent run with one child per",
        "variant \u2014 the ladder, the ablations, the overfit sweep, the board-context",
        "folds and the serve-time regime. Each child records the feature subset it",
        "actually fitted on, both cut instants and the embargo, the panel's path and",
        "sha256, its own parameters and metrics, and the git SHA that produced them.",
        "",
        tracking_note or "Tracking status not recorded.",
        "",
    ]
    gap_figure = Path("reports/figures/complexity_gap.png")
    if gap_figure.exists():
        lines[-1:-1] = [
            "",
            "## The gap, drawn",
            "",
            "The sweep opens a train/validation gap and closes it one knob at a time.",
            "A gap is a distance between two lines, and as a table it is two columns the",
            "reader has to subtract in their head, one row at a time.",
            "",
            f"![train against validation across complexity]"
            f"({gap_figure.parent.name}/{gap_figure.name})",
        ]

    path.write_text("\n".join(lines) + "\n")


def _track(args, frame, split, ladder, ablations, sweep, fold_evidence, serve_time, blocker) -> str:
    """Log the families, and return the sentence the report will carry.

    The return value is the point as much as the logging is. An untracked run
    that says nothing looks exactly like a tracked one in the report, and
    `CLAUDE.md` §4.6 asks that every run record its params, metrics, dataset
    version and git SHA — a requirement that quietly stops holding is worse than
    one that was never claimed.
    """
    if blocker is not None or split is None:
        return "Not tracked: nothing ran, so there is no run to log."
    if args.no_mlflow:
        return "**Not tracked**, by `--no-mlflow`. The numbers above are in this report only."
    if not tracking.available():
        print(
            "mlflow is not installed — the runs above were NOT logged. `pip install -e .[tracking]`"
        )
        return (
            "**Not tracked**: MLflow is not installed in this environment, so the runs "
            "above exist only in this file. `pip install -e '.[tracking]'` and re-run to "
            "record them."
        )

    from src import plots

    prov = provenance.collect(args.panel, len(frame), provenance.REAL)
    mlflow = tracking.start(args.experiment, args.tracking_uri)
    written = log_experiments(
        mlflow,
        split,
        prov,
        args.budget,
        ladder=ladder,
        ablations=ablations,
        sweep=sweep,
        fold_evidence=fold_evidence,
        serve_time=serve_time,
        figures_dir=plots.FIGURES_DIR,
    )
    print(f"logged {written} run(s) to experiment {args.experiment!r} at {args.tracking_uri}")
    return (
        f"Tracked: **{written} runs** in MLflow experiment `{args.experiment}`, as one parent "
        f"per family with a child per variant. Each child carries its own feature subset, the "
        f"split, the panel's sha256 and the git SHA."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    parser.add_argument("--experiment", default=tracking.DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=tracking.DEFAULT_TRACKING_URI)
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="skip experiment tracking deliberately. Without this flag the runs are "
        "logged, and if MLflow is not installed the report says the run was untracked "
        "rather than passing over it in silence",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    split = ladder = ablations = sweep = blocker = fold_evidence = serve_time = None
    try:
        split = temporal_split(frame, best_cuts(frame))
        ladder, _ = run_ladder(split, args.budget)
        boosted = fit_and_score(build_xgboost(split), split, args.budget)
        ladder = pd.concat(
            [
                ladder,
                pd.DataFrame(
                    [
                        {
                            "model": "xgboost",
                            "description": "gradient boosting",
                            "pr_auc": boosted["val_pr_auc"],
                            "brier": boosted["val_brier"],
                            "roc_auc": boosted["val_roc_auc"],
                            "precision": boosted["val_precision"],
                            "recall": boosted["val_recall"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        ablations = ablate(split, args.budget)
        sweep = overfit_sweep(split, args.budget)
        fold_evidence = board_context_folds(split, args.budget)
        serve_time = serve_time_regime(split, args.budget)
        print(ladder.to_string(index=False))
    except SplitTooShallow as error:
        blocker = (
            "No honest three-way split exists on this snapshot, so the ablation has no\n"
            "validation block to test a hypothesis against and the overfit sweep has no\n"
            "gap to open. The engineered features are built, audited and tested; what\n"
            "waits is the evidence for keeping or dropping each one.\n\n"
            "```\n" + str(error).split("\n\n")[0] + "\n```"
        )
        print(f"not run: {str(error).splitlines()[0]}")

    tracking_note = _track(
        args, frame, split, ladder, ablations, sweep, fold_evidence, serve_time, blocker
    )
    write_report(
        args.out,
        frame,
        ladder,
        ablations,
        sweep,
        blocker,
        fold_evidence,
        serve_time,
        tracking_note,
    )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
