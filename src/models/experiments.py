"""The replayable experiment history.

**Why this is a script and not a notebook of remembered runs.** By the time
there are ten of them you can no longer answer *which model was that? what
features? what `max_depth`? which snapshot? why did I keep it?* — and the answer
"I think it was the one with the deeper trees" is how a result becomes folklore.
Tracking arrives here rather than earlier for that reason — late enough that the
pain it removes is one you have felt. Every run logs params, metrics, the
dataset version and the git SHA.

**Why replayable rather than logged-as-you-go.** The usual MLflow setup records
runs as a side effect of whatever you happened to type, which makes the history
a transcript of your afternoon: rich in runs nobody can reproduce and missing
the two that mattered. Here the history is a *definition* — `RUNS` below — and
replaying it produces the whole story from scratch on any panel. That has three
consequences worth the constraint:

1. the real panel is currently too shallow to split (`src/data/split.py` refuses
   rather than returning an undefined metric), so today the history runs on a
   synthetic panel and is tagged `dataset=synthetic`. When the panel is deep
   enough, one command produces the real history — nothing has to be remembered
   or retyped;
2. a run is reproducible by construction, and `reproduce` checks it: refit from
   the logged params and compare against the logged metric;
3. the leak in run 06 stays in the record permanently instead of being a story
   about a number nobody can see any more.

**Runs 06 and 07 are the point of this module.** A history is worth keeping only
if it contains *the run that changed your mind about a feature*, and a history
of runs that all agreed with each other contains no such thing. Run 06 admits
two columns from
`src/features/leaky.py` that are computed over the whole panel; run 07 removes
them and is otherwise byte-identical to run 05. The gap between them is the size
of the lie, measured rather than asserted — and because 07 and 05 differ in
nothing at all, 07 reproducing 05's number exactly is the proof that the leak
was the only difference. `tests/test_experiments.py` asserts both properties.

**What this module does not do.** It never touches `split.test`. Selection is a
validation-block question and the test block is opened once, later, when the
pipeline is frozen — see `src/models/evaluate.py`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.split import Cuts, SplitResult, SplitTooShallow, crawl_waves, temporal_split
from src.features.leaky import LEAKY_COLUMNS, add_leaky_features
from src.features.preprocessing import (
    FEATURES,
    build_pipeline,
    feature_columns,
    features_and_target,
    fit_on_training_fold,
)
from src.models import provenance
from src.models.evaluate import cross_validate, summarise_folds, wave_forward_folds
from src.models.metrics import DEFAULT_ALERT_BUDGET, evaluate, expected_calibration_error
from src.models.train import _xgb_parameters
from src.models.train_baseline import DEFAULT_PANEL, RANDOM_STATE, _table, prediction_days

DEFAULT_EXPERIMENT = "shelf-life"
#: SQLite rather than the `./mlruns` directory. MLflow 3 put the filesystem
#: store into maintenance mode and refuses it without an opt-out environment
#: variable, and the database backend is what `mlflow ui` expects anyway:
#:
#:     mlflow ui --backend-store-uri sqlite:///mlflow.db
#:
#: The file is gitignored. It is a cache of runs that `replay` can rebuild, not
#: a source artifact — which is the same reason `data/processed/` is not
#: committed.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_REPORT = Path("reports/experiment_log.md")

#: The synthetic panel's "dataset version". Hashing the builder is the right
#: analogue of hashing the parquet: the fixture *is* the data, so editing it
#: produces a different dataset and the recorded version should say so. Pointing
#: synthetic runs at the real panel's path and sha256 would be worse than
#: recording nothing — it would attribute numbers to a file they never read.
SYNTHETIC_PANEL_SOURCE = Path("tests/panels.py")

#: The features that existed before `src/features/derive.py` did. Run 04 uses
#: only these so that run 05 measures what feature engineering bought, rather
#: than measuring it and the switch to boosting at the same time.
PANEL_NATIVE: tuple[str, ...] = tuple(column.name for column in FEATURES)


# --- the runs ---------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One entry in the history.

    `resolve` returns the estimator's hyperparameters and may *search* for them
    — run 08 does. `build` then constructs the pipeline from a params dict and
    nothing else, which is what makes `reproduce` possible: replaying a run
    means feeding `build` the params MLflow recorded, never re-running the
    search and hoping it lands in the same place.
    """

    number: int
    name: str
    question: str
    resolve: Callable[[SplitResult], dict]
    build: Callable[[SplitResult, dict], Pipeline]
    leaky: bool = False
    tuned: bool = False

    @property
    def run_name(self) -> str:
        return f"{self.number:02d}-{self.name}"


def _logistic_params(_: SplitResult) -> dict:
    return {"max_iter": 2000, "class_weight": "balanced"}


def _forest_params(_: SplitResult) -> dict:
    return {
        "n_estimators": 300,
        "min_samples_leaf": 5,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }


#: Searched by run 08, inside the training window only, on rolling-origin folds
#: — never against validation and never against test. A hyperparameter chosen
#: against the validation block has made that block part of the training
#: procedure, and its score stops being an estimate of anything.
#: Deliberately four candidates rather than forty: on a panel this small the
#: differences between neighbouring settings are inside the fold noise, and a
#: large grid would be an elaborate way of selecting on that noise.
TUNING_GRID: tuple[dict, ...] = (
    {"max_depth": 2, "learning_rate": 0.05, "min_child_weight": 10},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"max_depth": 4, "learning_rate": 0.10, "min_child_weight": 20},
    {"max_depth": 6, "learning_rate": 0.03, "min_child_weight": 5},
)


def tune(
    split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> tuple[dict, pd.DataFrame]:
    """Pick hyperparameters by cross-validation inside the training window.

    Expanding-window folds, the same ones `src/models/evaluate.py` cuts, because
    a hyperparameter chosen on shuffled k-fold has been chosen with the future
    in the training set. The selection is on the fold *mean*, and the spread is
    returned alongside so that "the best setting" can be read against how little
    separates it from the others.
    """
    folds = wave_forward_folds(split.train, split.embargo)
    rows = []
    for index, overrides in enumerate(TUNING_GRID):
        params = _xgb_parameters(split, **overrides)
        per_fold = cross_validate(
            lambda p=params: build_pipeline(XGBClassifier(**p)),
            split.train,
            folds,
            budget_per_day,
        )
        rows.append({"candidate": index, **overrides, **summarise_folds(per_fold)})

    table = pd.DataFrame(rows)
    scored = table.dropna(subset=["cv_pr_auc_mean"])
    if scored.empty:  # every fold undefined — fall back to the documented default
        return _xgb_parameters(split), table
    best = scored.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]
    return _xgb_parameters(split, **TUNING_GRID[int(best["candidate"])]), table


RUNS: tuple[RunSpec, ...] = (
    RunSpec(
        1,
        "prior",
        "does anything at all beat a constant?",
        lambda split: {"strategy": "prior"},
        lambda split, params: build_pipeline(DummyClassifier(**params)),
    ),
    RunSpec(
        2,
        "logistic",
        "is there linear signal in the allowed features?",
        _logistic_params,
        lambda split, params: build_pipeline(LogisticRegression(**params)),
    ),
    RunSpec(
        3,
        "random_forest",
        "does non-linearity buy anything over the logistic fit?",
        _forest_params,
        lambda split, params: build_pipeline(RandomForestClassifier(**params)),
    ),
    RunSpec(
        4,
        "xgboost_panel_native",
        "boosting, on the features that existed before any were engineered",
        lambda split: _xgb_parameters(split),
        lambda split, params: build_pipeline(XGBClassifier(**params), only=PANEL_NATIVE),
    ),
    RunSpec(
        5,
        "xgboost_engineered",
        "do the seven engineered features earn their place?",
        lambda split: _xgb_parameters(split),
        lambda split, params: build_pipeline(XGBClassifier(**params)),
    ),
    RunSpec(
        6,
        "xgboost_leaky",
        "what happens when a panel-wide observation count is admitted?",
        lambda split: _xgb_parameters(split),
        lambda split, params: build_pipeline(XGBClassifier(**params), include_leaky=True),
        leaky=True,
    ),
    RunSpec(
        7,
        "xgboost_leak_removed",
        "and what is left once it is taken away again?",
        lambda split: _xgb_parameters(split),
        lambda split, params: build_pipeline(XGBClassifier(**params)),
    ),
    RunSpec(
        8,
        "xgboost_tuned",
        "does tuning inside the training window buy anything?",
        lambda split: tune(split)[0],
        lambda split, params: build_pipeline(XGBClassifier(**params)),
        tuned=True,
    ),
)


#: Where the two cuts fall, as fractions of the available crawl waves. 60/20/20
#: before the embargo takes its share — the embargo is subtracted from the
#: evaluation blocks, not from the training window, because the training window
#: is what the rolling-origin folds are cut from and starving it costs error
#: bars on every run.
TRAIN_FRACTION, VAL_FRACTION = 0.6, 0.8


def default_cuts(waves: pd.Series) -> Cuts:
    """Two cut instants from a list of crawl waves.

    Not the `waves[0], waves[len//2]` pair the earlier components use. That pair
    was written for a panel with five waves, where it is the only choice that
    leaves anything on both sides; applied to a deeper panel it leaves a
    **one-wave training block**, which silently yields zero rolling-origin folds
    and therefore no error bars on any run. The failure is quiet — an empty fold
    table reads as "cross-validation ran and found nothing" — so the fractions
    are named here rather than left implicit at the call site.
    """
    n = len(waves)
    if n < 3:
        raise SplitTooShallow(f"{n} crawl waves cannot be cut three ways")
    train_index = min(max(int(n * TRAIN_FRACTION) - 1, 0), n - 3)
    val_index = min(max(int(n * VAL_FRACTION) - 1, train_index + 1), n - 2)
    return Cuts(waves.iloc[train_index], waves.iloc[val_index])


def spec_by_name(run_name: str) -> RunSpec:
    """Look a spec up by its MLflow run name, for `reproduce`."""
    for spec in RUNS:
        if spec.run_name == run_name or spec.name == run_name:
            return spec
    raise KeyError(f"no run spec named {run_name!r}; known: {[s.run_name for s in RUNS]}")


# --- executing one run ------------------------------------------------------


@dataclass
class RunResult:
    spec: RunSpec
    params: dict
    metrics: dict[str, float]
    per_fold: pd.DataFrame
    features: tuple[str, ...]
    tuning: pd.DataFrame | None = field(default=None)


def execute(
    spec: RunSpec,
    split: SplitResult,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    cross_validate_folds: bool = True,
    params: dict | None = None,
) -> RunResult:
    """Fit one run on the training fold and score it on validation.

    `params` overrides `spec.resolve` — that is the reproduction path, where the
    hyperparameters come from the tracking store rather than from a fresh
    search.

    Train and validation are both scored, always. The validation number says how
    good the model is; the pair says whether it is memorising, and on a panel
    where the same posting appears in both blocks that second question is the
    one that catches a fraud.
    """
    tuning = None
    if params is None:
        if spec.tuned:
            params, tuning = tune(split, budget_per_day)
        else:
            params = spec.resolve(split)

    model = spec.build(split, params)
    fit_on_training_fold(model, split)

    scored = {}
    for block_name in ("train", "val"):
        block = getattr(split, block_name)
        features, target = features_and_target(block)
        probabilities = model.predict_proba(features)[:, 1]
        summary = evaluate(
            target, probabilities, n_days=prediction_days(block), budget_per_day=budget_per_day
        )
        scored[block_name] = summary
        if block_name == "val":
            scored["val_ece"] = expected_calibration_error(target, probabilities)

    metrics = {
        "train_pr_auc": scored["train"]["pr_auc"],
        "val_pr_auc": scored["val"]["pr_auc"],
        "gap": scored["train"]["pr_auc"] - scored["val"]["pr_auc"],
        "val_brier": scored["val"]["brier"],
        "val_ece": float(scored["val_ece"]),
        "val_roc_auc": scored["val"]["roc_auc"],
        "val_precision": scored["val"]["precision"],
        "val_recall": scored["val"]["recall"],
        "val_base_rate": scored["val"]["base_rate"],
        "n_train": float(len(split.train)),
        "n_val": float(len(split.val)),
    }

    per_fold = pd.DataFrame()
    if cross_validate_folds:
        folds = wave_forward_folds(split.train, split.embargo)
        if folds:
            per_fold = cross_validate(
                lambda: spec.build(split, params), split.train, folds, budget_per_day
            )
            metrics.update(summarise_folds(per_fold))
        else:
            # A training window too narrow to cut a single fold from. Reported
            # as absent rather than as zero: no error bar is not a small error
            # bar, and a 0.0 here would be averaged into a comparison as if the
            # model had been scored and failed.
            metrics.update({"folds": 0.0, "folds_scored": 0.0})

    features = tuple(column.name for column in feature_columns(only=None, include_leaky=spec.leaky))
    if spec.name == "xgboost_panel_native":
        features = PANEL_NATIVE

    return RunResult(spec, dict(params), metrics, per_fold, features, tuning)


# --- MLflow -----------------------------------------------------------------


def _mlflow():
    """Imported lazily so the rest of `src/` does not depend on it.

    MLflow pulls in a large dependency tree and the modelling code has no need
    of it; a stranger who wants to run the ladder should not have to install a
    tracking server to do so.
    """
    import mlflow

    return mlflow


def start(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str = DEFAULT_TRACKING_URI):
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow


def log_run(mlflow, result: RunResult, prov: provenance.Provenance, budget_per_day: int) -> str:
    """Write one run to the tracking store. Returns its run id.

    Feature names go in as an artifact rather than a param: MLflow truncates
    long param values, and a truncated feature list is worse than none — it
    reads as complete.
    """
    spec = result.spec
    with mlflow.start_run(run_name=spec.run_name) as active:
        mlflow.log_params(
            {
                "run": spec.name,
                "n_features": len(result.features),
                "include_leaky": spec.leaky,
                "tuned": spec.tuned,
                "alert_budget_per_day": budget_per_day,
                **{f"model__{key}": value for key, value in result.params.items()},
            }
        )
        mlflow.log_metrics(
            {key: float(value) for key, value in result.metrics.items() if pd.notna(value)}
        )
        mlflow.set_tags(
            {
                "run_number": spec.number,
                "question": spec.question,
                "leaky": spec.leaky,
                **prov.as_tags(),
            }
        )
        mlflow.log_dict({"features": list(result.features)}, "features.json")
        if not result.per_fold.empty:
            mlflow.log_text(result.per_fold.to_csv(index=False), "per_fold.csv")
        if result.tuning is not None:
            mlflow.log_text(result.tuning.to_csv(index=False), "tuning.csv")
        return active.info.run_id


def replay(
    panel: pd.DataFrame,
    panel_path: Path,
    dataset: str = provenance.REAL,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    cuts: Cuts | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Run the whole history, in order, logging each run.

    The leaky columns are attached to the panel **before** the split, because
    that is where the real mistake happens — see `src/features/leaky.py`. Every
    run then sees the same frame and they differ only in which columns they are
    allowed to select, so run 06 against run 07 is a controlled comparison of
    one thing.
    """
    prepared = add_leaky_features(panel)
    waves = crawl_waves(prepared[prepared["label_observable"]])
    split = temporal_split(prepared, cuts or default_cuts(waves))

    mlflow = start(experiment, tracking_uri)
    prov = provenance.collect(panel_path, len(panel), dataset)

    rows, run_ids = [], []
    for spec in RUNS:
        result = execute(spec, split, budget_per_day)
        run_ids.append(log_run(mlflow, result, prov, budget_per_day))
        rows.append({"run": spec.run_name, "question": spec.question, **result.metrics})
        print(
            f"{spec.run_name:<28} val_pr_auc={result.metrics['val_pr_auc']:.4f}  "
            f"gap={result.metrics['gap']:+.4f}"
        )
    return pd.DataFrame(rows), run_ids


# --- reproducing a run ------------------------------------------------------


def reproduce(
    run_id: str,
    panel: pd.DataFrame,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    cuts: Cuts | None = None,
) -> dict[str, object]:
    """Refit a logged run from its logged params and compare the metric.

    This is the component's test made executable: *pick any run and reproduce
    its metric from its logged params and dataset version.* If it cannot be
    done, the run was logging too little, and the failure says which of the
    three reasons applies — different code, different data, or a params record
    too thin to rebuild the model from.

    The dataset check is a comparison, not a gate. Reproducing a run against a
    panel that has since grown is a legitimate thing to want; silently calling
    the result "reproduced" is not, so the mismatch is reported.
    """
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    logged = mlflow.get_run(run_id)

    spec = spec_by_name(logged.data.tags["mlflow.runName"])
    params = _decode_params(logged.data.params)

    prepared = add_leaky_features(panel)
    waves = crawl_waves(prepared[prepared["label_observable"]])
    split = temporal_split(prepared, cuts or default_cuts(waves))

    result = execute(spec, split, budget_per_day, cross_validate_folds=False, params=params)

    original = float(logged.data.metrics["val_pr_auc"])
    replayed = float(result.metrics["val_pr_auc"])
    current = provenance.collect(Path(logged.data.tags.get("panel_path", "")), len(panel))
    return {
        "run_name": spec.run_name,
        "logged_val_pr_auc": original,
        "replayed_val_pr_auc": replayed,
        "difference": replayed - original,
        "same_code": logged.data.tags.get("git_sha") == current.git_sha,
        "same_data": logged.data.tags.get("panel_sha256") == current.panel_sha256,
        "params": params,
    }


def _decode_params(params: dict[str, str]) -> dict:
    """Recover estimator kwargs from MLflow's all-strings param store.

    MLflow stores every param as a string, so `max_depth` comes back as `"4"`
    and `subsample` as `"0.8"`. Rebuilding the estimator from those without
    coercion gives a differently-configured model that raises nothing and scores
    differently — a reproduction that fails silently, which is worse than one
    that fails.
    """
    decoded = {}
    for key, value in params.items():
        if not key.startswith("model__"):
            continue
        name = key.removeprefix("model__")
        decoded[name] = json.loads(value) if _is_json(value) else value
    return decoded


def _is_json(value: str) -> bool:
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


# --- the report -------------------------------------------------------------


def leak_delta(table: pd.DataFrame) -> dict[str, float]:
    """How much the leaky columns were 'worth', and how much of run 05 came back.

    Two numbers, because they answer different questions. `inflation` is the
    size of the lie. `restored` is whether removing it returned the honest score
    exactly — if it did not, run 07 differs from run 05 in something other than
    the leak, and the comparison is not the controlled one it claims to be.
    """
    by_run = table.set_index("run")["val_pr_auc"]
    return {
        "honest": float(by_run["05-xgboost_engineered"]),
        "leaked": float(by_run["06-xgboost_leaky"]),
        "removed": float(by_run["07-xgboost_leak_removed"]),
        "inflation": float(by_run["06-xgboost_leaky"] - by_run["05-xgboost_engineered"]),
        "restored": float(by_run["07-xgboost_leak_removed"] - by_run["05-xgboost_engineered"]),
    }


def write_report(
    path: Path,
    table: pd.DataFrame | None,
    prov: provenance.Provenance | None,
    blocker: str | None,
) -> None:
    lines = [
        "# Experiment history",
        "",
        "Generated by `python -m src.models.experiments`. Regenerate rather than edit.",
        "",
        "## Why the history is a script",
        "",
        "The runs below are not a transcript of an afternoon — they are the definition in",
        "`src/models/experiments.py:RUNS`, replayed. That is the difference between a",
        "history you can point at and one you can only remember: every run here can be",
        "refitted from its logged parameters, and `reproduce` does exactly that.",
        "",
        "Each run logs its parameters, its metrics, the **git SHA** of the code that",
        "produced it and the **sha256 of the panel** it read. Neither of those last two",
        "can be recovered afterwards, which is why they are logged rather than inferred:",
        "the scraper adds a wave a day, so a PR-AUC with no dataset version is a number",
        "about an unknown quantity of data.",
        "",
    ]

    if blocker is not None:
        lines += ["## The real panel has not run", "", blocker, ""]

    if table is not None:
        assert prov is not None
        banner = (
            "**These are synthetic-panel runs.**"
            if prov.dataset == provenance.SYNTHETIC
            else "**These are runs on the pinned snapshot.**"
        )
        lines += [
            f"## The runs — `dataset={prov.dataset}`",
            "",
            banner,
            "",
        ]
        if prov.dataset == provenance.SYNTHETIC:
            lines += [
                "The real panel cannot yet be split three ways, so the history was replayed",
                "against `tests/panels.py`, whose label is drawn independently of every",
                "feature. **No number below is a finding about job postings.** They are here",
                "to show that the machinery runs end to end, and because one of them is not",
                "about the data at all — see the leak, which is a property of how a column is",
                "computed and shows up on any panel whatsoever.",
                "",
            ]
        lines += [
            _table(
                table,
                [
                    "run",
                    "question",
                    "cv_pr_auc_mean",
                    "cv_pr_auc_sd",
                    "val_pr_auc",
                    "train_pr_auc",
                    "gap",
                    "val_ece",
                ],
            ),
            "",
        ]

        delta = leak_delta(table)
        lines += [
            "## Run 06 — the run that changed my mind",
            "",
            "Run 06 admits two columns from `src/features/leaky.py`:",
            f"{', '.join(f'`{name}`' for name in LEAKY_COLUMNS)}. Both are aggregates over a",
            "posting's rows in the **whole panel** rather than in a window ending at `t`.",
            "",
            f"- run 05, honest features: **{delta['honest']:.4f}** validation PR-AUC",
            f"- run 06, leaky columns admitted: **{delta['leaked']:.4f}**",
            f"- run 07, leaky columns removed: **{delta['removed']:.4f}**",
            "",
            f"The leak was worth **{delta['inflation']:+.4f}** PR-AUC, and removing it returned",
            f"run 05's score to within **{abs(delta['restored']):.4f}** — runs 05 and 07 differ in",
            "nothing else, so that agreement is what makes this a controlled comparison rather",
            "than two numbers side by side.",
            "",
            "**Why the column leaks.** The panel is one row per (posting, crawl). A posting",
            "that stays on the board accrues a row per wave; one that is pulled stops",
            "accruing. So a count over the whole panel measures how long the posting",
            "survived, which is the label integrated. The same quantity windowed to end at",
            "`t` is perfectly legitimate — `board_growth` and `n_same_title_on_board` are",
            "exactly that — and the entire defect is the open right-hand end of the window.",
            "",
            "**What it would do in production, which is worse than the metric.** A posting",
            "arriving at `POST /predict` has been seen once, so `n_observations_total` is 1",
            "for every request a stranger ever makes. The model learned that low counts mean",
            "closing. It would flag everything.",
            "",
            "> **In my own words:** _(to be written. The table above is machinery; this",
            "> line is the deliverable, and it is not one a generator can produce.)_",
            "",
        ]

    if prov is not None:
        lines += [
            "## Provenance",
            "",
            f"- git `{prov.git_sha}` on `{prov.git_branch}`"
            + (" — **dirty tree**" if prov.git_dirty else ""),
            f"- panel `{prov.panel_path}`, {prov.panel_rows:,} rows, sha256 `{prov.panel_sha256}`",
            f"- snapshot `{prov.snapshot_date}`",
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


BLOCKER = (
    "No honest three-way split exists on this snapshot, so the history has no\n"
    "validation block to score against. The definition in `RUNS` is complete and\n"
    "tested; replaying it on the real panel is one command once the panel is deep\n"
    "enough, and `--synthetic` exercises it meanwhile.\n\n"
    "```\n{refusal}\n```"
)


def real_panel_blocker(panel_path: Path) -> str | None:
    """Why the real panel cannot be split, or `None` if it can.

    Cheap — it cuts the panel and fits nothing — so the synthetic report can
    carry the actual refusal rather than a claim that one exists. A report that
    says "the real panel is too shallow" without the receipt is asking to be
    believed; the two reports written before this one both quote the refusal,
    and this keeps them consistent.
    """
    if not panel_path.exists():
        return None
    frame = pd.read_parquet(panel_path)
    prepared = add_leaky_features(frame)
    waves = crawl_waves(prepared[prepared["label_observable"]])
    try:
        temporal_split(prepared, default_cuts(waves))
    except SplitTooShallow as error:
        return BLOCKER.format(refusal=str(error).split("\n\n")[0])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="replay against tests/panels.py instead of the real panel, for when "
        "the real one is too shallow to split",
    )
    args = parser.parse_args()

    table = prov = blocker = None

    if args.synthetic:
        from tests.panels import make_closing_panel

        panel = make_closing_panel()
        table, _ = replay(
            panel,
            SYNTHETIC_PANEL_SOURCE,
            provenance.SYNTHETIC,
            args.experiment,
            args.tracking_uri,
            args.budget,
        )
        prov = provenance.collect(SYNTHETIC_PANEL_SOURCE, len(panel), provenance.SYNTHETIC)
        blocker = real_panel_blocker(args.panel)
    else:
        panel = pd.read_parquet(args.panel)
        try:
            table, _ = replay(
                panel,
                args.panel,
                provenance.REAL,
                args.experiment,
                args.tracking_uri,
                args.budget,
            )
            prov = provenance.collect(args.panel, len(panel), provenance.REAL)
        except SplitTooShallow as error:
            blocker = BLOCKER.format(refusal=str(error).split("\n\n")[0])
            print(f"not run: {str(error).splitlines()[0]}")
            prov = provenance.collect(args.panel, len(panel), provenance.REAL)

    write_report(args.out, table, prov, blocker)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
