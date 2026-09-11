"""Model comparison, threshold defence and calibration.

The question this module exists to change. Not *"which model scores highest?"*
but **"which model is best on the metric that matters, by a margin that is
real?"** On a few thousand rows carrying fifty-odd positives, 0.71 against 0.73
is very often nothing at all, and a single validation number with no error bar
is not evidence — it is one draw from a distribution nobody has looked at.

So every comparison here reports **fold variance, not just a mean**, and the
model-versus-model question is answered by *paired* differences across the same
folds rather than by subtracting two averages. Two models scored on the same
fold share that fold's luck; differencing before averaging cancels it, and the
spread of those differences is what says whether a gap is real.

**Selection happens on validation only.** The test block is not read here, and
it is not read anywhere in `src/` outside the property that defines it —
`tests/test_evaluate.py` greps for that and fails if it stops being true. The
single test evaluation happens later, once the pipeline is frozen and can no
longer change in response to what it says. This is stricter than "evaluate on
test at the end of model selection", and the reason is that a test set looked at
twice is a validation set: the second look cannot be un-seen, and every decision
after it is contaminated by it.

**Why ROC-AUC is reported but not decisive.** With rare positives it flatters.
The false-positive rate has the true-negative count in its denominator — 5,618
of them against 75 positives on the 2026-09-05 snapshot — so a model can produce
a great many false alarms and barely move the x-axis. Precision puts those same false alarms over
the number of *flagged* rows, where they are impossible to hide. At a 1.2% base
rate the ROC curve is describing a decision nobody makes.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import cohort_audit
from src.data.split import (
    Cuts,
    SplitResult,
    SplitTooShallow,
    best_cuts,
    crawl_waves,
    feasible_cuts,
    temporal_split,
)
from src.features.assemble import horizon_banner
from src.features.preprocessing import features_and_target, fit_on_frame
from src.models import ledger, provenance

# `verdict` is aliased: `write_report` already has a `verdict` parameter holding
# the model-selection verdict, and two different verdicts under one name in one
# module is a bug waiting for someone to move a line.
from src.models.generalisation import (
    leave_one_board_out,
    report_tables,
)
from src.models.generalisation import (
    verdict as transfer_verdict,
)
from src.models.metrics import (
    DEFAULT_ALERT_BUDGET,
    alert_budget,
    confusion_at,
    evaluate,
    evaluate_by,
    expected_calibration_error,
    reliability_curve,
    threshold_for_budget,
)
from src.models.train import build_xgboost
from src.models.train_baseline import (
    DEFAULT_PANEL,
    HEURISTIC_RUNGS,
    LADDER,
    _table,
    analytic_reference,
    prediction_days,
)

DEFAULT_REPORT = Path("reports/model_comparison.md")

#: Alert budgets to report the operating point across, so that the chosen one is
#: defended against alternatives rather than asserted alone.
BUDGET_SWEEP: tuple[int, ...] = (5, 10, 20, 50, 100)

#: The columns `cross_validate` returns, always, including when it returns no
#: rows at all. A zero-fold panel is the ordinary state of this build while the
#: scraper is still accruing depth, and an empty frame carrying no columns is
#: not "no folds" — it is a frame whose schema depends on the data, which every
#: consumer then has to guess at. `summarise_folds` and `paired_fold_difference`
#: both already handle *no usable rows*; neither survived *no columns*, and the
#: first real-panel run turned that into a `KeyError: 'pr_auc'` two modules away
#: from the cause (DEBUGGING.md, 2026-09-09).
FOLD_COLUMNS: tuple[str, ...] = (
    "fold",
    "train_end",
    "val_start",
    "n_train",
    "n_val",
    "val_positives",
    "pr_auc",
    "brier",
    "roc_auc",
)


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold, described in time rather than in row numbers."""

    number: int
    train_index: np.ndarray
    val_index: np.ndarray
    train_end: pd.Timestamp
    val_start: pd.Timestamp

    @property
    def n_train(self) -> int:
        return int(self.train_index.size)

    @property
    def n_val(self) -> int:
        return int(self.val_index.size)


def wave_forward_folds(
    block: pd.DataFrame, embargo: pd.Timedelta, min_train_waves: int = 1
) -> list[Fold]:
    """Expanding-window folds over crawl waves, each honouring the embargo.

    Not `KFold`, and not `TimeSeriesSplit` either. `KFold` would put the future
    in the training set. `TimeSeriesSplit` splits on *row position*, which on a
    panel means it can cut through the middle of a single crawl wave and put
    half of one afternoon's postings on each side.

    So the folds are cut on waves, the window expands rather than slides — each
    fold trains on everything up to its origin, which is what a model in
    production would have — and the same embargo the outer split uses separates
    each fold's training data from its validation wave, because a fold's labels
    reach forward exactly as far as the outer split's do.
    """
    waves = crawl_waves(block)
    folds: list[Fold] = []
    positions = np.arange(len(block))
    # Kept as a pandas Series rather than converted to numpy: `np.datetime64`
    # silently drops the timezone, and a tz-naive/tz-aware comparison either
    # raises or, worse, compares the wrong instants.
    times = block["t"]

    for index in range(min_train_waves - 1, len(waves) - 1):
        train_end = waves.iloc[index]
        candidates = waves[waves > train_end + embargo]
        if candidates.empty:
            continue
        val_start = candidates.iloc[0]

        train_mask = (times <= train_end).to_numpy()
        val_mask = ((times > train_end + embargo) & (times <= val_start)).to_numpy()
        if not train_mask.any() or not val_mask.any():
            continue
        folds.append(
            Fold(
                number=len(folds),
                train_index=positions[train_mask],
                val_index=positions[val_mask],
                train_end=train_end,
                val_start=val_start,
            )
        )
    return folds


def cross_validate(
    build: Callable[[], object],
    block: pd.DataFrame,
    folds: Sequence[Fold],
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
) -> pd.DataFrame:
    """Score a model on each fold. One row per fold, no averaging yet.

    A fresh model per fold, built by `build` rather than passed in, because a
    fitted estimator reused across folds would carry the previous fold's
    parameters into the next one.

    A fold whose validation wave contains no positives yields `NaN` rather than
    zero — PR-AUC is undefined there, and a zero would be averaged in as if it
    were a bad score rather than an absent one.

    **A fold whose *training* slice is all one class is skipped the same way.**
    On an expanding window the earliest folds are the shallowest, and a training
    slice covering the first wave or two of a rare-event panel can easily
    contain no positives at all. A classifier fitted on one class has one class
    in `classes_`, so `predict_proba` returns a single column and indexing
    `[:, 1]` raises — the failure is an `IndexError` about array shape, which
    says nothing about the cause. The condition is checked before the fit rather
    than caught after it, because "this fold had nothing to learn from" is a
    fact about the fold, not an exception.
    """
    rows = []
    for fold in folds:
        train_block = block.iloc[fold.train_index]
        val_block = block.iloc[fold.val_index]

        if train_block["y"].nunique(dropna=True) < 2:
            rows.append(
                {
                    "fold": fold.number,
                    "train_end": fold.train_end,
                    "val_start": fold.val_start,
                    "n_train": fold.n_train,
                    "n_val": fold.n_val,
                    "val_positives": float((val_block["y"] == 1).sum()),
                    "pr_auc": float("nan"),
                    "brier": float("nan"),
                    "roc_auc": float("nan"),
                }
            )
            continue

        model = build()
        fit_on_frame(model, train_block)
        features, target = features_and_target(val_block)
        scores = model.predict_proba(features)[:, 1]

        summary = evaluate(
            target, scores, n_days=prediction_days(val_block), budget_per_day=budget_per_day
        )
        rows.append(
            {
                "fold": fold.number,
                "train_end": fold.train_end,
                "val_start": fold.val_start,
                "n_train": fold.n_train,
                "n_val": fold.n_val,
                "val_positives": summary["positives"],
                "pr_auc": summary["pr_auc"],
                "brier": summary["brier"],
                "roc_auc": summary["roc_auc"],
            }
        )
    return pd.DataFrame(rows, columns=list(FOLD_COLUMNS))


def summarise_folds(per_fold: pd.DataFrame, metric: str = "pr_auc") -> dict[str, float]:
    """Mean and spread across folds, and how many folds actually counted.

    `sd` uses the sample standard deviation and is `NaN` on a single fold, which
    is correct and worth leaving visible: one fold has no spread to report, and
    a `0.0` there would read as "no variance" rather than "no information".
    """
    usable = per_fold[per_fold[metric].notna()]
    return {
        "folds": float(len(per_fold)),
        "folds_scored": float(len(usable)),
        f"cv_{metric}_mean": float(usable[metric].mean()) if len(usable) else float("nan"),
        f"cv_{metric}_sd": float(usable[metric].std(ddof=1)) if len(usable) > 1 else float("nan"),
    }


def paired_fold_difference(
    per_fold_a: pd.DataFrame, per_fold_b: pd.DataFrame, metric: str = "pr_auc"
) -> dict[str, float]:
    """Is A better than B, or is that fold noise?

    Paired on fold, because two models scored on the same validation wave share
    whatever made that wave easy or hard. Differencing within a fold removes
    that shared component; averaging the two models separately and subtracting
    leaves it in, which is how a comparison manufactures a difference that is
    really a week's weather.

    Reports the mean difference, its spread, and **how many folds it won** —
    that last count is the one to read first on a handful of folds, where a mean
    ± sd invites more confidence than the sample size supports.
    """
    joined = per_fold_a[["fold", metric]].merge(
        per_fold_b[["fold", metric]], on="fold", suffixes=("_a", "_b")
    )
    joined = joined.dropna()
    if joined.empty:
        return {"folds": 0.0, "mean_difference": float("nan"), "sd": float("nan"), "wins": 0.0}

    difference = joined[f"{metric}_a"] - joined[f"{metric}_b"]
    return {
        "folds": float(len(difference)),
        "mean_difference": float(difference.mean()),
        "sd": float(difference.std(ddof=1)) if len(difference) > 1 else float("nan"),
        "wins": float((difference > 0).sum()),
    }


def threshold_sweep(
    y_true, y_score, n_days: int, budgets: Sequence[int] = BUDGET_SWEEP
) -> pd.DataFrame:
    """Precision and recall across alert budgets.

    The threshold is a modelling decision, and a decision is only defensible
    against the alternatives. `docs/design.md` §5 fixed the cost asymmetry — a
    false "removal soon" costs a rushed application measured in hours, a false
    "stays open" costs a job never applied to, which is unrecoverable — so the
    operating point leans to recall, and twenty a day is the largest list a
    person will actually read. This table is what that choice was made against.
    """
    truth = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    rows = []
    for budget in budgets:
        capped = alert_budget(truth.size, n_days, budget)
        threshold = threshold_for_budget(scores, capped)
        rows.append(
            {
                "budget_per_day": budget,
                "alerts": capped,
                **confusion_at(truth, scores, threshold),
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(y_true, y_score, n_bins: int = 10) -> dict[str, float]:
    """Brier and ECE together. Neither alone is enough — see the metric module."""
    reported = evaluate(y_true, y_score)
    return {
        "brier": reported["brier"],
        "expected_calibration_error": expected_calibration_error(y_true, y_score, n_bins),
        "mean_predicted": float(np.mean(np.asarray(y_score, dtype=float))),
        "observed_rate": float(np.mean(np.asarray(y_true, dtype=float))),
    }


def candidate_models(split: SplitResult) -> dict[str, Callable[[], object]]:
    """Everything in the ladder, plus the boosted rung."""
    candidates: dict[str, Callable[[], object]] = {rung.name: rung.build for rung in LADDER}
    candidates["xgboost"] = lambda: build_xgboost(split)
    return candidates


def compare_models(
    split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Cross-validate inside the training window, then score once on validation.

    Two numbers per model and they answer different questions. The CV mean ± sd
    says how the approach behaves across several origins — that is the one to
    select on. The validation score is a single draw at the real cut, reported
    so that a wild disagreement between the two is visible rather than averaged
    away.
    """
    folds = wave_forward_folds(split.train, split.embargo)
    rows, per_fold, val_scores = [], {}, {}

    for name, build in candidate_models(split).items():
        fold_scores = cross_validate(build, split.train, folds, budget_per_day)
        per_fold[name] = fold_scores

        model = build()
        fit_on_frame(model, split.train)
        features, target = features_and_target(split.val)
        scores = model.predict_proba(features)[:, 1]
        val_scores[name] = scores

        summary = evaluate(
            target, scores, n_days=prediction_days(split.val), budget_per_day=budget_per_day
        )
        rows.append(
            {
                "model": name,
                **summarise_folds(fold_scores),
                "val_pr_auc": summary["pr_auc"],
                "val_brier": summary["brier"],
                "val_ece": expected_calibration_error(target, scores),
                "val_roc_auc": summary["roc_auc"],
                "val_precision": summary["precision"],
                "val_recall": summary["recall"],
            }
        )
    return pd.DataFrame(rows), per_fold, val_scores


#: The fewest scored folds a selection may rest on. The same three that
#: `src/data/split.py` sizes the panel for, and for the same reason: two folds
#: can agree by accident, and the first thing worth seeing is a model losing a
#: fold it was expected to win.
MIN_SELECTION_FOLDS = 3


def ladder_order() -> tuple[str, ...]:
    """Every candidate, simplest first — complexity order, not score order.

    `LADDER` is already built to be climbed "in ascending seriousness" and
    `candidate_models` appends the boosted rung to it, so this is that same
    sequence read back rather than a second opinion about which model is more
    complicated than which. A name this does not know sorts last: an unknown
    candidate is treated as the most complex thing present, so parsimony can
    never be used to argue *for* it.
    """
    return tuple(rung.name for rung in LADDER) + ("xgboost",)


def _complexity(name: str) -> int:
    order = ladder_order()
    return order.index(name) if name in order else len(order)


def _scored_folds(per_fold: pd.DataFrame | None, metric: str = "pr_auc") -> int:
    if per_fold is None or len(per_fold) == 0 or metric not in per_fold:
        return 0
    return int(per_fold[metric].notna().sum())


def _indistinguishable(difference: dict[str, float]) -> bool:
    """Is a paired lead smaller than the fold-to-fold spread of that lead?"""
    return (
        difference["folds"] > 1
        and not np.isnan(difference["sd"])
        and abs(difference["mean_difference"]) <= difference["sd"]
    )


#: Which rungs the figures draw, chosen by **position on the ladder** — the
#: constant, a linear fit, and the boosted rung — and never by score. Picking the
#: three best-scoring candidates would be selection on the validation block
#: wearing a picture, which is the one thing this module may not do.
FIGURE_RUNGS: tuple[str, ...] = ("prior", "logistic", "xgboost")


def write_figures(
    split: SplitResult,
    val_scores: dict[str, np.ndarray],
    figures_dir: Path,
) -> list[Path]:
    """The PR curve and the calibration diagram, for a fixed spread of the ladder.

    **These do not need a chosen model, and that is deliberate.** Every table
    below the verdict waits for a selection the panel cannot yet support, but a
    diagnostic is not a verdict: the question "does anything here clear the base
    rate" is worth a picture long before the question "which one ships" has an
    answer. Drawing a fixed three rungs keeps it that way.
    """
    from sklearn.metrics import precision_recall_curve

    from src import plots

    target = split.val["y"].astype(int)
    drawn = [name for name in FIGURE_RUNGS if name in val_scores]

    curves = {}
    for name in drawn:
        precision, recall, _ = precision_recall_curve(target, val_scores[name])
        curves[name] = (recall, precision)

    pr_path = plots.precision_recall(
        curves, float(target.mean()), figures_dir / "precision_recall.png"
    )
    calibration_path = plots.calibration(
        {name: reliability_curve(target, val_scores[name]) for name in drawn},
        figures_dir / "calibration.png",
        eces={name: expected_calibration_error(target, val_scores[name]) for name in drawn},
    )
    return [pr_path, calibration_path]


def fold_census(panel: pd.DataFrame) -> pd.DataFrame:
    """Every legal cut of this panel, and how many folds each one yields.

    **Why an enumeration rather than a number.** "No fold is available" read as a
    property of *the* cut invites the obvious retort — try a different one. The
    cut is chosen to maximise folds, so the answer is already no, but that is a
    claim about a function rather than about the data and a reader has to take it
    on trust. This walks every cut the panel admits and reports the fold count of
    each, which turns the refusal into a table: not *this cut* yields nothing,
    but *no cut does*.

    It also makes `best_cuts`' contract checkable. That function deliberately does
    *not* maximise folds — past `DEFAULT_TARGET_FOLDS` it prefers blocks split in
    proportion, because maximising folds deepens training without limit and leaves
    validation and test one and two waves wide for ever. What it must do is reach
    the target when any cut can, and otherwise return the deepest cut available.
    Both are visible in this table, and while the panel is too shallow the second
    is the one that matters: it is the difference between waiting for depth and
    waiting for depth longer than necessary.
    """
    rows = []
    for _, candidate in feasible_cuts(panel).iterrows():
        if not candidate["valid"]:
            continue
        try:
            split = temporal_split(panel, Cuts(candidate["train_end"], candidate["val_end"]))
        except SplitTooShallow:
            continue
        rows.append(
            {
                "train_end": pd.Timestamp(candidate["train_end"]).date().isoformat(),
                "val_end": pd.Timestamp(candidate["val_end"]).date().isoformat(),
                "train_rows": int(len(split.train)),
                "val_rows": int(len(split.val)),
                "val_positives": int(split.val["y"].sum()),
                "folds": len(wave_forward_folds(split.train, split.embargo)),
            }
        )
    return pd.DataFrame(rows)


def select(summary: pd.DataFrame, per_fold: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Pick a model by a rule fixed before the numbers it would be applied to.

    **The rule, pre-registered in `docs/design.md` §14 on 2026-09-10, while every
    `folds_scored` in the comparison was still zero.** That timing is the entire
    source of its authority. A selection rule written after the fold scores are
    visible is not a rule, it is a description of the winner, and every degree of
    freedom in it — which metric, how to break a tie, how big a gap has to be —
    is a place to prefer the model one already likes.

    Three steps, in order:

    1. **Eligibility.** A candidate needs `MIN_SELECTION_FOLDS` scored folds. Below
       that there is no spread to judge a lead against, and the honest verdict is
       that nothing was selected — not that the leader wins by default.
    2. **The tied set.** Rank by `cv_pr_auc_mean`, take the leader, and collect
       every candidate whose paired difference from the leader is no larger than
       the spread of that difference. Paired, because two models scored on the
       same fold share whatever made it hard.
    3. **Parsimony decides among equals.** From the leader and everything tied
       with it, take the one *earliest in the ladder* — the simplest. Only a lead
       that survives fold variance can buy complexity.
    4. **The heuristic floor is a gate, not a footnote.** If the pick is a fitted
       model whose lead over the best `HEURISTIC_RUNGS` entry does not itself
       survive fold variance, the heuristic is selected instead.

    Step 3 is what the old implementation documented and did not do: it returned
    the highest mean whatever the spread, so a boosted model half a standard
    deviation ahead of a logistic regression was crowned, and `CLAUDE.md` §4.4 is
    explicit that this is backwards — *"the goal is not to reach the top rung, it
    is to learn whether increasing model complexity actually buys anything"*.

    **The floor needs both step 3 and step 4, and the reason is worth keeping.**
    `HEURISTIC_RUNGS` — the rules a person could follow without a computer — sit
    at the simple end of the ladder, so a fitted model that cannot separate from
    `age_ceiling` usually finds `age_ceiling` in its own tied set and loses to it
    on parsimony alone. But "tied with the leader" and "better than a rule" are
    different questions, and on a noisy label they come apart: the leader can
    fail to clear a heuristic that is nonetheless too far from it to be tied with
    it, and then every model tied with the leader inherits a failure none of them
    is measured against. Step 4 closes that. Verified on the synthetic panel whose
    label is drawn independently of every feature, where step 3 alone returned
    `age_only` — a fitted model that does not beat a constant — and the gate
    returns `prior`.

    Together they are `CLAUDE.md` §4.4 #12: *a model that does not beat the
    baseline is not a model, it is a slower baseline.*
    """
    scored = summary.dropna(subset=["cv_pr_auc_mean"]).copy()
    if scored.empty:
        return {"chosen": None, "reason": "no model was scored on any fold"}

    folds = {name: _scored_folds(per_fold.get(name)) for name in scored["model"]}
    eligible = scored[[folds[name] >= MIN_SELECTION_FOLDS for name in scored["model"]]]
    if eligible.empty:
        deepest = max(folds.values(), default=0)
        return {
            "chosen": None,
            "reason": (
                f"no model reached {MIN_SELECTION_FOLDS} scored folds — the deepest "
                f"managed {deepest}, which is a number without a spread to read it against"
            ),
        }

    ranked = eligible.sort_values("cv_pr_auc_mean", ascending=False)
    leader = str(ranked.iloc[0]["model"])

    if len(ranked) == 1:
        verdict: dict[str, object] = {
            "chosen": leader,
            "chosen_on": "sole candidate",
            "reason": f"{leader} was the only model with {MIN_SELECTION_FOLDS} scored folds",
        }
        return {**verdict, **_floor(leader, eligible, per_fold)}

    runner_up = str(ranked.iloc[1]["model"])
    difference = paired_fold_difference(per_fold[leader], per_fold[runner_up])
    separated = not _indistinguishable(difference) and difference["folds"] > 1

    tied = [
        str(name)
        for name in ranked["model"][1:]
        if _indistinguishable(paired_fold_difference(per_fold[leader], per_fold[name]))
    ]
    pool = [leader, *tied]
    chosen = min(pool, key=_complexity)

    if tied and chosen != leader:
        reason = (
            f"{leader} leads {runner_up} by {difference['mean_difference']:.4f} PR-AUC but "
            f"ties with {', '.join(tied)} inside fold variance, so the simplest of them "
            f"wins: {chosen}"
        )
        chosen_on = "parsimony"
    elif tied:
        reason = (
            f"{leader} ties with {', '.join(tied)} inside fold variance and is already the "
            f"simplest of them, so it stands"
        )
        chosen_on = "parsimony"
    else:
        reason = (
            f"{leader} leads {runner_up} by {difference['mean_difference']:.4f} PR-AUC, "
            f"winning {difference['wins']:.0f} of {difference['folds']:.0f} folds, and the "
            f"lead is larger than its own spread"
        )
        chosen_on = "separation"

    # Step 4, the floor. Parsimony picks the simplest of the models tied with the
    # leader, but "tied with the leader" is not the same question as "better than
    # a rule a person could follow" — on a label that is mostly noise the leader
    # itself may not clear one, and then everything tied with it inherits that
    # failure. Without this gate the rule returns the simplest member of a set
    # that nothing in it deserved to be in.
    floor = _floor(chosen, eligible, per_fold)
    if (
        floor.get("best_heuristic") is not None
        and not floor["chosen_is_heuristic"]
        and not floor.get("clears_heuristic_floor", False)
    ):
        beaten = chosen
        chosen = str(floor["best_heuristic"])
        chosen_on = "heuristic floor"
        reason = (
            f"no fitted model separated from {chosen}: {beaten} was the simplest of those "
            f"tied with {leader}, and its lead over {chosen} does not survive fold "
            f"variance either. CLAUDE.md §4.4 #12 — a model that does not beat the "
            f"baseline is not a model, it is a slower baseline."
        )
        floor = _floor(chosen, eligible, per_fold)

    return {
        "chosen": chosen,
        "chosen_on": chosen_on,
        "leader": leader,
        "runner_up": runner_up,
        "tied_with": tied,
        **{f"vs_runner_up_{key}": value for key, value in difference.items()},
        "separated": separated,
        "reason": reason,
        **floor,
    }


def _floor(chosen: str, eligible: pd.DataFrame, per_fold: dict[str, pd.DataFrame]) -> dict:
    """Did the pick clear the best rule a person could follow without a computer?

    Pure measurement, called twice by `select`: once to ask the question, and —
    if the answer was no and the gate swapped the pick for the heuristic — once
    more to describe what was finally chosen. Keeping it free of the decision
    means the numbers in the verdict always describe the model named beside them.
    """
    heuristics = eligible[eligible["model"].isin(HEURISTIC_RUNGS)]
    if heuristics.empty:
        return {"chosen_is_heuristic": chosen in HEURISTIC_RUNGS, "best_heuristic": None}

    best = str(heuristics.sort_values("cv_pr_auc_mean", ascending=False).iloc[0]["model"])
    floor: dict[str, object] = {
        "chosen_is_heuristic": chosen in HEURISTIC_RUNGS,
        "best_heuristic": best,
    }
    if chosen == best:
        return floor

    difference = paired_fold_difference(per_fold[chosen], per_fold[best])
    floor.update(
        {f"vs_best_heuristic_{key}": value for key, value in difference.items()},
    )
    floor["clears_heuristic_floor"] = bool(
        difference["folds"] > 1
        and not np.isnan(difference["sd"])
        and difference["mean_difference"] > difference["sd"]
    )
    return floor


def _generalisation_section(generalisation) -> list[str]:
    """Leave-one-board-out, or the reason none could be run.

    Placed directly after the per-source breakdown because the two are easy to
    confuse and one is much weaker evidence than it looks: scoring a board with
    a model fitted on it says nothing about a board the model has never seen.
    """
    lines = [
        "## Would it work on a board it has never seen?",
        "",
        "`docs/design.md` §4 excluded board identity on 2026-09-09, on per-board rates",
        "whose intervals all contain the pooled rate. That settles what goes *into* the",
        "model; it does not settle whether a model trained on these boards carries to a",
        "new one, and the model card's caveat is about the second. This is the",
        "measurement: hold a whole board out of the fit, then score its rows twice —",
        "once with a model that never saw it, once with a model that did. The **gap**",
        "is what board-specific learning was worth. A bare transfer score would not do,",
        "because boards have different base rates and a low number could be the board",
        "being harder rather than the model failing to carry over.",
        "",
    ]
    if generalisation is None:
        return lines + [
            "Not run on this snapshot: there is no split to fit on.",
            "",
        ]

    folds, skipped, scored, refused = generalisation
    lines += [transfer_verdict(folds), ""]
    if not scored.empty:
        lines += [
            _table(
                scored,
                [
                    "board",
                    "eval_rows",
                    "eval_positives",
                    "base_rate",
                    "train_share",
                    "transfer_pr_auc",
                    "ceiling_pr_auc",
                    "gap",
                ],
            ),
            "",
        ]
    if not refused.empty:
        lines += [
            "**Folds refused.** A board is scored only when it keeps enough positives to",
            "measure and leaves enough behind to fit on. Both guards exist because the",
            "alternative is a PR-AUC computed on a handful of events, which reads exactly",
            "like a real one.",
            "",
            _table(refused, ["board", "reason"]),
            "",
        ]
    return lines


def _complexity_reading(summary: pd.DataFrame, frame_for_caveat: pd.DataFrame) -> list[str]:
    """What the single validation draw shows about complexity — and why it is not the answer.

    **This is not a selection and must not read as one.** It reports a direction
    the one available draw points in, and says plainly that a direction is not
    evidence. The distinction is the whole discipline: a number with no spread
    cannot separate a real ordering from the ordering a single block happened to
    produce, which is why the rule requires folds before anything is chosen.

    It is written because the negative finding is worth stating early.
    `CLAUDE.md` §4.4: the goal is not to reach the top rung, it is to learn
    whether increasing complexity buys anything — and *it did not, here* is a
    finding about the data to be written down rather than tuned past. If it
    survives fold depth, step 4 of the rule returns a heuristic and this project
    ships a rule a person could follow unaided, which is a legitimate outcome and
    not a failure to produce a model.
    """
    if summary is None or summary.empty or "val_pr_auc" not in summary:
        return []

    heuristics = summary[summary["model"].isin(HEURISTIC_RUNGS)]
    fitted = summary[~summary["model"].isin(HEURISTIC_RUNGS)]
    if heuristics.empty or fitted.empty:
        return []

    best_heuristic = heuristics.loc[heuristics["val_pr_auc"].idxmax()]
    best_fitted = fitted.loc[fitted["val_pr_auc"].idxmax()]
    margin = float(best_fitted["val_pr_auc"] - best_heuristic["val_pr_auc"])

    if margin > 0:
        direction = (
            f"the best fitted rung, `{best_fitted['model']}` at "
            f"{best_fitted['val_pr_auc']:.4f}, is **ahead** of the best rule a person could "
            f"follow unaided, `{best_heuristic['model']}` at "
            f"{best_heuristic['val_pr_auc']:.4f} — by {margin:.4f}"
        )
        consequence = (
            "If that lead survives fold depth *and* clears the paired spread, step 3 of "
            "the rule buys the complexity and a fitted model ships. If it does not, step 4 "
            "returns the heuristic."
        )
    else:
        direction = (
            f"**no fitted rung beats a rule a person could follow unaided.** The best of "
            f"them, `{best_fitted['model']}`, scores {best_fitted['val_pr_auc']:.4f} against "
            f"`{best_heuristic['model']}` at {best_heuristic['val_pr_auc']:.4f} — behind by "
            f"{abs(margin):.4f}"
        )
        consequence = (
            "If that holds once folds exist, step 4 of the rule returns the heuristic and "
            "this project ships a rule a person could follow without a computer. That is a "
            "legitimate result and not a failure to produce a model: `CLAUDE.md` §4.4 asks "
            "whether increasing complexity buys anything, and *it did not* is an answer."
        )

    return [
        "### Does complexity earn its place? What the one draw points at",
        "",
        f"On the single validation draw above, {direction}.",
        "",
        "**That is a direction, not evidence, and the difference is the whole point.** One",
        "number has no spread, so it cannot separate a real ordering from the ordering this",
        "particular block happened to produce — which is exactly why the rule in",
        "`docs/design.md` §14 requires folds before anything is chosen. Reporting a winner",
        "from this table would be selection on the validation block.",
        "",
        consequence,
        "",
        *_lifespan_caveat(frame_for_caveat),
    ]


def _lifespan_caveat(panel: pd.DataFrame) -> list[str]:
    """The reading above is worthless if the target is measuring the crawl.

    Carried here rather than left in `reports/label_validity.md` because this is
    the table where somebody decides whether a model is working, and a score
    driven by how long a posting has been observed will look exactly like a model
    working. The two files being separate is how a reader ends up believing the
    optimistic one.

    Fires on the *rate ratio* between short-history and settled rows, the same
    test `lifespan_verdict` applies — not on the share of closures that fall in
    the short buckets. "Seen in" is counted as of each row's own `t`, so the
    short buckets hold most of every panel's rows and most of its closures
    whether or not anything is wrong; share fired on every panel. What is
    diagnostic is whether closures land on those rows at a *higher rate*.
    """
    from src.data.label_audit import SETTLED_OBSERVATIONS, lifespan_concentration

    census = lifespan_concentration(panel)
    if census.empty:
        return []

    settled_bucket = census["seen in"] == f"{SETTLED_OBSERVATIONS}+ runs"
    brief, settled = census[~settled_bucket], census[settled_bucket]
    if not brief["rows"].sum() or not settled["rows"].sum():
        return []
    brief_rate = float(brief["removals"].sum()) / float(brief["rows"].sum())
    settled_rate = float(settled["removals"].sum()) / float(settled["rows"].sum())
    if brief_rate < 10 * settled_rate:
        return []
    share = float(brief["removals"].sum()) / max(float(census["removals"].sum()), 1.0)

    return [
        f"**Read all of the above against the label first.** Rows whose posting had been "
        f"seen in fewer than {SETTLED_OBSERVATIONS} complete crawls as of their own `t` are "
        f"removed at {brief_rate:.1%} against {settled_rate:.1%} for the rest, and carry "
        f"{share:.0%} of "
        "the removals — see [`label_validity.md`](label_validity.md). Any feature tracking how "
        "long a posting has been around will separate those rows, so a lead built on "
        "`age_days` is mostly a lead on *observed lifespan*.",
        "",
        "**Whether that is a scraping defect is a separate question, and it was checked rather "
        "than argued.** [`label_check.md`](label_check.md) sampled postings the label calls "
        "removed and asked the boards: 59 of 60 are genuinely gone under their own id, against "
        "a control drift of 3.3%. What survives is the caveat that was always there: "
        "**removed is not filled**, and 12% of the verified removals had their title relisted "
        "under a new id within days.",
        "",
    ]


def _census_section(frame: pd.DataFrame) -> list[str]:
    """Every legal cut and its fold count, so the refusal is shown rather than asserted.

    "No fold is available" invites the retort *try a different cut*. This answers
    it in advance and exhaustively: the table below is every cut this panel
    admits, and the `folds` column is the whole argument.
    """
    census = fold_census(frame)
    if census.empty:
        return [
            "### Which cuts were tried",
            "",
            "**No cut of this panel is legal at all** — every candidate leaves the",
            "validation or the held-out block empty, so there is nothing to enumerate.",
            "`scripts/watch_depth.sh` reports the shortfall and the date it clears.",
            "",
        ]

    best = int(census["folds"].max())
    verdict = (
        f"The best any legal cut manages is **{best} fold(s)**, against the "
        f"{MIN_SELECTION_FOLDS} the rule requires."
        if best < MIN_SELECTION_FOLDS
        else f"The deepest legal cut yields **{best} fold(s)**."
    )
    return [
        "### Which cuts were tried",
        "",
        "Not one cut — every cut. *No fold is available* read as a property of the",
        "chosen cut invites the obvious retort, so here is the enumeration:",
        "",
        _table(census, list(census.columns)),
        "",
        verdict,
        "",
    ]


def _figure_section(figures: list[Path] | None) -> list[str]:
    """The two diagnostics, or the reason there are none.

    Placed above the sections that wait on a selection, because these two do
    not: a picture of whether anything clears the base rate is worth having
    while the question of *which* model ships is still unanswerable.
    """
    if not figures:
        return [
            "## Diagnostics",
            "",
            "_Figures not rendered: matplotlib is not installed. "
            "`pip install -e '.[plots]'` and re-run._",
            "",
        ]
    return [
        "## Diagnostics",
        "",
        "Drawn for a fixed spread of the ladder — the constant, the linear fit and the",
        "boosted rung — chosen by **position on the ladder, never by score**. Picking the",
        "three best-scoring candidates would be selection on the validation block wearing",
        "a picture.",
        "",
        *[f"![{path.stem.replace('_', ' ')}](figures/{path.name})\n" for path in figures],
        "The dashed line on the first is what a constant predictor scores. A curve below",
        "it has not beaten predicting the base rate for every posting.",
        "",
    ]


def write_report(
    path: Path,
    frame: pd.DataFrame,
    summary: pd.DataFrame | None,
    per_fold: dict[str, pd.DataFrame] | None,
    verdict: dict[str, object] | None,
    thresholds: pd.DataFrame | None,
    calibration: pd.DataFrame | None,
    by_source: pd.DataFrame | None,
    by_carryover: pd.DataFrame | None,
    blocker: str | None,
    generalisation: tuple | None = None,
    figures: list[Path] | None = None,
    prov: provenance.Provenance | None = None,
    by_first_observation: pd.DataFrame | None = None,
) -> None:
    reference = analytic_reference(frame)
    lines = [
        "# Model comparison, threshold and calibration",
        "",
        *provenance.header(prov, "python -m src.models.evaluate", horizon_banner(frame)),
        "## What is being compared, and on what",
        "",
        "**PR-AUC is the headline.** Accuracy is not reported: at a base rate of "
        f'{reference["base_rate"]:.4f}, predicting "stays open" for every posting scores '
        f"{100 * (1 - reference['base_rate']):.1f}% and has told you nothing.",
        "",
        "**ROC-AUC is reported but not decisive.** With rare positives it flatters. The",
        "false-positive rate divides by the true-negative count — "
        f"{reference['n'] - reference['positives']:,.0f} of them against "
        f"{reference['positives']:,.0f}",
        "positives — so a model can raise a great many false alarms without moving the",
        "x-axis. Precision divides the same false alarms by the number of rows flagged,",
        "where they cannot hide.",
        "",
        "**Selection is on validation only.** The test block is not read by this module,",
        "or anywhere in `src/` outside the property that defines it. A test set looked at",
        "twice is a validation set.",
        "",
        "**The rule below was fixed before any fold could be scored** — pre-registered in",
        "`docs/design.md` §14 on 2026-09-10, when every `folds_scored` in this table was",
        "zero at both horizons. A rule written after the scores are visible is not a rule,",
        "it is a description of the winner. It runs in four steps:",
        "",
        f"1. a candidate needs {MIN_SELECTION_FOLDS} scored folds, or nothing is selected;",
        "2. rank by `cv_pr_auc_mean`, and everything whose *paired* difference from the",
        "   leader is no bigger than that difference's own spread is tied with it;",
        "3. among the leader and everything tied with it, the **simplest** wins — only a",
        "   lead that survives fold variance buys complexity;",
        "4. and if that pick is a fitted model that cannot separate from the best rule a",
        "   person could follow unaided, the rule is selected instead.",
        "",
    ]

    if blocker is not None:
        lines += [
            "## Nothing below has run",
            "",
            blocker,
            "",
            "### What has been verified without it",
            "",
            "The comparison machinery is exercised on a twenty-wave synthetic panel whose",
            "label is drawn independently of every feature, so the correct answer is known:",
            "nothing should beat the base rate. Seven rolling-origin folds, base rate 0.10:",
            "",
            "| model | cv_pr_auc_mean | cv_pr_auc_sd | val_pr_auc | val_ece |",
            "|---|---|---|---|---|",
            "| prior | 0.1000 | 0.0000 | 0.1000 | 0.0000 |",
            "| board_hazard | 0.1000 | 0.0000 | 0.1000 | 0.0000 |",
            "| age_only | 0.1133 | 0.0340 | 0.1057 | 0.3801 |",
            "| logistic | 0.1846 | 0.0996 | 0.1339 | 0.3540 |",
            "| random_forest | 0.2078 | 0.0885 | 0.1441 | 0.2640 |",
            "| xgboost | 0.1974 | 0.1088 | 0.1198 | 0.4529 |",
            "",
            "**The random forest posts the best mean and the rule selects `prior`.** It",
            "leads by 0.0232 over xgboost, which is inside the spread of that lead, so",
            "xgboost, logistic and age_only are all tied with it; parsimony takes the",
            "simplest of those, `age_only`; and `age_only` in turn cannot separate from a",
            "constant, so the floor returns the constant. On a label drawn independently of",
            "every feature that is the only correct answer, and a comparison that reported",
            "the 0.2078 as a result would be inventing one.",
            "",
            "That is what the error bars are for. `tests/test_evaluate.py` pins this run,",
            "so a change that lets a fitted model win on noise fails CI rather than",
            "producing a better-looking report.",
            "",
            "Note also `prior`: PR-AUC exactly at the base rate and **ECE 0.0000** —",
            "perfectly calibrated and completely useless. That is why calibration is never",
            "reported on its own.",
            "",
        ]
    else:
        assert summary is not None and per_fold is not None and verdict is not None
        lines += [
            "## Models, with fold variance",
            "",
            "`cv_pr_auc_mean ± sd` across rolling-origin folds inside the training window;",
            "`val_pr_auc` is the single draw at the real cut. Select on the first, and read",
            "a disagreement between them as a warning rather than an average.",
            "",
            _table(
                summary,
                [
                    "model",
                    "folds_scored",
                    "cv_pr_auc_mean",
                    "cv_pr_auc_sd",
                    "val_pr_auc",
                    "val_brier",
                    "val_ece",
                    "val_roc_auc",
                ],
            ),
            "",
            "## The verdict",
            "",
            f"**Chosen: {verdict.get('chosen')}** — {verdict.get('reason')}",
            "",
            *_figure_section(figures),
        ]

    if blocker is None and thresholds is None:
        # A comparison ran but chose nothing. That is not a blocker — the table
        # above is real — and it is not a selection either, so everything that
        # needs a chosen model stops here rather than quietly running on a
        # default nobody picked.
        lines += [
            "## No threshold, calibration or breakdown yet",
            "",
            "Every section below this point describes *one* model, and no model has been",
            "selected: with no rolling-origin fold available inside the training window,",
            "there is no evidence to select on. The `cv_pr_auc_mean` column above is empty",
            "for exactly that reason.",
            "",
            "Picking one anyway — the best validation score, say, or the top of the ladder —",
            "would be selection on the validation block, which is the thing the block exists",
            "to prevent. So the threshold sweep, the calibration curve, the per-source table",
            "and the leave-one-board-out transfer measurement all wait for fold depth. They",
            "are exercised meanwhile against the synthetic panel in `tests/`.",
            "",
            *_census_section(frame),
            *_complexity_reading(summary, frame),
        ]
    elif blocker is None:
        lines += [
            "## Threshold",
            "",
            'The operating point is set by an alert budget, not by 0.5. A false "closing',
            'soon" costs a rushed application, measured in hours; a false "stays open"',
            "costs a job never applied to, which is unrecoverable. The second is worse, so",
            "the point leans to recall — and twenty a day is the longest list a person",
            "will actually read. This table is what that was chosen against.",
            "",
            _table(
                thresholds,
                [
                    "budget_per_day",
                    "alerts",
                    "threshold",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                ],
            ),
            "",
            "## Calibration",
            "",
            "A probability that is not calibrated is a score wearing a percent sign.",
            "",
            _table(calibration, ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"]),
            "",
            "## Per source",
            "",
            "arbeitnow is excluded from labelled rows, but the remaining boards still",
            "differ. A model that works on one board is a per-board model.",
            "",
            _table(by_source, ["source", "n", "positives", "base_rate", "pr_auc", "brier"]),
            "",
            "**That table scores each board with a model fitted on it**, which answers",
            "*does it work here* rather than *would it work somewhere new*. The section",
            "below answers the second question by removing a board from the fit.",
            "",
            *_generalisation_section(generalisation),
            "## Carried-over postings against unseen ones",
            "",
            _table(
                by_carryover, ["seen_in_train", "n", "positives", "base_rate", "pr_auc", "brier"]
            ),
            "",
            *_first_observation_section(by_first_observation),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _first_observation_section(table: pd.DataFrame | None) -> list[str]:
    """The narrower product promise, scored on its own rows.

    `design.md` §15: the product ranks a day's whole board, and a posting scored
    on the day it first appears is a case the same model must handle — reported
    separately so a model that only works on the incumbent stock cannot hide
    inside the board-wide number. Absent when the block holds no such rows,
    which the section says rather than leaving a gap.
    """
    lines = [
        "## The day a posting first appears",
        "",
        "Rows that are an incident posting's first sighting, against the rest of the",
        "block. `design.md` §15 makes the board ranking the product and this the slice",
        "it must not fail on.",
        "",
    ]
    if table is None or table.empty or not bool(table["first_observation"].any()):
        return [
            *lines,
            "_The validation block holds no first observations at this cut — no posting",
            "first appeared inside it after the embargo. `reports/cohort_audit.md` §7 shows",
            "which cuts do._",
            "",
        ]
    return [
        *lines,
        _table(table, ["first_observation", "n", "positives", "base_rate", "pr_auc", "brier"]),
        "",
    ]


def _draw(val_scores, split, panel_path: Path, n_rows: int) -> list[Path]:
    """Render the figures and log them, or say why not.

    Never fails the report over a picture, and never over a tracking server
    either: both are diagnostics of the comparison, and the comparison's tables
    are the artifact that has to survive.
    """
    from src import plots
    from src.models import provenance, tracking

    if not plots.available():
        print("matplotlib is not installed — no figures. `pip install -e '.[plots]'`")
        return []

    figures = write_figures(split, val_scores, plots.FIGURES_DIR)
    run = tracking.log_figure_run(
        "comparison diagnostics",
        figures,
        provenance.collect(panel_path, n_rows, provenance.REAL),
        params={"rungs_drawn": ", ".join(FIGURE_RUNGS)},
    )
    print(
        f"figures -> {plots.FIGURES_DIR}"
        + (f", logged to MLflow run {run[:8]}" if run else " (not logged: mlflow absent)")
    )
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    figures: list[Path] = []
    summary = per_fold = verdict = thresholds = calibration = None
    generalisation = None
    by_source = by_carryover = by_first_observation = blocker = None
    try:
        split = temporal_split(frame, best_cuts(frame))
        summary, per_fold, val_scores = compare_models(split, args.budget)
        verdict = select(summary, per_fold)
        figures = _draw(val_scores, split, args.panel, len(frame))
        print(summary.to_string(index=False))
        print(f"\nchosen: {verdict['chosen']} — {verdict['reason']}")

        # `select` returns no model when no fold could be scored, which on a panel
        # still accruing depth is the ordinary answer rather than a failure. Every
        # section below needs *a* model, and the only ways to supply one here —
        # best validation score, or top of the ladder — are respectively selection
        # on the validation block and skipping the ladder. So the comparison table
        # is written, and the rest waits. This used to read `str(verdict["chosen"])`
        # and die on `KeyError: 'None'` (DEBUGGING.md, 2026-09-09).
        chosen = verdict["chosen"]
        if chosen is not None:
            chosen = str(chosen)
            scores = val_scores[chosen]
            target = split.val["y"].astype(int)
            thresholds = threshold_sweep(target, scores, prediction_days(split.val))
            calibration = reliability_curve(target, scores)
            by_source = evaluate_by(split.val, scores, "source", n_days=prediction_days(split.val))
            by_carryover = evaluate_by(
                split.val, scores, "seen_in_train", n_days=prediction_days(split.val)
            )
            # The cohort columns are as-of-`t` and derived from the whole panel,
            # so the block is annotated from the frame rather than from itself —
            # and joined on keys, not index: `temporal_split` renumbers its
            # blocks, and `.loc` on the frame's index would pick other rows.
            by_first_observation = evaluate_by(
                cohort_audit.attach_first_observation(split.val, frame),
                scores,
                "first_observation",
                n_days=prediction_days(split.val),
            )
            # Refits per board, so it is a modelling activity and stays on the
            # validation block.
            folds, skipped = leave_one_board_out(split, lambda: build_xgboost(split), args.budget)
            generalisation = (folds, skipped, *report_tables(folds, skipped))

            row = summary[summary["model"] == chosen].iloc[0]
            labelled = frame[frame["label_observable"]]
            ledger.write_report(
                ledger.append(
                    ledger.record(
                        stage=ledger.VALIDATION,
                        provenance=provenance.collect(args.panel, len(frame)),
                        labelled_waves=len(crawl_waves(labelled)),
                        labelled_rows=len(labelled),
                        positives=int((labelled["y"] == 1).sum()),
                        folds=len(per_fold[chosen]),
                        chosen=chosen,
                        pr_auc=row["val_pr_auc"],
                        cv_pr_auc_mean=row["cv_pr_auc_mean"],
                        cv_pr_auc_sd=row["cv_pr_auc_sd"],
                        block_positives=int(target.sum()),
                    )
                )
            )
    except SplitTooShallow as error:
        blocker = (
            "No honest three-way split exists on this snapshot, so there is no validation\n"
            "block to select on and no training window deep enough to cut rolling-origin\n"
            "folds from. Comparison, threshold and calibration all wait on panel depth.\n\n"
            "```\n" + str(error).split("\n\n")[0] + "\n```"
        )
        print(f"not run: {str(error).splitlines()[0]}")

    write_report(
        args.out,
        frame,
        summary,
        per_fold,
        verdict,
        thresholds,
        calibration,
        by_source,
        by_carryover,
        blocker,
        generalisation,
        figures=figures,
        prov=provenance.collect(args.panel, len(frame), provenance.REAL),
        by_first_observation=by_first_observation,
    )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
