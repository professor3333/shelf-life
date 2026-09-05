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

from src.data.split import Cuts, SplitResult, SplitTooShallow, crawl_waves, temporal_split
from src.features.preprocessing import features_and_target, fit_on_frame
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
    LADDER,
    _table,
    analytic_reference,
    prediction_days,
)

DEFAULT_REPORT = Path("reports/model_comparison.md")

#: Alert budgets to report the operating point across, so that the chosen one is
#: defended against alternatives rather than asserted alone.
BUDGET_SWEEP: tuple[int, ...] = (5, 10, 20, 50, 100)


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
    return pd.DataFrame(rows)


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
    false "closing soon" costs a rushed application measured in hours, a false
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


def select(summary: pd.DataFrame, per_fold: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Pick a model, and say whether the pick is defensible.

    The best CV mean wins — but the verdict also carries the paired comparison
    against the runner-up, because "best" is only meaningful if the gap survives
    fold variance. When it does not, the honest report is that the two are
    indistinguishable on this data and the simpler one should be preferred.
    """
    ranked = summary.dropna(subset=["cv_pr_auc_mean"]).sort_values(
        "cv_pr_auc_mean", ascending=False
    )
    if ranked.empty:
        return {"chosen": None, "reason": "no model was scored on any fold"}
    if len(ranked) == 1:
        return {"chosen": ranked.iloc[0]["model"], "reason": "only one model scored"}

    best, runner_up = ranked.iloc[0]["model"], ranked.iloc[1]["model"]
    difference = paired_fold_difference(per_fold[best], per_fold[runner_up])
    separated = (
        difference["folds"] > 1
        and not np.isnan(difference["sd"])
        and abs(difference["mean_difference"]) > difference["sd"]
    )
    return {
        "chosen": best,
        "runner_up": runner_up,
        **{f"vs_runner_up_{key}": value for key, value in difference.items()},
        "separated": separated,
        "reason": (
            f"{best} leads {runner_up} by {difference['mean_difference']:.4f} PR-AUC, "
            f"winning {difference['wins']:.0f} of {difference['folds']:.0f} folds"
            + ("" if separated else " — inside one standard deviation, so treat them as tied")
        ),
    }


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
) -> None:
    reference = analytic_reference(frame)
    lines = [
        "# Model comparison, threshold and calibration",
        "",
        "Generated by `python -m src.models.evaluate`. Regenerate rather than edit.",
        "",
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
            "| logistic | 0.1937 | 0.1294 | 0.1560 | 0.3849 |",
            "| random_forest | 0.2532 | 0.1844 | 0.1412 | 0.2156 |",
            "| xgboost | 0.1910 | 0.1083 | 0.1538 | 0.1899 |",
            "",
            "The random forest posts the best mean and the verdict still refuses it:",
            "*random_forest leads logistic by 0.0595 PR-AUC, winning 4 of 7 folds — inside",
            "one standard deviation, so treat them as tied.* That is the whole point of the",
            "error bars. On a label that is pure noise, every score above 0.10 is noise, and",
            "a comparison that reported the 0.2532 as a result would be inventing one.",
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
            "## Carried-over postings against unseen ones",
            "",
            _table(
                by_carryover, ["seen_in_train", "n", "positives", "base_rate", "pr_auc", "brier"]
            ),
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    waves = crawl_waves(frame[frame["label_observable"]])

    summary = per_fold = verdict = thresholds = calibration = None
    by_source = by_carryover = blocker = None
    try:
        split = temporal_split(frame, Cuts(waves.iloc[0], waves.iloc[len(waves) // 2]))
        summary, per_fold, val_scores = compare_models(split, args.budget)
        verdict = select(summary, per_fold)

        chosen = str(verdict["chosen"])
        scores = val_scores[chosen]
        target = split.val["y"].astype(int)
        thresholds = threshold_sweep(target, scores, prediction_days(split.val))
        calibration = reliability_curve(target, scores)
        by_source = evaluate_by(split.val, scores, "source", n_days=prediction_days(split.val))
        by_carryover = evaluate_by(
            split.val, scores, "seen_in_train", n_days=prediction_days(split.val)
        )
        print(summary.to_string(index=False))
        print(f"\nchosen: {verdict['chosen']} — {verdict['reason']}")
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
    )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
