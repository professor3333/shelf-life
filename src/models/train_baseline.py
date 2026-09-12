"""The baseline ladder: climb one rung at a time and write down what each buys.

    prior  →  age_days only  →  per-board hazard  →  logistic  →  tree  →  forest

**The point of the ladder is not the top rung.** It is finding out whether
increasing complexity buys anything on this problem. If the random forest ties
the logistic regression, that is a finding about the data and it gets written
down rather than tuned past. Jumping straight to the last rung hides whether the
problem needed a model at all.

**Every rung is scored on the validation block, never on test.** The test block
is opened once, at the end of the build, and `SplitResult.test` is not read
anywhere in this module.

**Every rung is fitted through `fit_on_training_fold`**, so no rung can see
validation data while learning — including the two baselines, which would
otherwise be the easiest place to cheat without noticing: a base rate computed
over the whole frame is a leak that looks like a constant.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.split import (
    Cuts,
    SplitResult,
    SplitTooShallow,
    best_cuts,
    temporal_split,
)
from src.features.assemble import panel_path
from src.features.preprocessing import build_pipeline, features_and_target, fit_on_training_fold
from src.models import provenance
from src.models.baselines import (
    BoardHazardBaseline,
    RuleBaseline,
    SingleFeatureCeiling,
    older_than_thirty_days,
    posted_this_week,
)
from src.models.metrics import DEFAULT_ALERT_BUDGET, evaluate, evaluate_by, reliability_curve

DEFAULT_PANEL = panel_path()
DEFAULT_REPORT = Path("reports/baseline_results.md")

#: Fixed everywhere a model can be seeded. Two runs of this file on one snapshot
#: must produce the same table, or none of the comparisons below mean anything.
RANDOM_STATE = 0


@dataclass(frozen=True)
class Rung:
    name: str
    description: str
    build: Callable[[], Pipeline]


LADDER: tuple[Rung, ...] = (
    Rung(
        "prior",
        "constant base rate",
        lambda: build_pipeline(DummyClassifier(strategy="prior")),
    ),
    # --- rules a person could follow, before any estimator ------------------
    # Placed here on purpose. The ladder is meant to be climbed in ascending
    # seriousness, and a hand-written rule is more serious than a constant and
    # less serious than a fit. Putting them after the models would invite the
    # reading that they are an afterthought; the whole point is that everything
    # below has to beat them.
    Rung(
        "rule_older_than_30d",
        "rule: up more than a month, so not about to close",
        lambda: RuleBaseline(
            rule=older_than_thirty_days,
            statement="A posting that has been up more than a month is not about to close.",
        ),
    ),
    Rung(
        "rule_posted_this_week",
        "rule: fresh postings move, stale ones have stalled",
        lambda: RuleBaseline(
            rule=posted_this_week,
            statement="A posting still up after a week has stalled; the fresh ones move.",
        ),
    ),
    Rung(
        "age_ceiling",
        "the best any age-only rule could do (deciles, non-monotone)",
        lambda: SingleFeatureCeiling(
            column="age_days",
            bins=10,
            statement="Age used as well as any rule could use it.",
        ),
    ),
    Rung(
        "age_only",
        "age_days alone, logistic",
        lambda: build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced"), only=("age_days",)
        ),
    ),
    Rung(
        "board_hazard",
        "per-board historical rate",
        BoardHazardBaseline,
    ),
    Rung(
        "logistic",
        "logistic regression, all allowed features",
        lambda: build_pipeline(LogisticRegression(max_iter=2000, class_weight="balanced")),
    ),
    Rung(
        "decision_tree",
        "decision tree, depth-limited",
        lambda: build_pipeline(
            DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
    ),
    Rung(
        "random_forest",
        "random forest, 300 trees",
        lambda: build_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
        ),
    ),
)


#: Rungs a person could act on without a computer, plus the ceiling that bounds
#: what any age-only rule could buy. Everything not named here is a fitted model
#: and has to earn its place against the best of these.
HEURISTIC_RUNGS: tuple[str, ...] = (
    "prior",
    "rule_older_than_30d",
    "rule_posted_this_week",
    "age_ceiling",
    "board_hazard",
)


def heuristic_verdict(results: pd.DataFrame, budget_per_day: int) -> list[str]:
    """Does the modelling buy anything over a rule someone could follow?

    The comparison the ladder above cannot make. Climbing from logistic to
    XGBoost shows that one estimator beats another, which is a statement about
    scikit-learn; a reader wants to know whether any of it beats *skip anything
    that has been up a month*.

    **Precision is the comparison, not PR-AUC.** The operating point is an alert
    budget — a fixed number of postings a person will actually read each day — so
    what the reader experiences is how many of those alerts were worth opening.
    PR-AUC is reported beside it because a rule and a model can tie at the budget
    while ranking very differently underneath, and a model that ranks better has
    somewhere to go when the budget changes.
    """
    if results.empty or "precision" not in results.columns:
        return []
    rules = results[results["model"].isin(HEURISTIC_RUNGS)]
    models = results[~results["model"].isin(HEURISTIC_RUNGS)]
    if rules.empty or models.empty:
        return []

    best_rule = rules.loc[rules["precision"].idxmax()]
    best_model = models.loc[models["precision"].idxmax()]
    gain = float(best_model["precision"]) - float(best_rule["precision"])

    if gain > 0:
        reading = (
            f"The modelling buys **{gain:+.4f}** precision at the budget over the best "
            f"rule. Whether that is worth the pipeline is a judgement, but it is a "
            f"judgement about a real number rather than about a model comparison."
        )
    elif gain == 0:
        reading = (
            "The best model and the best rule are **exactly level** at the budget. "
            "Everything the pipeline adds is currently invisible to the person "
            "reading the alerts."
        )
    else:
        reading = (
            f"**The best rule wins**, by {-gain:.4f} precision at the budget. That is a "
            "finding about this problem, not a bug to tune away: on this data a "
            "person following one sentence does better than the fitted models, and "
            "the honest report of that is worth more than a model that edges past it "
            "after enough attempts."
        )

    return [
        "### Does the modelling beat a rule?",
        "",
        f"At {budget_per_day} alerts per prediction day:",
        "",
        f"- best rule — `{best_rule['model']}`, precision **{best_rule['precision']:.4f}**, "
        f"PR-AUC {best_rule['pr_auc']:.4f}",
        f"- best model — `{best_model['model']}`, precision **{best_model['precision']:.4f}**, "
        f"PR-AUC {best_model['pr_auc']:.4f}",
        "",
        reading,
        "",
        "Both numbers are single draws on one validation block. A difference smaller "
        "than the fold spread in `reports/model_comparison.md` is not a difference, "
        "and at this panel depth most of them will be.",
        "",
    ]


def prediction_days(block: pd.DataFrame) -> int:
    """Distinct calendar days of predictions in a block, for the alert budget."""
    return int(block["t"].dt.date.nunique())


def score_rung(rung: Rung, split: SplitResult, budget_per_day: int) -> tuple[dict, pd.Series]:
    """Fit one rung on the training fold and score it on validation."""
    model = rung.build()
    fit_on_training_fold(model, split)

    validation = split.val
    features, target = features_and_target(validation)
    scores = pd.Series(model.predict_proba(features)[:, 1], index=validation.index)

    summary = evaluate(
        target, scores, n_days=prediction_days(validation), budget_per_day=budget_per_day
    )
    return {"model": rung.name, "description": rung.description, **summary}, scores


def run_ladder(
    split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Every rung, in order. Returns the results table and each rung's scores."""
    rows, scores = [], {}
    for rung in LADDER:
        summary, rung_scores = score_rung(rung, split, budget_per_day)
        rows.append(summary)
        scores[rung.name] = rung_scores
    return pd.DataFrame(rows), scores


def analytic_reference(frame: pd.DataFrame) -> dict[str, float]:
    """What a constant predictor scores, derived rather than fitted.

    This is the number every later model must beat, and it is available without
    a split: for a constant score, average precision equals the base rate,
    ROC-AUC is exactly 0.5, and the Brier score of predicting the base rate `p`
    is `p(1−p)`. `tests/test_baseline.py` asserts the fitted dummy reproduces
    these, which is how a wrongly wired metric gets caught on a model whose
    answer can be computed by hand.
    """
    labelled = frame[frame["label_observable"]]
    positives = float((labelled["y"] == 1).sum())
    n = float(len(labelled))
    base_rate = positives / n if n else float("nan")
    return {
        "n": n,
        "positives": positives,
        "base_rate": base_rate,
        "pr_auc": base_rate,
        "roc_auc": 0.5,
        "brier": base_rate * (1.0 - base_rate),
    }


def _cell(value) -> str:
    if isinstance(value, float):
        return "—" if pd.isna(value) else f"{value:.4f}".rstrip("0").rstrip(".")
    return "—" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    """A markdown table, written here rather than via `to_markdown`.

    `DataFrame.to_markdown` needs `tabulate`, and a formatting convenience is a
    poor reason to add a runtime dependency to a project a stranger has to
    install.
    """
    shown = [column for column in columns if column in frame.columns]
    header = "| " + " | ".join(shown) + " |"
    rule = "|" + "|".join("---" for _ in shown) + "|"
    body = [
        "| " + " | ".join(_cell(row[column]) for column in shown) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *body])


def write_report(
    path: Path,
    frame: pd.DataFrame,
    split: SplitResult | None,
    results: pd.DataFrame | None,
    scores: dict[str, pd.Series] | None,
    blocker: str | None,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    prov: provenance.Provenance | None = None,
) -> None:
    """Write `reports/baseline_results.md`, whether or not the ladder could run."""
    reference = analytic_reference(frame)
    horizon = int(pd.unique(frame["horizon_days"])[0])
    basis = str(pd.unique(frame["horizon_basis"])[0])

    lines = [
        "# Baseline results",
        "",
        *provenance.header(
            prov, "python -m src.models.train_baseline", f"H = {horizon} ({basis} basis)"
        ),
        "## The number every model must beat",
        "",
        "For a constant predictor, average precision **is** the base rate, ROC-AUC is",
        "exactly 0.5, and the Brier score of predicting `p` is `p(1-p)`. These are",
        "derived, not fitted, so they are available before any split is possible.",
        "",
        f"- labelled rows: **{reference['n']:,.0f}**",
        f"- positives: **{reference['positives']:,.0f}**",
        f"- base rate: **{reference['base_rate']:.4f}**",
        f"- constant-predictor PR-AUC: **{reference['pr_auc']:.4f}**",
        f"- constant-predictor Brier: **{reference['brier']:.4f}**",
        "",
        "Accuracy is not reported at any point in this file. At this base rate, always",
        f'predicting "stays open" scores {100 * (1 - reference["base_rate"]):.1f}%.',
        "",
    ]

    if blocker is not None:
        lines += [
            "## The fitted ladder has not run",
            "",
            blocker,
            "",
        ]
    else:
        assert split is not None and results is not None and scores is not None
        lines += [
            "## The split",
            "",
            f"- cut at `{split.cuts.train_end}` and `{split.cuts.val_end}`",
            f"- embargo `{split.embargo}` between blocks",
            f"- train {len(split.train):,} rows / validation {len(split.val):,} rows",
            f"- {split.n_embargoed:,} rows discarded to the embargo, "
            f"{split.n_unlabelled:,} unlabelled rows dropped before splitting",
            "",
            "The test block is not scored here. It is opened once, at the end of the build.",
            "",
            "## The ladder, scored on validation",
            "",
            _table(
                results,
                [
                    "model",
                    "description",
                    "pr_auc",
                    "brier",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "alert_budget",
                    "flagged",
                    "tp",
                    "fp",
                    "fn",
                ],
            ),
            "",
            "The first rungs are rules, not fits: `rule_older_than_30d` and",
            "`rule_posted_this_week` are one sentence each, and `age_ceiling` bounds what",
            "any age-only rule could buy by binning age into deciles and predicting each",
            "bin's training rate — free to be non-monotone, which `age_only`'s logistic",
            "fit is not. Everything below them has to beat them.",
            "",
            *heuristic_verdict(results, budget_per_day),
            "## Per source",
            "",
            _table(
                evaluate_by(split.val, scores["logistic"], "source"),
                ["source", "n", "positives", "base_rate", "pr_auc", "brier"],
            ),
            "",
            "## Carried-over postings against unseen ones",
            "",
            "`design.md` §8 accepted subject overlap across the cut and promised this",
            "breakdown as the memorisation diagnosis. A large gap is the finding.",
            "",
            _table(
                evaluate_by(split.val, scores["logistic"], "seen_in_train"),
                ["seen_in_train", "n", "positives", "base_rate", "pr_auc", "brier"],
            ),
            "",
            "## Calibration, logistic regression",
            "",
            _table(
                reliability_curve(split.val["y"].astype(int), scores["logistic"]),
                ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"],
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
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--val-end", type=str, default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    reference = analytic_reference(frame)
    print(f"labelled rows      : {reference['n']:,.0f}")
    print(f"positives          : {reference['positives']:,.0f}")
    print(f"base rate          : {reference['base_rate']:.4f}")
    print(f"constant PR-AUC    : {reference['pr_auc']:.4f}   <- the number to beat")
    print(f"constant Brier     : {reference['brier']:.4f}")

    split = results = scores = blocker = None
    try:
        if args.train_end or args.val_end:
            chosen = best_cuts(frame)
            cuts = Cuts(
                train_end=pd.Timestamp(args.train_end) if args.train_end else chosen.train_end,
                val_end=pd.Timestamp(args.val_end) if args.val_end else chosen.val_end,
            )
        else:
            cuts = best_cuts(frame)
        split = temporal_split(frame, cuts)
        results, scores = run_ladder(split, args.budget)
        print()
        print(results.to_string(index=False))
    except SplitTooShallow as error:
        blocker = (
            "No honest three-way split exists on this snapshot, so no rung above the\n"
            "constant predictor has a validation block to be scored on. The refusal:\n\n"
            "```\n" + str(error) + "\n```\n\n"
            "This is panel depth, not a cut that can be moved. The scraper adds a wave\n"
            "a day; re-running this command is the readiness check."
        )
        print()
        print(f"ladder not run: {error}".split("\n")[0])

    prov = provenance.collect(args.panel, len(frame), provenance.REAL)
    write_report(args.out, frame, split, results, scores, blocker, args.budget, prov)
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
