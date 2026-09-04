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

import pandas as pd
from xgboost import XGBClassifier

from src.data.split import Cuts, SplitResult, SplitTooShallow, crawl_waves, temporal_split
from src.features.derive import DERIVED_COLUMNS
from src.features.preprocessing import (
    DERIVED,
    FEATURES,
    build_pipeline,
    feature_columns,
    features_and_target,
    fit_on_training_fold,
)
from src.models.metrics import DEFAULT_ALERT_BUDGET, evaluate
from src.models.train_baseline import (
    DEFAULT_PANEL,
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
    feature: str
    hypothesis: str


ABLATIONS: tuple[Ablation, ...] = tuple(
    Ablation(name, hypothesis)
    for name, hypothesis in (
        ("title_seniority", "how senior a role is relates to how long it takes to fill"),
        ("title_is_manager", "management roles have longer hiring processes"),
        ("title_words", "terse titles are boilerplate on high-volume reqs and churn faster"),
        ("title_chars", "as title_words, by another measure"),
        ("location_is_remote", "remote roles draw a larger pool and close faster"),
        ("n_locations", "a posting open in several places is a wider net"),
        ("salary_band", "pay level relates to fill speed, non-linearly"),
    )
)


def ablate(split: SplitResult, budget_per_day: int = DEFAULT_ALERT_BUDGET) -> pd.DataFrame:
    """Refit without each engineered feature in turn.

    Leave-one-out rather than add-one-in, because the question a hypothesis
    poses is "does the model need this?", and a feature can be redundant with
    another without being useless on its own. `delta` is the validation PR-AUC
    the feature is worth: positive means removing it hurt, so it earned its
    place.
    """
    everything = [column.name for column in feature_columns()]
    baseline = fit_and_score(build_xgboost(split), split, budget_per_day)

    rows = [{"removed": "nothing", "hypothesis": "—", **baseline, "delta": 0.0}]
    for ablation in ABLATIONS:
        kept = tuple(name for name in everything if name != ablation.feature)
        model = build_pipeline(XGBClassifier(**_xgb_parameters(split)), only=kept)
        scored = fit_and_score(model, split, budget_per_day)
        rows.append(
            {
                "removed": ablation.feature,
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


def write_report(
    path: Path,
    frame: pd.DataFrame,
    ladder: pd.DataFrame | None,
    ablations: pd.DataFrame | None,
    sweep: pd.DataFrame | None,
    blocker: str | None,
) -> None:
    reference = analytic_reference(frame)
    derived = ", ".join(f"`{name}`" for name in DERIVED_COLUMNS)
    lines = [
        "# Model results — engineered features and XGBoost",
        "",
        "Generated by `python -m src.models.train`. Regenerate rather than edit.",
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
            "Each row refits without one engineered feature. `delta` is the validation",
            "PR-AUC that feature is worth: positive means removing it hurt.",
            "",
            _table(
                ablations, ["removed", "hypothesis", "val_pr_auc", "delta", "train_pr_auc", "gap"]
            ),
            "",
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
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    waves = crawl_waves(frame[frame["label_observable"]])

    ladder = ablations = sweep = blocker = None
    try:
        split = temporal_split(frame, Cuts(waves.iloc[0], waves.iloc[len(waves) // 2]))
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

    write_report(args.out, frame, ladder, ablations, sweep, blocker)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
