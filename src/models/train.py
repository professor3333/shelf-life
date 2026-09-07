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

from src.data.split import SplitResult, SplitTooShallow, best_cuts, temporal_split
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
        "board context is worth having at all — the question `docs/design.md` §12 "
        "is open on, and the only one that matches what a caller actually loses"
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
    everything = [column.name for column in feature_columns()]
    baseline = fit_and_score(build_xgboost(split), split, budget_per_day)

    rows = [{"removed": "nothing", "n_removed": 0, "hypothesis": "—", **baseline, "delta": 0.0}]
    for ablation in ABLATIONS:
        withheld = set(ablation.features)
        kept = tuple(name for name in everything if name not in withheld)
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

    everything = [column.name for column in feature_columns()]
    withheld = set(BOARD_CONTEXT_ABLATION.features)
    kept = tuple(name for name in everything if name not in withheld)
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


def _board_context_verdict(ablations: pd.DataFrame) -> list[str]:
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
        "The number prices one thing only: what a caller who supplies no board "
        "context loses. It does not price a caller who can supply it — a board owner "
        "scoring their own requisitions has all four, and for them the imputation "
        "branch never runs.",
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
            *_board_context_verdict(ablations),
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
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    args = parser.parse_args()

    frame = pd.read_parquet(args.panel)
    ladder = ablations = sweep = blocker = fold_evidence = None
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

    write_report(args.out, frame, ladder, ablations, sweep, blocker, fold_evidence)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
