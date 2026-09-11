"""Freeze one pipeline, then open the test set. **Once.**

This module is the end of the experimental phase and it is deliberately awkward
to use, because everything it does is irreversible.

**It will not choose the model for you.** `--run` is required and names a spec
from `src/models/experiments.py`. There is no default, and the omission is the
point: which model ships is a judgement made against the validation evidence in
`reports/model_comparison.md` and recorded in `docs/design.md`, not a constant
somebody typed here. A default would be that decision, made silently, by
whoever wrote this file.

**It is the only place in `src/` that reads the test block.** Everything else —
the ladder, the ablation, the tuning, the calibration — selects on validation,
and `tests/test_evaluate.py` parses `src/` to keep that true. This module is the
second and last name on that list. Its whole shape follows from a single rule:
*a test set looked at twice is a validation set*, so the pipeline is fitted and
the threshold is fixed **before** the test block is touched, and nothing after
the touch may change either. If the number disappoints, that is a finding about
the validation discipline, and it goes in the README as it came out.

**Both numbers are reported, side by side.** A validation score alone is the one
the model was selected on and therefore optimistic; a test score alone hides how
optimistic. The gap between them is the finding, and on a panel this small it is
also an error bar nobody computed — which the report says out loud.

The threshold is chosen on validation at the alert budget from `docs/design.md`
§5, and travels inside the artifact. The test block is scored twice: once at
that frozen threshold, which is what the deployed model will actually do, and
once at a threshold recomputed from the test block's own budget, which is what
the model could have done had the operating point been chosen with hindsight.
The difference between those two is the price of choosing an operating point in
advance, and it is a real cost that is usually left unmeasured.

Usage::

    python -m src.models.freeze --run 05-xgboost_engineered
    python -m src.models.freeze --run 05-xgboost_engineered --synthetic

Exit codes: ``0`` the artifact was written · ``3`` declined on depth, and
``reports/test_results.md`` says which of the two waits it declined on ·
``4`` declined because the working tree is not clean; nothing is written.

**Three refusals, not one.** ``SplitTooShallow`` asks whether a three-way cut is
legal; ``NoFoldEvidence`` asks whether anything could have *chosen* the model
being tested. The second arrives later than the first — five days later on the
real panel — and for most of that gap this step was the only one in the pipeline
that would run, which put the single unrepeatable action behind the weakest
check. Both are decisions, so both exit 3 rather than raising. ``DirtyWorktree``
is the third and is about the code, not the data: on a real panel the tree
must be clean, because the commit recorded on the artifact has to reproduce it
(`design.md` §16). It exits 4 and writes no report.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.split import (
    SplitResult,
    SplitTooShallow,
    crawl_waves,
    depth_report,
    temporal_split,
)
from src.features.assemble import horizon_banner
from src.features.preprocessing import features_and_target, fit_on_frame
from src.inference import artifact as artifact_module
from src.models import ledger, provenance
from src.models.evaluate import calibration_summary, wave_forward_folds
from src.models.experiments import (
    SYNTHETIC_PANEL_SOURCE,
    default_cuts,
    spec_by_name,
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
from src.models.train_baseline import DEFAULT_PANEL, RANDOM_STATE, _table, prediction_days
from src.models.uncertainty import bootstrap_block, format_table, fragility_note

DEFAULT_REPORT = Path("reports/test_results.md")

#: What the pipeline is fitted on before test is opened. `train` ships the
#: object whose validation number is known, so the test score describes exactly
#: the artifact that gets served. `train+val` ships a model fitted on more and
#: more recent data — the better deployment story — at the cost that its
#: validation number belongs to a slightly different object. Default `train`,
#: because the first property is the one the README has to be able to claim.
FIT_BLOCKS: dict[str, tuple[str, ...]] = {"train": ("train",), "train+val": ("train", "val")}


@dataclass(frozen=True)
class FrozenModel:
    """A fitted pipeline, the threshold it ships with, and both scores."""

    pipeline: object
    threshold: float
    validation: dict[str, float]
    test: dict[str, float]
    test_at_frozen_threshold: dict[str, float]
    #: Posting-clustered bootstrap intervals on the test block. The headline is
    #: PR-AUC on roughly twenty positives, and a point estimate at that count is
    #: not a result on its own — `src/models/uncertainty.py` says why the
    #: resampling unit is the posting rather than the row.
    test_intervals: dict[str, object]
    #: The fragility note, or None when the block is big enough not to need one.
    test_fragility: str | None
    by_source: pd.DataFrame
    by_seen_in_train: pd.DataFrame
    calibration: dict[str, float]
    #: The binned reliability curve, not just its summary. `calibration` gives
    #: Brier and ECE, which say *how far* the probabilities are from the truth
    #: and not *which way* — a model that is uniformly overconfident and one that
    #: is confident in the wrong direction can score the same. Validation has
    #: carried the curve since Component 10; the test block reported only the
    #: two scalars until 2026-09-09, which meant the block opened once and the
    #: shape of its calibration was never looked at.
    reliability: pd.DataFrame
    #: Prediction days in the test block, so a daily rate can be quoted. Counted
    #: here because the block is not reachable from the report.
    test_days: int
    params: dict
    features: tuple[str, ...]
    fitted_on: str


def _fit_block(split: SplitResult, fitted_on: str) -> pd.DataFrame:
    """The rows the pipeline is fitted on, assembled from named blocks only.

    Concatenation rather than a wider cut, so that the embargo between train and
    val is still discarded and the test block still cannot be reached by any
    combination of the names.
    """
    blocks = [getattr(split, name) for name in FIT_BLOCKS[fitted_on]]
    return pd.concat(blocks, ignore_index=True) if len(blocks) > 1 else blocks[0]


def _score(model, block: pd.DataFrame) -> pd.Series:
    features, _ = features_and_target(block)
    return pd.Series(model.predict_proba(features)[:, 1], index=block.index)


def freeze(
    split: SplitResult,
    run_name: str,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    fitted_on: str = "train",
    params: dict | None = None,
) -> FrozenModel:
    """Fit, fix the threshold on validation, then read test exactly once.

    The order of the statements below is the whole discipline, so it is worth
    reading as an order rather than as a list: fit, score validation, choose the
    threshold, and only then touch `split.test`. Nothing after that line feeds
    back into anything before it.
    """
    spec = spec_by_name(run_name)
    if spec.leaky:
        raise ValueError(
            f"{spec.run_name} is a deliberately leaky run (src/features/leaky.py). "
            "It exists to measure the size of a lie, and must never be served."
        )

    resolved = spec.resolve(split) if params is None else dict(params)
    model = spec.build(split, resolved)
    fit_on_frame(model, _fit_block(split, fitted_on))

    validation_scores = _score(model, split.val)
    validation = evaluate(
        split.val["y"], validation_scores, prediction_days(split.val), budget_per_day
    )
    validation["ece"] = expected_calibration_error(split.val["y"], validation_scores)

    # The operating point, chosen here and never again. From validation, at the
    # budget a person can actually read in a day — `docs/design.md` §5.
    threshold = threshold_for_budget(
        validation_scores,
        alert_budget(len(split.val), prediction_days(split.val), budget_per_day),
    )

    # ---------------------------------------------------------------------
    # The test block is opened on the next line. Everything above is frozen.
    # ---------------------------------------------------------------------
    test_block = split.test
    test_scores = _score(model, test_block)
    test = evaluate(test_block["y"], test_scores, prediction_days(test_block), budget_per_day)
    test["ece"] = expected_calibration_error(test_block["y"], test_scores)

    return FrozenModel(
        pipeline=model,
        threshold=float(threshold),
        validation=validation,
        test=test,
        test_at_frozen_threshold=confusion_at(test_block["y"], test_scores, threshold),
        test_intervals=bootstrap_block(test_block, test_scores, float(threshold)),
        test_fragility=fragility_note(test_block),
        by_source=evaluate_by(
            test_block,
            test_scores,
            "source",
            n_days=prediction_days(test_block),
            budget_per_day=budget_per_day,
        ),
        by_seen_in_train=evaluate_by(
            test_block,
            test_scores,
            "seen_in_train",
            n_days=prediction_days(test_block),
            budget_per_day=budget_per_day,
        ),
        calibration=calibration_summary(test_block["y"], test_scores),
        reliability=reliability_curve(test_block["y"], test_scores),
        test_days=int(prediction_days(test_block)),
        params=resolved,
        features=tuple(spec_features(spec, split)),
        fitted_on=fitted_on,
    )


def spec_features(spec, split: SplitResult) -> tuple[str, ...]:
    """The feature names the spec's pipeline actually selects.

    Read off the built pipeline rather than recomputed from the feature registry,
    because a spec may restrict the set — run 04 does — and a metadata field that
    lists features the model never saw is worse than no field.
    """
    pipeline = spec.build(split, spec.resolve(split))
    return tuple(pipeline.named_steps["select"].kw_args["columns"])


def build_metadata(
    frozen: FrozenModel,
    run_name: str,
    panel: pd.DataFrame,
    panel_path: Path,
    dataset: str,
    budget_per_day: int,
    selection_folds: int = -1,
) -> artifact_module.Metadata:
    spec = spec_by_name(run_name)
    horizons = pd.unique(panel["horizon_days"])
    bases = pd.unique(panel["horizon_basis"])
    return artifact_module.Metadata(
        run_name=spec.run_name,
        question=spec.question,
        params={key: str(value) for key, value in frozen.params.items()},
        features=frozen.features,
        fitted_on=frozen.fitted_on,
        threshold=frozen.threshold,
        budget_per_day=budget_per_day,
        horizon_days=int(horizons[0]),
        horizon_basis=str(bases[0]),
        metrics={
            **{f"val_{key}": float(value) for key, value in frozen.validation.items()},
            **{f"test_{key}": float(value) for key, value in frozen.test.items()},
        },
        provenance=provenance.collect(panel_path, len(panel), dataset).as_tags(),
        dataset=dataset,
        selection_folds=selection_folds,
        rules_version=int(panel["rules_version"].iloc[0]) if "rules_version" in panel else -1,
        lock_sha256=provenance.lock_sha256(),
        seed=RANDOM_STATE,
    )


def what_the_user_gets(frozen: FrozenModel, budget_per_day: int) -> dict[str, float]:
    """The frozen numbers restated as the decision they inform.

    `docs/problem_definition.md` §8 names the user and the decision: someone
    reading a daily list of postings ranked by "likely gone within a week",
    deciding *what to apply to tonight*. Precision and recall answer that
    question only after arithmetic, and doing the arithmetic in a reader's head
    is where a metric quietly stops meaning anything — 0.21 precision sounds
    poor until it is set against a base rate of 0.10, and 0.44 recall sounds
    fine until it is restated as *the majority of closures are missed*.

    So both readings are computed and reported together, along with the honest
    counterfactual: what the same person would get by reading the same number of
    postings off the top of the board without a model. That comparison is the
    one that decides whether any of this was worth building, and it is the one
    a table of metrics never quite states.
    """
    confusion = frozen.test_at_frozen_threshold
    days = max(1, frozen.test_days)
    base_rate = float(frozen.test["base_rate"])

    flagged = float(confusion["flagged"])
    true_positives = float(confusion["tp"])
    missed = float(confusion["fn"])
    precision = float(confusion["precision"])

    return {
        "alerts_per_day": flagged / days,
        "real_closures_caught_per_day": true_positives / days,
        "false_alarms_per_day": float(confusion["fp"]) / days,
        "closures_missed_per_day": missed / days,
        "share_of_closures_caught": float(confusion["recall"]),
        # What the same reading effort returns with no model at all: the budget
        # drawn from a board whose closure rate is the base rate.
        "unaided_closures_caught_per_day": (flagged * base_rate) / days,
        "lift_over_unaided": (precision / base_rate) if base_rate else float("nan"),
    }


NOT_RUN = """## Nothing below has run

The test set has not been opened, because no honest three-way split exists on
this snapshot yet. That is the correct outcome rather than a failure: a test
block with no positives is not a hard test set, it is an undefined metric, and
freezing a model against one would produce a README number that means nothing.

```
{refusal}
```

**How much longer.** {depth}

The packaging around it is built and tested — `src/inference/artifact.py`,
`src/inference/contract.py` and `src/inference/predict.py`, exercised end to end
on the synthetic panel — so what waits here is panel depth, not code.
"""


NO_FOLDS = """## Nothing below has run

A legal three-way split exists on this snapshot, but not an *evaluable* one, and
freeze refuses on the difference.

```
{refusal}
```

**How much longer.** {depth}

**Why legality is not enough.** The test block is opened once and spent. What it
buys is a check on a model that validation already chose — so if nothing chose
one, there is nothing for it to check. Rolling-origin folds are cut from the
training window, and at this depth that window is a single wave: every model
comparison is one number with no spread, and picking the leader from a column of
numbers that have no error bars is picking noise. Spending the test set to
measure that pick would produce a README figure with a decimal point and no
meaning behind it.

`scripts/watch_depth.sh` reports the shortfall daily and runs the rehearsal on
the day it clears. Pass `--accept-no-folds` to override this, which spends the
held-out block on a model nothing selected; the report and the artifact both
record that it was used.
"""


class DirtyWorktree(RuntimeError):
    """The tree has uncommitted changes, so the SHA would not describe the run.

    A report from a dirty tree says so and is understood as provisional. An
    artifact from one is not provisional — it ships — and a SHA that does not
    reproduce it is a SHA that names nothing. Real panels only: a synthetic
    freeze is a rehearsal and says `dataset=synthetic` on every response.
    """


class NoFoldEvidence(RuntimeError):
    """The split is legal, but no fold could be cut to choose a model on.

    Separate from `SplitTooShallow` because the two are different waits with
    different remedies, and collapsing them is how "the panel is too shallow"
    stops being read once it is half true.
    """


def _what_it_means_section(frozen: FrozenModel, budget_per_day: int) -> list[str]:
    """The last section on purpose. Every number above is an input to it."""
    reading = what_the_user_gets(frozen, budget_per_day)
    base_rate = float(frozen.test["base_rate"])
    return [
        "## What this means for the person using it",
        "",
        "`docs/problem_definition.md` §8: someone reads a daily list of postings",
        "ranked by *likely gone within a week* and decides what to apply to tonight.",
        "Restating the table above as that day:",
        "",
        f"- The list runs to **{reading['alerts_per_day']:.0f} postings a day**.",
        f"- About **{reading['real_closures_caught_per_day']:.1f} of them are genuinely "
        f"about to be removed**; the other {reading['false_alarms_per_day']:.1f} are not.",
        f"- That is **{reading['share_of_closures_caught']:.0%} of the removals** that "
        f"happen — so **{reading['closures_missed_per_day']:.1f} a day are missed**, and "
        "missing one is the expensive error: a rushed application costs hours, a job "
        "never applied to is unrecoverable.",
        "",
        "**Against reading the same number of postings with no model at all**, off a "
        f"board losing {base_rate:.1%} of its postings a day: that person would find "
        f"**{reading['unaided_closures_caught_per_day']:.1f}** real ones against this "
        f"model's {reading['real_closures_caught_per_day']:.1f} — a lift of "
        f"**{reading['lift_over_unaided']:.1f}×**.",
        "",
        "**That ratio is the whole case for the model, and it is the number to argue",
        "about.** Below about 1.5× the honest report is that a person would do nearly",
        "as well reading the board directly, whatever the PR-AUC says — and the",
        "interval on precision above is wide enough that the lift carries one too.",
        "The claim is *removed from the board*, never *filled*: a posting also comes",
        "down when it expires, is reposted, or the page is reorganised.",
        "",
    ]


def write_report(
    path: Path,
    frozen: FrozenModel | None,
    metadata: artifact_module.Metadata | None,
    budget_per_day: int,
    blocker: str | None,
    panel: pd.DataFrame | None = None,
    prov: provenance.Provenance | None = None,
) -> None:
    lines = [
        "# Test-set results",
        "",
        *provenance.header(
            prov,
            "python -m src.models.freeze",
            horizon_banner(panel) if panel is not None else None,
        ),
        "**This is the number, whatever it is.** The test block is opened once, after",
        "the pipeline and its threshold are frozen, and nothing downstream of it may",
        "change either. A disappointing figure here is evidence about the validation",
        "discipline, not a licence to tune — the tuning would be selection on test,",
        "and the next number would mean less than this one.",
        "",
    ]

    if blocker is not None:
        lines += [blocker, ""]
    else:
        assert frozen is not None and metadata is not None
        both = pd.DataFrame(
            [
                {"block": "validation", **frozen.validation},
                {"block": "test", **frozen.test},
            ]
        )
        lines += [
            f"Model: **{metadata.run_name}** — *{metadata.question}*",
            f"Fitted on: `{frozen.fitted_on}`. "
            f"Horizon: {metadata.horizon_days} day(s), {metadata.horizon_basis} basis.",
            "",
            "## Validation and test, side by side",
            "",
            _table(
                both,
                [
                    "block",
                    "n",
                    "positives",
                    "base_rate",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "ece",
                    "precision",
                    "recall",
                    "f1",
                ],
            ),
            "",
            f"The gap in PR-AUC is "
            f"**{frozen.validation['pr_auc'] - frozen.test['pr_auc']:+.4f}** "
            "(validation minus test). Validation is the block the model was selected on,",
            "so it is optimistic by construction; the gap is a measurement of how much.",
            "",
            "## How much of this is signal",
            "",
            "The headline is a point estimate on a block with tens of positives, so it",
            "comes with a **95% interval from resampling postings** — not rows. The panel",
            "is one row per (posting, crawl) at about six rows per posting, so rows are",
            "not independent draws: the board accumulates postings and observes them",
            "daily. `src/models/uncertainty.py` records the measurement showing the two",
            "resampling schemes disagree, and why the design argument settles it rather",
            "than the direction of the disagreement.",
            "",
            "`precision` and `recall` are measured at the **frozen** threshold, held fixed",
            "across resamples. Recomputing the budget threshold inside each one would mix",
            "how well the model separates with where the operating point happened to land,",
            "and the artifact ships one threshold rather than a distribution of them.",
            "",
            *format_table(frozen.test_intervals),
            "",
            *([frozen.test_fragility, ""] if frozen.test_fragility else []),
            "**Read the width, not the centre.** If an interval spans the baseline, the",
            "honest report is that this test set could not tell the model from the",
            "baseline — which is a result about the sample size and not a failure of the",
            "model. Fold-to-fold variation is the companion number and lives in",
            "`reports/model_comparison.md`; the two answer different questions, one about",
            "this block and one about which block you happened to get.",
            "",
            "## The operating point, applied as frozen",
            "",
            f"Threshold **{frozen.threshold:.6f}**, chosen on validation at "
            f"{budget_per_day} alerts per prediction day and shipped inside the artifact.",
            "Applied unchanged to the test block:",
            "",
            _table(
                pd.DataFrame([frozen.test_at_frozen_threshold]),
                ["threshold", "flagged", "tp", "fp", "fn", "tn", "precision", "recall", "f1"],
            ),
            "",
            "The row above is what the deployed model does. The `precision`/`recall` in",
            "the first table come from a threshold recomputed on the test block's own",
            "budget — what the model *could* have done had the operating point been",
            "chosen with hindsight. The difference between the two is the price of",
            "fixing a threshold in advance, and it is a real cost that is usually left",
            "unmeasured.",
            "",
            "## Per source",
            "",
            "A model that only works on one board is a per-board model.",
            "",
            _table(
                frozen.by_source,
                ["source", "n", "positives", "base_rate", "pr_auc", "precision", "recall"],
            ),
            "",
            "## Carried over from training, or not",
            "",
            "`seen_in_train` is true for postings the model was also fitted on, in an",
            "earlier wave. A model scoring well on those and badly on the rest has",
            "memorised postings rather than learned duration dependence.",
            "",
            _table(
                frozen.by_seen_in_train,
                ["seen_in_train", "n", "positives", "base_rate", "pr_auc", "precision", "recall"],
            ),
            "",
            "## Calibration on test",
            "",
            "Brier and ECE say how far the probabilities sit from the truth. They do",
            "not say which way, and a model that is uniformly overconfident scores the",
            "same as one that is confident in the wrong direction — so the curve is",
            "here too, and it is the one to read.",
            "",
            _table(pd.DataFrame([frozen.calibration]), list(frozen.calibration)),
            "",
            _table(
                frozen.reliability,
                ["bin_low", "bin_high", "n", "mean_predicted", "observed_rate"],
            ),
            "",
            "A bin whose `observed_rate` sits above its `mean_predicted` is one where",
            "the model was too cautious; below, too confident. Bins with a handful of",
            "rows say nothing either way — read `n` first.",
            "",
            *_what_it_means_section(frozen, budget_per_day),
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="which spec from src/models/experiments.py to freeze. No default: "
        "choosing the model that ships is a decision recorded in docs/design.md",
    )
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact", type=Path, default=artifact_module.DEFAULT_ARTIFACT)
    parser.add_argument("--budget", type=int, default=DEFAULT_ALERT_BUDGET)
    parser.add_argument("--fit", choices=sorted(FIT_BLOCKS), default="train")
    parser.add_argument(
        "--accept-no-folds",
        action="store_true",
        help="freeze even though no rolling-origin fold exists to have chosen the "
        "model on. Spends the held-out block, once, on a pick nothing selected",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="freeze against tests/panels.py instead of the real panel, for when "
        "the real one is too shallow to split",
    )
    args = parser.parse_args()

    if args.synthetic:
        from tests.panels import make_closing_panel

        panel, panel_path, dataset = (
            make_closing_panel(),
            SYNTHETIC_PANEL_SOURCE,
            provenance.SYNTHETIC,
        )
    else:
        panel, panel_path, dataset = (
            pd.read_parquet(args.panel),
            args.panel,
            provenance.REAL,
        )

    frozen = metadata = blocker = None
    try:
        split = temporal_split(panel, default_cuts(panel))

        # Before `freeze`, which opens the test block. `SplitTooShallow` covers
        # legality only, and legality arrives days before evaluability — on the
        # real panel, five. Between those two dates every gate upstream of here
        # declines and this one used to wave the run through, which put the
        # single irreversible step behind the weakest check in the pipeline.
        n_folds = len(wave_forward_folds(split.train, split.embargo))
        if n_folds == 0 and not args.accept_no_folds:
            raise NoFoldEvidence(
                f"the training window yields {n_folds} rolling-origin fold(s), so no "
                "model comparison on this snapshot has an error bar and nothing has "
                "selected a model to test."
            )

        # The last check before the irreversible step, and the one that is
        # about the code rather than the data (`design.md` §16).
        if dataset == provenance.REAL and not provenance.worktree_is_clean():
            raise DirtyWorktree(
                "the working tree has uncommitted changes, so the commit recorded on "
                "the artifact would not reproduce it. Commit (or stash) everything — "
                "the regenerated reports included — and run freeze again."
            )
        frozen = freeze(split, args.run, args.budget, args.fit)
        metadata = build_metadata(
            frozen, args.run, panel, panel_path, dataset, args.budget, n_folds
        )
        artifact_module.save(frozen.pipeline, metadata, args.artifact)
        labelled = panel[panel["label_observable"]]
        ledger.write_report(
            ledger.append(
                ledger.record(
                    stage=ledger.HELD_OUT,
                    provenance=provenance.collect(panel_path, len(panel), dataset),
                    labelled_waves=len(crawl_waves(labelled)),
                    labelled_rows=len(labelled),
                    positives=int((labelled["y"] == 1).sum()),
                    folds=len(wave_forward_folds(split.train, split.embargo)),
                    chosen=args.run,
                    pr_auc=frozen.test["pr_auc"],
                    pr_auc_low=frozen.test_intervals["pr_auc"].low,
                    pr_auc_high=frozen.test_intervals["pr_auc"].high,
                    block_positives=int(split.frame.loc[split.frame["split"] == "test", "y"].sum()),
                )
            )
        )
        print(
            f"val pr_auc {frozen.validation['pr_auc']:.4f} -> "
            f"test pr_auc {frozen.test['pr_auc']:.4f}"
        )
        print(f"threshold  {frozen.threshold:.6f}")
        print(f"wrote -> {args.artifact}")
    except SplitTooShallow as error:
        blocker = NOT_RUN.format(refusal=str(error).split("\n\n")[0], depth=depth_report(panel))
        print(f"not run: {str(error).splitlines()[0]}")
    except NoFoldEvidence as error:
        blocker = NO_FOLDS.format(refusal=str(error), depth=depth_report(panel))
        print(f"not run: {error}")
        print("`--accept-no-folds` overrides this and spends the held-out block.")
    except DirtyWorktree as error:
        # No report: writing one would dirty the tree further, and this is not
        # a finding about the data. Its own exit code, so a caller can tell
        # "commit first" from "wait for depth".
        print(f"not run: {error}")
        raise SystemExit(4) from None

    write_report(
        args.out,
        frozen,
        metadata,
        args.budget,
        blocker,
        panel=panel,
        prov=provenance.collect(panel_path, len(panel), dataset),
    )
    print(f"wrote -> {args.out}")

    # A refusal that exits 0 reads as success to every caller that checks `$?`,
    # and this is the step whose success means an artifact exists. 3 is what
    # `scripts/rehearse.sh` already uses for "declined, nothing was run", so a
    # caller can tell a decision apart from a crash.
    if blocker is not None:
        raise SystemExit(3)


if __name__ == "__main__":  # pragma: no cover
    main()
