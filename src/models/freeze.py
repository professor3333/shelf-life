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
from src.features.preprocessing import features_and_target, fit_on_frame
from src.inference import artifact as artifact_module
from src.models import provenance
from src.models.evaluate import calibration_summary
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
    threshold_for_budget,
)
from src.models.train_baseline import DEFAULT_PANEL, _table, prediction_days

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
    by_source: pd.DataFrame
    by_seen_in_train: pd.DataFrame
    calibration: dict[str, float]
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
    )


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


def write_report(
    path: Path,
    frozen: FrozenModel | None,
    metadata: artifact_module.Metadata | None,
    budget_per_day: int,
    blocker: str | None,
) -> None:
    lines = [
        "# Test-set results",
        "",
        "Generated by `python -m src.models.freeze`. Regenerate rather than edit.",
        "",
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
            _table(pd.DataFrame([frozen.calibration]), list(frozen.calibration)),
            "",
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
        waves = crawl_waves(panel[panel["label_observable"]])
        split = temporal_split(panel, default_cuts(waves))
        frozen = freeze(split, args.run, args.budget, args.fit)
        metadata = build_metadata(frozen, args.run, panel, panel_path, dataset, args.budget)
        artifact_module.save(frozen.pipeline, metadata, args.artifact)
        print(
            f"val pr_auc {frozen.validation['pr_auc']:.4f} -> "
            f"test pr_auc {frozen.test['pr_auc']:.4f}"
        )
        print(f"threshold  {frozen.threshold:.6f}")
        print(f"wrote -> {args.artifact}")
    except SplitTooShallow as error:
        blocker = NOT_RUN.format(refusal=str(error).split("\n\n")[0], depth=depth_report(panel))
        print(f"not run: {str(error).splitlines()[0]}")

    write_report(args.out, frozen, metadata, args.budget, blocker)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
