"""Can the production features name the board, with board identity removed?

    python -m src.models.board_fingerprint            # H=1 panel, the one with a legal cut
    python -m src.models.board_fingerprint --panel <parquet> --out <md>

`design.md` §4 excludes board identity — `source`, `company`, `url`'s domain —
because the product scores a posting from a board it has never seen. Excluding
the columns is not the same as excluding the information. Some fields are
systematically unavailable on one board (`departments` is null only for
python_org, `age_days` only there too), and the pipeline's missing-value
policy encodes that absence as a category or a sentinel, which is legitimate
information at serve time and also a fingerprint. If `departments ==
__missing__` *means* `source == python_org`, the column was removed and the
identity stayed.

This module measures that directly rather than arguing it: fit a classifier to
predict `source` from the production feature matrix — the same `Pipeline`,
fitted on the training block only, with the estimator swapped for one whose
target is the board — and score it on the validation block. Then the same
question three more ways: from missingness alone, from each feature alone,
and with each feature removed, so the number comes with the names of the
columns that carry it.

**What the number means, and does not.** A representation from which the board
is recoverable is not thereby wrong: at serve time a python_org posting
genuinely has no `departments`, and the model is allowed to know that. What the
fingerprint costs is *transfer* — a model that has learned "this is python_org"
as a proxy for hazard has learned nothing it can use on a board it has not
seen — and transfer is measured elsewhere, by `generalisation.leave_one_board_out`.
This file says how much identity the matrix carries and where; that one says
whether the model spent it. Neither substitutes for the other.

Reads train and validation only. The closure label is not used at all — the
fingerprint is a property of the feature matrix, not of the horizon — so the
H=1 panel is the default: it is the panel with a legal cut today, and the cut
only decides which rows the imputers are fitted on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from src.data.split import SplitResult, SplitTooShallow, best_cuts, temporal_split
from src.features.assemble import horizon_banner, panel_path
from src.features.preprocessing import build_pipeline, feature_columns
from src.inference.contract import BOARD_CONTEXT
from src.models import provenance

DEFAULT_REPORT = Path("reports/board_fingerprint.md")
TARGET = "source"

#: The classifier that asks "can *any* model recover the board". A forest, not a
#: linear model, because sentinels and one-hot missing levels are exactly the
#: kind of split a tree finds first and a logistic regression can miss.
FOREST = dict(n_estimators=200, class_weight="balanced", random_state=0, n_jobs=-1)
#: One feature at a time gets a shallow tree: enough depth to carve a sentinel
#: out, not enough to memorise the block.
STUMP = dict(max_depth=6, class_weight="balanced", random_state=0)


def _fit_identity(split: SplitResult, estimator, **pipeline_kwargs):
    """The production pipeline with its estimator's target swapped for the board.

    Fitted on `split.train` and nothing else — the imputers, the category
    tables and the estimator all see the training block only. This is the
    third caller of `Pipeline.fit` on a block after the two `preprocessing`
    names, and it is a diagnostic rather than a model: nothing fitted here is
    scored on the label, persisted or served.
    """
    pipeline = build_pipeline(estimator, **pipeline_kwargs)
    return pipeline.fit(split.train, split.train[TARGET])


def _score(pipeline, block: pd.DataFrame) -> dict[str, float]:
    predicted = pipeline.predict(block)
    truth = block[TARGET].to_numpy()
    return {
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def majority_baseline(split: SplitResult) -> dict[str, float]:
    """What guessing the training block's commonest board scores on validation."""
    commonest = split.train[TARGET].value_counts().idxmax()
    truth = split.val[TARGET].to_numpy()
    predicted = np.full(len(truth), commonest, dtype=object)
    return {
        "board": str(commonest),
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def full_matrix(split: SplitResult) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """The headline: the whole production feature set against the board.

    Returns the scores, the per-board recall, and the confusion table.
    """
    pipeline = _fit_identity(split, RandomForestClassifier(**FOREST))
    scores = _score(pipeline, split.val)
    predicted = pipeline.predict(split.val)
    truth = split.val[TARGET]
    per_board = (
        pd.DataFrame({"board": truth.to_numpy(), "hit": predicted == truth.to_numpy()})
        .groupby("board")
        .agg(rows=("hit", "size"), recall=("hit", "mean"))
        .reset_index()
    )
    confusion = pd.crosstab(
        pd.Series(truth.to_numpy(), name="actual"), pd.Series(predicted, name="predicted")
    )
    return scores, per_board, confusion


def without_board_context(split: SplitResult) -> dict[str, float]:
    """The §12 "absent" set: production features minus the four board-context
    columns. The nearest thing the design has to a board-independent feature
    set, scored the same way, so "would removing board context make the matrix
    board-neutral" is answered with a number rather than a hope."""
    names = tuple(column.name for column in feature_columns() if column.name not in BOARD_CONTEXT)
    pipeline = _fit_identity(split, RandomForestClassifier(**FOREST), only=names)
    return _score(pipeline, split.val)


def missingness_only(split: SplitResult) -> dict[str, float]:
    """The board from *which fields are null*, and nothing else.

    The question the missing-value policy raises in its sharpest form. Every
    production feature is replaced by its null indicator, computed on the raw
    block (before imputation, which is what erases the pattern into a sentinel)
    — so this needs no fitted transformer and cannot leak.
    """
    names = [column.name for column in feature_columns() if column.name in split.train]
    train = split.train[names].isna().astype(int)
    val = split.val[names].isna().astype(int)
    model = DecisionTreeClassifier(**STUMP).fit(train, split.train[TARGET])
    predicted = model.predict(val)
    truth = split.val[TARGET].to_numpy()
    return {
        "accuracy": float((predicted == truth).mean()),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "indicators_with_any_null": int((train.sum() > 0).sum()),
    }


def per_feature(split: SplitResult) -> pd.DataFrame:
    """Each production feature alone, through its own branch of the pipeline.

    `only=(name,)` keeps the feature on the same imputation and encoding it
    gets in production, so a sentinel that fingerprints the board fingerprints
    it here for the same reason. Sorted by accuracy: the top of the table is
    the list of columns that carry identity.
    """
    rows = []
    for column in feature_columns():
        pipeline = _fit_identity(split, DecisionTreeClassifier(**STUMP), only=(column.name,))
        rows.append({"feature": column.name, "fill": column.fill, **_score(pipeline, split.val)})
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def drop_one(split: SplitResult, full_accuracy: float) -> pd.DataFrame:
    """The full set with each feature removed: how much identity is *only* there.

    A feature whose removal costs nothing is redundant with the rest for this
    purpose — its fingerprint is carried elsewhere too — which is why this
    table and the one above disagree, and why both are shown.
    """
    names = tuple(column.name for column in feature_columns())
    rows = []
    for name in names:
        rest = tuple(other for other in names if other != name)
        pipeline = _fit_identity(split, RandomForestClassifier(**FOREST), only=rest)
        scores = _score(pipeline, split.val)
        rows.append(
            {
                "removed": name,
                "accuracy_without": scores["accuracy"],
                "drop": full_accuracy - scores["accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values("drop", ascending=False).reset_index(drop=True)


# --- the report --------------------------------------------------------------


def _md(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return ["_empty_"]
    shown = table.copy()
    for col in shown.columns:
        if shown[col].dtype.kind == "f":
            shown[col] = shown[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return [header, "|" + "---|" * len(shown.columns), *body]


def _verdict(full: dict[str, float], baseline: dict[str, float], missing: dict[str, float]) -> str:
    if full["accuracy"] >= 0.95:
        strength = (
            f"**The board is recoverable from the production features: {full['accuracy']:.1%} "
            f"accuracy, macro-F1 {full['macro_f1']:.3f}, against {baseline['accuracy']:.1%} for "
            f"guessing `{baseline['board']}`.** Removing `source` removed a column, not the "
            "information."
        )
    elif full["accuracy"] >= baseline["accuracy"] + 0.10:
        strength = (
            f"The board is partly recoverable: {full['accuracy']:.1%} accuracy, macro-F1 "
            f"{full['macro_f1']:.3f}, against {baseline['accuracy']:.1%} for guessing "
            f"`{baseline['board']}`."
        )
    else:
        strength = (
            f"The board is not recoverable beyond the majority guess: {full['accuracy']:.1%} "
            f"against {baseline['accuracy']:.1%}."
        )
    missing_line = (
        f" Missingness alone — which fields are null, before any imputation — scores "
        f"{missing['accuracy']:.1%} (macro-F1 {missing['macro_f1']:.3f}) from "
        f"{missing['indicators_with_any_null']} indicator(s)."
    )
    return strength + missing_line


def render(panel: pd.DataFrame, prov: provenance.Provenance | None = None) -> str:
    try:
        split = temporal_split(panel, best_cuts(panel))
    except SplitTooShallow as exc:
        return (
            "\n".join(
                [
                    "# Board fingerprint — can the production features name the board?",
                    "",
                    *provenance.header(
                        prov, "python -m src.models.board_fingerprint", horizon_banner(panel)
                    ),
                    f"_No legal cut on this panel yet ({exc}), so there is no training block",
                    "to fit the imputers on. The H=1 panel has one; pass it with `--panel`._",
                    "",
                ]
            )
            + "\n"
        )

    baseline = majority_baseline(split)
    full, per_board, confusion = full_matrix(split)
    no_context = without_board_context(split)
    missing = missingness_only(split)
    singles = per_feature(split)
    drops = drop_one(split, full["accuracy"])

    lines = [
        "# Board fingerprint — can the production features name the board?",
        "",
        *provenance.header(prov, "python -m src.models.board_fingerprint", horizon_banner(panel)),
        "`design.md` §4 excludes board identity so the model can score a posting from a",
        "board it has never seen. This asks whether the identity left with the columns.",
        "A random forest is fitted on the **training block only** — the production",
        "`Pipeline`, same imputation and encoding, with the estimator's target swapped for",
        f"`{TARGET}` — and asked to name the board of every validation row. The closure",
        "label is not used: a fingerprint is a property of the feature matrix.",
        "",
        "## The headline",
        "",
        _verdict(full, baseline, missing),
        "",
        f"Training block {len(split.train):,} rows, validation {len(split.val):,}; "
        f"{split.train[TARGET].nunique()} boards in training.",
        "",
        "### Three feature sets, one question",
        "",
        "| feature set | accuracy | macro-F1 |",
        "|---|---|---|",
        f"| guess the commonest board (`{baseline['board']}`) | {baseline['accuracy']:.3f} | "
        f"{baseline['macro_f1']:.3f} |",
        f"| missingness only — which fields are null, no values | {missing['accuracy']:.3f} | "
        f"{missing['macro_f1']:.3f} |",
        f'| production features minus board context (`design.md` §12 "absent") | '
        f"{no_context['accuracy']:.3f} | {no_context['macro_f1']:.3f} |",
        f"| **production features** | **{full['accuracy']:.3f}** | **{full['macro_f1']:.3f}** |",
        "",
        "### Per board",
        "",
        *_md(per_board),
        "",
        "### Confusion (rows actual, columns predicted)",
        "",
        *_md(confusion.reset_index()),
        "",
        "## Where the identity is",
        "",
        "### Each feature alone",
        "",
        "One shallow tree per feature, on that feature's own production branch —",
        "so a sentinel that fingerprints the board in the model fingerprints it here",
        "for the same reason. `fill` is the missing-value policy that produced it.",
        "",
        *_md(singles),
        "",
        "### The full set with one feature removed",
        "",
        "How much identity is carried *only* by that feature. A removal that costs",
        "nothing means the rest carries it too — which is why this table and the one",
        "above disagree, and why both are shown.",
        "",
        *_md(drops),
        "",
        "## What this means for the model, and what it does not",
        "",
        "A matrix from which the board is recoverable is not thereby wrong. A python_org",
        "posting genuinely has no `departments` at serve time, and the model is allowed",
        "to know that; `design.md` §4a is the policy. What a fingerprint can cost is",
        "**transfer**: a model that has learned *this is python_org* as a proxy for",
        "hazard has learned nothing it can use on a board it has not seen. That is",
        "measured by `generalisation.leave_one_board_out` in `model_comparison.md`,",
        "which removes a board from the fit and scores it cold. This file says how much",
        "identity the matrix carries and where; that one says whether the model spent",
        "it. Neither substitutes for the other, and the second needs positives per",
        "board that this panel does not yet have.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=panel_path(1))
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    prov = provenance.collect(args.panel, len(panel), provenance.REAL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = render(panel, prov)
    args.out.write_text(text)
    headline = next(
        (line for line in text.splitlines() if line.startswith(("**The board", "The board"))), ""
    )
    print(headline[:160])
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
