"""Does the model learn about job postings, or about these seven boards?

`docs/design.md` §4 leaves "is `source` a feature?" open, and the model card
says plainly that this is essentially a Greenhouse model. Both are statements
about the same worry, and neither is measured by the per-source breakdown
`CLAUDE.md` §4.5 already requires — that scores each board with a model **fitted
on it**, which answers "does it work here" rather than "would it work somewhere
new".

This module answers the second question by holding a whole board out of the fit.

**The comparison needs a control, and the obvious version of the experiment
omits it.** "Train on four boards, test on Duolingo, PR-AUC 0.09" is
uninterpretable on its own: boards have different base rates — 0.0090 on figma
against 0.0173 on discord — so a low score on an unseen board may be the board
being harder rather than the model failing to transfer. So each fold is run
twice on the same evaluation rows:

* **transfer** — fitted on every *other* board, scored on the held-out one;
* **ceiling** — fitted on every board including the held-out one, scored on the
  same rows.

The gap between them is what board-specific learning was worth. A gap near zero
means the model is using properties of postings that carry across boards; a
large gap means it learned this board. That is a statement neither number
supports alone, which is the whole reason both are computed.

**Folds are refused, not reported, when the held-out board has too few
positives.** On the 2026-09-07 panel that is most of them: airtable has none at
all, python_org three, duolingo five. A PR-AUC on five positives is a number
whose value is set by which five, and printing it next to a real one invites a
comparison that cannot be made.

**Evaluation is on the validation block.** This refits models, which makes it a
modelling activity, and `docs/design.md` §12 and the AST guard in
`tests/test_evaluate.py` both hold the line that anything which refits stays
away from the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.split import SplitResult
from src.features.preprocessing import features_and_target, fit_on_frame
from src.models.metrics import DEFAULT_ALERT_BUDGET, average_precision

#: A held-out board needs at least this many positives in the evaluation block
#: before its fold is scored. Ten is already thin — it buys an interval wide
#: enough to swallow most differences — and it is the point below which the
#: number stops being about the model at all.
MIN_HELD_OUT_POSITIVES = 10

#: And the remaining boards need to keep at least this share of the training
#: positives. Holding out anthropic removes 52% of them, so the transfer arm is
#: then fitted on half the data *and* one fewer board, and the two effects
#: cannot be told apart in the result.
MIN_TRAIN_SHARE = 0.6


@dataclass(frozen=True)
class BoardFold:
    """One board held out, scored twice on the same rows."""

    board: str
    eval_rows: int
    eval_positives: int
    train_positives: int
    train_share: float
    base_rate: float
    transfer_pr_auc: float
    ceiling_pr_auc: float

    @property
    def gap(self) -> float:
        """What seeing this board at fit time was worth, in PR-AUC."""
        return self.ceiling_pr_auc - self.transfer_pr_auc


@dataclass(frozen=True)
class Skipped:
    """A board whose fold was not run, and why."""

    board: str
    reason: str


def _score(model, block: pd.DataFrame) -> float:
    features, target = features_and_target(block)
    return average_precision(target, model.predict_proba(features)[:, 1])


def leave_one_board_out(
    split: SplitResult,
    build,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    min_positives: int = MIN_HELD_OUT_POSITIVES,
    min_train_share: float = MIN_TRAIN_SHARE,
) -> tuple[list[BoardFold], list[Skipped]]:
    """Hold each board out of the fit in turn; return the folds and the refusals.

    `build` is a zero-argument factory rather than a fitted model, because every
    fold needs its own fit and a reused estimator would carry the previous
    board's parameters into the next one.

    Both arms are fitted on the **training block only** and scored on the
    validation block, so the temporal discipline is unchanged: holding a board
    out removes rows from the fit, it does not license training on the future.
    """
    train, validation = split.train, split.val
    boards = sorted(set(validation["source"]) & set(train["source"]))
    total_positives = int((train["y"] == 1).sum())

    folds: list[BoardFold] = []
    skipped: list[Skipped] = []

    for board in boards:
        held_out = validation[validation["source"] == board]
        positives = int((held_out["y"] == 1).sum())
        if positives < min_positives:
            skipped.append(
                Skipped(board, f"{positives} positive(s) in validation, need {min_positives}")
            )
            continue

        remaining = train[train["source"] != board]
        kept = int((remaining["y"] == 1).sum())
        share = kept / total_positives if total_positives else 0.0
        if share < min_train_share:
            skipped.append(
                Skipped(
                    board,
                    f"holding it out leaves {share:.0%} of training positives, "
                    f"under {min_train_share:.0%} — the fit and the board would "
                    "change together",
                )
            )
            continue

        transfer = fit_on_frame(build(), remaining)
        ceiling = fit_on_frame(build(), train)

        folds.append(
            BoardFold(
                board=board,
                eval_rows=len(held_out),
                eval_positives=positives,
                train_positives=kept,
                train_share=share,
                base_rate=float((held_out["y"] == 1).mean()),
                transfer_pr_auc=_score(transfer, held_out),
                ceiling_pr_auc=_score(ceiling, held_out),
            )
        )

    return folds, skipped


def verdict(folds: list[BoardFold]) -> str:
    """What the folds say about transfer, or that they say nothing yet."""
    if not folds:
        return (
            "**No board could be held out.** Every fold was refused — see the table "
            "below for which and why. This is panel depth, not a result: the question "
            "is unanswered rather than answered negatively."
        )
    if len(folds) == 1:
        fold = folds[0]
        return (
            f"**One board only** (`{fold.board}`), so this is an observation rather "
            f"than a pattern. Seeing it at fit time was worth {fold.gap:+.4f} PR-AUC "
            f"on its own rows."
        )

    gaps = pd.Series([fold.gap for fold in folds])
    mean, spread = gaps.mean(), gaps.std(ddof=1)
    if abs(mean) <= spread:
        return (
            f"**Transfer looks intact.** Across {len(folds)} boards, seeing a board at "
            f"fit time was worth {mean:+.4f} PR-AUC on average against a spread of "
            f"{spread:.4f} — inside one standard deviation, so the model is not "
            "measurably better on boards it has seen. On this evidence it is using "
            "properties of postings rather than of boards."
        )
    if mean > 0:
        return (
            f"**Board-specific learning is doing work.** Seeing a board at fit time "
            f"was worth {mean:+.4f} PR-AUC across {len(folds)} boards, against a spread "
            f"of {spread:.4f}. A posting from a board the model has never seen is "
            "scored measurably worse, which is what the model card's Greenhouse "
            "caveat predicts and is the number that should sit next to it."
        )
    return (
        f"**The model does better on boards it did not see** ({mean:+.4f} across "
        f"{len(folds)}), which is not a result to celebrate — it usually means the "
        "held-out fits differ from the full fit in some way other than the board, "
        "and it is worth understanding before it is reported."
    )


def report_tables(
    folds: list[BoardFold], skipped: list[Skipped]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The folds and the refusals, as two frames for the report."""
    scored = pd.DataFrame(
        [
            {
                "board": fold.board,
                "eval_rows": fold.eval_rows,
                "eval_positives": fold.eval_positives,
                "base_rate": round(fold.base_rate, 4),
                "train_share": round(fold.train_share, 2),
                "transfer_pr_auc": round(fold.transfer_pr_auc, 4),
                "ceiling_pr_auc": round(fold.ceiling_pr_auc, 4),
                "gap": round(fold.gap, 4),
            }
            for fold in folds
        ]
    )
    refused = pd.DataFrame([{"board": s.board, "reason": s.reason} for s in skipped])
    return scored, refused
