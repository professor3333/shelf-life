"""Whether the default public model keeps the four board-context features.

`docs/design.md` §12 kept `board_size_at_t`, `board_growth`,
`n_same_title_on_board` and `n_same_req_on_board` on a measured cost of
removal, with serve-time imputation for a caller who cannot supply them —
which, for someone holding one job advert, is every caller. §4a then made
that conditional on transfer holding under the imputed regime. What neither
said was what to do at the freeze if the model that was fitted *with* those
columns turns out to do worse, once they are imputed, than a model that never
had them. That decision is made here, by a rule written on 2026-09-12 while
the only numbers it could be applied to are H=1 or synthetic.

**Three validation PR-AUCs for the chosen candidate.**

    supplied   the full model, board context present     — what `/rank` gets
               from a caller who can describe the board
    imputed    the same model, board context withheld    — what `/predict`
               gets from everyone else (`generalisation.without_board_context`)
    without    a refit with the four columns removed     — the alternative
               default

**The rule.** Ship the refit as the default public model iff

    imputed + fold_sd  <  without

— the full model, once its board columns are imputed to the training fold's
constants, does worse than a model that never had them by more than the
candidate's own fold-to-fold spread. Otherwise the full model ships, with
imputation, as §12 decided. The spread is the noise scale the selection rule
already uses (`design.md` §14); a difference inside it is not a difference.

**What it deliberately does not do.** It does not compare `supplied` against
`without` — that answers "are the columns worth having when present", which
§12 answered and which is the wrong question for a default model most callers
reach with the columns absent. And it does not decide transfer: that is the
gate in `generalisation.assess`, run on whichever pipeline this rule ships.

Fitted on the training block only, scored on validation; nothing here reads
the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from src.data.split import SplitResult
from src.features.preprocessing import build_pipeline, features_and_target, fit_on_frame
from src.inference.contract import BOARD_CONTEXT
from src.models.generalisation import without_board_context
from src.models.metrics import average_precision

FULL, WITHOUT = "full", "without_board_context"

RULE = "ship the no-board-context refit iff imputed_pr_auc + fold_sd < without_pr_auc"


@dataclass(frozen=True)
class Decision:
    ship: str
    supplied: float
    imputed: float
    without: float
    fold_sd: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "ship": self.ship,
            "rule": RULE,
            "val_pr_auc_supplied": self.supplied,
            "val_pr_auc_imputed": self.imputed,
            "val_pr_auc_without": self.without,
            "fold_sd": self.fold_sd,
            "board_context_features": list(BOARD_CONTEXT),
            "reason": self.reason,
        }


def without_board_context_pipeline(pipeline: Pipeline, leaky: bool = False) -> Pipeline:
    """The same estimator, the same feature set minus the four board columns.

    Read the selection off the built pipeline rather than off the registry, so
    a spec that already restricts its columns (run 04) is narrowed from its own
    set and not from everything.
    """
    selected = tuple(pipeline.named_steps["select"].kw_args["columns"])
    kept = tuple(column for column in selected if column not in BOARD_CONTEXT)
    estimator = clone(pipeline.steps[-1][1])
    return build_pipeline(estimator, only=kept, include_leaky=leaky)


def _pr_auc(model, block) -> float:
    features, target = features_and_target(block)
    return float(average_precision(target, model.predict_proba(features)[:, 1]))


def decide(split: SplitResult, build, fold_sd: float, leaky: bool = False) -> Decision:
    """Apply the rule to a candidate; `build` is the spec's zero-argument builder."""
    full = fit_on_frame(build(), split.train)
    supplied = _pr_auc(full, split.val)
    imputed = _pr_auc(full, without_board_context(split.val))
    reduced = fit_on_frame(without_board_context_pipeline(build(), leaky), split.train)
    without = _pr_auc(reduced, split.val)
    sd = 0.0 if fold_sd is None or np.isnan(fold_sd) else float(fold_sd)

    if imputed + sd < without:
        return Decision(
            WITHOUT,
            supplied,
            imputed,
            without,
            sd,
            f"with board context imputed the candidate scores {imputed:.4f}, and even "
            f"{sd:.4f} of fold spread above that is under the {without:.4f} a refit "
            "without the four columns scores — the default caller is better served by "
            "a model that never had them",
        )
    return Decision(
        FULL,
        supplied,
        imputed,
        without,
        sd,
        f"with board context imputed the candidate scores {imputed:.4f} against "
        f"{without:.4f} for a refit without the columns; inside {sd:.4f} of fold spread, "
        "so §12 stands and the full model ships with imputation",
    )
