"""Whether to recalibrate the frozen model, decided by a rule written before the number.

The output is consumed as a probability, so Brier and the reliability curve are
co-primary with PR-AUC (`docs/design.md` §5). What §5 did not say, until
2026-09-12, was what happens when the curve comes back bent: a decision left
to the afternoon the numbers appear is a decision made by the numbers. So the
rule is here, fixed while every validation ECE in the repository is H=1 or
synthetic, and `evaluate` reports what it *would* do for the chosen model
before `freeze` does it.

**The rule.** Recalibrate iff, on the validation block,

    ECE  >  RECALIBRATION_ECE_RATIO × base rate      and      positives ≥ MIN_POSITIVES

The bound is relative to the base rate rather than absolute, because at a 7.7%
positive rate an ECE of 0.02 is a quarter of everything there is to predict,
while at 50% it would be noise. The positives floor is the same one the
bootstrap uses to call an interval fragile: a monotone curve fitted to fewer
than thirty events is fitted to those events.

**Why recalibration cannot change who is flagged.** The method is isotonic
regression — monotone non-decreasing — fitted on validation scores, wrapped
around the pipeline's final estimator so the served artifact stays one
`Pipeline`. The operating point is a rank statistic, the budget-th validation
score (`threshold_for_budget`), and a monotone map preserves ranks. So the
alert list under the recalibrated model is the alert list under the raw one,
with one honest exception: isotonic regression pools neighbouring scores into
steps, so postings that straddled the operating point can tie on it, and the
rule `probability >= threshold` includes ties. Recalibration can therefore
add a posting at the boundary, never remove one, and never reorder any. What
changes is what the number *means* — which is the whole point, since the
number is shown to a person as a percentage.

**What it costs.** The recalibrated pipeline carries the validation block's
curve inside it. The test block is opened after, as before, and its ECE is the
honest measure of whether the curve transferred; the report says which model
it is scoring. Fitted on validation only, never on train (it would learn the
training fit's overconfidence) and never on test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from src.models.metrics import expected_calibration_error
from src.models.uncertainty import FRAGILE_POSITIVES

#: The pre-registered bound: recalibrate when validation ECE exceeds this
#: fraction of the validation base rate.
RECALIBRATION_ECE_RATIO = 0.25

#: Fewer positives than this and no curve is fitted, whatever the ECE says.
MIN_POSITIVES = FRAGILE_POSITIVES

METHOD = "isotonic"


@dataclass(frozen=True)
class Decision:
    """What the rule said, and the numbers it said it about."""

    recalibrate: bool
    ece: float
    base_rate: float
    bound: float
    positives: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "recalibrate": self.recalibrate,
            "method": METHOD if self.recalibrate else None,
            "rule": f"ece > {RECALIBRATION_ECE_RATIO} * base_rate and positives >= {MIN_POSITIVES}",
            "validation_ece": self.ece,
            "validation_base_rate": self.base_rate,
            "bound": self.bound,
            "validation_positives": self.positives,
            "reason": self.reason,
        }


def decide(y_true, y_score) -> Decision:
    """Apply the rule to a validation block's scores."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positives = int(truth.sum())
    base_rate = float(truth.mean()) if len(truth) else float("nan")
    ece = float(expected_calibration_error(truth, scores))
    bound = RECALIBRATION_ECE_RATIO * base_rate
    if positives < MIN_POSITIVES:
        return Decision(
            False,
            ece,
            base_rate,
            bound,
            positives,
            f"{positives} validation positives is under the {MIN_POSITIVES} needed to fit a curve",
        )
    if ece > bound:
        return Decision(
            True,
            ece,
            base_rate,
            bound,
            positives,
            f"validation ECE {ece:.4f} exceeds {RECALIBRATION_ECE_RATIO} × base rate "
            f"{base_rate:.4f} = {bound:.4f}",
        )
    return Decision(
        False,
        ece,
        base_rate,
        bound,
        positives,
        f"validation ECE {ece:.4f} is within {RECALIBRATION_ECE_RATIO} × base rate "
        f"{base_rate:.4f} = {bound:.4f}",
    )


def recalibrate(pipeline: Pipeline, validation: pd.DataFrame, target: pd.Series) -> Pipeline:
    """The same pipeline with its final estimator wrapped in an isotonic map.

    Everything before the estimator is left as fitted; the validation block is
    pushed through those steps and the curve is fitted on the estimator's
    scores for it. The result is still a `Pipeline` with the same step names,
    so the artifact contract (`assert_is_full_pipeline`) and the serving path
    are unchanged.
    """
    head = Pipeline(pipeline.steps[:-1])
    name, estimator = pipeline.steps[-1]
    transformed = head.transform(validation)
    calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method=METHOD)
    calibrated.fit(transformed, np.asarray(target, dtype=int))
    return Pipeline(pipeline.steps[:-1] + [(name, calibrated)])
