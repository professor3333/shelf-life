"""The metric suite, and the threshold rule.

`docs/design.md` §5 chose these and said why. This module is the choice in code:

**average precision (PR-AUC)** is primary: the positive class is rare and it is
the interesting one. **Brier score and a reliability curve** are co-primary,
because the output is consumed as a probability and ranking well while
calibrated badly fails the actual use. **Precision and recall at the alert
budget** are the operational read — they mirror the real decision, which is a
shortlist a person reads rather than a full ranking. **ROC-AUC** is reported for
comparability with other people's numbers and is not decisive, being insensitive
to the imbalance that matters here.

**Accuracy is not computed anywhere in this file, deliberately.** At a 1.2%
positive rate a model that predicts "stays open" for every posting scores 98.8%,
and there is no threshold at which that number carries information. A test
asserts it is absent from the reported keys, so it cannot creep back in as a
convenience.

**The threshold is not 0.5.** `docs/design.md` §5 fixed the cost asymmetry: a
false "closing soon" costs a rushed application, measured in hours; a false
"stays open" costs a job never applied to, which is unrecoverable. The second is
worse, so the operating point leans to recall and the threshold is set by a
fixed **alert budget** — twenty flagged postings a day, a list a person can
actually read — rather than by a probability that happens to be half.

`average_precision` is implemented here rather than imported, and
`tests/test_metrics.py` checks it against scikit-learn on random inputs. The
point is not distrust of scikit-learn; it is that a metric you cannot derive is
a metric you cannot defend, and the analytic values for a constant predictor
(AP = the base rate, ROC-AUC = 0.5, Brier = p(1-p)) are the check that catches a
metric wired up to the wrong argument.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

#: Postings a person will actually look at in a day. `docs/design.md` §5.
DEFAULT_ALERT_BUDGET = 20


def average_precision(y_true, y_score) -> float:
    """Area under the precision-recall curve, by the step-wise sum.

    ``AP = Σ (R_n − R_{n−1}) · P_n`` over the distinct score thresholds, which
    is the definition scikit-learn uses and is *not* the trapezoid rule — the PR
    curve's interpolation is misleading between points, so the rectangle is the
    honest reading.

    Ties are taken as one group, which is what makes a constant predictor score
    exactly the base rate instead of something that depends on row order.
    """
    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    if truth.size == 0:
        return float("nan")
    positives = truth.sum()
    if positives == 0:
        return float("nan")  # undefined, and saying so beats returning 0.0

    order = np.argsort(-score, kind="stable")
    truth, score = truth[order], score[order]

    # One cut per distinct score, taking tied rows together.
    last_of_group = np.r_[np.where(np.diff(score))[0], truth.size - 1]
    true_positives = np.cumsum(truth)[last_of_group]
    flagged = (np.arange(truth.size) + 1.0)[last_of_group]

    precision = true_positives / flagged
    recall = true_positives / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier_score(y_true, y_score) -> float:
    """Mean squared error of the probability. Lower is better.

    For a constant prediction `q` against a base rate `p` this is
    ``p(1−q)² + (1−p)q²``, minimised at `q = p` where it equals `p(1−p)`. That
    identity is the analytic check on the whole suite.
    """
    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    return float(np.mean((score - truth) ** 2))


def reliability_curve(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, by bin.

    The table behind the calibration plot. A bin whose postings were given 30%
    should contain about 30% positives; the gap between `mean_predicted` and
    `observed_rate` is the miscalibration, and `n` says whether to believe it.
    """
    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.digitize(score, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for index in range(n_bins):
        mask = which == index
        rows.append(
            {
                "bin_low": edges[index],
                "bin_high": edges[index + 1],
                "n": int(mask.sum()),
                "mean_predicted": float(score[mask].mean()) if mask.any() else float("nan"),
                "observed_rate": float(truth[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Average gap between predicted probability and observed frequency.

    The single number behind the reliability curve: each bin's
    ``|mean predicted − observed rate|``, weighted by how many rows fall in it.
    Zero is perfect calibration.

    It is reported *alongside* the Brier score, not instead of it, because the
    two fail differently. Brier mixes calibration with discrimination, so a
    model can improve it by ranking better while staying badly calibrated. ECE
    ignores ranking entirely, so a model that predicts the base rate for every
    posting scores a perfect 0.0 — useless and perfectly calibrated. Either
    number alone can be gamed; the pair cannot.
    """
    curve = reliability_curve(y_true, y_score, n_bins)
    populated = curve[curve["n"] > 0]
    if populated.empty:
        return float("nan")
    gaps = (populated["mean_predicted"] - populated["observed_rate"]).abs()
    return float((gaps * populated["n"]).sum() / populated["n"].sum())


def alert_budget(n_rows: int, n_days: int, budget_per_day: int = DEFAULT_ALERT_BUDGET) -> int:
    """How many postings may be flagged across an evaluation block.

    The budget is per day because that is how the list is read. A block spanning
    three prediction days gets three days of budget, capped at the rows that
    exist.
    """
    return int(min(n_rows, max(1, budget_per_day * max(1, n_days))))


def threshold_for_budget(y_score, budget: int) -> float:
    """The score at which exactly `budget` postings are flagged.

    Returned as a value rather than a mask so that the operating point is a
    number that can be written down, argued about, and applied unchanged at
    serve time — which is what makes it a decision rather than a default.
    """
    score = np.asarray(y_score, dtype=float)
    if score.size == 0:
        return float("nan")
    budget = int(np.clip(budget, 1, score.size))
    return float(np.sort(score)[::-1][budget - 1])


def confusion_at(y_true, y_score, threshold: float) -> dict[str, float]:
    """Counts and rates at one operating point. No accuracy — see the module docstring."""
    truth = np.asarray(y_true, dtype=float)
    flagged = np.asarray(y_score, dtype=float) >= threshold

    true_positives = float(np.sum(flagged & (truth == 1)))
    false_positives = float(np.sum(flagged & (truth == 0)))
    false_negatives = float(np.sum(~flagged & (truth == 1)))
    true_negatives = float(np.sum(~flagged & (truth == 0)))

    precision = (
        true_positives / (true_positives + false_positives) if flagged.any() else float("nan")
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else float("nan")
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and not np.isnan(precision) and not np.isnan(recall)
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "flagged": float(flagged.sum()),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tn": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    y_true,
    y_score,
    n_days: int = 1,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
) -> dict[str, float]:
    """The full suite at the budget-derived operating point."""
    truth = np.asarray(y_true, dtype=float)
    score = np.asarray(y_score, dtype=float)
    positives = float(truth.sum())

    budget = alert_budget(truth.size, n_days, budget_per_day)
    threshold = threshold_for_budget(score, budget)

    try:
        roc = float(roc_auc_score(truth, score)) if 0 < positives < truth.size else float("nan")
    except ValueError:  # pragma: no cover - guarded by the bounds above
        roc = float("nan")

    return {
        "n": float(truth.size),
        "positives": positives,
        "base_rate": positives / truth.size if truth.size else float("nan"),
        "pr_auc": average_precision(truth, score),
        "brier": brier_score(truth, score),
        "roc_auc": roc,
        "alert_budget": float(budget),
        **confusion_at(truth, score, threshold),
    }


def evaluate_by(
    frame: pd.DataFrame,
    y_score,
    group: str,
    target: str = "y",
    n_days: int = 1,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
) -> pd.DataFrame:
    """The suite, broken down by a column.

    Two breakdowns are not optional. **Per source**, because arbeitnow aside the
    boards differ and a model that only works on anthropic is a per-board model
    (`docs/leakage_audit.md`). **By `seen_in_train`**, because `docs/design.md`
    §8 accepted subject overlap and promised this arm as the memorisation
    diagnosis: a large gap between carried-over and unseen postings is the
    finding.

    The threshold is recomputed within each group, so each row answers "how does
    this model behave on this slice", not "how does the global operating point
    land here".
    """
    scores = pd.Series(np.asarray(y_score, dtype=float), index=frame.index)
    rows = []
    for value, block in frame.groupby(group, sort=True, dropna=False):
        summary = evaluate(block[target], scores.loc[block.index], n_days, budget_per_day)
        rows.append({group: value, **summary})
    return pd.DataFrame(rows)
