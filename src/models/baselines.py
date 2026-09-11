"""Baselines that are not models.

`docs/design.md` §7 names three, in ascending order of seriousness, and a model
must beat **all three** to have earned anything:

1. **the constant base rate** — `DummyClassifier(strategy="prior")`. Beating it
   proves only that features exist;
2. **`age_days` alone** — the honest baseline. Most of the signal in a survival
   problem is duration dependence, and if the full model does not beat this then
   the posting's content contributes nothing. That is a real finding, not a
   failure;
3. **per-board hazard** — this file. It predicts each board's historical removal
   rate and knows nothing else. A content model that cannot beat it has learned
   only which company posted the job.

The third is the sharpest of the three because of what `docs/leakage_audit.md`
found: board identity is spread across `source`, `company`, `url` and the
missingness of every archive-derived column, so a model can absorb it without
ever being handed `source`. This baseline is the yardstick that makes that
visible — it *is* the board-identity model, so the gap between it and a real
model is the part that is not board identity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class BoardHazardBaseline(ClassifierMixin, BaseEstimator):
    """Predict each board's removal rate, learned from the training fold only.

    Takes the raw job-day frame rather than a preprocessed matrix, because
    `source` is deliberately not a feature by default (`docs/design.md` §4) and
    this estimator needs it. That is the point of it: it is the model you get if
    board identity is *all* you use.

    A board unseen at fit time falls back to the pooled training rate, which is
    the honest answer to "what do you predict for a board you have never seen" —
    and is exactly the situation the deployment story in §4 has not settled.
    """

    def __init__(self, group: str = "source"):
        self.group = group

    def fit(self, X: pd.DataFrame, y) -> BoardHazardBaseline:
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.pooled_rate_ = float(target.mean())
        self.rates_ = target.groupby(X[self.group]).mean().to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rates = X[self.group].map(self.rates_).astype("float64").fillna(self.pooled_rate_)
        positive = rates.to_numpy()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class RuleBaseline(ClassifierMixin, BaseEstimator):
    """A rule a person could follow, priced on the metric the model is priced on.

    The ladder above this one is a ladder of *estimators*, and a model that beats
    every rung of it has only been shown to beat other models. The question a
    reader actually has is whether any of it earns its keep against something a
    job seeker could do in their head — "skip anything that has been up a month",
    "watch the boards that turn over". Until that comparison exists, "XGBoost
    beats logistic regression" is a statement about scikit-learn.

    **The rule is calibrated, not just applied.** A person holding a rule holds a
    rate with it — *old postings rarely close* means something like "maybe one in
    a hundred" — so `fit` learns each group's closure rate on the training fold
    and `predict_proba` returns it. That keeps the comparison honest in both
    directions: the rule gets a real probability, so Brier and calibration mean
    something for it, and it gets exactly one parameter per group rather than
    being handicapped by an arbitrary constant.

    A group unseen at fit time falls back to the pooled training rate, which is
    the honest answer to "what do you predict for a bucket you have never seen".
    """

    def __init__(self, rule=None, statement: str = ""):
        self.rule = rule
        self.statement = statement

    def _groups(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.rule(X), index=X.index)

    def fit(self, X: pd.DataFrame, y) -> RuleBaseline:
        target = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.classes_ = np.array([0, 1])
        self.pooled_rate_ = float(target.mean())
        self.rates_ = target.groupby(self._groups(X)).mean().to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rates = self._groups(X).map(self.rates_).astype("float64").fillna(self.pooled_rate_)
        positive = rates.to_numpy()
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class SingleFeatureCeiling(RuleBaseline):
    """The best *any* rule over one column could do, which is not a rule at all.

    A threshold rule picks one cut point and lives with it. This bins the column
    into quantiles on the training fold and predicts each bin's rate, so it is
    free to be non-monotone and to put its cut points wherever the data wants
    them. No human would apply it, and that is the point: it bounds what a human
    rule over that column could possibly buy, so a model beating a hand-written
    threshold by a little can be checked against a much harder target.

    It matters here because `age_only` is a *logistic* fit on `age_days`, which
    can only express "closure rises with age" or "closure falls with age". The
    observed relationship is neither — the closure rate across age buckets runs
    1.13%, 1.59%, 0.86%, 1.49%, 1.77%, 0.82% — so a monotone fit can report
    "no signal" for a column that has a shape it cannot represent. This one can
    see the shape if there is one.

    Bin edges are learned on the training fold and reused unchanged, because
    quantiles computed on the evaluation block would be a statistic of the
    future. Duplicate edges are dropped rather than raising: a column with a
    heavy point mass has fewer distinct bins than requested, which is a fact
    about the column and not an error.
    """

    def __init__(self, column: str = "age_days", bins: int = 10, statement: str = ""):
        self.column = column
        self.bins = bins
        self.statement = statement
        super().__init__(rule=None, statement=statement)

    def fit(self, X: pd.DataFrame, y) -> SingleFeatureCeiling:
        values = pd.to_numeric(X[self.column], errors="coerce")
        _, self.edges_ = pd.qcut(values, self.bins, retbins=True, duplicates="drop")
        # Open the outer edges so a validation row beyond the training range
        # lands in the nearest bin instead of becoming NaN and taking the
        # pooled-rate fallback for a reason that is not about the column.
        self.edges_ = np.asarray(self.edges_, dtype=float)
        self.edges_[0], self.edges_[-1] = -np.inf, np.inf
        return super().fit(X, y)

    def _groups(self, X: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(X[self.column], errors="coerce")
        return pd.Series(pd.cut(values, self.edges_).astype(str), index=X.index, name=self.column)


# --- the rules themselves, each expressible in one sentence -------------------


def older_than_thirty_days(frame: pd.DataFrame) -> pd.Series:
    """ "A posting that has been up more than a month is not about to be removed."

    Stated as a hypothesis worth testing rather than as a fact: on the
    2026-09-07 panel it is false and slightly backwards, 1.30% against 1.14%.
    Kept in the ladder for exactly that reason — a plausible rule that the data
    does not support is a result, and a reader who holds this belief is better
    served by seeing it priced than by not finding it.
    """
    return pd.to_numeric(frame["age_days"], errors="coerce") > 30


def posted_this_week(frame: pd.DataFrame) -> pd.Series:
    """ "A posting still up after a week has stalled; the fresh ones move."

    The opposite prior to the one above, and at most one of them can be right.
    Both are widely held.
    """
    return pd.to_numeric(frame["age_days"], errors="coerce") <= 7
