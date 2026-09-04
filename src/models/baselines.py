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
