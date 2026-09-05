"""Raw posting in, probability out. The whole serving path, and nothing else.

    {"title": "Senior Data Engineer", ...}  ->  0.041, below threshold, keep

Four steps, and three of them are somebody else's code on purpose:
`contract.validate` checks the payload, `contract.build_row` shapes it with the
training frame's own functions, the artifact's `Pipeline` does every derivation,
imputation and encoding exactly as it did at fit time, and this module compares
the resulting probability to the threshold stored beside it. There is no feature
logic here to drift.

**The threshold travels with the model.** It is a decision about which error is
more expensive — `docs/design.md` §5 — and it was chosen on validation at a
stated alert budget, so shipping it separately from the model that produced it
would let the two disagree silently. A caller may override it per request when
their budget differs from the one it was set for; the response always reports
which threshold was actually applied.

**A probability is not a decision.** The response carries both, and it also
carries `board_context_supplied`, because a prediction made without board
context came from a model whose four board features were imputed to constants —
see `src/inference/contract.py`. A caller who never sees that flag cannot tell
the two regimes apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.inference.artifact import DEFAULT_ARTIFACT, Artifact, load
from src.inference.contract import board_context_supplied, build_row


@dataclass(frozen=True)
class Prediction:
    """One answer, with everything needed to read it."""

    probability: float
    threshold: float
    closing_soon: bool
    horizon_days: int
    board_context_supplied: bool
    model: str
    t: str

    def as_dict(self) -> dict:
        return asdict(self)


class Predictor:
    """A loaded artifact, ready to score.

    A class rather than a function because loading is the expensive part —
    unpickling a booster and its transformer — and the API loads once at startup
    and serves from memory. Construction is the only place that touches disk.
    """

    def __init__(self, artifact: Artifact):
        self.artifact = artifact

    @classmethod
    def load(cls, path: Path = DEFAULT_ARTIFACT) -> Predictor:
        return cls(load(path))

    @property
    def metadata(self):
        return self.artifact.metadata

    def predict(
        self,
        payload: dict,
        t: pd.Timestamp | str | None = None,
        threshold: float | None = None,
    ) -> Prediction:
        """Score one posting.

        `t` is the prediction instant — the moment the caller is standing at,
        which is what `age_days` and `days_since_update` are measured from. It
        defaults to now, and it is an argument rather than always-now so that a
        caller can score a posting as of a past instant and get the same number
        twice. A test that could not pin `t` could not pin a probability.
        """
        moment = pd.Timestamp(t) if t is not None else pd.Timestamp(datetime.now(UTC))
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")

        row = build_row(payload, moment)
        probability = float(self.artifact.pipeline.predict_proba(row)[0, 1])
        applied = self.metadata.threshold if threshold is None else float(threshold)

        return Prediction(
            probability=probability,
            threshold=applied,
            closing_soon=probability >= applied,
            horizon_days=self.metadata.horizon_days,
            board_context_supplied=board_context_supplied(payload),
            model=self.metadata.run_name,
            t=moment.isoformat(),
        )


def predict(payload: dict, path: Path = DEFAULT_ARTIFACT, **kwargs) -> Prediction:
    """One-shot convenience: load, score, discard. For the CLI and for tests."""
    return Predictor.load(path).predict(payload, **kwargs)
