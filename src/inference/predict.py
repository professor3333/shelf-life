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
carries `dataset` — real panel or synthetic fixture, because a rehearsal must
never be mistaken for a result — and `board_context_supplied`, because a
prediction made without board context came from a model whose four board
features were imputed to constants —
see `src/inference/contract.py`. A caller who never sees that flag cannot tell
the two regimes apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.artifact import DEFAULT_ARTIFACT, Artifact, load
from src.inference.board_snapshot import DERIVED, SUPPLIED_OR_IMPUTED, derive_board_context
from src.inference.contract import board_context_supplied, build_row

#: An opaque caller-supplied handle that rides along and comes back on the
#: matching result. Never a feature: it is stripped before a row is built,
#: and the contract's `assert_known_columns` would refuse it if it were not.
#: It exists because "results come back in input order" is a contract that
#: survives a single synchronous call and little else — a CSV row, a retry,
#: a deduplication, a page of a board all want a handle, not a position.
CLIENT_ID = "client_id"


def _split_client_id(payload: dict) -> tuple[dict, str | None]:
    """The payload without its handle, and the handle."""
    handle = payload.get(CLIENT_ID)
    return {k: v for k, v in payload.items() if k != CLIENT_ID}, (
        None if handle is None else str(handle)
    )


def predicts(horizon_days: int) -> str:
    """The sentence every prediction carries about what it predicts."""
    unit = "day" if horizon_days == 1 else "days"
    return f"removal from the board within {horizon_days} {unit} — not filled"


@dataclass(frozen=True)
class Prediction:
    """One answer, with everything needed to read it."""

    probability: float
    threshold: float
    removal_flagged: bool
    horizon_days: int
    #: What the number is a probability *of*, in the response itself. The label
    #: is removal from the board; "filled" is the overclaim every caller is one
    #: paraphrase away from, and a caveat that lives only in documentation is not
    #: read by client code.
    predicts: str
    board_context_supplied: bool
    model: str
    dataset: str
    t: str
    #: The caller's handle, echoed. `None` when none was sent.
    client_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ranked:
    """One posting's place in a ranked batch.

    `rank` is 1-based because the caller reads it as a position in a list, not
    as an array index, and an off-by-one in a watch list is a posting someone
    does not open.
    """

    rank: int
    probability: float
    watch: bool
    board_context_supplied: bool
    #: The caller's handle, echoed on the matching result — the join key a
    #: client should use instead of array position.
    client_id: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


#: The largest batch `/rank` accepts. Derived from measurement rather than
#: chosen, because a cap picked for looking round is a cap that is wrong on the
#: hardware it runs on.
#:
#: Measured on the XGBoost artifact at this size: **1.42 s** on a full core,
#: peak RSS 211 MB with no growth across the batch. The free instance has
#: 0.1 vCPU, and the cold-start work in `docs/design.md` §7d ran **16x** slower
#: there than locally. Applying that factor puts 250 at roughly 23 s, and a
#: request may already have spent the cold start waking the instance — which
#: the three-cycle baseline of 2026-09-12 puts at **52 s worst case**, not the
#: 32.65 s single sample this arithmetic was first done with
#: (`reports/cold_start_baseline.md`). So: **~75 s worst case against the 90 s
#: stop rule** §7e sets, with 211 MB against 512 MB. Doubling the cap would put
#: that at ~98 s, over the rule. The cap is tighter than it was, not looser.
#:
#: A full day's board is about 1,150 postings, so this deliberately does not
#: rank a whole day in one request — the caller pages, and `/health` reports
#: this number as `rank_max_batch` so a caller pages by the service's cap and
#: not a copy. `app/client.py` `rank_board` is the reference paging client: it
#: merges the pages by the rule `rank` applies to a batch and applies the
#: budget once, and a test holds its result equal to this method's on a board
#: larger than a page. Raising the cap is a measurement against the deployed
#: instance, not an edit to this line.
MAX_BATCH = 250


@dataclass(frozen=True)
class RankedBatch:
    """A scored, ordered batch and the operating point that was applied."""

    postings: list[Ranked]
    threshold_applied: float
    threshold_source: str
    budget: int
    horizon_days: int
    model: str
    dataset: str
    t: str
    #: Where the four board-context fields came from for this batch: derived
    #: from a declared snapshot, or (per posting) supplied by the caller or
    #: imputed. Said at the batch level because in snapshot mode it is one fact
    #: about the batch, not a property of each posting.
    board_context_source: str = "supplied per posting or imputed"
    #: The board size the snapshot implied, so a caller who declared a partial
    #: board as whole can see the number the model was handed.
    board_size: int | None = None


#: Where `threshold_applied` came from. Named rather than inferred, because the
#: two can differ and silently substituting one for the other would change what
#: `watch` means without changing the response's shape.
FROZEN, BATCH_BUDGET = "frozen", "batch_budget"


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

        payload, client_id = _split_client_id(payload)
        row = build_row(payload, moment)
        probability = float(self.artifact.pipeline.predict_proba(row)[0, 1])
        applied = self.metadata.threshold if threshold is None else float(threshold)

        return Prediction(
            probability=probability,
            threshold=applied,
            removal_flagged=probability >= applied,
            horizon_days=self.metadata.horizon_days,
            predicts=predicts(self.metadata.horizon_days),
            board_context_supplied=board_context_supplied(payload),
            model=self.metadata.run_name,
            dataset=self.metadata.dataset,
            t=moment.isoformat(),
            client_id=client_id,
        )

    def rank(
        self,
        payloads: Sequence[dict],
        t: pd.Timestamp | str | None = None,
        budget: int | None = None,
        board_snapshot: bool = False,
        previous_board_size: int | None = None,
    ) -> RankedBatch:
        """Score many postings and order them, applying the alert budget.

        **This is the shape the operating point was designed for.** The frozen
        threshold is `threshold_for_budget` — literally the score at which
        exactly `budget` postings are flagged — so it is a rank statistic taken
        over a whole day's board. `predict` applies that number to one posting in
        isolation, which is coherent but is the degenerate case; this is the
        case it came from.

        **Board context is imputed unless the batch is declared a snapshot.** A
        batch is not automatically the board: deriving `board_size_at_t` from
        fifty postings would manufacture a board of fifty, and the model was
        fitted on boards of several hundred. So by default the four fields are
        imputed exactly as in `predict`, or used if a caller supplied them. With
        `board_snapshot=True` the caller declares the batch *is* one board,
        whole, and the four are derived from it by the panel's own definitions
        (`src/inference/board_snapshot.py`) — the ranking mode `docs/design.md`
        §12 names as the honest home for board context.

        **Scoring is one `predict_proba` call over one frame**, built from the
        same `build_row` and run through the same fitted pipeline object as
        `predict`. Not a loop over `predict`: measured at 1,000 postings, the
        loop costs 18.4 s against 4.8 s batched, because the per-call pipeline
        overhead dominates a single row. The two agree to about 6e-17 — batching
        changes the order of floating-point reductions and nothing else — and
        `test_a_posting_scores_the_same_through_rank_as_through_predict` pins
        that at a tolerance far tighter than any real divergence would be.
        """
        if not payloads:
            raise ValueError("no postings to rank")
        if len(payloads) > MAX_BATCH:
            raise ValueError(
                f"{len(payloads)} postings exceeds the {MAX_BATCH}-posting limit; "
                "send them in pages, or as a board (POST /boards)"
            )

        moment = self.moment(t)
        if board_snapshot:
            payloads = derive_board_context(payloads, previous_board_size)
            context_source = DERIVED
        else:
            context_source = SUPPLIED_OR_IMPUTED

        probabilities = self.score(payloads, moment)
        return self.order(
            payloads,
            probabilities,
            moment,
            budget,
            context_source,
            len(payloads) if board_snapshot else None,
        )

    @staticmethod
    def moment(t: pd.Timestamp | str | None) -> pd.Timestamp:
        """The prediction instant, UTC, now by default."""
        moment = pd.Timestamp(t) if t is not None else pd.Timestamp(datetime.now(UTC))
        return moment.tz_localize("UTC") if moment.tzinfo is None else moment

    def score(self, payloads: Sequence[dict], moment: pd.Timestamp) -> np.ndarray:
        """Probabilities for a page of payloads — one frame, one `predict_proba`.

        The cap is the page's, not a board's: a board larger than it is scored
        page by page by `src/inference/boards.py`, which calls this.
        """
        if len(payloads) > MAX_BATCH:
            raise ValueError(f"{len(payloads)} postings exceeds the {MAX_BATCH}-posting page")
        rows = [build_row(_split_client_id(payload)[0], moment) for payload in payloads]
        frame = pd.concat(rows, ignore_index=True)
        return self.artifact.pipeline.predict_proba(frame)[:, 1]

    def order(
        self,
        payloads: Sequence[dict],
        probabilities: np.ndarray,
        moment: pd.Timestamp,
        budget: int | None,
        context_source: str,
        board_size: int | None,
    ) -> RankedBatch:
        """The ranking rule over already-scored postings, however many.

        Separate from `score` so a board scored in pages is ordered once, here,
        by the same rule a single batch is — descending probability, ties by
        input order, the budget applied to the whole.
        """
        # The budget decides how many are flagged; the *threshold* is then
        # whichever operating point that implies, and the response says which.
        if budget is None:
            applied, source = float(self.metadata.threshold), FROZEN
            effective = int((probabilities >= applied).sum())
        else:
            effective = int(min(max(budget, 0), len(payloads)))
            applied = (
                float(np.sort(probabilities)[::-1][effective - 1])
                if effective > 0
                else float("inf")
            )
            source = BATCH_BUDGET

        order = np.argsort(-probabilities, kind="stable")
        ranked: list[Ranked | None] = [None] * len(payloads)
        for position, index in enumerate(order, start=1):
            ranked[index] = Ranked(
                rank=position,
                probability=float(probabilities[index]),
                watch=position <= effective,
                board_context_supplied=board_context_supplied(payloads[index]),
                client_id=_split_client_id(payloads[index])[1],
            )

        return RankedBatch(
            postings=[item for item in ranked if item is not None],
            threshold_applied=applied,
            threshold_source=source,
            budget=effective,
            horizon_days=self.metadata.horizon_days,
            model=self.metadata.run_name,
            dataset=self.metadata.dataset,
            t=moment.isoformat(),
            board_context_source=context_source,
            board_size=board_size,
        )


def predict(payload: dict, path: Path = DEFAULT_ARTIFACT, **kwargs) -> Prediction:
    """One-shot convenience: load, score, discard. For the CLI and for tests."""
    return Predictor.load(path).predict(payload, **kwargs)
