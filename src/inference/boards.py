"""A board as one logical collection: uploaded in pages, scored in pages, ranked once.

The product is *rank today's board and say which twenty to read*. A board is
about 1,150 postings; a request is capped at 250 (`predict.MAX_BATCH`, a
measurement against a tenth of a CPU, where 1,150 in one call would take
about 106 s against a 90 s rule). So the whole-board operation is
necessarily several requests — and until 2026-09-12 the logic *between*
them lived in the client: page, merge, apply the budget, and in snapshot
mode derive the board context over the whole board before any page. That
was a second copy of the ranking rule and of the derivation, which `/rank`
existed to make unnecessary.

Here the server owns the collection. A `Board` is created with its
prediction instant and its snapshot declaration; postings are appended a
page at a time; `score_next` scores the next unscored page, deriving board
context over the *whole* stored board first, once; `rank` refuses until every
page is scored and then orders the whole board by the same rule `/rank`
applies to a batch. The client pages uploads and loops `score` — it never
sees a probability until the ranking.

**What the free tier makes true, said here rather than found out.** One
instance, boards held in memory, gone when the service sleeps, restarts or
redeploys, and after `TTL_SECONDS` regardless. A caller who is told
`board expired` starts over; nothing partial is ever ranked. Caps keep a
single caller from holding the instance's 512 MB: `MAX_POSTINGS` per board,
`MAX_BOARDS` live at once (oldest evicted).
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.inference.board_snapshot import DERIVED, SUPPLIED_OR_IMPUTED, derive_board_context
from src.inference.contract import InvalidPayload
from src.inference.predict import MAX_BATCH, Predictor, RankedBatch

#: A board older than this is forgotten. An hour covers a slow client paging a
#: large board through a sleeping service several times over.
TTL_SECONDS = 3600.0
#: Postings per board. Four times a day's board, so a caller cannot fill memory.
MAX_POSTINGS = 5000
#: Live boards. Beyond this the oldest is evicted, which on one free instance
#: is a fair description of what memory is for.
MAX_BOARDS = 32


class BoardError(InvalidPayload):
    """The caller asked for something the board's state does not allow."""


class BoardExpired(KeyError):
    """No board by that id — expired, evicted, or the service restarted."""


@dataclass
class Board:
    board_id: str
    moment: pd.Timestamp
    is_snapshot: bool
    previous_board_size: int | None
    created_at: float = field(default_factory=time.monotonic)
    postings: list[dict] = field(default_factory=list)
    #: Filled on the first `score_next`, over the whole board, when a snapshot.
    scored_payloads: list[dict] | None = None
    scores: list[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def total(self) -> int:
        return len(self.postings)

    @property
    def scored(self) -> int:
        return len(self.scores)

    @property
    def done(self) -> bool:
        return self.total > 0 and self.scored == self.total

    @property
    def sealed(self) -> bool:
        """Once scoring starts, the board's membership is fixed — in snapshot
        mode the derived context is a function of the whole board."""
        return self.scored_payloads is not None

    def add(self, page: Sequence[dict]) -> int:
        with self.lock:
            if self.sealed:
                raise BoardError("scoring has begun; postings can no longer be added")
            if not page:
                raise BoardError("an empty page adds nothing")
            if len(page) > MAX_BATCH:
                raise BoardError(f"a page holds at most {MAX_BATCH} postings")
            if self.total + len(page) > MAX_POSTINGS:
                raise BoardError(f"a board holds at most {MAX_POSTINGS} postings")
            self.postings.extend(dict(p) for p in page)
            return self.total

    def score_next(self, predictor: Predictor) -> tuple[int, int]:
        """Score the next unscored page. Returns (scored, total)."""
        with self.lock:
            if self.total == 0:
                raise BoardError("the board has no postings")
            if self.scored_payloads is None:
                self.scored_payloads = (
                    derive_board_context(self.postings, self.previous_board_size)
                    if self.is_snapshot
                    else list(self.postings)
                )
            if self.done:
                return self.scored, self.total
            page = self.scored_payloads[self.scored : self.scored + MAX_BATCH]
            self.scores.extend(float(x) for x in predictor.score(page, self.moment))
            return self.scored, self.total

    def rank(self, predictor: Predictor, budget: int | None) -> RankedBatch:
        with self.lock:
            if not self.done:
                raise BoardError(
                    f"{self.scored} of {self.total} postings are scored; call score until done"
                )
            assert self.scored_payloads is not None
            batch = predictor.order(
                self.scored_payloads,
                np.asarray(self.scores, dtype=float),
                self.moment,
                budget,
                DERIVED if self.is_snapshot else SUPPLIED_OR_IMPUTED,
                self.total if self.is_snapshot else None,
            )
            return batch


class BoardStore:
    """The live boards, in memory, with a TTL and a cap."""

    def __init__(self, ttl_seconds: float = TTL_SECONDS, max_boards: int = MAX_BOARDS):
        self._boards: dict[str, Board] = {}
        self._ttl = ttl_seconds
        self._max = max_boards
        self._lock = threading.Lock()

    def create(
        self,
        moment: pd.Timestamp,
        is_snapshot: bool = False,
        previous_board_size: int | None = None,
    ) -> Board:
        with self._lock:
            self._sweep()
            while len(self._boards) >= self._max:
                oldest = min(self._boards.values(), key=lambda b: b.created_at)
                del self._boards[oldest.board_id]
            board = Board(secrets.token_urlsafe(12), moment, is_snapshot, previous_board_size)
            self._boards[board.board_id] = board
            return board

    def get(self, board_id: str) -> Board:
        with self._lock:
            self._sweep()
            try:
                return self._boards[board_id]
            except KeyError:
                raise BoardExpired(
                    f"no board {board_id!r}: it expired, was evicted, or the service "
                    "restarted since it was created. Boards live in memory on one instance; "
                    "create it again and re-upload"
                ) from None

    def delete(self, board_id: str) -> None:
        with self._lock:
            self._boards.pop(board_id, None)

    def __len__(self) -> int:
        with self._lock:
            self._sweep()
            return len(self._boards)

    def _sweep(self) -> None:
        now = time.monotonic()
        for key in [k for k, b in self._boards.items() if now - b.created_at > self._ttl]:
            del self._boards[key]
