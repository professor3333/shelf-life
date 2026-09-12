"""Board context derived from a board snapshot, by the panel's own definitions.

Two questions have lived under one endpoint. Someone holding one job advert
asks *will this posting be gone soon?* and cannot know how many postings the
board carried, how many shared its title, or how the board grew since
yesterday — so `/predict` never accepts those four fields, and imputes them.
Someone holding the whole board asks *which of these should I read first?*,
and for them the four are not unknowable: they are properties of the very
batch being sent. `/rank` with `is_board_snapshot=true` derives them here.

**The definitions are the panel's, not a paraphrase of them.**
`src/features/assemble.py:_board_context` computes each within one crawl of
one board; the same arithmetic on a declared snapshot gives the value the
training rows carried:

    board_size_at_t        rows in the snapshot
    n_same_title_on_board  rows in the snapshot with the same `title`, exactly
    n_same_req_on_board    rows sharing the posting's `requisition_id`;
                           absent when the posting has none, as in the panel
    board_growth           board_size_at_t − previous_board_size, when the
                           caller supplies yesterday's size; absent otherwise

`tests/test_board_snapshot.py` holds this equal to `_board_context` on a
one-crawl frame, so the two cannot drift apart without a test saying so.

**What a snapshot must be.** One board — a batch that names two `source`
values is refused, because `board_size_at_t` of a union is nobody's board —
and complete: a partial board declared whole yields a small board size the
model never saw, which is the caller's error and is said in the response. And
it must not also carry the four fields per posting: one source of truth, or
the response could not say where the context came from.

`requisition_id` is accepted in this mode for the count and nothing else. It
is an identifier and is never a feature (`docs/leakage_audit.md`); the panel
used it the same way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from src.inference.contract import BOARD_CONTEXT, InvalidPayload

#: The identifier used for the requisition-group count, and for nothing else.
REQUISITION_ID = "requisition_id"

DERIVED, SUPPLIED_OR_IMPUTED = "derived from the snapshot", "supplied per posting or imputed"


def derive_board_context(
    payloads: Sequence[dict], previous_board_size: int | None = None
) -> list[dict]:
    """The payloads with the four board fields set from the snapshot itself."""
    if not payloads:
        raise InvalidPayload("a board snapshot holds at least one posting")
    for index, payload in enumerate(payloads):
        present = [name for name in BOARD_CONTEXT if payload.get(name) is not None]
        if present:
            raise InvalidPayload(
                f"posting {index} supplies {present}, but in a board snapshot the service "
                "derives board context from the batch; send one or the other"
            )
    sources = {payload.get("source") for payload in payloads if payload.get("source")}
    if len(sources) > 1:
        raise InvalidPayload(f"a board snapshot is one board; this batch names {sorted(sources)}")

    size = len(payloads)
    titles = Counter(payload.get("title") for payload in payloads)
    requisitions = Counter(
        payload.get(REQUISITION_ID) for payload in payloads if payload.get(REQUISITION_ID)
    )
    growth = None if previous_board_size is None else float(size - int(previous_board_size))

    derived = []
    for payload in payloads:
        row = {key: value for key, value in payload.items() if key != REQUISITION_ID}
        row["board_size_at_t"] = float(size)
        row["n_same_title_on_board"] = float(titles[payload.get("title")])
        requisition = payload.get(REQUISITION_ID)
        if requisition:
            row["n_same_req_on_board"] = float(requisitions[requisition])
        if growth is not None:
            row["board_growth"] = growth
        derived.append(row)
    return derived
