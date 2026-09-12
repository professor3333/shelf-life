"""The request and response models — the leakage audit, made executable.

A field that cannot appear in the request cannot be a feature. That is the whole
design rule for this file, and it runs in the uncomfortable direction: if the
model needs a column a caller cannot send, the *model* is wrong, not the schema.
`observation_count` would raise the score handsomely and it is not here, because
nobody standing in front of a fresh posting knows how many times it will be
seen.

**The vocabulary is not restated here.** `src/inference/contract.py:FIELDS` is
the single list of what a caller may send, what type it is, and — the part that
matters — whether a person holding one job posting could know it. This module
turns that list into pydantic, and `tests/test_api.py` fails if the two ever
disagree. Two hand-maintained copies of a schema is how a field gets quietly
dropped from one of them.

**`as_of` is the exception, and it is not a feature.** It is the prediction
instant: the moment the caller is standing at, which `age_days` and
`days_since_update` are measured from. The panel treats `t` as an axis and never
as an input (`docs/leakage_audit.md`), and so does this. It is exposed because a
prediction that cannot be pinned to an instant cannot be reproduced — including
by the test that pins one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.inference.contract import BOARD_CONTEXT, FIELDS_BY_NAME
from src.inference.predict import MAX_BATCH


def _why(name: str) -> str:
    """The serve-time availability note, straight from the contract.

    It becomes the field's description in `/docs`, so the reason a field exists
    is visible to whoever is filling in the form rather than buried in a module
    nobody reading the API will open.
    """
    return FIELDS_BY_NAME[name].availability


class PostingRequest(BaseModel):
    """One job posting, as somebody looking at it could describe it.

    **The individual-posting mode.** Only what a person holding one advert can
    know — so the four board-context fields (`board_size_at_t`, `board_growth`,
    `n_same_title_on_board`, `n_same_req_on_board`) are not fields here at all.
    Until 2026-09-12 they were accepted on `/predict` and almost always
    imputed, which mixed two questions under one endpoint: a caller who did
    type them in was scoring a posting with numbers the model expects to
    describe a whole board. They belong to the board-ranking mode, where the
    service derives them from the snapshot it is given (`BoardPosting`,
    `RankRequest.is_board_snapshot`). Sent here, they are a 422 that says so.

    `extra="forbid"` makes an unknown field a 422 rather than a shrug. Ignoring
    it is friendlier exactly until someone sends `salary` instead of
    `salary_raw`, gets a confident probability computed without it, and has no
    way to find out.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "title": "Senior Data Engineer",
                "location": "Berlin",
                "salary_raw": "120000 - 160000 USD",
                "departments": "Eng",
                "offices": "HQ",
                "content_chars": 1400,
                "first_published": "2026-08-20T00:00:00Z",
            }
        },
    )

    title: str = Field(min_length=1, description=_why("title"))
    location: str | None = Field(default=None, description=_why("location"))
    salary_raw: str | None = Field(default=None, description=_why("salary_raw"))
    departments: str | None = Field(default=None, description=_why("departments"))
    offices: str | None = Field(default=None, description=_why("offices"))
    n_offices: float | None = Field(default=None, ge=0, description=_why("n_offices"))
    n_metadata: float | None = Field(default=None, ge=0, description=_why("n_metadata"))
    content_chars: float | None = Field(default=None, ge=0, description=_why("content_chars"))
    first_published: datetime | None = Field(default=None, description=_why("first_published"))
    updated_at: datetime | None = Field(default=None, description=_why("updated_at"))
    source: str | None = Field(default=None, description=_why("source"))
    company: str | None = Field(default=None, description=_why("company"))

    as_of: datetime | None = Field(
        default=None,
        description="The prediction instant. Defaults to now. Not a feature — it "
        "is the moment `age_days` is measured from, and it is exposed so that a "
        "prediction can be reproduced exactly.",
    )

    @model_validator(mode="before")
    @classmethod
    def _board_context_belongs_to_the_ranking_mode(cls, data):
        """Say *why* the four fields are refused here, not just that they are.

        `extra="forbid"` would reject them as unknown, which reads as a typo.
        They are not unknown; they are the other mode's. `BoardPosting`
        declares them and skips this.
        """
        if cls is PostingRequest and isinstance(data, dict):
            sent = [name for name in BOARD_CONTEXT if name in data]
            if sent:
                raise ValueError(
                    f"{sent} describe the board, not the posting, and are not accepted on "
                    "/predict. Send the whole board to /rank with is_board_snapshot=true "
                    "and the service derives them; or supply them per posting there."
                )
        return data

    def payload(self) -> dict:
        """The contract's vocabulary, with `as_of` removed.

        `exclude_none` matters: an omitted field and an explicit null must reach
        the pipeline the same way, as a value the training fold's imputer fills.
        """
        return self.model_dump(exclude_none=True, exclude={"as_of"})


class BoardPosting(PostingRequest):
    """One posting inside a board batch — the board-ranking mode's row.

    Everything `PostingRequest` takes, plus what only a caller holding the
    board can say: the four board-context fields, if they computed them
    themselves; or `requisition_id`, which the service uses for the
    requisition-group count when the batch is a declared snapshot and for
    nothing else — it is an identifier, never a feature.
    """

    board_size_at_t: float | None = Field(default=None, ge=0, description=_why("board_size_at_t"))
    board_growth: float | None = Field(default=None, description=_why("board_growth"))
    n_same_title_on_board: float | None = Field(
        default=None, ge=0, description=_why("n_same_title_on_board")
    )
    n_same_req_on_board: float | None = Field(
        default=None, ge=0, description=_why("n_same_req_on_board")
    )
    requisition_id: str | None = Field(
        default=None,
        description="The employer's requisition id, for the requisition-group count in a "
        "declared board snapshot. An identifier; never reaches the model.",
    )


class PredictionResponse(BaseModel):
    """A probability, and everything needed to read it as a decision.

    Most of these fields exist because a bare probability is not an answer.
    `threshold` is the operating point it was compared against, chosen on
    validation at a stated alert budget. `horizon_days` is what "soon" means.
    `board_context_supplied` says whether the four board-level features carried
    information for this request or were imputed to constants
    (`docs/design.md` §12). `model` names the run, so a number can be traced to
    the artifact that produced it. `dataset` says whether that run was fitted on
    the real panel or on the synthetic fixture — a service serving a rehearsal
    must not look like a service serving a model, and while the panel is too
    shallow to freeze against, that distinction is the difference between a
    number and a placeholder.

    **What the probability is not.** It is the chance the posting is *removed
    from the board* within the horizon — not the chance it is filled. A posting
    can be pulled, expire, or move. `docs/problem_definition.md` says so at
    greater length and the README says it in plain words — and `predicts` says
    it in the response, because a caveat that lives only in documentation is
    not read by client code. `removal_flagged` is the threshold decision: the
    probability is at or above `threshold`, so the posting is on the alert list.
    """

    probability: float = Field(ge=0.0, le=1.0)
    threshold: float
    removal_flagged: bool
    horizon_days: int
    predicts: str
    board_context_supplied: bool
    model: str
    dataset: str
    t: str


class RankRequest(BaseModel):
    """Many postings, and how many of them the caller will actually read.

    `budget` is optional. Supplied, it means *flag this many* and the operating
    point becomes the batch's budget-th score. Omitted, the frozen threshold
    applies and however many clear it are flagged. The response always says
    which happened, because the two are different decisions wearing the same
    field name.
    """

    model_config = ConfigDict(extra="forbid")

    postings: list[BoardPosting] = Field(min_length=1, max_length=MAX_BATCH)
    budget: int | None = Field(default=None, ge=1)
    as_of: datetime | None = None
    #: The board-ranking mode proper. `true` declares that `postings` is one
    #: board, whole, as of `as_of`, and the service derives the four
    #: board-context fields from it by the panel's own definitions — a
    #: posting may not then also carry them. `false` (the default) leaves them
    #: to the caller: supplied per posting, or imputed. A partial board
    #: declared whole yields a board size the model never saw; the response
    #: reports the size it was handed.
    is_board_snapshot: bool = False
    #: Yesterday's board size, for `board_growth` in snapshot mode. Without it
    #: growth is imputed, as in the panel's first crawl of a board.
    previous_board_size: int | None = Field(default=None, ge=0)


class RankedPosting(BaseModel):
    """One posting's place in the list. `rank` is 1-based."""

    rank: int
    probability: float = Field(ge=0.0, le=1.0)
    watch: bool
    board_context_supplied: bool


class RankResponse(BaseModel):
    """The ranked batch, and the operating point that produced `watch`.

    `threshold_source` is the field that keeps this honest. `frozen` is the
    model's calibrated operating point, chosen on validation at a stated alert
    budget; `batch_budget` is the batch's own budget-th score, which is a
    property of what was submitted and not of the model. They can differ widely,
    and a response that reported only a number would let a caller read one as
    the other.

    Postings come back **in the order they were sent**, so a caller can zip the
    response against their own list; `rank` carries the ordering.
    """

    postings: list[RankedPosting]
    threshold_applied: float
    threshold_source: Literal["frozen", "batch_budget"]
    budget: int
    horizon_days: int
    model: str
    dataset: str
    t: str
    #: Where the board-context fields came from for this batch, and — in
    #: snapshot mode — the board size the model was handed.
    board_context_source: str
    board_size: int | None = None


class BoardCreateRequest(BaseModel):
    """Open a board: one logical collection, uploaded and scored in pages.

    The whole-board ranking is several requests on the free instance — a day's
    board is about 1,150 postings and a request is capped at 250 — and this is
    the object the requests are about. Its prediction instant and its snapshot
    declaration are fixed here, once, for every page that follows.
    """

    model_config = ConfigDict(extra="forbid")

    as_of: datetime | None = None
    is_board_snapshot: bool = False
    previous_board_size: int | None = Field(default=None, ge=0)


class BoardCreated(BaseModel):
    board_id: str
    #: Boards live in memory on one instance and are forgotten after this
    #: many seconds, or sooner if the service sleeps or restarts.
    ttl_seconds: float
    page_size: int


class BoardPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    postings: list[BoardPosting] = Field(min_length=1, max_length=MAX_BATCH)


class BoardProgress(BaseModel):
    """Where the board is: how many postings it holds, how many are scored."""

    board_id: str
    total: int
    scored: int
    done: bool


class BoardRankRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget: int | None = Field(default=None, ge=1)


class BoardRankResponse(RankResponse):
    """The ranking over the whole board, plus how it got there."""

    board_id: str
    pages_scored: int


class HealthResponse(BaseModel):
    """Is the process up, and does it have a model?

    Two questions rather than one, and the answer is 200 either way. A model-less
    process is still answering, and collapsing "cannot reach the service" into
    "the artifact is missing" costs the one distinction that tells you which
    thing to go and fix. `status` carries the difference, and `/predict` refuses
    with a 503 while `model_loaded` is false.
    """

    status: str
    model_loaded: bool
    artifact: str
    #: The release tag this image was built from, from `MODEL_TAG` at build time.
    #: It answers a question the artifact's own metadata cannot: *which published
    #: version is this process serving?* `run_name` says which experiment was
    #: frozen, not which release reached the URL, and after a deploy those are
    #: exactly the two facts that need distinguishing. `None` when the image was
    #: built without one — a local build, or the deliberate no-artifact image.
    artifact_tag: str | None = None
    model: str | None = None
    dataset: str | None = None
    horizon_days: int | None = None
    threshold: float | None = None
    fitted_on: str | None = None
    created_at: str | None = None
    detail: str | None = None
    #: What this process cost to become ready, measured by the process itself,
    #: for `scripts/cold_start.sh` to print beside its outside timing. The
    #: acceptance criterion is applied to the outside number only; these say
    #: where it went. `load_seconds` is the unpickle — the term the no-artifact
    #: baseline omits (`docs/design.md` §7e); `ready_after_seconds` is process
    #: start to model ready, so the difference from `load_seconds` is the
    #: interpreter and the imports; `rss_mb` is peak resident memory, against
    #: the free instance's 512 MB.
    load_seconds: float | None = None
    ready_after_seconds: float | None = None
    rss_mb: float | None = None
    #: The largest batch `/rank` accepts — `src.inference.predict.MAX_BATCH`,
    #: which is a measurement against the instance rather than a round number.
    #: Reported so a caller ranking a whole board can page by the service's
    #: actual cap instead of a copy of it that would drift (`app/client.py`
    #: `rank_board`). Present whether or not a model is loaded.
    rank_max_batch: int = MAX_BATCH
