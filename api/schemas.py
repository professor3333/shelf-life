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

from pydantic import BaseModel, ConfigDict, Field

from src.inference.contract import FIELDS_BY_NAME


def _why(name: str) -> str:
    """The serve-time availability note, straight from the contract.

    It becomes the field's description in `/docs`, so the reason a field exists
    is visible to whoever is filling in the form rather than buried in a module
    nobody reading the API will open.
    """
    return FIELDS_BY_NAME[name].availability


class PostingRequest(BaseModel):
    """One job posting, as somebody looking at it could describe it.

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
    board_size_at_t: float | None = Field(default=None, ge=0, description=_why("board_size_at_t"))
    board_growth: float | None = Field(default=None, description=_why("board_growth"))
    n_same_title_on_board: float | None = Field(
        default=None, ge=0, description=_why("n_same_title_on_board")
    )
    n_same_req_on_board: float | None = Field(
        default=None, ge=0, description=_why("n_same_req_on_board")
    )

    as_of: datetime | None = Field(
        default=None,
        description="The prediction instant. Defaults to now. Not a feature — it "
        "is the moment `age_days` is measured from, and it is exposed so that a "
        "prediction can be reproduced exactly.",
    )

    def payload(self) -> dict:
        """The contract's vocabulary, with `as_of` removed.

        `exclude_none` matters: an omitted field and an explicit null must reach
        the pipeline the same way, as a value the training fold's imputer fills.
        """
        return self.model_dump(exclude_none=True, exclude={"as_of"})


class PredictionResponse(BaseModel):
    """A probability, and everything needed to read it as a decision.

    Four of these six fields exist because a bare probability is not an answer.
    `threshold` is the operating point it was compared against, chosen on
    validation at a stated alert budget. `horizon_days` is what "closing" means.
    `board_context_supplied` says whether the four board-level features carried
    information for this request or were imputed to constants
    (`docs/design.md` §12). `model` names the run, so a number can be traced to
    the artifact that produced it. `dataset` says whether that run was fitted on
    the real panel or on the synthetic fixture — a service serving a rehearsal
    must not look like a service serving a model, and while the panel is too
    shallow to freeze against, that distinction is the difference between a
    number and a placeholder.

    **What the probability is not.** It is the chance the posting *leaves the
    board* within the horizon — not the chance it is filled. A posting can be
    pulled, expire, or move. `docs/problem_definition.md` says so at greater
    length and the README says it in plain words.
    """

    probability: float = Field(ge=0.0, le=1.0)
    threshold: float
    closing_soon: bool
    horizon_days: int
    board_context_supplied: bool
    model: str
    dataset: str
    t: str


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
    model: str | None = None
    dataset: str | None = None
    horizon_days: int | None = None
    threshold: float | None = None
    fitted_on: str | None = None
    created_at: str | None = None
    detail: str | None = None
