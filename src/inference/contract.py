"""What a caller must send, and how it becomes the row the pipeline expects.

The training frame is a *job-day*: a posting as one crawl saw it, sitting on a
board whose size and composition that crawl also measured. A caller at serve
time has none of that. They have a posting. So this module is where the panel's
44 columns are cut down to the fields a stranger can actually supply, and where
the rest are reconstructed — by calling the same functions the training frame
was built with, never by restating them.

**Three kinds of column, and only the third is a problem.**

*Supplied.* `title`, `location`, `salary_raw`, `departments`, `offices` and the
payload counts are properties of the posting. A caller has them because they are
looking at the posting.

*Row-local.* `age_days`, `days_since_update`, `posted_dow` and the parsed salary
columns are arithmetic on a supplied field and `t`. They are computed here by
`src.features.assemble.row_local_features` — the function that built them for
training. Two implementations of `age_days` is the definition of training/serving
skew, so there is one.

*Board context.* `board_size_at_t`, `board_growth`, `n_same_title_on_board` and
`n_same_req_on_board` describe the board, not the posting, and **a single
posting does not contain them**. They are accepted when a caller has them — a
board owner scoring their own requisitions does — and left missing when it does
not, which routes them through the training fold's imputers like any other
absent value.

That last case deserves its name said out loud: a median-filled `board_size_at_t`
is a constant, so for a caller who cannot supply it the feature is *inert*, and
the model at serve time is not quite the model that was validated. The
prediction says so — `board_context_supplied` rides on every response — and
`docs/design.md` §11 records the decision. The alternative, dropping the four
features from the fitted model, is a live option that costs whatever they are
worth; the ablation in `reports/model_results.md` is what will price it, and it
has not run yet.

Nothing here is fitted. It is a data-shaping step, and every statistic it might
have learned instead lives in the `ColumnTransformer` inside the artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.features.assemble import row_local_features

Kind = Literal["text", "timestamp", "number", "category"]
Origin = Literal["supplied", "board"]


@dataclass(frozen=True)
class Field:
    """One thing a caller may send.

    `availability` is the serve-time half of the leakage audit: not *can the
    model use this* — `docs/leakage_audit.md` settled that — but *can the person
    holding a single posting know it*. A field that passes the first test and
    fails the second is a feature that only exists in training.
    """

    name: str
    kind: Kind
    origin: Origin
    required: bool
    availability: str


#: The request schema, and the only vocabulary a payload may use. `POST /predict`
#: is generated from this tuple, so a field that is not here cannot be sent and a
#: column that is not derivable from it cannot be a feature.
FIELDS: tuple[Field, ...] = (
    Field(
        "title",
        "text",
        "supplied",
        True,
        "on the posting. Required: four derived features read it, and a posting "
        "with no title is not a posting",
    ),
    Field("location", "text", "supplied", False, "on the posting"),
    Field(
        "salary_raw",
        "text",
        "supplied",
        False,
        "on the posting when stated at all; absence is itself the signal that "
        "`salary_stated` carries",
    ),
    Field("departments", "category", "supplied", False, "from the board's own payload"),
    Field("offices", "category", "supplied", False, "from the board's own payload"),
    Field("n_offices", "number", "supplied", False, "count of the offices field"),
    Field("n_metadata", "number", "supplied", False, "count of the payload's metadata entries"),
    Field("content_chars", "number", "supplied", False, "length of the posting's own body text"),
    Field(
        "first_published",
        "timestamp",
        "supplied",
        False,
        "the employer's publication instant, not when this project first saw it",
    ),
    Field("updated_at", "timestamp", "supplied", False, "the employer's last-edit instant"),
    Field(
        "source",
        "category",
        "supplied",
        False,
        "which board. Reaches the model only when the artifact was fitted with "
        "board identity admitted, which `docs/design.md` §4 defaults to off",
    ),
    Field("company", "category", "supplied", False, "as source, and the same information"),
    Field(
        "board_size_at_t",
        "number",
        "board",
        False,
        "how many postings the board carried at `t`. Unknowable from one posting",
    ),
    Field(
        "board_growth",
        "number",
        "board",
        False,
        "change in board size since the previous crawl. Unknowable from one posting",
    ),
    Field(
        "n_same_title_on_board",
        "number",
        "board",
        False,
        "duplicate-title count on the board at `t`. Unknowable from one posting",
    ),
    Field(
        "n_same_req_on_board",
        "number",
        "board",
        False,
        "requisition-group size on the board at `t`. Unknowable from one posting",
    ),
)

FIELDS_BY_NAME: dict[str, Field] = {field.name: field for field in FIELDS}

#: Fields describing the board rather than the posting. Named so that a response
#: can report whether any of them arrived, instead of the caller having to infer
#: it from a probability that quietly moved.
BOARD_CONTEXT: tuple[str, ...] = tuple(f.name for f in FIELDS if f.origin == "board")

#: pandas dtype per field kind. Nullable extension dtypes throughout, because
#: every field but `title` may legitimately be absent and `float64`/`object` have
#: no way to say so that survives `select_columns`.
_DTYPES: dict[Kind, str] = {
    "text": "string",
    "category": "string",
    "number": "Float64",
    "timestamp": "datetime64[ns, UTC]",
}


class InvalidPayload(ValueError):
    """The payload cannot be turned into a row. Always the caller's fault.

    A distinct type because the API has to answer it with a 4xx and everything
    else with a 5xx, and telling those two apart by parsing a message is how a
    bad request ends up reported as a broken server.
    """


def _coerce(field: Field, value: object) -> object:
    if value is None:
        return None
    try:
        if field.kind == "number":
            number = float(value)  # type: ignore[arg-type]
            if not np.isfinite(number):
                raise InvalidPayload(f"{field.name} must be a finite number, got {value!r}")
            return number
        if field.kind == "timestamp":
            moment = pd.Timestamp(value)  # type: ignore[arg-type]
            return moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
        if isinstance(value, bool):
            raise InvalidPayload(f"{field.name} must be text, got {value!r}")
        return str(value)
    except InvalidPayload:
        raise
    except (TypeError, ValueError) as error:
        raise InvalidPayload(f"{field.name} is not a valid {field.kind}: {value!r}") from error


def validate(payload: dict) -> dict:
    """Check the keys and coerce the values. Raises `InvalidPayload`, never 500s.

    Unknown keys are refused rather than ignored. Ignoring them is friendlier
    right up until someone sends `salary` instead of `salary_raw`, gets a
    confident probability computed without it, and has no way to discover that
    the field was dropped on the floor.
    """
    if not isinstance(payload, dict):
        raise InvalidPayload(f"payload must be an object, got {type(payload).__name__}")

    unknown = sorted(set(payload) - set(FIELDS_BY_NAME))
    if unknown:
        raise InvalidPayload(f"unknown field(s) {unknown}; accepted: {sorted(FIELDS_BY_NAME)}")

    missing = [f.name for f in FIELDS if f.required and payload.get(f.name) in (None, "")]
    if missing:
        raise InvalidPayload(f"missing required field(s) {missing}")

    return {name: _coerce(FIELDS_BY_NAME[name], value) for name, value in payload.items()}


def build_row(payload: dict, t: pd.Timestamp) -> pd.DataFrame:
    """One validated payload plus a prediction instant -> a one-row frame.

    The frame carries every column the fitted pipeline reads and nothing it does
    not. Fields the caller omitted are present and null, which is what sends them
    to the imputers the training fold fitted; a *missing column* would instead
    raise inside `select_columns`, and the difference between "absent value" and
    "absent column" is the difference between a prediction and a stack trace.
    """
    clean = validate(payload)

    row: dict[str, object] = {field.name: clean.get(field.name) for field in FIELDS}
    row["t"] = pd.Timestamp(t)
    if row["t"].tzinfo is None:
        row["t"] = row["t"].tz_localize("UTC")

    frame = pd.DataFrame(index=[0])
    for field in FIELDS:
        # Typed explicitly, one column at a time. A `DataFrame([row])` built
        # from a dict gives an omitted numeric field `object` dtype holding a
        # single `None`, and `select_columns` routes on dtype: the column would
        # arrive at a median imputer as text. Every serve-time payload omits
        # *something*, so this is the ordinary path, not an edge case.
        frame[field.name] = pd.Series([row[field.name]], index=[0], dtype=_DTYPES[field.kind])
    frame["t"] = pd.Series([row["t"]], index=[0], dtype="datetime64[ns, UTC]")

    return row_local_features(frame)


def board_context_supplied(payload: dict) -> bool:
    """Did the caller send any board-level field?

    Any, not all: one supplied field already makes this prediction unlike the
    all-missing case, and a caller who sent three of four should not be told
    their context was ignored.
    """
    return any(payload.get(name) is not None for name in BOARD_CONTEXT)


def describe() -> pd.DataFrame:
    """The contract as a table, for the README and the API's docs page."""
    return pd.DataFrame(
        [
            {
                "field": f.name,
                "type": f.kind,
                "required": f.required,
                "origin": f.origin,
                "serve-time availability": f.availability,
            }
            for f in FIELDS
        ]
    )
