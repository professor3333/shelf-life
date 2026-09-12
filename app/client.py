"""Everything the UI does that is not drawing: HTTP, payloads, and wording.

Separated from `streamlit_app.py` for one reason — **this file can be tested and
that one cannot.** A Streamlit script runs top to bottom inside its own runtime;
there is no function to call and no return value to assert. So the logic that
could be wrong lives here, in ordinary functions, and the script is left holding
only widget calls.

**The UI never imports the model.** Not `src.inference`, not the artifact, not
the pipeline. It knows a URL. That separation is the whole point of the
component — UI ≠ API ≠ model ≠ training pipeline — and it is enforced by a test
that parses this package for forbidden imports rather than trusted to good
intentions. The temptation is real and it always looks like a shortcut: calling
`Predictor.load()` here would work on a laptop and would mean the deployed UI
and the deployed API were scoring with two different artifacts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests

#: Where the API lives. An environment variable because the UI and the API are
#: deployed separately and the UI has no way to guess the other one's hostname.
API_URL_ENV = "SHELF_LIFE_API"
DEFAULT_API_URL = "http://localhost:8000"

#: Seconds. Generous, because a free tier that has slept the container needs to
#: start Python, import XGBoost and unpickle a booster before it can answer —
#: and a UI that gives up at five seconds reports that as an outage.
#:
#: **Raised from 30 to 90 on 2026-09-06**, when the API moved to a host that
#: spins down after 15 idle minutes and documents "about one minute" to come
#: back, on 0.1 of a CPU (`docs/design.md` §7b). Against that, 30 seconds is a
#: timeout that fires on the *normal* case and reports a working service as a
#: dead one. The number is not a guess about how slow the wake is; it is a
#: deliberate over-estimate until `scripts/cold_start.sh` measures the real one.
TIMEOUT = 90.0


class ApiError(RuntimeError):
    """The API said no, or could not be reached.

    Carries the API's own message where there is one. A UI that replaces
    "unknown field 'salary'" with "something went wrong" has thrown away the
    only part of the response the person can act on.
    """


def api_url_from(explicit: str = "", secret: str | None = None) -> str:
    """Where the API lives, in precedence order, as an ordinary testable function.

    `explicit` (a session override) beats `secret` beats the environment beats
    localhost. The middle rung exists because of how the deployed UI is actually
    configured: Streamlit Community Cloud takes the URL as a *secret*, and the
    fact that root-level secrets are also exported to `os.environ` is a
    documented convenience rather than a guarantee — it says nothing about
    secrets nested under a section, and it is not the interface Streamlit tells
    you to read. Depending on it made the deployed UI fall back to localhost and
    say so on screen, which was the correct behaviour of a wrong assumption.

    Reading `st.secrets` explicitly is the supported path; the environment stays
    as a rung so `SHELF_LIFE_API=... streamlit run` keeps working locally and so
    nothing here has to import Streamlit.
    """
    return explicit or secret or os.environ.get(API_URL_ENV) or DEFAULT_API_URL


@dataclass(frozen=True)
class Api:
    """A thin client. No retries, no caching, no cleverness."""

    base_url: str = ""

    def __post_init__(self) -> None:
        url = self.base_url or os.environ.get(API_URL_ENV, DEFAULT_API_URL)
        object.__setattr__(self, "base_url", url.rstrip("/"))

    def _get(self, path: str) -> object:
        try:
            response = requests.get(f"{self.base_url}{path}", timeout=TIMEOUT)
        except requests.RequestException as error:
            raise ApiError(f"cannot reach the API at {self.base_url}: {error}") from error
        return _unwrap(response)

    def health(self) -> dict:
        return self._get("/health")  # type: ignore[return-value]

    def contract(self) -> list[dict]:
        return self._get("/contract")  # type: ignore[return-value]

    def predict(self, payload: dict) -> dict:
        return self._post("/predict", payload)

    def rank(
        self,
        postings: list[dict],
        budget: int | None = None,
        as_of: str | None = None,
        is_board_snapshot: bool = False,
        previous_board_size: int | None = None,
    ) -> dict:
        """One `/rank` call: at most the service's page of postings."""
        body: dict = {"postings": postings}
        if budget is not None:
            body["budget"] = budget
        if as_of is not None:
            body["as_of"] = as_of
        if is_board_snapshot:
            body["is_board_snapshot"] = True
        if previous_board_size is not None:
            body["previous_board_size"] = previous_board_size
        return self._post("/rank", body)

    def _post(self, path: str, body: dict) -> dict:
        try:
            response = requests.post(f"{self.base_url}{path}", json=body, timeout=TIMEOUT)
        except requests.RequestException as error:
            raise ApiError(f"cannot reach the API at {self.base_url}: {error}") from error
        return _unwrap(response)  # type: ignore[return-value]


#: The four fields the service derives from a declared board snapshot, and
#: that a person holding one advert cannot know. Named here because the
#: client has to know which fields a snapshot may not also carry.
BOARD_CONTEXT_FIELDS = (
    "board_size_at_t",
    "board_growth",
    "n_same_title_on_board",
    "n_same_req_on_board",
)

#: Accepted in a snapshot for the requisition-group count and then dropped —
#: an identifier, never a feature. Mirrors `src/inference/board_snapshot.py`.
REQUISITION_ID = "requisition_id"

DERIVED_ACROSS_PAGES = "derived from the snapshot (by the client, across pages)"


def board_context_for(postings: list[dict], previous_board_size: int | None = None) -> list[dict]:
    """The four board fields for every posting, computed over the whole board.

    The service derives these itself for a batch declared a snapshot — but a
    batch is at most one page, and a board is often larger. `board_size_at_t`
    of a page is not the board's size, so for a paged board the client has to
    do the arithmetic over the whole board and send the result per posting.
    The arithmetic is the panel's (`src/features/assemble.py:_board_context`)
    and the service's (`src/inference/board_snapshot.py`), and a test holds
    this copy equal to the service's: rows in the board; rows sharing the
    exact title; rows sharing `requisition_id`, absent when the posting has
    none; growth against yesterday's size when given.
    """
    if not postings:
        raise ValueError("a board snapshot holds at least one posting")
    for index, posting in enumerate(postings):
        present = [name for name in BOARD_CONTEXT_FIELDS if posting.get(name) is not None]
        if present:
            raise ValueError(
                f"posting {index} supplies {present}; a snapshot derives board context "
                "from the board itself — send one or the other"
            )
    sources = {p.get("source") for p in postings if p.get("source")}
    if len(sources) > 1:
        raise ValueError(f"a board snapshot is one board; this batch names {sorted(sources)}")

    from collections import Counter

    size = len(postings)
    titles = Counter(p.get("title") for p in postings)
    requisitions = Counter(p.get(REQUISITION_ID) for p in postings if p.get(REQUISITION_ID))
    growth = None if previous_board_size is None else float(size - int(previous_board_size))
    out = []
    for posting in postings:
        row = {k: v for k, v in posting.items() if k != REQUISITION_ID}
        row["board_size_at_t"] = float(size)
        row["n_same_title_on_board"] = float(titles[posting.get("title")])
        if posting.get(REQUISITION_ID):
            row["n_same_req_on_board"] = float(requisitions[posting[REQUISITION_ID]])
        if growth is not None:
            row["board_growth"] = growth
        out.append(row)
    return out


#: What `/rank` calls the operating point when a batch's own budget set it. The
#: merged, whole-board ranking below reports its own name for the same idea, so
#: a reader can tell "the service ranked this batch" from "the client merged
#: several batches" — they agree by construction, but they are not the same call.
BOARD_BUDGET = "board_budget"


def rank_board(
    api: Api,
    postings: list[dict],
    budget: int,
    as_of: str | None = None,
    page_size: int | None = None,
    is_board_snapshot: bool = False,
    previous_board_size: int | None = None,
) -> dict:
    """Rank a whole board under one budget, in pages the service will accept.

    `/rank` caps a request at `rank_max_batch` postings (250 on the free
    instance — `src.inference.predict.MAX_BATCH`, a measurement, not a round
    number), and a day's board is about 1,150. So the board is sent in pages.
    Paging is only honest if the merged answer is the answer one big call would
    have given, and with a budget it is not automatically so: the budget-th
    score of each *page* is not the board's budget-th score. What makes the
    merge exact is that `/rank` returns every posting's probability, so the
    pages can be ranked here by the same rule the service uses on a batch —
    descending probability, ties by input order — and the budget applied once,
    to the union. `tests/test_app.py` holds this equal to the service's own
    unpaged ranking on a board larger than a page.

    Each page is sent *without* a budget, so the service applies its frozen
    threshold to it; that threshold is returned as `frozen_threshold` beside
    the board's budget-th score, which is `threshold_applied` here, because the
    two are different operating points and a caller should see both.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if not postings:
        raise ValueError("no postings to rank")
    # Snapshot mode, over the whole board rather than per page — see
    # `board_context_for`. The pages then carry the context per posting and
    # the service is not told they are snapshots, because they are not.
    board_size = len(postings) if is_board_snapshot else None
    if is_board_snapshot:
        postings = board_context_for(postings, previous_board_size)
    size = page_size or int(api.health().get("rank_max_batch") or 1)
    if size < 1:
        raise ValueError("page size must be at least 1")

    scored: list[tuple[int, float, dict]] = []
    frozen_threshold: float | None = None
    meta: dict = {}
    pages = 0
    for start in range(0, len(postings), size):
        page = postings[start : start + size]
        response = api.rank(page, as_of=as_of)
        pages += 1
        if frozen_threshold is None:
            frozen_threshold = float(response["threshold_applied"])
            meta = {
                key: response[key]
                for key in ("horizon_days", "model", "dataset", "t", "board_context_source")
            }
        for offset, item in enumerate(response["postings"]):
            scored.append((start + offset, float(item["probability"]), item))

    # The service's rule, applied once to the whole board: descending
    # probability, ties broken by the order the postings were sent in.
    service_source = meta.pop("board_context_source", "supplied per posting or imputed")
    context_source = DERIVED_ACROSS_PAGES if is_board_snapshot else service_source
    order = sorted(scored, key=lambda row: (-row[1], row[0]))
    effective = min(budget, len(order))
    threshold_applied = order[effective - 1][1]
    ranked: list[dict | None] = [None] * len(postings)
    for position, (index, probability, item) in enumerate(order, start=1):
        ranked[index] = {
            "rank": position,
            "probability": probability,
            "watch": position <= effective,
            "board_context_supplied": bool(item["board_context_supplied"]),
        }
    return {
        "postings": ranked,
        "threshold_applied": threshold_applied,
        "threshold_source": BOARD_BUDGET,
        "frozen_threshold": frozen_threshold,
        "budget": effective,
        "pages": pages,
        "page_size": size,
        "board_context_source": context_source,
        "board_size": board_size,
        **meta,
    }


def _unwrap(response: requests.Response) -> object:
    """Return the body, or raise with whatever the API said was wrong.

    Status code rather than `response.ok`, because that attribute is `requests`-
    specific and the test harness swaps in an ASGI transport whose responses are
    httpx's. A client that only works against one HTTP library is a client that
    cannot be tested without a socket.
    """
    if response.status_code < 400:
        return response.json()
    raise ApiError(_detail(response))


def _detail(response: requests.Response) -> str:
    """Pull a human sentence out of an error body.

    FastAPI answers a schema failure with a list of per-field errors and a
    handler failure with a string, so both shapes have to be read here or the UI
    shows a JSON dump to somebody who did not ask for one.
    """
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:200]}"

    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):
        parts = []
        for item in detail:
            location = ".".join(str(piece) for piece in item.get("loc", []) if piece != "body")
            parts.append(f"{location}: {item.get('msg', '')}".strip(": "))
        return "; ".join(parts) or f"HTTP {response.status_code}"
    return f"{detail}" if detail else f"HTTP {response.status_code}"


def build_payload(form: dict) -> dict:
    """Form state -> request body, dropping everything the person left alone.

    Blank fields are *omitted*, never sent as empty strings. The API treats an
    absent field as "impute this from the training fold" and an empty string as
    a category called `""` — a real level, one the model has never seen. An
    untouched text box would otherwise quietly become a feature value.
    """
    payload: dict = {}
    for name, value in form.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        payload[name] = value.strip() if isinstance(value, str) else value
    return payload


def verdict(prediction: dict) -> tuple[str, str]:
    """The headline and the sentence under it.

    The caveat is not optional and it is not a footnote: the label is *removed
    from the board*, which is not *filled*. A UI that says "82% likely to be
    filled" has overclaimed in a way the data cannot support, and it is the
    easiest overclaim in this project to make by accident.
    """
    horizon = prediction["horizon_days"]
    days = "day" if horizon == 1 else "days"
    if prediction["removal_flagged"]:
        headline = f"LIKELY TO BE REMOVED from the board within {horizon} {days}"
        explanation = (
            f"The probability is at or above the threshold of "
            f"{prediction['threshold']:.3f}, so this posting is on the alert list."
        )
    else:
        headline = f"NOT flagged for removal within {horizon} {days}"
        explanation = (
            f"The probability is below the threshold of "
            f"{prediction['threshold']:.3f}, so this posting is not on the alert list."
        )
    return headline, explanation


def parse_board(text: str, allowed: set[str]) -> list[dict]:
    """A pasted or uploaded board -> the list `/rank` takes.

    Two shapes, because the two people who would use this hold the board in
    different forms: JSON (a list of postings, or `{"postings": [...]}`) from
    anything that already speaks the API, and CSV with one column per field
    for everything else. Columns the contract does not list are dropped rather
    than sent — the API refuses unknown fields on purpose, and a board export
    always carries columns that are nobody's business at prediction time.
    Blank cells are omitted the way blank form fields are (`build_payload`).
    """
    import io
    import json

    stripped = text.strip()
    if not stripped:
        return []
    rows: list[dict]
    if stripped[0] in "[{":
        loaded = json.loads(stripped)
        rows = loaded["postings"] if isinstance(loaded, dict) else loaded
        if not isinstance(rows, list):
            raise ValueError('JSON must be a list of postings or {"postings": [...]}')
    else:
        import pandas as pd

        frame = pd.read_csv(io.StringIO(stripped))
        rows = frame.astype(object).where(frame.notna(), None).to_dict("records")
    postings = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every posting must be an object")
        kept = build_payload({k: v for k, v in row.items() if k in allowed})
        if "title" not in kept:
            raise ValueError("every posting needs a title")
        postings.append(kept)
    return postings


#: Five postings a stranger can rank without a board of their own: the same
#: batch the deployed smoke test sends, so what the UI shows is what CI checked.
EXAMPLE_BOARD_CSV = """title,location,salary_raw,content_chars,first_published
Senior Data Engineer,Berlin,120000 - 160000 USD,1400,2026-08-20
Werkstudent Marketing (m/w/d),München,,600,2026-09-01
"Staff Software Engineer, Infrastructure",Remote,200000 - 260000 USD,2800,2026-08-10
Customer Support Specialist,Dublin,,900,2026-09-05
Head of Product,London,,2100,2026-07-28
"""


#: Shown under every result. Wording taken from the problem definition rather
#: than paraphrased, because paraphrasing a caveat is how it gets softened.
CAVEAT = (
    "**Removed from the board is not the same as filled.** A posting can be pulled, "
    "expire, or be reposted elsewhere. The model predicts removal — disappearance from "
    "the board it was seen on — and no claim beyond that is supported by the label."
)


def warnings_for(health: dict, prediction: dict | None = None) -> list[str]:
    """Things the person must be told before they read a number.

    Both of these are conditions where the prediction is *arithmetically fine*
    and *interpretively worthless*, which is the dangerous combination: nothing
    looks wrong on screen.
    """
    notes = []
    dataset = (prediction or health).get("dataset")
    if dataset == "synthetic":
        notes.append(
            "This service is serving a model fitted on the **synthetic fixture**, not "
            "on real postings. The numbers below exercise the pipeline and mean "
            "nothing about any real job."
        )
    if prediction is not None and not prediction["board_context_supplied"]:
        notes.append(
            "No board context was supplied, so the four board-level features were "
            "imputed to constants. The prediction rests on the posting alone."
        )
    return notes
