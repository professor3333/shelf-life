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


def rank_board(
    api: Api,
    postings: list[dict],
    budget: int,
    as_of: str | None = None,
    is_board_snapshot: bool = False,
    previous_board_size: int | None = None,
    on_progress=None,
) -> dict:
    """Rank a whole board through the server's board flow, whatever its size.

    `/rank` caps a request at the service's page (250 on the free instance,
    `src.inference.predict.MAX_BATCH`, a measurement), and a day's board is
    about 1,150 — so the whole-board ranking is several requests. The server
    owns everything between them: `POST /boards` opens the collection with
    its instant and its snapshot declaration; pages are appended; `score` is
    called until `done`, the server deriving board context over the *whole*
    board first when it is a snapshot; `rank` orders the whole board by the
    same rule `/rank` applies to a batch. This client pages and loops. It
    holds no copy of the ranking rule and none of the derivation — until
    2026-09-12 it held both, which was the reimplementation `/rank` existed
    to make unnecessary.

    A board that fits in one page still goes this way, so there is one path.
    Boards live in memory on the instance and are gone when it sleeps: a 404
    mid-flow is retried from the start, once.
    """
    if budget < 1:
        raise ValueError("budget must be at least 1")
    if not postings:
        raise ValueError("no postings to rank")

    for attempt in (1, 2):
        try:
            return _drive(
                api, postings, budget, as_of, is_board_snapshot, previous_board_size, on_progress
            )
        except ApiError as error:
            if attempt == 1 and "re-upload" in str(error):
                continue  # the board expired mid-flow; once more from the top
            raise
    raise AssertionError("unreachable")


def _drive(api, postings, budget, as_of, is_board_snapshot, previous_board_size, on_progress):
    create: dict = {"is_board_snapshot": bool(is_board_snapshot)}
    if as_of is not None:
        create["as_of"] = as_of
    if previous_board_size is not None:
        create["previous_board_size"] = previous_board_size
    opened = api._post("/boards", create)
    board_id, page_size = opened["board_id"], int(opened["page_size"])

    for start in range(0, len(postings), page_size):
        api._post(f"/boards/{board_id}/postings", {"postings": postings[start : start + page_size]})
    while True:
        progress = api._post(f"/boards/{board_id}/score", {})
        if on_progress is not None:
            on_progress(progress["scored"], progress["total"])
        if progress["done"]:
            break
    ranking = api._post(f"/boards/{board_id}/rank", {"budget": budget})
    try:
        requests.delete(f"{api.base_url}/boards/{board_id}", timeout=TIMEOUT)
    except requests.RequestException:
        pass  # it expires on its own
    return ranking


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
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("every posting must be an object")
        kept = build_payload({k: v for k, v in row.items() if k in allowed or k == "client_id"})
        if "title" not in kept:
            raise ValueError("every posting needs a title")
        # The join key. A board's own `client_id` column is kept; without one,
        # the row number stands in — so results are matched by handle, never
        # by array position, even for a pasted CSV.
        kept.setdefault("client_id", f"row-{index + 1}")
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
