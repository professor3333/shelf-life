"""The page somebody who does not read JSON can use: a board, ranked; or one posting.

    Streamlit  ──HTTP──>  FastAPI  ──>  the frozen pipeline

**Ranking a board comes first**, because that is what the model is for
(`docs/design.md` §15): give it today's board and a budget — how many postings
a person can actually read — and get back the ones most likely to be gone
within the horizon. The single-posting form is the second mode, kept because it
is the form of the question a person holding one job ad asks. The board is
uploaded to the service as one board (`POST /boards`, in pages the service
will accept), scored there page by page and ranked there once, under one
budget — the client pages and loops, and holds no copy of the ranking rule
(`app/client.py` `rank_board`).

Three rules this file keeps, and each one is a line it would be easy to cross:

**It calls the API, never the model.** No import from `src/` appears here or in
`app/client.py`, and a test enforces it. Loading the artifact directly would
work perfectly on a laptop and would mean the deployed UI and the deployed API
were two different models wearing one name.

**It sends what the person typed, and nothing else.** Blank boxes are omitted
rather than sent as empty strings, so an untouched field stays a missing value
the API imputes instead of becoming a category the model has never seen.

**It shows the caveat on screen.** The label is removal from the board, not
filled. That sentence belongs where the number is, not only in the README —
whoever reads a probability is the person who needs it.

Run it::

    SHELF_LIFE_API=http://localhost:8000 streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# `streamlit run app/streamlit_app.py` puts *this file's* directory on `sys.path`,
# not the repository root, so `app` is not importable as a package under the very
# command the file is meant to be launched with. Put the root back before the
# import that needs it. Everything else in this project is imported normally,
# because everything else is launched with `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The host re-runs *this* script when the repository changes, but it does not
# necessarily re-import the modules this script imports: on 2026-09-12 a push
# that added names to `app/client.py` left Community Cloud's process holding
# the old module, and every visitor saw an ImportError on the line below until
# the process was replaced. So if the helper module is already loaded and is
# missing a name this script needs, reload it before importing from it. Locally
# the branch is never taken; on the host it is what turns a push into a deploy.
import importlib  # noqa: E402

import app.client as _client  # noqa: E402

if not hasattr(_client, "rank_board"):
    _client = importlib.reload(_client)

from app.client import (  # noqa: E402
    API_URL_ENV,
    CAVEAT,
    EXAMPLE_BOARD_CSV,
    Api,
    ApiError,
    api_url_from,
    build_payload,
    parse_board,
    rank_board,
    verdict,
    warnings_for,
)

st.set_page_config(page_title="shelf-life", page_icon="📋", layout="centered")


def _iso(value: dt.date | None) -> str | None:
    """A date widget's value as an instant the API will accept, or nothing.

    `None` rather than today's date when the person did not pick one: the API
    imputes an absent `first_published`, and defaulting it to today would invent
    an `age_days` of zero for every posting whose publication date is unknown.
    """
    return dt.datetime.combine(value, dt.time.min, dt.UTC).isoformat() if value else None


def _secret_api_url() -> str | None:
    """`SHELF_LIFE_API` from Streamlit's secrets, or nothing.

    Wrapped because reading `st.secrets` is an error, not an empty mapping, when
    no secrets are configured at all — which is the normal case for
    `streamlit run` on a laptop. A missing secret must leave the other rungs of
    `api_url_from` intact rather than replacing the form with a traceback.
    """
    try:
        return st.secrets.get(API_URL_ENV) or None
    except Exception:  # noqa: BLE001 - any secrets failure means "no secret"
        return None


api = Api(api_url_from(st.session_state.get("api_url", ""), _secret_api_url()))

st.title("shelf-life")
st.caption(
    "Will this job posting come off the board soon? A prediction made from what "
    "is knowable the moment the posting is first seen."
)

# --- is there anything to talk to? ------------------------------------------

# This call is also the wake-up. The API is hosted on a free tier that spins the
# container down after 15 idle minutes and takes about a minute to come back, so
# the first request of a visit pays for a cold start whether it asks for one or
# not. Making that request *here*, while the form is still being read and filled
# in, spends the wait on time the visitor was going to use anyway — which is the
# only cold-start mitigation this architecture gets for free, now that there is
# no warm-instance knob to buy (`docs/design.md` §7e).
#
# The spinner text is the honest version rather than a bare spinner: a minute of
# silence reads as broken, and the same minute with a sentence explaining it
# reads as a free tier.
try:
    with st.spinner("Waking the prediction service — the free tier takes up to a minute…"):
        health = api.health()
except ApiError as error:
    st.error(f"{error}\n\nStart the API with `uvicorn api.main:app`, or set `SHELF_LIFE_API`.")
    st.stop()

# No model is the deliberate state until the panel is deep enough to freeze one
# (`docs/design.md` §13), and it used to stop the page here — so a visitor to
# the public URL saw a warning and nothing of what the product does. The page
# now draws both modes anyway, with the buttons disabled and the warning kept,
# so the workflow is visible before there is a model to run it. Nothing below
# calls the API for a prediction while `MODEL_LOADED` is false.
MODEL_LOADED = bool(health["model_loaded"])
if not MODEL_LOADED:
    st.warning(
        f"The API is up but has no model loaded: {health.get('detail')}\n\n"
        "Build one with `python -m src.models.freeze --run <spec>`. Until then the "
        "page below shows the workflow and runs nothing."
    )

for note in warnings_for(health):
    st.warning(note)

with st.sidebar:
    st.subheader("The model behind this form")
    if MODEL_LOADED:
        st.write(
            pd.Series(
                {
                    "run": health["model"],
                    "fitted on": f"{health['fitted_on']} block, {health['dataset']} data",
                    "horizon": f"{health['horizon_days']} day(s)",
                    "threshold": f"{health['threshold']:.4f}",
                    "frozen": health["created_at"],
                }
            )
        )
        st.caption(
            "The threshold was chosen on the validation block at a fixed alert "
            "budget — the number of postings a person can actually read in a day — "
            "not left at 0.5."
        )
    else:
        st.caption(
            "None yet. The panel is not deep enough to select and freeze a model "
            "honestly; the readiness report in the repository says how far off that is."
        )

# --- which question ----------------------------------------------------------

RANK, ONE = "Rank a board", "One posting"
mode = st.radio("What do you have?", (RANK, ONE), horizontal=True)

# --- a board, ranked under a budget -------------------------------------------

if mode == RANK:
    st.subheader("Today's board")
    st.caption(
        'Paste or upload the postings — JSON (a list, or `{"postings": [...]}`) or CSV '
        "with one column per field. Fields the API does not accept are dropped; "
        "`/contract` lists the ones it does. The board is uploaded to the service in "
        f"pages of {health.get('rank_max_batch', '?')}, scored there page by page, and "
        "ranked there once, under one budget."
    )
    uploaded = st.file_uploader("Upload a board", type=("csv", "json"))
    pasted = st.text_area(
        "…or paste it",
        height=160,
        placeholder=EXAMPLE_BOARD_CSV,
        help="The placeholder is a real five-posting board; the download below has it.",
    )
    st.download_button(
        "Download the example board (CSV)",
        EXAMPLE_BOARD_CSV,
        file_name="example_board.csv",
        mime="text/csv",
    )
    left, right = st.columns(2)
    with left:
        budget = st.number_input(
            "Budget — postings you will read today",
            min_value=1,
            value=20,
            step=1,
            help="The operating point is the budget-th score on this board: exactly this "
            "many postings are flagged, and the threshold that implies is shown.",
        )
    with right:
        as_of_date = st.date_input("As of", value=None, format="YYYY-MM-DD")
    # The board-ranking mode proper. Board context — how many postings the
    # board carries, how many share a title, how it grew — is unknowable from
    # one advert and is imputed for one, but for a whole board it is arithmetic
    # over the board itself. The caller says whether this is the whole board.
    is_snapshot = st.checkbox(
        "These postings are the whole board — derive board context from them",
        value=False,
        help="Off: the four board-level features are imputed from the training data "
        "(or used if a column supplies them). On: they are computed from this batch by "
        "the panel's own definitions — size, same-title count, requisition-group count "
        "(from a `requisition_id` column, if present) and growth against yesterday's "
        "size below. A partial board declared whole hands the model a board size it "
        "never saw.",
    )
    previous_size = st.number_input(
        "Board size at the previous crawl (for growth; optional)",
        min_value=0,
        value=None,
        step=1,
        disabled=not is_snapshot,
    )
    rank_it = st.button("Rank", type="primary", disabled=not MODEL_LOADED)

    # Read the board as soon as there is one, so the count is on screen
    # before anything is ranked — and so a malformed board fails at paste
    # time, not after a minute of waiting on the free tier.
    text = uploaded.getvalue().decode("utf-8") if uploaded is not None else pasted
    board: list[dict] = []
    if text.strip():
        try:
            allowed = {row["field"] for row in api.contract()}
            if is_snapshot:
                allowed = allowed | {"requisition_id"}
            board = parse_board(text, allowed)
            st.caption(f"{len(board)} posting{'s' if len(board) != 1 else ''} read.")
        except (ApiError, ValueError) as error:
            st.error(f"could not read the board: {error}")
            st.stop()

    if rank_it:
        if not board:
            st.error("No postings. Paste a board, upload one, or use the example.")
            st.stop()

        try:
            with st.spinner(f"Ranking {len(board)} postings…"):
                ranking = rank_board(
                    api,
                    board,
                    budget=int(budget),
                    as_of=_iso(as_of_date),
                    is_board_snapshot=bool(is_snapshot),
                    previous_board_size=int(previous_size) if previous_size is not None else None,
                    on_progress=lambda scored, total: st.toast(f"scored {scored} of {total}"),
                )
        except (ApiError, ValueError) as error:
            st.error(str(error))
            st.stop()

        # Joined on `client_id`, the handle every parsed posting carries, not on
        # array position — the position contract holds for one synchronous call
        # and nothing else, and this table should not depend on it.
        by_id = {posting["client_id"]: posting for posting in board}
        results = ranking["postings"]
        table = pd.DataFrame(
            {
                "rank": [row["rank"] for row in results],
                "flagged": [row["watch"] for row in results],
                "probability": [row["probability"] for row in results],
                "title": [by_id[row["client_id"]].get("title") for row in results],
                "company": [by_id[row["client_id"]].get("company") for row in results],
                "location": [by_id[row["client_id"]].get("location") for row in results],
                "client_id": [row["client_id"] for row in results],
            }
        ).sort_values("rank")
        flagged = table[table["flagged"]]

        horizon = ranking["horizon_days"]
        st.subheader(
            f"{len(flagged)} of {len(board)} flagged as likely to be removed "
            f"within {horizon} day{'s' if horizon != 1 else ''}"
        )
        columns = st.columns(3)
        columns[0].metric("Budget", ranking["budget"])
        columns[1].metric("Threshold on this board", f"{ranking['threshold_applied']:.3f}")
        columns[2].metric("Frozen threshold", f"{health['threshold']:.3f}")
        pages = ranking["pages_scored"]
        st.caption(
            "The board threshold is the budget-th score of what you sent — a property of "
            "this board. The frozen threshold is the model's own operating point, chosen "
            "on validation at the stated budget. They are different decisions, and "
            f"this ranking used the first (source: `{ranking['threshold_source']}`; "
            f"scored by the service in {pages} page{'s' if pages != 1 else ''})."
        )
        st.dataframe(
            flagged.drop(columns="flagged").style.format({"probability": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
        with st.expander("The whole board, ranked"):
            st.dataframe(
                table.style.format({"probability": "{:.1%}"}),
                hide_index=True,
                use_container_width=True,
            )
        for note in warnings_for(health):
            st.warning(note)
        if ranking["board_size"] is not None:
            st.caption(
                f"Board context derived from the snapshot: {ranking['board_size']} postings "
                f"on the board ({ranking['board_context_source']})."
            )
        elif not all(row["board_context_supplied"] for row in ranking["postings"]):
            st.warning(
                "Board context was not supplied for every posting, so the board-level "
                "features were imputed to constants there. A submitted batch is not the "
                "board unless you say it is — tick the snapshot box if it is."
            )
        st.info(CAVEAT)
    st.stop()

# --- the form ---------------------------------------------------------------

with st.form("posting"):
    st.subheader("The posting")
    title = st.text_input("Title", placeholder="Senior Backend Engineer")
    location = st.text_input("Location", placeholder="Berlin, Germany")
    salary_raw = st.text_input(
        "Salary, as written on the posting",
        placeholder="90.000 € bis 130.000 €",
        help="Sent verbatim. The API parses it the same way the training data was parsed, "
        "and a posting that states no pay is itself a signal.",
    )

    left, right = st.columns(2)
    with left:
        departments = st.text_input("Department", placeholder="Engineering")
        company = st.text_input("Company", placeholder="Wolt")
        published = st.date_input("First published", value=None, format="YYYY-MM-DD")
    with right:
        offices = st.text_input("Office", placeholder="Berlin")
        source = st.text_input("Board", placeholder="greenhouse:acme")
        updated = st.date_input("Last updated", value=None, format="YYYY-MM-DD")

    description = st.text_area(
        "Description",
        placeholder="Paste the posting body here.",
        help="Only its length is used. The model reads no text beyond the title; "
        "embeddings are a later stage of this project, not this one.",
    )

    st.caption(
        "Only what you can know from the advert itself. The four board-level "
        "features — how many postings the board carries, how many share this title, "
        "how it grew — are imputed here, and the result says so; if you hold the whole "
        "board, *Rank a board* derives them from it."
    )

    submitted = st.form_submit_button("Predict", type="primary", disabled=not MODEL_LOADED)

# --- the answer -------------------------------------------------------------

if submitted:
    payload = build_payload(
        {
            "title": title,
            "location": location,
            "salary_raw": salary_raw,
            "departments": departments,
            "offices": offices,
            "company": company,
            "source": source,
            # The body itself is never sent — only how long it is, which is the
            # feature the panel actually carries.
            "content_chars": float(len(description)) if description.strip() else None,
            "first_published": _iso(published),
            "updated_at": _iso(updated),
        }
    )

    try:
        prediction = api.predict(payload)
    except ApiError as error:
        st.error(str(error))
        st.stop()

    headline, explanation = verdict(prediction)
    probability = prediction["probability"]

    st.subheader(headline)
    columns = st.columns(3)
    columns[0].metric("Probability", f"{probability:.1%}")
    columns[1].metric("Threshold", f"{prediction['threshold']:.3f}")
    columns[2].metric("Horizon", f"{prediction['horizon_days']} day(s)")
    st.progress(min(max(probability, 0.0), 1.0))
    st.write(explanation)

    for note in warnings_for(health, prediction):
        st.warning(note)
    st.info(CAVEAT)

    with st.expander("What was actually sent"):
        st.json(payload)
