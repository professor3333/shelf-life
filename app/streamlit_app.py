"""The form somebody who does not read JSON can use.

    Streamlit  ──HTTP──>  FastAPI  ──>  the frozen pipeline

Three rules this file keeps, and each one is a line it would be easy to cross:

**It calls the API, never the model.** No import from `src/` appears here or in
`app/client.py`, and a test enforces it. Loading the artifact directly would
work perfectly on a laptop and would mean the deployed UI and the deployed API
were two different models wearing one name.

**It sends what the person typed, and nothing else.** Blank boxes are omitted
rather than sent as empty strings, so an untouched field stays a missing value
the API imputes instead of becoming a category the model has never seen.

**It shows the caveat on screen.** "Closed" means removed from the board, not
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

from app.client import (  # noqa: E402
    CAVEAT,
    Api,
    ApiError,
    build_payload,
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


api = Api(st.session_state.get("api_url", ""))

st.title("shelf-life")
st.caption(
    "Will this job posting come off the board soon? A prediction made from what "
    "is knowable the moment the posting is first seen."
)

# --- is there anything to talk to? ------------------------------------------

try:
    health = api.health()
except ApiError as error:
    st.error(f"{error}\n\nStart the API with `uvicorn api.main:app`, or set `SHELF_LIFE_API`.")
    st.stop()

if not health["model_loaded"]:
    st.warning(
        f"The API is up but has no model loaded: {health.get('detail')}\n\n"
        "Build one with `python -m src.models.freeze --run <spec>`."
    )
    st.stop()

for note in warnings_for(health):
    st.warning(note)

with st.sidebar:
    st.subheader("The model behind this form")
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

    with st.expander("Board context — only if you run the board"):
        st.caption(
            "These four describe the board rather than the posting, so somebody "
            "holding one job ad cannot know them. Left empty they are imputed "
            "from the training data, and the result says so."
        )
        board_size = st.number_input("Postings on the board", min_value=0, value=None, step=1)
        board_growth = st.number_input("Change since the previous crawl", value=None, step=1)
        same_title = st.number_input("Postings with this same title", min_value=0, value=None)
        same_req = st.number_input("Postings in this requisition group", min_value=0, value=None)

    submitted = st.form_submit_button("Predict", type="primary")

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
            "board_size_at_t": board_size,
            "board_growth": board_growth,
            "n_same_title_on_board": same_title,
            "n_same_req_on_board": same_req,
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
