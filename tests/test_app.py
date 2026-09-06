"""The UI's logic, and the separation it exists to demonstrate.

A Streamlit script cannot be called, so everything that could be wrong was moved
into `app/client.py` and is tested here as ordinary functions. What remains in
`app/streamlit_app.py` is widget calls, and the one thing worth asserting about
that file is what it does *not* import.

**`test_the_ui_calls_the_api_and_never_the_model` is the point of the
component.** UI ≠ API ≠ model ≠ training pipeline. The failure it prevents is not
a crash — importing `Predictor` into the UI works fine on one machine. It is a
deployment where the UI and the API load two different artifacts and nothing
says so.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import requests
from fastapi.testclient import TestClient
from test_inference import EXPECTED_PROBABILITY, FIXED_POSTING, FIXED_T

from api.main import create_app
from app.client import (
    Api,
    ApiError,
    _detail,
    api_url_from,
    build_payload,
    verdict,
    warnings_for,
)


class _FakeResponse:
    """The two error shapes FastAPI produces, without a server to produce them."""

    def __init__(self, status_code: int, body, text: str = ""):
        self.status_code = status_code
        self._body = body
        self.text = text or str(body)

    def json(self):
        if self._body is _UNPARSEABLE:
            raise ValueError("no json")
        return self._body


_UNPARSEABLE = object()


# --- payload building -------------------------------------------------------


def test_blank_fields_are_omitted_not_sent_as_empty_strings():
    """An untouched text box must not become a category the model never saw."""
    payload = build_payload({"title": "Engineer", "location": "  ", "company": None})
    assert payload == {"title": "Engineer"}


def test_values_are_stripped_and_non_strings_pass_through():
    payload = build_payload({"title": "  Engineer  ", "board_size_at_t": 300, "n_offices": 0})
    assert payload == {"title": "Engineer", "board_size_at_t": 300, "n_offices": 0}


# --- wording ----------------------------------------------------------------


def test_the_verdict_names_the_horizon_and_the_threshold():
    headline, explanation = verdict(
        {"closing_soon": True, "threshold": 0.4, "horizon_days": 1, "probability": 0.9}
    )
    assert "within 1 day" in headline
    assert "0.400" in explanation


def test_a_probability_below_the_threshold_is_not_flagged():
    headline, _ = verdict(
        {"closing_soon": False, "threshold": 0.4, "horizon_days": 7, "probability": 0.1}
    )
    assert "NOT flagged" in headline and "7 days" in headline


def test_a_synthetic_model_is_announced_before_the_number_is_read():
    notes = warnings_for({"dataset": "synthetic"})
    assert notes and "synthetic" in notes[0]


def test_a_real_model_with_board_context_warns_about_nothing():
    health = {"dataset": "real"}
    prediction = {"dataset": "real", "board_context_supplied": True}
    assert warnings_for(health, prediction) == []


def test_missing_board_context_is_reported_to_the_person_reading_it():
    prediction = {"dataset": "real", "board_context_supplied": False}
    notes = warnings_for({"dataset": "real"}, prediction)
    assert len(notes) == 1 and "imputed" in notes[0]


# --- error handling ---------------------------------------------------------


def test_a_field_error_keeps_the_field_name():
    """FastAPI's 422 body is a list. "Something went wrong" throws away the fix."""
    body = {"detail": [{"loc": ["body", "title"], "msg": "Field required"}]}
    assert _detail(_FakeResponse(422, body)) == "title: Field required"


def test_a_handler_error_keeps_its_sentence():
    body = {"detail": "no model loaded from models/shelf_life.joblib"}
    assert "no model loaded" in _detail(_FakeResponse(503, body))


def test_an_unparseable_error_body_still_says_something_useful():
    detail = _detail(_FakeResponse(502, _UNPARSEABLE, text="<html>bad gateway</html>"))
    assert "502" in detail


def test_an_unreachable_api_is_an_ApiError_not_a_traceback(monkeypatch):
    def refuse(*args, **kwargs):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)
    with pytest.raises(ApiError, match="cannot reach the API"):
        Api("http://127.0.0.1:1").health()


# --- the round trip, without a network --------------------------------------


@pytest.fixture
def wired_api(monkeypatch, synthetic_artifact):
    """The real client against the real app, with `requests` routed to the ASGI
    test transport. Everything but the socket is exercised."""
    client = TestClient(create_app(synthetic_artifact))
    client.__enter__()

    def get(url, **kwargs):
        return client.get(url.replace("http://api", ""))

    def post(url, json=None, **kwargs):
        return client.post(url.replace("http://api", ""), json=json)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    yield Api("http://api")
    client.__exit__(None, None, None)


def test_the_client_gets_the_pinned_probability_through_the_api(wired_api):
    prediction = wired_api.predict({**FIXED_POSTING, "as_of": FIXED_T})
    assert prediction["probability"] == pytest.approx(EXPECTED_PROBABILITY, abs=1e-6)


def test_the_client_surfaces_the_apis_own_refusal(wired_api):
    with pytest.raises(ApiError, match="salary"):
        wired_api.predict({"title": "Engineer", "salary": "lots"})


def test_health_and_contract_come_back_through_the_client(wired_api):
    assert wired_api.health()["model_loaded"] is True
    assert any(row["field"] == "title" for row in wired_api.contract())


# --- the separation ---------------------------------------------------------


def test_the_ui_calls_the_api_and_never_the_model():
    """`app/` may not import `src/`. See the module docstring."""
    for path in sorted(Path("app").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            module = getattr(node, "module", None)
            names = [alias.name for alias in getattr(node, "names", [])]
            assert not (module or "").startswith("src"), f"{path} imports {module}"
            assert not any(name.startswith("src") for name in names), f"{path} imports src"


# --- the script itself, rendered ---------------------------------------------

#: Absolute, because `AppTest.from_file` resolves a relative path against the
#: *calling* file rather than the working directory.
APP_SCRIPT = str(Path("app/streamlit_app.py").resolve())


@pytest.fixture
def rendered(monkeypatch, wired_api):
    """The Streamlit script, executed in Streamlit's own test runtime.

    `AppTest` runs the file top to bottom the way `streamlit run` does and hands
    back the widget tree, so the script is covered by tests rather than by having
    been opened in a browser once. `wired_api` has already routed `requests`
    into the FastAPI app, so this exercises UI -> HTTP -> API -> pipeline with no
    server and no socket.
    """
    pytest.importorskip("streamlit", reason="the UI extra is not installed")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SHELF_LIFE_API", wired_api.base_url)
    test = AppTest.from_file(APP_SCRIPT, default_timeout=30)
    return test.run()


def test_the_form_renders_without_error(rendered):
    assert not rendered.exception
    assert rendered.title[0].value == "shelf-life"
    labels = {widget.label for widget in rendered.text_input}
    assert {"Title", "Location"} <= labels


def test_a_synthetic_model_is_announced_on_screen(rendered):
    """The banner from `warnings_for`, in the rendered page rather than in a unit
    test of the string that produces it."""
    assert any("synthetic" in warning.value for warning in rendered.warning)


def test_submitting_the_form_shows_a_probability_and_the_caveat(rendered):
    for widget in rendered.text_input:
        if widget.label == "Title":
            widget.set_value("Senior Data Engineer")
    result = rendered.button[0].click().run()

    assert not result.exception
    assert result.metric[0].label == "Probability"
    assert any("not the same as filled" in info.value for info in result.info)


# --- where the API is, which is what broke the first deployed UI -------------


def test_the_url_falls_back_to_localhost_when_nothing_says_otherwise(monkeypatch):
    monkeypatch.delenv("SHELF_LIFE_API", raising=False)
    assert api_url_from() == "http://localhost:8000"


def test_a_secret_is_used_when_the_environment_is_empty(monkeypatch):
    """The deployed case. Community Cloud configures the URL as a secret."""
    monkeypatch.delenv("SHELF_LIFE_API", raising=False)
    assert api_url_from("", "https://api.example") == "https://api.example"


def test_the_environment_still_works_for_a_local_run(monkeypatch):
    monkeypatch.setenv("SHELF_LIFE_API", "http://localhost:9000")
    assert api_url_from() == "http://localhost:9000"


def test_a_secret_outranks_the_environment(monkeypatch):
    """Both present means a deployed app whose host also exports the variable.

    The secret is the one the operator set on purpose, so it wins.
    """
    monkeypatch.setenv("SHELF_LIFE_API", "http://localhost:9000")
    assert api_url_from("", "https://api.example") == "https://api.example"


def test_an_explicit_url_outranks_everything(monkeypatch):
    monkeypatch.setenv("SHELF_LIFE_API", "http://localhost:9000")
    assert api_url_from("https://override", "https://api.example") == "https://override"


def test_an_empty_secret_does_not_shadow_the_environment(monkeypatch):
    """A blank secret box is 'unset', not 'use the empty string'."""
    monkeypatch.setenv("SHELF_LIFE_API", "http://localhost:9000")
    assert api_url_from("", "") == "http://localhost:9000"
