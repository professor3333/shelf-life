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
    rank_board,
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
        {"removal_flagged": True, "threshold": 0.4, "horizon_days": 1, "probability": 0.9}
    )
    assert "within 1 day" in headline
    assert "REMOVED" in headline and "CLOSE" not in headline.upper().replace("DISCLOSE", "")
    assert "0.400" in explanation


def test_a_probability_below_the_threshold_is_not_flagged():
    headline, _ = verdict(
        {"removal_flagged": False, "threshold": 0.4, "horizon_days": 7, "probability": 0.1}
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

    def delete(url, **kwargs):
        return client.delete(url.replace("http://api", ""))

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(requests, "delete", delete)
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


# --- a whole board, in pages, ranked once --------------------------------------


def _board(n: int) -> list[dict]:
    """`n` distinct postings, varied enough that scores are not all equal."""
    titles = ("Senior Data Engineer", "Werkstudent Marketing", "Staff Engineer", "Head of Product")
    return [
        {
            **FIXED_POSTING,
            "title": f"{titles[i % len(titles)]} {i}",
            "content_chars": 400 + (i * 37) % 2600,
            "first_published": f"2026-0{7 + i % 2}-{1 + i % 27:02d}T00:00:00Z",
        }
        for i in range(n)
    ]


def test_ranking_a_board_through_the_server_equals_ranking_it_at_once(
    wired_api, synthetic_artifact, monkeypatch
):
    """The property the design rests on, now with the server owning the pages:
    a board larger than a page, uploaded and scored in pages and ranked once
    by the service, equals `Predictor.rank` on the whole board with the cap
    lifted. The client holds no copy of the rule."""
    from src.inference import predict as predict_module
    from src.inference.predict import MAX_BATCH, Predictor

    board = _board(2 * MAX_BATCH + 100)
    progress = []
    paged = rank_board(
        wired_api, board, budget=20, as_of=FIXED_T, on_progress=lambda s, t: progress.append(s)
    )
    monkeypatch.setattr(predict_module, "MAX_BATCH", len(board))
    whole = Predictor.load(synthetic_artifact).rank(board, t=FIXED_T, budget=20)

    assert paged["pages_scored"] == 3 and progress[-1] == len(board)
    assert paged["budget"] == whole.budget == 20
    assert paged["threshold_applied"] == pytest.approx(whole.threshold_applied)
    assert [p["rank"] for p in paged["postings"]] == [p.rank for p in whole.postings]
    assert [p["watch"] for p in paged["postings"]] == [p.watch for p in whole.postings]
    assert sum(p["watch"] for p in paged["postings"]) == 20


def test_a_snapshot_larger_than_a_page_is_derived_by_the_server_over_the_whole_board(wired_api):
    from src.inference.predict import MAX_BATCH

    board = _board(MAX_BATCH + 50)
    out = rank_board(wired_api, board, budget=5, as_of=FIXED_T, is_board_snapshot=True)
    assert out["board_size"] == len(board)
    assert out["board_context_source"] == "derived from the snapshot"
    assert out["pages_scored"] == 2


def test_the_client_holds_no_copy_of_the_ranking_rule_or_the_derivation():
    """What `/boards` exists to remove. A test, because the copies were there."""
    import ast

    source = (Path("app") / "client.py").read_text()
    tree = ast.parse(source)
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "board_context_for" not in names
    assert "argsort" not in source and "sorted(scored" not in source
    assert "n_same_title_on_board" not in source


def test_an_expired_board_mid_flow_is_retried_once_from_the_top(monkeypatch):
    calls = []

    class Flaky(Api):
        def _post(self, path, body):
            calls.append(path)
            if path == "/boards":
                return {"board_id": f"b{len(calls)}", "page_size": 250}
            if path.endswith("/postings"):
                return {"total": 1, "scored": 0, "done": False}
            if path.endswith("/score"):
                if len([c for c in calls if c.endswith("/score")]) == 1:
                    raise ApiError("no board 'b1': it expired ... create it again and re-upload")
                return {"total": 1, "scored": 1, "done": True}
            return {
                "postings": [
                    {"rank": 1, "probability": 0.5, "watch": True, "board_context_supplied": False}
                ],
                "threshold_applied": 0.5,
                "threshold_source": "batch_budget",
                "budget": 1,
                "horizon_days": 7,
                "model": "m",
                "dataset": "synthetic",
                "t": "t",
                "board_context_source": "supplied per posting or imputed",
                "board_size": None,
                "board_id": "b",
                "pages_scored": 1,
            }

    monkeypatch.setattr(requests, "delete", lambda *a, **k: None)
    out = rank_board(Flaky("http://api"), [{"title": "x"}], budget=1)
    assert out["budget"] == 1
    assert calls.count("/boards") == 2, "opened twice: once before the expiry, once after"


def test_a_board_is_read_from_json_or_csv_and_unknown_columns_are_dropped():
    from app.client import EXAMPLE_BOARD_CSV, parse_board

    allowed = {"title", "location", "salary_raw", "content_chars", "first_published"}
    from_csv = parse_board(EXAMPLE_BOARD_CSV, allowed)
    assert len(from_csv) == 5
    assert from_csv[0]["title"] == "Senior Data Engineer"
    assert "salary_raw" not in from_csv[1], "a blank cell is omitted, not sent as NaN"
    assert from_csv[0]["content_chars"] == 1400

    as_list = parse_board('[{"title": "A", "url": "https://x", "location": "B"}]', allowed)
    assert as_list == [{"title": "A", "location": "B"}], "url is not a field the API takes"
    wrapped = parse_board('{"postings": [{"title": "A"}]}', allowed)
    assert wrapped == [{"title": "A"}]
    assert parse_board("   ", allowed) == []
    with pytest.raises(ValueError, match="title"):
        parse_board("location\nBerlin\n", allowed)


def test_rank_board_refuses_nothing_and_a_zero_budget():
    with pytest.raises(ValueError):
        rank_board(Api("http://api"), [], budget=1)
    with pytest.raises(ValueError):
        rank_board(Api("http://api"), [{"title": "x"}], budget=0)


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


def _one_posting_mode(rendered):
    """Switch the page to the single-posting form. The board comes first."""
    rendered.radio[0].set_value("One posting")
    return rendered.run()


def test_the_board_comes_first(rendered):
    """The page opens on the product's question — a board, ranked under a
    budget — and the single-posting form is the second mode."""
    assert not rendered.exception
    assert rendered.title[0].value == "shelf-life"
    assert rendered.radio[0].value == "Rank a board"
    assert any(widget.label.startswith("Budget") for widget in rendered.number_input)
    assert not any(widget.label == "Title" for widget in rendered.text_input)


def test_ranking_the_example_board_flags_the_budget_and_shows_both_thresholds(rendered):
    """Paste the example board, ask for two, get exactly two flagged — through
    the real client, the real API and the pipeline, with no socket."""
    from app.client import EXAMPLE_BOARD_CSV

    rendered.text_area[0].set_value(EXAMPLE_BOARD_CSV)
    for widget in rendered.number_input:
        if widget.label.startswith("Budget"):
            widget.set_value(2)
    rank = next(button for button in rendered.button if button.label == "Rank")
    result = rank.click().run()

    assert not result.exception
    assert any(sub.value.startswith("2 of 5 flagged") for sub in result.subheader)
    assert "company" in result.dataframe[0].value.columns
    labels = {metric.label for metric in result.metric}
    assert {"Budget", "Threshold on this board", "Frozen threshold"} <= labels
    assert any("not the same as filled" in info.value for info in result.info)


def test_the_snapshot_switch_derives_board_context_on_the_page(rendered):
    from app.client import EXAMPLE_BOARD_CSV

    rendered.text_area[0].set_value(EXAMPLE_BOARD_CSV)
    rendered.checkbox[0].set_value(True)
    for widget in rendered.number_input:
        if widget.label.startswith("Budget"):
            widget.set_value(2)
    rank = next(button for button in rendered.button if button.label == "Rank")
    result = rank.click().run()

    assert not result.exception
    assert any("derived from the snapshot: 5 postings" in c.value for c in result.caption)


def test_the_one_posting_form_takes_no_board_context(rendered):
    rendered = _one_posting_mode(rendered)
    labels = {widget.label for widget in rendered.number_input}
    assert not any("board" in label.lower() for label in labels)
    assert not any("requisition" in label.lower() for label in labels)


@pytest.fixture
def rendered_without_a_model(monkeypatch):
    """The page against an API that has no model — the public URL's state."""
    pytest.importorskip("streamlit", reason="the UI extra is not installed")
    from streamlit.testing.v1 import AppTest

    client = TestClient(create_app("/nonexistent/shelf_life.joblib"))
    client.__enter__()
    monkeypatch.setattr(requests, "get", lambda url, **k: client.get(url.replace("http://api", "")))
    monkeypatch.setattr(
        requests,
        "post",
        lambda url, json=None, **k: client.post(url.replace("http://api", ""), json=json),
    )
    monkeypatch.setenv("SHELF_LIFE_API", "http://api")
    yield AppTest.from_file(APP_SCRIPT, default_timeout=30).run()
    client.__exit__(None, None, None)


def test_the_workflow_is_visible_before_there_is_a_model(rendered_without_a_model):
    """Until 2026-09-12 the page stopped at the no-model warning, so a visitor
    to the public URL saw nothing of the product. Now both modes draw, with
    their buttons disabled, and the warning stays."""
    page = rendered_without_a_model
    assert not page.exception
    assert any("no model loaded" in w.value for w in page.warning)
    assert page.radio[0].value == "Rank a board"
    rank = next(button for button in page.button if button.label == "Rank")
    assert rank.disabled
    assert any(widget.label.startswith("Budget") for widget in page.number_input)


def test_the_count_is_shown_as_soon_as_a_board_is_read(rendered):
    from app.client import EXAMPLE_BOARD_CSV

    rendered.text_area[0].set_value(EXAMPLE_BOARD_CSV)
    page = rendered.run()
    assert any(c.value == "5 postings read." for c in page.caption)


def test_the_form_renders_without_error(rendered):
    rendered = _one_posting_mode(rendered)
    assert not rendered.exception
    labels = {widget.label for widget in rendered.text_input}
    assert {"Title", "Location"} <= labels


def test_a_synthetic_model_is_announced_on_screen(rendered):
    """The banner from `warnings_for`, in the rendered page rather than in a unit
    test of the string that produces it."""
    assert any("synthetic" in warning.value for warning in rendered.warning)


def test_submitting_the_form_shows_a_probability_and_the_caveat(rendered):
    rendered = _one_posting_mode(rendered)
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
