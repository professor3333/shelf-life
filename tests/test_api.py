"""The HTTP surface: the number survives the round trip, and bad input is a 4xx.

Two claims are worth stating before the tests make them.

**The API must not be able to change the prediction.** `test_the_api_returns_the
_pinned_probability` asserts the value from `tests/test_inference.py` — the same
constant, imported rather than copied — comes back through JSON unchanged. If
serialisation, the schema, or a well-meaning default in `api/` ever shifts it,
that test fails and names the culprit. A second constant here would just be a
second thing to update in lockstep.

**A caller's mistake is never a 500.** Every malformed request below asserts the
status code explicitly, because "it returned an error" is not the requirement:
422 says *you sent something wrong and here is what*, 500 says *this service is
broken*, and a service that spends the second signal on typos has no way left to
report its own failures.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_inference import EXPECTED_PROBABILITY, FIXED_POSTING, FIXED_T

from api.main import create_app
from api.schemas import PostingRequest
from src.inference.contract import FIELDS


@pytest.fixture(scope="module")
def client(synthetic_artifact):
    with TestClient(create_app(synthetic_artifact)) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def modelless_client(tmp_path_factory):
    """An app whose artifact does not exist. The state a fresh deploy is in."""
    missing = tmp_path_factory.mktemp("empty") / "absent.joblib"
    with TestClient(create_app(missing)) as test_client:
        yield test_client


# --- health -----------------------------------------------------------------


def test_health_reports_the_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["dataset"] == "synthetic"
    assert body["horizon_days"] == 1
    assert 0.0 <= body["threshold"] <= 1.0


def test_health_is_200_even_with_no_model(modelless_client):
    """Up-but-empty is a different fact from unreachable, and both are useful."""
    body = modelless_client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "freeze" in body["detail"]


def test_predict_without_a_model_is_a_503_not_a_crash(modelless_client):
    response = modelless_client.post("/predict", json={"title": "Engineer"})
    assert response.status_code == 503
    assert "freeze" in response.json()["detail"]


# --- the prediction ---------------------------------------------------------


def test_the_api_returns_the_pinned_probability(client):
    """The number from `tests/test_inference.py`, through HTTP, unchanged."""
    response = client.post("/predict", json={**FIXED_POSTING, "as_of": FIXED_T})
    assert response.status_code == 200
    assert response.json()["probability"] == pytest.approx(EXPECTED_PROBABILITY, abs=1e-6)


def test_the_response_carries_what_makes_a_probability_readable(client):
    body = client.post("/predict", json={**FIXED_POSTING, "as_of": FIXED_T}).json()
    assert body["closing_soon"] == (body["probability"] >= body["threshold"])
    assert body["horizon_days"] == 1
    assert body["board_context_supplied"] is False
    assert body["model"] == "05-xgboost_engineered"
    assert body["dataset"] == "synthetic"


def test_board_context_is_reported_when_supplied(client):
    payload = {**FIXED_POSTING, "as_of": FIXED_T, "board_size_at_t": 300}
    assert client.post("/predict", json=payload).json()["board_context_supplied"] is True


def test_a_title_alone_is_a_valid_request(client):
    """Every optional field omitted. The ordinary case, not an edge case."""
    response = client.post("/predict", json={"title": "Engineer"})
    assert response.status_code == 200
    assert 0.0 <= response.json()["probability"] <= 1.0


def test_a_category_unseen_at_fit_time_does_not_crash(client):
    payload = {**FIXED_POSTING, "as_of": FIXED_T, "location": "Ulaanbaatar"}
    assert client.post("/predict", json=payload).status_code == 200


# --- bad input --------------------------------------------------------------


def test_a_missing_title_is_422(client):
    assert client.post("/predict", json={"location": "Berlin"}).status_code == 422


def test_an_empty_title_is_422(client):
    """`min_length=1`: a blank string is not a title, and four features read it."""
    assert client.post("/predict", json={"title": ""}).status_code == 422


def test_an_unknown_field_is_422_not_silently_dropped(client):
    """`extra="forbid"`. Accepting `salary` and ignoring it is the bad outcome."""
    response = client.post("/predict", json={"title": "Engineer", "salary": "lots"})
    assert response.status_code == 422
    assert "salary" in response.text


def test_a_wrongly_typed_field_is_422(client):
    response = client.post("/predict", json={"title": "Engineer", "content_chars": "quite long"})
    assert response.status_code == 422


def test_a_negative_count_is_422(client):
    assert client.post("/predict", json={"title": "E", "n_offices": -1}).status_code == 422


def test_a_non_finite_number_is_422_from_the_contract(client):
    """The path that reaches `InvalidPayload` rather than pydantic.

    `Infinity` is valid JSON to Python's parser and passes `ge=0`, so it arrives
    at the contract intact — which is exactly what the exception handler is for.
    Without it this request would be a 500 raised out of numpy.
    """
    response = client.post(
        "/predict",
        content='{"title": "Engineer", "n_offices": Infinity}',
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422


def test_a_malformed_body_is_422(client):
    response = client.post(
        "/predict", content="not json at all", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


# --- the schema is the audit ------------------------------------------------


def test_the_request_schema_is_exactly_the_contract():
    """A field in one and not the other is a feature nobody can send, or a field
    that reaches nothing. `as_of` is the documented exception: it is the
    prediction instant, an axis, never an input."""
    assert set(PostingRequest.model_fields) - {"as_of"} == {field.name for field in FIELDS}


def test_the_contract_endpoint_publishes_the_audit(client):
    rows = client.get("/contract").json()
    assert {row["field"] for row in rows} == {field.name for field in FIELDS}
    assert all(row["serve-time availability"] for row in rows)


def test_the_api_contains_no_feature_logic():
    """`api/` may not import the feature modules.

    The requirement is that serving uses the *same fitted pipeline object* as
    training. The way that requirement dies is quietly: someone needs one
    derived column in a handler, imports `src.features.derive`, and now there are
    two implementations that agree until one of them changes. Importing
    `src.inference` is fine — that is the artifact and its contract.
    """
    forbidden = {"src.features", "src.models", "src.data"}
    for path in sorted(Path("api").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(node.module.startswith(prefix) for prefix in forbidden), (
                    f"{path} imports {node.module}: serving must go through the "
                    "frozen artifact, not the feature modules"
                )
