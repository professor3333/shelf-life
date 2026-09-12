"""The two courtesy limits: a body cap before parsing, a per-address budget on expensive routes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from test_inference import FIXED_POSTING, FIXED_T

from api import protection
from api.main import create_app
from api.protection import TokenBuckets


@pytest.fixture
def tight(synthetic_artifact):
    """Three expensive requests at once, then one every ten seconds."""
    with TestClient(
        create_app(synthetic_artifact, buckets=TokenBuckets(burst=3, per_minute=6))
    ) as c:
        yield c


def _postings(n):
    return [{**FIXED_POSTING, "title": f"T {i}"} for i in range(n)]


def test_the_bucket_refills_at_its_rate():
    buckets = TokenBuckets(burst=2, per_minute=60)  # one a second
    assert buckets.take("a", now=0.0) == 0.0
    assert buckets.take("a", now=0.0) == 0.0
    wait = buckets.take("a", now=0.0)
    assert wait == pytest.approx(1.0)
    assert buckets.take("a", now=1.0) == 0.0, "one token back after a second"
    assert buckets.take("b", now=0.0) == 0.0, "another address has its own bucket"


def test_expensive_routes_are_named_and_cheap_ones_are_not():
    assert protection.is_expensive("/rank", "POST")
    assert protection.is_expensive("/boards/abc/score", "POST")
    assert protection.is_expensive("/boards/abc/rank", "POST")
    assert not protection.is_expensive("/boards/abc/postings", "POST")
    assert not protection.is_expensive("/predict", "POST")
    assert not protection.is_expensive("/health", "GET")
    assert not protection.is_expensive("/rank", "GET")


def test_the_budget_answers_429_with_retry_after_and_leaves_cheap_routes_alone(tight):
    body = {"postings": _postings(3), "as_of": FIXED_T}
    codes = [tight.post("/rank", json=body).status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    refused = tight.post("/rank", json=body)
    assert "Retry-After" in refused.headers and int(refused.headers["Retry-After"]) >= 1
    assert "no key to raise this" in refused.json()["detail"]
    # Cheap routes are untouched by the budget.
    assert tight.get("/health").status_code == 200
    assert tight.post("/predict", json={**FIXED_POSTING, "as_of": FIXED_T}).status_code == 200


def test_an_oversized_body_is_refused_before_it_is_parsed(tight):
    huge = "x" * (protection.MAX_BODY_BYTES + 10)
    response = tight.post("/predict", content=huge, headers={"content-type": "application/json"})
    assert response.status_code == 413
    assert "send fewer postings" in response.json()["detail"]


def test_the_body_limit_admits_a_full_page_at_the_text_cap():
    """The limit is sized from the largest legitimate request, twice over."""
    import json

    from api.schemas import TEXT_MAX
    from src.inference.predict import MAX_BATCH

    page = {
        "postings": [
            {
                **{
                    k: "x" * TEXT_MAX
                    for k in (
                        "title",
                        "location",
                        "salary_raw",
                        "departments",
                        "offices",
                        "company",
                        "source",
                    )
                }
            }
            for _ in range(MAX_BATCH)
        ]
    }
    assert len(json.dumps(page)) < protection.MAX_BODY_BYTES


def test_the_address_behind_the_proxy_is_the_first_forwarded_hop(tight):
    body = {"postings": _postings(2), "as_of": FIXED_T}
    for _ in range(3):
        tight.post("/rank", json=body, headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"})
    blocked = tight.post("/rank", json=body, headers={"x-forwarded-for": "203.0.113.5, 10.0.0.1"})
    other = tight.post("/rank", json=body, headers={"x-forwarded-for": "198.51.100.7"})
    assert blocked.status_code == 429 and other.status_code == 200
