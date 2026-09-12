"""The service benchmark, in-process: the same functions, no socket, a light pass."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.protection import TokenBuckets
from benchmarks import service


def generous() -> TokenBuckets:
    """A bucket that never refuses: these tests are about the routes, not the
    limiter, which has its own tests in `test_protection.py`."""
    return TokenBuckets(burst=1_000_000, per_minute=6e7)


@pytest.fixture(scope="module")
def in_process(synthetic_artifact):
    client = TestClient(create_app(synthetic_artifact, buckets=generous()))
    client.__enter__()

    def post(path, body):
        started = time.perf_counter()
        response = client.post(path, json=body)
        seconds = time.perf_counter() - started
        try:
            parsed = response.json()
        except ValueError:
            parsed = None
        return response.status_code, seconds, parsed

    def get_health():
        return client.get("/health").json()

    yield post, get_health
    client.__exit__(None, None, None)


def test_every_section_passes_in_process(in_process):
    """Concurrency, latency shape, memory, abuse and the burst — all against
    the real app with the synthetic artifact, small enough for CI."""
    post, get_health = in_process
    batch = int(get_health()["rank_max_batch"])
    sections = service.run(post, get_health, batch, concurrency=4, light=True)
    failed = [s for s in sections if not s.passed]
    assert not failed, "\n".join(f"{s.name}: {s.detail}" for s in failed)
    names = {s.name for s in sections}
    assert any("concurrent /rank" in n for n in names)
    assert any("malformed" in n for n in names)
    assert any("no budget" in n for n in names)


def test_abuse_is_always_a_4xx(in_process):
    post, get_health = in_process
    section = service.malformed_and_oversized(post, int(get_health()["rank_max_batch"]))
    assert section.passed, section.detail
    assert "5 MB title" in section.detail and "2,000-posting body" in section.detail


def test_the_budget_section_sees_a_refusal_against_a_real_bucket(synthetic_artifact):
    """The shared fixture never refuses; this one runs the budget section
    against the default bucket, where a burst past BURST must meet a 429."""
    client = TestClient(create_app(synthetic_artifact))
    client.__enter__()
    try:

        def post(path, body):
            started = time.perf_counter()
            response = client.post(path, json=body)
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
            return response.status_code, time.perf_counter() - started, parsed

        section = service.budget_on_an_expensive_route(post, batch=250)
        assert section.passed, section.detail
        assert section.numbers["refused_429"] >= 1
    finally:
        client.__exit__(None, None, None)


def test_the_report_renders_with_a_failure_visible():
    sections = [
        service.Section("a", True, "fine", {"p50_ms": 1.0}),
        service.Section("b", False, "5xx seen", {"p50_ms": 2.0, "growth_mb": 60.0}),
    ]
    text = service.render(sections, "http://x", "test", {"model": "m", "dataset": "synthetic"})
    assert "| b | **NO** | 5xx seen |" in text
    assert "## What is deliberately not here" in text
