"""Tests for the board verification. Nothing here touches the network.

The fetcher is injected, so every case below is a scripted board: what happens
when a posting is listed, when it is gone, when its title reappears under a new
id, and when the source cannot be asked at all. The one thing a live check could
not give reliably is a *failing* instrument, and that is the case worth pinning
— see the first test.
"""

from __future__ import annotations

import json

import pandas as pd

from src.data import label_check


def _board(*jobs: tuple[str, str]) -> tuple[int, bytes]:
    payload = {"jobs": [{"id": int(i), "title": t} for i, t in jobs]}
    return 200, json.dumps(payload).encode()


def _sample(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- the instrument that would have agreed with anything --------------------


def test_a_board_that_answers_nothing_is_unverifiable_not_removed():
    """The failure this guards is the one that looks like success.

    An instrument that reports *gone* whenever it cannot see a posting agrees
    perfectly with any label, including a wrong one. The first Greenhouse
    instrument tried here did exactly that — the posting URL answers 200 and
    redirects to the board index for live and dead jobs alike — so a source that
    cannot be asked must produce `unverifiable`, never a verdict.
    """

    def refuse(url: str) -> tuple[int, bytes]:
        return 503, b""

    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    results = label_check.verify(sample, get=refuse, delay=0)
    assert results.iloc[0]["verdict"] == "unverifiable"
    assert results.iloc[0]["listed_now"] is None


def test_unparseable_json_is_unverifiable_too():
    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    results = label_check.verify(sample, get=lambda url: (200, b"<html>"), delay=0)
    assert results.iloc[0]["verdict"] == "unverifiable"


# --- the verdicts -----------------------------------------------------------


def test_a_posting_absent_from_the_board_is_a_genuine_removal():
    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    results = label_check.verify(sample, get=lambda url: _board(("2", "Other")), delay=0)
    assert results.iloc[0]["verdict"] == "genuine removal"


def test_a_posting_still_listed_is_a_label_error():
    """Ids are not reused, so a posting listed now was not removed then."""
    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    results = label_check.verify(sample, get=lambda url: _board(("1", "Eng")), delay=0)
    assert results.iloc[0]["verdict"] == "still listed — label wrong"


def test_a_title_reappearing_under_a_new_id_is_a_relisting():
    """Still a removal of that posting — and the clearest case of removed != filled."""
    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    results = label_check.verify(sample, get=lambda url: _board(("9", "Eng")), delay=0)
    assert results.iloc[0]["verdict"] == "removed, title relisted"
    assert "9" in results.iloc[0]["note"]


def test_the_control_arm_gets_its_own_verdicts():
    """A control posting is not scored against the label; it measures drift."""
    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": "1",
                "url": "u",
                "arm": "still listed at last crawl",
                "title": "Eng",
            },
            {
                "source": "greenhouse:acme",
                "source_id": "7",
                "url": "u",
                "arm": "still listed at last crawl",
                "title": "Gone",
            },
        ]
    )
    results = label_check.verify(sample, get=lambda url: _board(("1", "Eng")), delay=0)
    assert set(results["verdict"]) == {"still listed", "gone since the last crawl"}


def test_one_request_per_board_however_many_postings():
    """The politeness property, asserted rather than intended.

    A hundred postings from one board must cost one request, not a hundred.
    """
    calls: list[str] = []

    def counting(url: str) -> tuple[int, bytes]:
        calls.append(url)
        return _board(("1", "Eng"))

    sample = _sample(
        [
            {
                "source": "greenhouse:acme",
                "source_id": str(i),
                "url": "u",
                "arm": "removed",
                "title": "Eng",
            }
            for i in range(25)
        ]
    )
    label_check.verify(sample, get=counting, delay=0)
    assert len(calls) == 1, f"asked the board {len(calls)} times for 25 postings"


def test_python_org_uses_the_status_code():
    sample = _sample(
        [
            {
                "source": "python_org",
                "source_id": "1",
                "url": "https://www.python.org/jobs/1/",
                "arm": "removed",
                "title": "Eng",
            }
        ]
    )
    gone = label_check.verify(sample, get=lambda url: (404, b""), delay=0)
    assert gone.iloc[0]["verdict"] == "genuine removal"
    live = label_check.verify(sample, get=lambda url: (200, b"x"), delay=0)
    assert live.iloc[0]["verdict"] == "still listed — label wrong"


# --- the interval -----------------------------------------------------------


def test_the_interval_cannot_exceed_one():
    """Why Wilson and not the normal approximation.

    59 of 60 is the number this check reports, and it sits close enough to the
    edge that the normal interval runs past 1 — which is not a bound on a
    proportion.
    """
    low, high = label_check.wilson(59, 60)
    assert 0.0 <= low < high <= 1.0
    assert high < 1.0, "an interval that includes certainty from one dissent is not an interval"
    assert low > 0.85


def test_the_interval_is_undefined_without_trials():
    low, high = label_check.wilson(0, 0)
    assert pd.isna(low) and pd.isna(high)


# --- sampling ---------------------------------------------------------------


def test_both_arms_are_sampled_because_one_arm_proves_nothing():
    """Agreement on the positives alone is compatible with an instrument that
    answers *gone* for everything."""
    panel = pd.DataFrame(
        {
            "source": ["greenhouse:a"] * 6,
            "source_id": list("123456"),
            "url": ["u"] * 6,
            "title": ["t"] * 6,
            "y": [1, 1, 1, 0, 0, 0],
            "label_observable": [True] * 6,
        }
    )
    alive = {"greenhouse:a|4", "greenhouse:a|5", "greenhouse:a|6"}
    sample = label_check.sample_for_check(panel, alive, per_arm=10)
    assert set(sample["arm"]) == {"removed", "still listed at last crawl"}
