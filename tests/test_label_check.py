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


# --- the paths the live run takes and the scripted boards above never did -------


def test_the_real_fetcher_returns_the_status_and_body_from_a_local_server():
    """`_get` against a server on localhost: a 200 with its body, a 404 as the
    status alone. No external network; the point is that the HTTPError path
    returns a status rather than raising, because 404 is python.org's *answer*."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/live":
                body = b'{"jobs": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def log_message(self, *args):  # keep the test output quiet
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert label_check._get(f"{base}/live") == (200, b'{"jobs": []}')
        assert label_check._get(f"{base}/gone") == (404, b"")
    finally:
        server.shutdown()
        server.server_close()


def test_python_org_answering_anything_but_200_or_404_is_unverifiable():
    """A 503 is not *gone*; reading it as gone would agree with the label for free."""
    import pytest

    with pytest.raises(label_check.Unverifiable, match="503"):
        label_check.python_org_listed("https://python.org/jobs/1/", lambda url: (503, b""))


def test_a_source_with_no_instrument_is_recorded_as_such_not_guessed():
    sample = _sample(
        [{"source": "arbeitnow", "source_id": "x", "url": "u", "arm": "removed", "title": "t"}]
    )
    results = label_check.verify(sample, get=lambda url: (200, b""), delay=0)
    assert results.iloc[0]["verdict"] == "unverifiable"
    assert results.iloc[0]["note"] == "no instrument"
    assert results.iloc[0]["listed_now"] is None or pd.isna(results.iloc[0]["listed_now"])


def test_sampling_is_proportional_to_each_sources_share_of_the_arm():
    """Sixty removed postings, three quarters of them from one board: the sample
    keeps that ratio rather than taking one per source, and never exceeds the
    arm size. Seeded, so the draw is the same every run."""
    panel = pd.DataFrame(
        {
            "source": ["greenhouse:big"] * 45 + ["python_org"] * 15,
            "source_id": [str(i) for i in range(60)],
            "url": ["u"] * 60,
            "title": ["t"] * 60,
            "y": [1] * 60,
            "label_observable": [True] * 60,
        }
    )
    sample = label_check.sample_for_check(panel, alive_at_final_crawl=set(), per_arm=20)
    assert set(sample["arm"]) == {"removed"}  # no survivor was alive: no control arm
    assert len(sample) == 20
    shares = sample["source"].value_counts()
    assert shares["greenhouse:big"] == 15 and shares["python_org"] == 5
    again = label_check.sample_for_check(panel, alive_at_final_crawl=set(), per_arm=20)
    assert sample["source_id"].tolist() == again["source_id"].tolist()


def test_summarise_gives_each_verdict_its_share_of_its_arm():
    results = pd.DataFrame(
        {
            "label": ["removed"] * 4 + ["still listed at last crawl"] * 2,
            "verdict": ["genuine removal"] * 3
            + ["still listed — label wrong"]
            + ["still listed"] * 2,
        }
    )
    table = label_check.summarise(results).set_index(["label", "verdict"])
    assert table.loc[("removed", "genuine removal"), "share"] == 0.75
    assert table.loc[("removed", "genuine removal"), "arm_total"] == 4
    assert table.loc[("still listed at last crawl", "still listed"), "share"] == 1.0
    assert label_check.summarise(pd.DataFrame()).empty


def test_the_report_counts_a_relisted_title_as_a_removal_and_excludes_the_unverifiable(
    tmp_path,
):
    """The rate is `gone / checked` where gone includes relistings (the posting
    left under its own id) and checked excludes what could not be asked. 9 of 10
    with one unverifiable: 90.0%, and the Wilson interval beside it."""
    results = pd.DataFrame(
        [
            *[dict(label="removed", verdict="genuine removal")] * 7,
            *[dict(label="removed", verdict="removed, title relisted")] * 2,
            dict(label="removed", verdict="still listed — label wrong"),
            dict(label="removed", verdict="unverifiable"),
            *[dict(label="still listed at last crawl", verdict="still listed")] * 8,
            *[dict(label="still listed at last crawl", verdict="gone since the last crawl")] * 2,
        ]
    )
    out = tmp_path / "label_check.md"
    label_check.write_report(out, results, prov=None, horizon="H=7", sampled=len(results))
    text = out.read_text()
    low, high = label_check.wilson(9, 10)
    assert "Of 10 postings the label calls removed, **9 are gone from the board" in text
    assert f"90.0% (95% Wilson {low:.1%}–{high:.1%})" in text
    assert "2 of those 9 have the same title listed under a *different* id" in text
    assert "1 is still listed under its own id" in text
    assert "1 could not be checked and are excluded from the rate." in text
    assert "8 are still listed and 2 have gone since — a drift of 20.0%" in text
    assert "Provenance | **not recorded**" in text
    assert "| Horizon | H=7 |" in text


def test_the_report_with_nothing_checked_says_so(tmp_path):
    out = tmp_path / "label_check.md"
    empty = pd.DataFrame(columns=["label", "verdict"])
    label_check.write_report(out, empty, prov=None, horizon=None, sampled=0)
    text = out.read_text()
    assert "_Nothing checked._" in text
    assert "### The positives" not in text and "### The control" not in text
