"""Tests for the label-validity audit.

Two of these exist because the module made the mistake they now forbid. The
first version returned a bare relisting rate, which reads as a label-error rate
and is not one. The second controlled the rate and then over-read the
difference, calling 12.0%-against-11.0% *elevated* on twenty events. Both are
the same failure at different levels: a number stated with more confidence than
its denominator supports.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.label_audit import (
    Comparison,
    board_stability,
    closure_dispersion,
    compare_relisting,
    render,
)

WAVE0 = pd.Timestamp("2026-08-31T03:45:00Z")
DAY = pd.Timedelta(days=1)


def _panel(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("Int8")
    return frame


def _row(source_id, wave, *, y=0, observable=True, req=None, title="Engineer", company="Acme"):
    return {
        "source": "greenhouse:acme",
        "source_id": source_id,
        "t": WAVE0 + wave * DAY,
        "y": y,
        "label_observable": observable,
        "requisition_id": req,
        "title": title,
        "company": company,
    }


# --- the verdict, which is the whole point -----------------------------------


def test_a_difference_inside_the_noise_is_not_called_elevated():
    """The real numbers that exposed this: 12/100 against 8/73. One point of
    separation on twenty events was reported as contamination of the target,
    with a recommendation to change the labelling rule."""
    comparison = Comparison("title", closed_hits=12, closed_n=100, open_hits=8, open_n=73)
    assert comparison.verdict == "indistinguishable"


def test_a_real_elevation_is_still_called():
    """A guard that can only say 'indistinguishable' is not a guard."""
    comparison = Comparison("requisition_id", closed_hits=60, closed_n=100, open_hits=5, open_n=100)
    assert comparison.verdict == "elevated"


def test_a_rate_below_its_control_is_named_as_such_not_as_absence():
    """'No elevation' and 'below control' are different facts, and collapsing
    them hides the case where closed postings relist *less* than survivors."""
    comparison = Comparison("requisition_id", closed_hits=2, closed_n=100, open_hits=40, open_n=100)
    assert comparison.verdict == "below control"


def test_an_empty_arm_is_not_assessable():
    assert Comparison("title", 0, 0, 3, 50).verdict == "not assessable"
    assert Comparison("title", 3, 50, 0, 0).verdict == "not assessable"


def test_the_same_rate_in_both_arms_is_never_elevated():
    for n in (10, 100, 10_000):
        assert Comparison("title", n // 10, n, n // 10, n).verdict == "indistinguishable"


def test_the_bar_tightens_as_counts_grow():
    """The same one-point difference is noise at 100 and a finding at 100,000 —
    which is the property that makes this audit sharpen as closures accumulate
    rather than needing to be rewritten."""
    small = Comparison("title", 12, 100, 11, 100)
    large = Comparison("title", 12_000, 100_000, 11_000, 100_000)
    assert small.verdict == "indistinguishable"
    assert large.verdict == "elevated"


# --- the measurement ---------------------------------------------------------


def test_a_relisting_under_a_new_id_is_counted():
    panel = _panel(
        [
            _row("old", 0, y=1, req="R-1"),
            _row("new", 1, y=0, req="R-1"),
            _row("stays", 0, y=0, req="R-2"),
            _row("stays", 1, y=0, req="R-2"),
        ]
    )
    assert compare_relisting(panel, "requisition_id").closed_hits == 1


def test_the_same_posting_reappearing_is_not_a_relisting():
    """A posting coming back under its *own* id is a resurrection, which the
    corroboration rule in `compute_labels` already handles. This audit is about
    the case that rule cannot see: a new id carrying the same role."""
    panel = _panel([_row("same", 0, y=1, req="R-1"), _row("same", 2, y=0, req="R-1")])
    assert compare_relisting(panel, "requisition_id").closed_hits == 0


def test_a_relisting_after_the_window_does_not_count():
    panel = _panel([_row("old", 0, y=1, req="R-1"), _row("new", 5, y=0, req="R-1")])
    assert compare_relisting(panel, "requisition_id").closed_hits == 0


def test_a_missing_requisition_id_is_excluded_rather_than_counted_as_no_match():
    """Otherwise the denominator silently absorbs every posting the employer
    never keyed, and the rate falls for a reason that has nothing to do with
    relisting."""
    panel = _panel(
        [_row("a", 0, y=1, req=None), _row("b", 0, y=1, req="R-1"), _row("c", 1, y=0, req="R-1")]
    )
    comparison = compare_relisting(panel, "requisition_id")
    assert comparison.closed_n == 1 and comparison.closed_hits == 1


def test_a_title_match_needs_the_company_to_agree():
    panel = _panel(
        [
            _row("old", 0, y=1, title="Engineer", company="Acme"),
            _row("other", 1, y=0, title="Engineer", company="Umbrella"),
        ]
    )
    assert compare_relisting(panel, "title").closed_hits == 0


def test_the_control_arm_is_drawn_from_the_same_crawl_instants():
    """Otherwise the two arms come from different weeks of a growing panel and
    the comparison measures the calendar."""
    panel = _panel(
        [_row("closed", 0, y=1, req="R-1")]
        + [_row(f"open{i}", 3, y=0, req=f"S-{i}") for i in range(5)]
    )
    assert compare_relisting(panel, "requisition_id").open_n == 0


# --- the surrounding tables ---------------------------------------------------


def test_board_stability_reports_a_cliff_as_a_fall():
    panel = _panel([_row(f"a{i}", 0) for i in range(100)] + [_row(f"a{i}", 1) for i in range(10)])
    assert board_stability(panel)["worst_daily_fall"].iloc[0] == "-90.0%"


def test_closure_dispersion_is_empty_when_nothing_has_closed():
    assert closure_dispersion(_panel([_row("a", 0)])).empty


def _both_arms_panel() -> pd.DataFrame:
    """A closed posting and survivors sharing its crawl instant, so both arms of
    the comparison are populated and the report reaches its verdict line."""
    # The control is matched on *last sighting*, not on presence, so survivors
    # must have their final observed row at the same instant as the closure —
    # give them a row at wave 1 and they no longer match wave 0 and the arm
    # empties. That matching is the point: it keeps the two arms drawn from the
    # same board state instead of from different weeks of a growing panel.
    return _panel(
        [_row("closed", 0, y=1, req="R-1"), _row("relisted", 1, y=0, req="R-1")]
        + [_row(f"open{i}", 0, y=0, req=f"S-{i}") for i in range(5)]
    )


def _flat(text: str) -> str:
    """Whitespace-normalised, because the report is wrapped prose and a phrase
    may straddle a line break."""
    return " ".join(text.lower().split())


def test_the_report_never_counts_an_outcome_it_cannot_observe():
    """The row this audit deliberately does not have. 'Filled' is not observable
    from outside the employer, and a count of it would be the strongest-sounding
    number in the repository resting on nothing."""
    text = _flat(render(_both_arms_panel()))
    assert "not distinguishable" in text
    for invented in ("actually filled", "| filled |", "cancelled:"):
        assert invented not in text


def test_the_report_states_the_arithmetic_behind_every_verdict():
    """A verdict without its numbers is an opinion, and this file exists because
    an opinion got mistaken for a measurement twice."""
    text = render(_both_arms_panel())
    assert "±" in text
    assert "indistinguishable" in text


@pytest.mark.parametrize("key", ["requisition_id", "title"])
def test_an_empty_panel_does_not_raise(key):
    empty = _panel([_row("a", 0)])
    assert compare_relisting(empty, key).closed_n == 0


# --- closures against observed lifespan --------------------------------------


def test_closures_piled_onto_postings_the_crawl_barely_held_are_called_out():
    """The measurement that stopped the H=7 target on 2026-09-10.

    A posting seen once and never again is, in the panel, indistinguishable from
    one filled the next morning. When *every* short-lived posting is a closure
    and the settled ones almost never are, the target has stopped being about
    hiring — and `age_days`, which tracks exactly how long a posting has been
    around, will predict it beautifully and mean nothing.
    """
    from src.data.label_audit import lifespan_verdict

    rows = []
    for short in range(8):  # seen once, closed
        rows.append(_row(f"brief-{short}", 0, y=1))
    for settled in range(8):  # seen in every run, never closed
        rows += [_row(f"settled-{settled}", wave) for wave in range(8)]

    verdict = lifespan_verdict(_panel(rows))
    assert "dominated by postings that barely existed" in verdict
    assert "must not be read as model quality" in verdict


def test_closures_spread_across_lifespans_are_not_called_out():
    """A check that can only say 'contaminated' is not a check.

    Closures here fall on postings the crawl held throughout, which is what the
    label is supposed to be measuring.
    """
    from src.data.label_audit import lifespan_verdict

    rows = []
    for i in range(10):
        rows += [_row(f"settled-{i}", wave, y=1 if wave == 7 and i < 3 else 0) for wave in range(8)]
    rows += [_row("brief", 0, y=0)]

    verdict = lifespan_verdict(_panel(rows))
    assert "Flat enough" in verdict


def test_the_lifespan_table_counts_rows_not_postings():
    """The denominator is job-days, like every other rate in the report."""
    from src.data.label_audit import lifespan_concentration

    rows = [_row("brief", 0, y=1)] + [_row("settled", wave) for wave in range(8)]
    table = lifespan_concentration(_panel(rows))
    by_bucket = dict(zip(table["seen in"], table["rows"], strict=True))
    assert by_bucket["1 run"] == 1
    assert by_bucket["6+ runs"] == 8
