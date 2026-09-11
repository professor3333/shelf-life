"""The cohort audit: what age encodes, and where a score would come from.

The fixtures here are built by hand rather than from `panels.make_panel`, whose
postings are all present at wave 0 — an all-incumbent panel is exactly the
shape this audit exists to look past.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from src.data.cohort_audit import (
    INCIDENT,
    INCUMBENT,
    _age_verdict,
    _cohort_verdict,
    age_distribution,
    age_rank,
    age_rank_by_slice,
    annotate,
    cohort_table,
    labelled,
    render,
    rolling_age_rank,
    runs_seen_table,
    validation_block,
)

DAY = pd.Timedelta(days=1)
WAVE0 = pd.Timestamp("2026-08-31T03:45:00Z")


def _panel(postings: dict[str, tuple[int, int, int]], n_waves: int, age0: float = 30.0):
    """`postings` maps an id to (first wave, last wave, y-on-every-labelled-row).

    Age runs from `age0` at the posting's first wave for incumbents and from 0
    for incidents, so the two cohorts' age ranges are separated the way a real
    panel's are unless a test says otherwise.
    """
    rows = []
    for source_id, (first, last, y) in postings.items():
        base_age = age0 if first == 0 else 0.0
        for wave in range(first, last + 1):
            rows.append(
                {
                    "source": "greenhouse:acme",
                    "source_id": source_id,
                    "run_index": wave,
                    "t": WAVE0 + wave * DAY,
                    "y": y,
                    "label_observable": wave < n_waves - 1,
                    "age_days": base_age + (wave - first),
                    "horizon_days": 1,
                    "horizon_basis": "calendar",
                }
            )
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("Int8")
    return frame


# --- the columns everything is cut on ---------------------------------------


def test_cohort_and_runs_seen_are_as_of_t():
    frame = annotate(_panel({"stock": (0, 4, 0), "late": (2, 4, 0)}, n_waves=5))
    stock = frame[frame["source_id"] == "stock"].sort_values("run_index")
    late = frame[frame["source_id"] == "late"].sort_values("run_index")

    assert set(stock["cohort"]) == {INCUMBENT}
    assert set(late["cohort"]) == {INCIDENT}
    # Counted up to the row, never over the posting's life.
    assert stock["runs_seen"].tolist() == [1, 2, 3, 4, 5]
    assert late["runs_seen"].tolist() == [1, 2, 3]
    # The first-observation flag marks the day an *incident* posting appeared.
    # An incumbent's wave-0 row is the day the panel first looked, not that.
    assert not stock["first_observation"].any()
    assert late["first_observation"].tolist() == [True, False, False]


def test_incumbency_is_per_board():
    """A board whose first complete run is later than another's has its own wave 0."""
    a = _panel({"a": (0, 3, 0)}, n_waves=4)
    b = _panel({"b": (1, 3, 0)}, n_waves=4).assign(source="greenhouse:other")
    frame = annotate(pd.concat([a, b], ignore_index=True))
    assert set(frame.loc[frame["source_id"] == "b", "cohort"]) == {INCUMBENT}


# --- the bug's signature is one table ---------------------------------------


def test_a_label_that_keys_on_arrival_is_called_out():
    """What 2026-09-11 looked like: every incident row positive, no incumbent
    row positive, and age separating the two perfectly. The verdict has to say
    the label is not indifferent to cohort — that is the whole point of the
    file existing."""
    postings = {f"stock-{i}": (0, 5, 0) for i in range(20)}
    postings |= {f"late-{i}": (2, 5, 1) for i in range(5)}
    rows = labelled(annotate(_panel(postings, n_waves=6)))

    verdict = _cohort_verdict(cohort_table(rows))
    assert "not indifferent to cohort" in verdict
    assert "2026-09-11 it was a bug" in verdict


def test_a_label_indifferent_to_cohort_is_said_to_be():
    postings = {f"stock-{i}": (0, 5, int(i < 2)) for i in range(20)}
    postings |= {f"late-{i}": (2, 5, int(i < 1)) for i in range(10)}
    rows = labelled(annotate(_panel(postings, n_waves=6)))
    assert "not keying on which population" in _cohort_verdict(cohort_table(rows))


def test_non_overlapping_age_ranges_mean_age_is_cohort():
    postings = {f"stock-{i}": (0, 5, 0) for i in range(5)} | {"late": (2, 5, 0)}
    rows = labelled(annotate(_panel(postings, n_waves=6, age0=100.0)))
    dist = age_distribution(rows)
    assert "do not overlap" in _age_verdict(dist)

    overlapping = labelled(annotate(_panel(postings, n_waves=6, age0=1.0)))
    assert "overlap (" in _age_verdict(age_distribution(overlapping))


# --- what age alone can do --------------------------------------------------


def test_age_rank_scores_both_directions_and_refuses_degenerate_blocks():
    block = pd.DataFrame({"y": [1, 1, 0, 0], "age_days": [1.0, 2.0, 30.0, 40.0]})
    scores = age_rank(block)
    assert scores["young_first"] == pytest.approx(1.0)
    assert scores["old_first"] < scores["young_first"]
    assert scores["base_rate"] == pytest.approx(0.5)

    for y in ([0, 0, 0], [1, 1, 1]):
        degenerate = age_rank(pd.DataFrame({"y": y, "age_days": [1.0, 2.0, 3.0]}))
        assert np.isnan(degenerate["young_first"]) and np.isnan(degenerate["old_first"])
    assert age_rank(pd.DataFrame({"y": [], "age_days": []}))["n"] == 0


def test_missing_ages_are_ranked_oldest_not_dropped():
    block = pd.DataFrame({"y": [1, 0, 0], "age_days": [np.nan, 5.0, 50.0]})
    scores = age_rank(block)
    assert scores["n"] == 3
    # Unknown is old: young-first puts the positive last, old-first puts it first.
    assert scores["old_first"] > scores["young_first"]


def test_slices_include_the_first_observation_rows():
    postings = {f"stock-{i}": (0, 3, int(i == 0)) for i in range(6)}
    postings |= {"late": (1, 3, 1)}  # waves 1 and 2 labelled; wave 3 is the unsettled tail
    val = labelled(annotate(_panel(postings, n_waves=4))).assign(seen_in_train=True)
    val.loc[val["source_id"] == "late", "seen_in_train"] = False
    table = age_rank_by_slice(val).set_index("slice")

    assert table.loc["first observation only", "n"] == 1
    assert table.loc["first observation only", "positives"] == 1
    assert table.loc[f"{INCIDENT} only", "n"] == 2
    assert table.loc["unseen in training", "n"] == 2
    assert table.loc["whole block", "n"] == len(val)


# --- the report, with and without a legal cut ---------------------------------


def test_runs_seen_table_never_uses_the_lifetime_count():
    postings = {"stock": (0, 5, 0), "late": (3, 5, 0)}
    table = runs_seen_table(labelled(annotate(_panel(postings, n_waves=6)))).set_index("runs_seen")
    # `stock` contributes one row per count 1..5; `late` contributes 1..2.
    assert table.loc[1, "rows"] == 2
    assert table.loc[5, "rows"] == 1


def test_a_shallow_panel_renders_without_a_validation_block():
    postings = {"stock": (0, 2, 0), "late": (1, 2, 1)}
    frame = _panel(postings, n_waves=3)
    assert validation_block(annotate(frame)) is None
    text = render(frame)
    assert "No legal three-way cut exists at this depth" in text
    assert "## 1. Incumbent stock against incident flow" in text
    assert "## 7." not in text


def test_the_rolling_table_walks_every_legal_cut_and_reads_no_held_out_rows():
    sys.path.insert(0, "tests")
    from panels import make_panel

    frame = annotate(make_panel(n_waves=12, random_labels=True))
    table = rolling_age_rank(frame)
    assert not table.empty
    assert table["val_end"].is_monotonic_increasing
    assert {"first_obs_rows", "incident_rows", "young_first", "old_first"} <= set(table.columns)
    # Every validation block ends where its cut says; nothing later is scored.
    assert (table["n"] > 0).all()


def test_render_on_a_deep_panel_carries_every_section():
    sys.path.insert(0, "tests")
    from panels import make_panel

    text = render(make_panel(n_waves=12, random_labels=True))
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert heading in text
    assert "design.md` §15" in text


def test_first_observation_is_attached_by_key_not_by_index():
    """`temporal_split` renumbers its blocks. An index-aligned lookup against the
    panel picks other rows and raises nothing — measured 2026-09-11: 28 first
    observations by index against 2 by key on the same block."""
    from src.data.cohort_audit import attach_first_observation

    panel = _panel({"stock": (0, 3, 0), "late": (2, 3, 0)}, n_waves=4)
    block = panel[panel["run_index"] == 2].reset_index(drop=True)  # renumbered, like a split
    out = attach_first_observation(block, panel)
    by_id = dict(zip(out["source_id"], out["first_observation"], strict=True))
    assert by_id == {"stock": False, "late": True}
