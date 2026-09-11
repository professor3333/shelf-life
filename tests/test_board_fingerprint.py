"""Can the production features name the board, with board identity removed?"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

sys.path.insert(0, "tests")
from panels import make_panel  # noqa: E402

from src.data.split import best_cuts, temporal_split  # noqa: E402
from src.models.board_fingerprint import (  # noqa: E402
    full_matrix,
    majority_baseline,
    missingness_only,
    per_feature,
    render,
    without_board_context,
)


def _two_boards(distinguishable: bool) -> pd.DataFrame:
    """Two boards with identical postings — except, when asked, for board size.

    `make_panel` builds one board. The second is the same board renamed, with
    its ids offset so the two do not share postings; when `distinguishable`,
    its `board_size_at_t` is shifted by a constant, which is the shape of the
    real fingerprint (a board's size is its name in integer form).
    """
    a = make_panel(n_waves=10, random_labels=True, seed=0)
    b = make_panel(n_waves=10, random_labels=True, seed=1).assign(source="greenhouse:other")
    b["source_id"] = "q" + b["source_id"].astype(str)
    if distinguishable:
        b["board_size_at_t"] = b["board_size_at_t"] + 500.0
    return pd.concat([a, b], ignore_index=True)


@pytest.fixture(scope="module")
def distinguishable_split():
    panel = _two_boards(distinguishable=True)
    return temporal_split(panel, best_cuts(panel))


@pytest.fixture(scope="module")
def identical_split():
    panel = _two_boards(distinguishable=False)
    return temporal_split(panel, best_cuts(panel))


def test_a_board_written_into_a_feature_is_recovered(distinguishable_split):
    scores, per_board, confusion = full_matrix(distinguishable_split)
    assert scores["accuracy"] == pytest.approx(1.0)
    assert (per_board["recall"] == 1.0).all()
    # The confusion table is aligned on values, not on the block's index —
    # the first draft aligned on index and rendered empty.
    assert confusion.to_numpy().trace() == len(distinguishable_split.val)


def test_identical_boards_are_not_told_apart(identical_split):
    """A check that could only say 'fingerprinted' would not be a check."""
    scores, _, _ = full_matrix(identical_split)
    baseline = majority_baseline(identical_split)
    assert scores["accuracy"] <= baseline["accuracy"] + 0.15


def test_the_carrier_is_named_by_the_single_feature_table(distinguishable_split):
    table = per_feature(distinguishable_split).set_index("feature")
    assert table.loc["board_size_at_t", "accuracy"] == pytest.approx(1.0)
    # Every other feature is drawn identically on both boards.
    others = table.drop(index="board_size_at_t")["accuracy"]
    assert (others < 0.75).all()


def test_removing_board_context_removes_this_fingerprint(distinguishable_split):
    """On the synthetic pair the only carrier is a board-context column, so the
    §12 'absent' set falls back to chance. On the real panel it does not — the
    employer's template carries the board too — which is the finding."""
    scores = without_board_context(distinguishable_split)
    baseline = majority_baseline(distinguishable_split)
    assert scores["accuracy"] <= baseline["accuracy"] + 0.15


def test_missingness_only_uses_no_values(distinguishable_split):
    """Nothing is null on the synthetic panel, so null indicators carry nothing
    — the board differs in a *value*, and this arm must not see values."""
    scores = missingness_only(distinguishable_split)
    baseline = majority_baseline(distinguishable_split)
    assert scores["accuracy"] <= baseline["accuracy"] + 1e-9


def test_a_shallow_panel_renders_a_refusal():
    text = render(make_panel(n_waves=3))
    assert "No legal cut on this panel yet" in text
    assert "## The headline" not in text


def test_the_report_carries_every_section(distinguishable_split):
    text = render(_two_boards(distinguishable=True))
    for heading in (
        "## The headline",
        "### Three feature sets, one question",
        "### Per board",
        "### Confusion",
        "### Each feature alone",
        "### The full set with one feature removed",
        "## What this means for the model",
    ):
        assert heading in text
    assert "design.md` §4a" in text
