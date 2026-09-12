"""Board context derived from a snapshot equals what the panel would have computed."""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.assemble import _board_context
from src.inference.board_snapshot import derive_board_context
from src.inference.contract import InvalidPayload


def _snapshot() -> list[dict]:
    return [
        {"title": "Engineer", "requisition_id": "R1", "location": "Berlin"},
        {"title": "Engineer", "requisition_id": "R1"},
        {"title": "Engineer", "requisition_id": "R2"},
        {"title": "Designer"},
        {"title": "Designer", "requisition_id": "R3"},
    ]


def test_the_derivation_is_the_panels_own_arithmetic():
    """One crawl of one board through `assemble._board_context`, against the
    same postings through `derive_board_context`. The definitions live in the
    panel; this is what keeps the serve-time copy honest."""
    postings = _snapshot()
    frame = pd.DataFrame(
        [
            {
                "source": "greenhouse:acme",
                "source_id": f"p{i}",
                "run_index": 0,
                "title": p["title"],
                "requisition_id": p.get("requisition_id"),
            }
            for i, p in enumerate(postings)
        ]
    )
    panel = _board_context(frame).sort_values("source_id").reset_index(drop=True)
    derived = derive_board_context(postings)

    assert [row["board_size_at_t"] for row in derived] == panel["board_size_at_t"].tolist()
    assert [row["n_same_title_on_board"] for row in derived] == (
        panel["n_same_title_on_board"].tolist()
    )
    for row, expected in zip(derived, panel["n_same_req_on_board"].tolist(), strict=True):
        if pd.isna(expected):
            assert "n_same_req_on_board" not in row, "no requisition, no count — as in the panel"
        else:
            assert row["n_same_req_on_board"] == expected
    # One crawl has no previous run, so the panel's growth is NA — and so is ours
    # unless the caller says what yesterday's size was.
    assert panel["board_growth"].isna().all()
    assert all("board_growth" not in row for row in derived)


def test_growth_comes_from_the_previous_size_the_caller_supplies():
    derived = derive_board_context(_snapshot(), previous_board_size=8)
    assert all(row["board_growth"] == -3.0 for row in derived)


def test_the_requisition_id_is_used_for_the_count_and_then_dropped():
    derived = derive_board_context(_snapshot())
    assert all("requisition_id" not in row for row in derived), "an identifier, never a feature"
    assert derived[0]["n_same_req_on_board"] == 2.0 and derived[2]["n_same_req_on_board"] == 1.0


def test_a_snapshot_is_one_board():
    mixed = [{"title": "A", "source": "greenhouse:a"}, {"title": "B", "source": "greenhouse:b"}]
    with pytest.raises(InvalidPayload, match="one board"):
        derive_board_context(mixed)


def test_a_snapshot_may_not_also_carry_board_context_per_posting():
    with pytest.raises(InvalidPayload, match="derives board context"):
        derive_board_context([{"title": "A", "board_size_at_t": 500}])


def test_an_empty_snapshot_is_refused():
    with pytest.raises(InvalidPayload):
        derive_board_context([])
