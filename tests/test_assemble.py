"""Tests for job-day assembly and the label rule.

The panel is built by hand so every expected label can be reasoned about from
the fixture rather than from the real data. The label cases follow the three
branches of `docs/problem_definition.md` §4, including the censoring branch,
which is the one that is silently wrong if you get it right by accident.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.assemble import (
    _board_context,
    build_observations,
    complete_runs,
    compute_labels,
)

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2026-09-01T03:45:00Z")


def _runs(n=5, source="greenhouse:acme", status="ok", rules_version=2, spacing=DAY):
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "source": [source] * n,
            "started_at": [T0 + i * spacing for i in range(n)],
            "status": [status] * n,
            "rules_version": [rules_version] * n,
        }
    )


def _snapshot_rows(presence: dict[str, list[int]], runs):
    """`presence` maps a posting id to the run indices that saw it."""
    times = runs.sort_values("started_at")["started_at"].tolist()
    rows = []
    for source_id, indices in presence.items():
        for i in indices:
            rows.append(
                {
                    "observed_at": times[i],
                    "source": runs["source"].iloc[0],
                    "source_id": source_id,
                    "title": f"Role {source_id}",
                    "company": "Acme",
                    "location": "Berlin",
                    "remote": None,
                    "salary_min": None,
                    "salary_max": None,
                    "currency": None,
                    "salary_raw": None,
                    "posted_at": T0 - 30 * DAY,
                    "url": f"https://example.test/{source_id}",
                }
            )
    return pd.DataFrame(rows).astype({"source_id": "string"})


def _panel(presence, runs=None, horizon=1, basis="calendar"):
    runs = _runs() if runs is None else runs
    rc = complete_runs(runs)
    panel = build_observations(_snapshot_rows(presence, runs), rc)
    return compute_labels(panel, rc, horizon, basis=basis)


def test_only_complete_runs_enter_the_panel():
    """A `partial` run did not see the whole board, and a run from an earlier
    rules_version is not comparable — neither may establish an absence."""
    runs = pd.concat(
        [
            _runs(2),
            _runs(1, status="partial").assign(id=[90]),
            _runs(1, rules_version=1).assign(id=[91]),
        ]
    )
    assert len(complete_runs(runs)) == 2


def test_run_index_is_per_source_and_time_ordered():
    runs = pd.concat([_runs(3), _runs(2, source="greenhouse:other").assign(id=[10, 11])])
    rc = complete_runs(runs)
    assert rc.groupby("source")["run_index"].max().to_dict() == {
        "greenhouse:acme": 2,
        "greenhouse:other": 1,
    }


def test_removal_corroborated_by_two_absences_is_a_positive():
    """Seen at runs 0-1, absent at 2 and 3: gone at run 2."""
    out = _panel({"a": [0, 1]})
    row = out[out["run_index"] == 1].iloc[0]
    assert row["y"] == 1 and row["label_observable"]


def test_a_single_absence_is_not_a_removal():
    """Absent at run 2 but back at run 3 — the corroboration rule exists exactly
    so a one-run blip is not read as a removal."""
    out = _panel({"a": [0, 1, 3, 4]})
    assert set(out["y"].dropna().unique()) == {0}


def test_survivor_is_a_negative():
    out = _panel({"a": [0, 1, 2, 3, 4]})
    assert out.loc[out["run_index"] == 0, "y"].iloc[0] == 0


def test_row_whose_horizon_has_not_elapsed_is_dropped_not_zeroed():
    """The branch that is silently wrong if you get it right by accident.
    The last row's horizon reaches past the final run, so its outcome is
    unknown — it must be dropped, never labelled 0."""
    out = _panel({"a": [0, 1, 2, 3, 4]})
    last = out[out["run_index"] == 4].iloc[0]
    assert not last["label_observable"]
    assert pd.isna(last["y"])
    assert (out["label_observable"] | out["y"].isna()).all()


def test_horizon_basis_changes_the_answer_when_runs_are_jittered():
    """Runs are not evenly spaced: 14 of 27 real gaps exceed 24h. Under instant
    arithmetic a removal confirmed by the very next daily run can fall outside a
    1-day horizon; under calendar comparison it does not."""
    jittered = _runs(4, spacing=pd.Timedelta(hours=34))
    seen = {"a": [0, 1]}
    instant = _panel(seen, runs=jittered, horizon=1, basis="instant")
    calendar = _panel(seen, runs=jittered, horizon=1, basis="calendar")
    assert (instant["y"].fillna(0) == 1).sum() == 0
    assert (calendar["y"].fillna(0) == 1).sum() == 1


def test_unknown_basis_is_rejected():
    with pytest.raises(ValueError, match="basis"):
        _panel({"a": [0, 1]}, basis="whenever")


def test_board_context_is_computed_within_a_run_only():
    """Board size at t must count the board as it stood at t. If the window ran
    past t these numbers would be identical across runs, which is the leak."""
    out = _board_context(
        _panel({"a": [0, 1, 2], "b": [0, 1], "c": [0]}).assign(requisition_id=None)
    )
    sizes = out.drop_duplicates(["run_index"]).set_index("run_index")["board_size_at_t"]
    assert sizes.loc[0] == 3 and sizes.loc[1] == 2 and sizes.loc[2] == 1


def test_observation_count_accumulates_only_up_to_t():
    out = _board_context(_panel({"a": [0, 1, 2]}).assign(requisition_id=None))
    counts = out.sort_values("run_index")["n_complete_runs_observed"].tolist()
    assert counts == [1, 2, 3], "must count runs seen so far, not the total"


def test_no_feature_column_is_derived_from_the_future():
    """A structural check: the assembled frame must not carry any column whose
    value is only knowable after t."""
    out = _board_context(_panel({"a": [0, 1, 2]}).assign(requisition_id=None))
    forbidden = {"last_seen", "content_hash", "parser_version", "hash_version", "rows_parsed"}
    assert forbidden.isdisjoint(out.columns)
