"""Tests for job-day assembly and the label rule.

The panel is built by hand so every expected label can be reasoned about from
the fixture rather than from the real data. The label cases follow the three
branches of `docs/problem_definition.md` §4, including the censoring branch,
which is the one that is silently wrong if you get it right by accident.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.features.assemble import (
    _board_context,
    assemble,
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


def test_a_closure_inside_an_unelapsed_window_is_dropped_too():
    """The asymmetry that makes a base rate lie.

    A closure is knowable the moment it happens; survival is only knowable once
    the whole window has elapsed. Label on outcome alone and the newest cohorts
    contain nothing but closures — not because closures cluster there, but
    because the survivors beside them are still censored.

    Eight runs, a three-run window. `gone` is seen through run 5 and absent at
    runs 6 and 7, so its removal is corroborated and `t_gone` is run 6. At run 5
    its outcome is therefore already known — but run 5's window ends at run 8,
    which does not exist, so no survivor at run 5 can be labelled. Keep the
    closure and run 5's labelled rows are 100% positive.
    """
    runs = _runs(n=8)
    out = _panel({"stays": list(range(8)), "gone": [0, 1, 2, 3, 4, 5]}, runs=runs, horizon=3)

    wave5 = out[out["run_index"] == 5]
    assert set(wave5["source_id"]) == {"stays", "gone"}, "the fixture must put both at run 5"
    assert not wave5["label_observable"].any(), (
        "run 5's window has not elapsed, so it must contribute nothing — "
        "including the closure whose outcome is already known"
    )

    # Every cohort that does survive could have shown either answer.
    labelled = out[out["label_observable"]]
    assert not labelled.empty
    for wave, block in labelled.groupby("run_index"):
        assert not (block["y"] == 1).all(), f"wave {wave} is all closures"


def test_the_base_rate_does_not_climb_as_partial_waves_arrive():
    """The property the fix exists for, stated as what a reader cares about.

    A crawl landing *inside* the horizon of the newest rows settles nothing. It
    must not add labelled rows at all — because the only rows it could add are
    the closures, and each one lifts the reported rate. Measured at H=7 on the
    real 2026-09-08 panel: 7.76% against 13.58%, the inflated figure sitting
    past the 11.3% planning estimate it was meant to replace, which is how a
    bias gets mistaken for a confirmation.
    """
    presence = {
        "stays": list(range(7)),
        "gone": [0, 1, 2, 3, 4],  # absent at 5 and 6, so t_gone is run 5
    }
    out = _panel(presence, runs=_runs(n=7), horizon=3)

    labelled = out[out["label_observable"]]
    # Runs 0-3 have a run at their deadline (3, 4, 5, 6); run 4's is run 7, absent.
    assert set(labelled["run_index"]) == {0, 1, 2, 3}, (
        "only cohorts whose deadline run exists may be labelled"
    )
    assert int((labelled["y"] == 1).sum()) > 0, "the fixture must contain a real closure"
    assert int((labelled["y"] == 0).sum()) > 0, "and a real survivor"

    # Run 4 holds `gone`'s last row and its outcome is known, which is exactly
    # the row the outcome-only rule would have kept.
    wave4 = out[out["run_index"] == 4]
    assert "gone" in set(wave4["source_id"])
    assert not wave4["label_observable"].any()


def test_the_newest_labelled_wave_can_never_carry_a_positive():
    """The structural fact `src.data.split.minimum_waves` is built on.

    A wave becomes *labelled* as soon as one further run exists — that is all a
    negative needs. A positive needs more: `t_gone` is only defined where the
    posting is absent at a run **and** at the run after it, so a wave's
    positives are not observable until two runs beyond its horizon. The gap is
    one wave wide and it sits at the newest end of the panel, which is exactly
    where the test block goes.

    Here runs 0-5 exist. Postings vanish after runs 1, 2, 3 and 4 respectively,
    so waves 1, 2 and 3 each get a positive — and wave 4, the newest labelled
    one, gets none even though `g5` did disappear at run 5. Confirmed on the
    real panel: the 2026-09-06 snapshot's newest labelled wave had 1,160
    labelled rows and 0 positives, against 14-22 for every wave before it.
    """
    out = _panel(
        {
            "s": [0, 1, 2, 3, 4, 5],
            "g2": [0, 1],
            "g3": [0, 1, 2],
            "g4": [0, 1, 2, 3],
            "g5": [0, 1, 2, 3, 4],
        },
        runs=_runs(n=6),
    )
    labelled = out[out["label_observable"]]
    newest = labelled["run_index"].max()
    assert newest == 4  # run 5's rows have no forward run at all

    positives = labelled[labelled["y"] == 1]
    assert set(positives["run_index"]) == {1, 2, 3}
    assert (labelled[labelled["run_index"] == newest]["y"] == 0).all()

    # and the posting that did vanish at run 5 is dropped, not called a survivor
    g5_last = out[(out["source_id"] == "g5") & (out["run_index"] == 4)].iloc[0]
    assert not g5_last["label_observable"]


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


def test_seconds_of_clock_drift_do_not_decide_the_label():
    """The damning case is not the 34h outage, it is ordinary cron drift. Real
    gaps landed 2.6s under 24h and 27.0s over it, and under instant arithmetic
    that difference alone decides whether a removal is a positive, a negative or
    discarded. The label must not be a function of the scheduler's punctuality."""
    early = _panel({"a": [0, 1]}, runs=_runs(4, spacing=DAY - pd.Timedelta(seconds=3)), horizon=1)
    late = _panel({"a": [0, 1]}, runs=_runs(4, spacing=DAY + pd.Timedelta(seconds=27)), horizon=1)
    assert (early["y"].fillna(0) == 1).sum() == (late["y"].fillna(0) == 1).sum() == 1


def test_calendar_is_the_default_basis():
    """docs/design.md §10. The default is the decision; a caller who wants the
    rejected comparison has to name it."""
    assert inspect.signature(compute_labels).parameters["basis"].default == "calendar"
    assert inspect.signature(assemble).parameters["basis"].default == "calendar"


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


# --- the resurrection window (design.md §11) --------------------------------


def test_a_label_is_final_once_corroborated_even_if_the_posting_returns():
    """`docs/design.md` §11, decided 2026-09-09.

    The rule used to be *absent at two consecutive runs **and never seen
    again***, which reads the whole remaining panel: a training label was never
    final, it could flip as depth accrued, and no embargo of any width could
    seal one from the evaluation period because the reach was unbounded. The
    embargo has always been computed as horizon plus one run, so that arithmetic
    was false rather than merely tight.

    Measured on the 2026-09-08 snapshot: 145 postings vanished and returned,
    144 of them after a single absent run — which corroboration already absorbs
    — exactly one after two, and none after three or more. So the cost of a
    final label is one posting in 1,530.

    Here `back` is absent at runs 2 and 3 and returns at run 4. It is gone.
    """
    out = _panel({"stays": [0, 1, 2, 3, 4, 5], "back": [0, 1, 4, 5]}, runs=_runs(n=6), horizon=1)

    row = out[(out["source_id"] == "back") & (out["run_index"] == 1)].iloc[0]
    assert row["label_observable"]
    assert row["y"] == 1, "two consecutive absences is a removal, and the return is too late"


def test_a_single_absence_still_is_not_a_removal():
    """The companion, and the reason the change costs so little: the common case
    by two orders of magnitude is a one-run gap, and corroboration never let that
    count as a removal in the first place."""
    out = _panel(
        {"stays": [0, 1, 2, 3, 4, 5], "blips": [0, 1, 3, 4, 5]}, runs=_runs(n=6), horizon=1
    )

    row = out[(out["source_id"] == "blips") & (out["run_index"] == 1)].iloc[0]
    assert row["y"] == 0, "one absent run is a scrape artefact, not a closure"


# --- one horizon, named once ------------------------------------------------


def test_no_module_spells_out_a_panel_filename():
    """Six copies of `job_days_h1_calendar.parquet` is six places for the
    horizon to disagree, and on 2026-09-09 it did: `scripts/rehearse.sh`
    assembled one panel while every module beneath it read another, so a gate
    report about H=7 sat above a ladder scored on H=1.

    `docs/design.md` §2 decided H=7 and calls H=1 a smoke test in the same
    sentence, so a hardcoded `h1` path is a module reporting the smoke test
    under the build's name.

    What is banned is a *literal* horizon in a path — `job_days_h1_...`. The
    template inside `panel_path` itself is the one place the shape is allowed to
    be written down, and prose may name a file without constructing one, so
    docstrings are read as documentation rather than as code.
    """
    import ast
    import re
    from pathlib import Path

    literal = re.compile(r"job_days_h\d")
    offenders = []
    for path in sorted(Path("src").rglob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = {
            ast.get_docstring(node, clean=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if node.value in docstrings or not literal.search(node.value):
                continue
            offenders.append(f"{path}:{node.lineno}: {node.value!r}")
    assert not offenders, "derive the path from `assemble.panel_path` instead:\n" + "\n".join(
        offenders
    )


def test_every_default_panel_names_the_decided_horizon():
    from src.data.label_audit import DEFAULT_PANEL as audit_panel
    from src.features.assemble import DEFAULT_HORIZON, panel_path
    from src.models.freeze import DEFAULT_PANEL as freeze_panel
    from src.models.train_baseline import DEFAULT_PANEL as ladder_panel

    assert DEFAULT_HORIZON == 7, "docs/design.md §2 decided H=7 on 2026-09-04"
    assert {audit_panel, freeze_panel, ladder_panel} == {panel_path()}


def test_the_scripts_pass_the_panel_to_every_step_they_run():
    """`HORIZON=1 ./scripts/rehearse.sh` has to reach the ladder, not just the
    gate. Without `--panel` each module falls back to its own default, which
    happens to agree at H=7 and silently disagrees everywhere else — the failure
    mode being a report whose header and body describe different data."""
    from pathlib import Path

    script = Path("scripts/rehearse.sh").read_text()
    for module in (
        "src.data.label_audit",
        "src.models.train_baseline",
        "src.models.train",
        "src.models.experiments",
        "src.models.evaluate",
    ):
        line = next(one for one in script.splitlines() if f"-m {module}" in one)
        assert '--panel "${PANEL}"' in line, f"{module} is run without --panel"
