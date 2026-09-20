"""The data profile — `src/data/profile.py` — on a snapshot CI can build itself.

The report is the factual half of the data dictionary, and until now it was
only ever run against the real 80 MB snapshot, which CI does not have. These
tests build the same miniature SQLite database `test_load.py` uses and check
the three things the report exists to say: the missingness fingerprint per
source, which runs hit their page cap, and how many jobs were seen exactly
once. The numbers are hand-countable from the fixture rows.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data import profile
from src.data.load import load_snapshot
from tests.test_load import JOBS, OBSERVATIONS, RUNS, _build_db


@pytest.fixture
def snapshot(tmp_path):
    snapshot_dir = tmp_path / "2026-08-30"
    snapshot_dir.mkdir()
    _build_db(snapshot_dir / "jobs.db")
    (snapshot_dir / "manifest.json").write_text(
        json.dumps({"snapshot_date": "2026-08-30", "sha256": "f" * 64})
    )
    return snapshot_dir


def test_coverage_by_source_is_the_missingness_fingerprint(snapshot):
    """The greenhouse row has a salary, the arbeitnow row does not — the table
    says 100 and 0, not a pooled 50, because pooled is the number that hides it."""
    jobs = load_snapshot(snapshot)["jobs"]
    table = profile.coverage_by_source(jobs)
    assert set(table.index) == {"arbeitnow", "greenhouse:acme"}
    assert table.loc["greenhouse:acme", "salary_min"] == 100
    assert table.loc["arbeitnow", "salary_min"] == 0
    assert table.loc["greenhouse:acme", "remote"] == 0  # structurally absent
    assert table.loc["arbeitnow", "remote"] == 100
    assert table["jobs"].sum() == len(JOBS)


def test_run_completeness_counts_the_capped_runs(snapshot):
    """arbeitnow fetched 8 of a cap of 8: that run did not see the whole board."""
    runs = load_snapshot(snapshot)["runs"]
    table = profile.run_completeness(runs)
    assert table.loc["arbeitnow", "capped_runs"] == 1
    assert table.loc["arbeitnow", "partial"] == 1
    assert table.loc["greenhouse:acme", "capped_runs"] == 0
    assert table.loc["greenhouse:acme", "ok"] == 1
    assert table["runs"].sum() == len(RUNS)
    assert table.loc["arbeitnow", "first_run"] == "2026-08-30"


def test_panel_shape_counts_jobs_seen_once(snapshot):
    """Both runs were on the same day, so every job was seen on exactly one day
    however many runs observed it; a job seen once has no trajectory."""
    frames = load_snapshot(snapshot)
    table = profile.panel_shape(frames["jobs"], frames["runs"], frames["job_observations"])
    assert int(table["jobs"].sum()) == len({job_id for job_id, _ in OBSERVATIONS})
    assert (table["max_days_seen"] == 1).all()
    assert (table["seen_once_pct"] == 100.0).all()


def test_column_summary_reports_null_rates_per_column(snapshot):
    jobs = load_snapshot(snapshot)["jobs"]
    summary = profile.column_summary(jobs)
    assert summary.index.name == "column"
    assert summary.loc["salary_min", "null_pct"] == 50.0
    assert summary.loc["title", "null_pct"] == 0.0
    assert summary.loc["title", "n_unique"] == len(JOBS)


def test_the_report_is_deterministic_and_names_its_snapshot(snapshot):
    """No wall clock anywhere in it: two runs on the same snapshot are one file."""
    first = profile.build_report(snapshot, figures_dir=None)
    second = profile.build_report(snapshot, figures_dir=None)
    assert first == second
    assert "snapshot `2026-08-30`" in first
    assert "`ffffffffffff…`" in first  # the manifest's sha256, truncated
    for heading in (
        "## Columns (`jobs`)",
        "## Coverage by source",
        "## Run completeness",
        "## Panel shape",
    ):
        assert heading in first
    assert "![" not in first  # no figures were asked for, so none are promised


def test_the_report_says_when_matplotlib_is_missing_rather_than_omitting_silently(
    snapshot, tmp_path, monkeypatch
):
    monkeypatch.setattr(profile.plots, "available", lambda: False)
    report = profile.build_report(snapshot, figures_dir=tmp_path / "figures")
    assert "Figures not rendered: matplotlib is not installed" in report
    assert not (tmp_path / "figures").exists()


def test_the_markdown_table_formats_by_column_not_by_row():
    """Iterating rows would upcast the integer column to float and print 1.0."""
    frame = pd.DataFrame(
        {"count": [1234, None], "share": [0.5, None], "flag": [True, False], "name": ["a", None]},
        index=pd.Index(["x", "y"], name="row"),
    )
    frame["count"] = frame["count"].astype("Int64")
    rendered = profile._md_table(frame)
    lines = rendered.splitlines()
    assert lines[0] == "| row | count | share | flag | name |"
    assert lines[2] == "| x | 1,234 | 0.5 | True | a |"
    assert lines[3] == "| y |  |  | False |  |"


def test_the_dictionary_scaffold_lists_every_column_and_asks_the_leakage_question():
    text = profile.scaffold_dictionary(["title", "last_seen"])
    assert "| `title` |" in text and "| `last_seen` |" in text
    assert "available at prediction point?" in text
