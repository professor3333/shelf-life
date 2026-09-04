"""Tests for the ingestion layer.

These build their own tiny SQLite database rather than reading the real
snapshot, so they need no scraper, no network, and no 80 MB fixture — and they
still exercise the schema check, the dtype coercion and the determinism
guarantee that the pipeline depends on.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from src.data.load import SchemaError, load_snapshot, validate, write_processed

RUNS = [
    # id, source, started_at, status, page_cap, pages_fetched
    (1, "greenhouse:acme", "2026-08-30T03:45:00+00:00", "ok", 8, 2),
    (2, "arbeitnow", "2026-08-30T03:45:00+00:00", "partial", 8, 8),
]

JOBS = [
    # id, source, source_id, title, company, location, remote, salary_min,
    # salary_max, currency, salary_raw, posted_at, url, description,
    # first_seen, last_seen, content_hash, seniority
    (
        1,
        "greenhouse:acme",
        "a1",
        "Data Engineer",
        "Acme",
        "Berlin",
        None,
        90000,
        130000,
        "EUR",
        "90.000 € bis 130.000 €",
        None,
        "https://example.test/1",
        "desc one",
        "2026-08-30T03:45:00+00:00",
        "2026-09-01T03:45:00+00:00",
        "h1",
        "mid",
    ),
    (
        2,
        "arbeitnow",
        "b2",
        "Werkstudent",
        "Wolt - English",
        None,
        1,
        None,
        None,
        None,
        None,
        None,
        "https://example.test/2",
        "desc two",
        "2026-08-30T03:45:00+00:00",
        "2026-08-30T03:45:00+00:00",
        "h2",
        None,
    ),
]

OBSERVATIONS = [(1, 1), (1, 2), (2, 2)]


def _build_db(path, *, drop_column: str | None = None):
    """Create a miniature snapshot database, optionally with a column removed."""
    job_columns = [
        "id INTEGER PRIMARY KEY",
        "source TEXT NOT NULL",
        "source_id TEXT NOT NULL",
        "title TEXT NOT NULL",
        "company TEXT NOT NULL",
        "location TEXT",
        "remote INTEGER",
        "salary_min INTEGER",
        "salary_max INTEGER",
        "currency TEXT",
        "salary_raw TEXT",
        "posted_at TEXT",
        "url TEXT NOT NULL",
        "description TEXT",
        "first_seen TEXT NOT NULL",
        "last_seen TEXT NOT NULL",
        "content_hash TEXT NOT NULL",
        "seniority TEXT",
    ]
    keep = [c for c in job_columns if not (drop_column and c.startswith(drop_column + " "))]
    job_values = [
        tuple(
            v
            for c, v in zip(job_columns, row, strict=True)
            if not (drop_column and c.startswith(drop_column + " "))
        )
        for row in JOBS
    ]

    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE runs (id INTEGER PRIMARY KEY, source TEXT NOT NULL, "
        "started_at TEXT NOT NULL, status TEXT NOT NULL, page_cap INTEGER, "
        "pages_fetched INTEGER)"
    )
    conn.execute(f"CREATE TABLE jobs ({', '.join(keep)})")
    conn.execute("CREATE TABLE job_observations (job_id INTEGER, run_id INTEGER)")
    conn.executemany("INSERT INTO runs VALUES (?,?,?,?,?,?)", RUNS)
    conn.executemany(f"INSERT INTO jobs VALUES ({','.join('?' * len(keep))})", job_values)
    conn.executemany("INSERT INTO job_observations VALUES (?,?)", OBSERVATIONS)
    conn.commit()
    conn.close()


@pytest.fixture
def snapshot(tmp_path):
    snapshot_dir = tmp_path / "2026-08-30"
    snapshot_dir.mkdir()
    _build_db(snapshot_dir / "jobs.db")
    return snapshot_dir


def test_loads_expected_row_counts(snapshot):
    frames = load_snapshot(snapshot)
    assert len(frames["jobs"]) == len(JOBS)
    assert len(frames["runs"]) == len(RUNS)
    assert len(frames["job_observations"]) == len(OBSERVATIONS)


def test_raises_on_missing_column(tmp_path):
    snapshot_dir = tmp_path / "2026-08-30"
    snapshot_dir.mkdir()
    _build_db(snapshot_dir / "jobs.db", drop_column="seniority")
    with pytest.raises(SchemaError, match="seniority"):
        load_snapshot(snapshot_dir)


def test_missing_snapshot_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path / "nope")


def test_dtypes_are_coerced(snapshot):
    jobs = load_snapshot(snapshot)["jobs"]
    assert jobs["id"].dtype == "int64"
    assert jobs["salary_min"].dtype == "Int64"
    assert jobs["remote"].dtype == "boolean"
    assert isinstance(jobs["first_seen"].dtype, pd.DatetimeTZDtype)


def test_missing_remote_stays_missing_and_is_not_false(snapshot):
    """Greenhouse never populates `remote`. If NULL collapsed to False the model
    would read 'this job is on-site' where the truth is 'nobody said'."""
    jobs = load_snapshot(snapshot)["jobs"]
    assert pd.isna(jobs.loc[jobs["id"] == 1, "remote"].iloc[0])
    assert bool(jobs.loc[jobs["id"] == 2, "remote"].iloc[0]) is True


def test_run_provenance_survives_loading(snapshot):
    """A run that stopped at its page cap did not see the whole board. If that
    fact is dropped here, no later step can tell a truncated scrape from a
    complete one, and 'absent' gets misread as 'removed'."""
    runs = load_snapshot(snapshot)["runs"]
    truncated = runs[runs["pages_fetched"] >= runs["page_cap"]]
    assert set(truncated["source"]) == {"arbeitnow"}


def test_validate_reports_duplicates_without_dropping_rows(snapshot):
    frames = load_snapshot(snapshot)
    report = validate(frames)
    assert report.row_counts["jobs"] == len(JOBS)
    assert report.duplicate_source_ids == 0
    assert report.truncated_runs == 1


def test_validate_rejects_an_empty_table(snapshot):
    frames = load_snapshot(snapshot)
    frames["jobs"] = frames["jobs"].iloc[0:0]
    with pytest.raises(ValueError, match="empty table"):
        validate(frames)


def test_row_order_is_deterministic(snapshot):
    first = load_snapshot(snapshot)["jobs"]["id"].tolist()
    second = load_snapshot(snapshot)["jobs"]["id"].tolist()
    assert first == second == sorted(first)


def test_written_output_is_byte_identical_across_runs(snapshot, tmp_path):
    """The Component 3 contract: delete the output, re-run, get the same bytes."""
    frames = load_snapshot(snapshot)
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    write_processed(frames, first)
    write_processed(load_snapshot(snapshot), second)
    for path in sorted(first.iterdir()):
        assert path.read_bytes() == (second / path.name).read_bytes(), path.name
