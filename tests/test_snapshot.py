"""Tests for snapshot pinning.

There were none until 2026-09-07, which is why `pin` shipped naming snapshots
after the wall clock. The bug is in `DEBUGGING.md`; these are the tests that
would have caught it, and they were all about the *name* rather than the copy,
because the copy did not look like the part that could be wrong. On 2026-10-14
it was: a byte copy of a database mid-transaction is a database no commit
produced, and the last test here is the one that shows it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import src.data.snapshot as snapshot
from src.data.snapshot import newest_run_date, pin

REQUIRED = ("runs", "jobs", "job_observations", "job_changes")


@pytest.fixture(autouse=True)
def _isolated_raw_root(tmp_path, monkeypatch):
    """Every test writes into its own tree.

    Autouse rather than per-test because the first draft of this file forgot it
    on one test, which then pinned into the repository's real `data/raw/`. A
    test that can reach live data is a test that will eventually change it.
    """
    monkeypatch.setattr(snapshot, "RAW_ROOT", tmp_path / "raw")


def _db(path: Path, run_times: list[str]) -> Path:
    """A database with the four required tables and the given run instants."""
    with sqlite3.connect(path) as conn:
        for table in REQUIRED:
            conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, started_at TEXT)")
        for index, when in enumerate(run_times, start=1):
            conn.execute("INSERT INTO runs (id, started_at) VALUES (?, ?)", (index, when))
    return path


def test_the_snapshot_is_named_after_the_newest_run_it_contains(tmp_path):
    """Not after today. A pin taken before the day's crawl contains yesterday's
    data, and calling it today's is a lie the immutability rule then enforces."""
    source = _db(tmp_path / "jobs.db", ["2026-09-05T03:45:00+00:00", "2026-09-06T03:45:43+00:00"])
    directory = pin(source, date=None)
    assert directory.name == "2026-09-06"
    assert json.loads((directory / "manifest.json").read_text())["snapshot_date"] == "2026-09-06"


def test_an_early_pin_collides_with_yesterday_instead_of_squatting_on_today(tmp_path):
    """The exact failure of 2026-09-07, as a test.

    The scraper fires at 03:45 UTC. A pin at 02:36 holds yesterday's data. Under
    the old rule it took today's name, and the real 03:45 wave then had nowhere
    to go — the name it needed was occupied by a copy that did not contain it.
    Under the new rule the early pin resolves to yesterday, collides with
    yesterday's snapshot and is refused, which is correct: an early pin has
    nothing new to record.
    """
    yesterday = _db(tmp_path / "a.db", ["2026-09-06T03:45:43+00:00"])
    assert pin(yesterday, date=None).name == "2026-09-06"

    # run again before today's crawl: same newest run, so the same name
    early = _db(tmp_path / "b.db", ["2026-09-06T03:45:43+00:00"])
    with pytest.raises(FileExistsError):
        pin(early, date=None)

    # and once the crawl lands, today's name is free and correct
    after = _db(tmp_path / "c.db", ["2026-09-06T03:45:43+00:00", "2026-09-07T03:45:42+00:00"])
    assert pin(after, date=None).name == "2026-09-07"


def test_two_snapshots_can_never_hold_the_same_newest_run(tmp_path):
    """The invariant the naming rule buys: one directory per crawl wave. On
    2026-09-07 two directories held identical row counts, which is how a
    longitudinal record quietly stops being one."""
    times = ["2026-09-06T03:45:43+00:00"]
    pin(_db(tmp_path / "a.db", times), date=None)
    with pytest.raises(FileExistsError):
        pin(_db(tmp_path / "b.db", times), date=None)

    names = sorted(p.name for p in (tmp_path / "raw").iterdir())
    assert names == ["2026-09-06"]


def test_an_explicit_date_still_wins(tmp_path):
    """Renaming a historical snapshot by hand is a legitimate repair, and this
    function should not be what prevents it."""
    source = _db(tmp_path / "jobs.db", ["2026-09-06T03:45:43+00:00"])
    assert pin(source, date="2026-01-01").name == "2026-01-01"


def test_a_database_with_no_runs_falls_back_to_today(tmp_path):
    """A fresh database has no newest run to be named after. Falling back beats
    refusing: the snapshot is still a real copy of a real file."""
    import datetime as dt

    source = _db(tmp_path / "jobs.db", [])
    assert newest_run_date(source) is None
    assert pin(source, date=None).name == dt.date.today().isoformat()


def test_newest_run_date_reads_the_date_not_the_instant(tmp_path):
    source = _db(tmp_path / "jobs.db", ["2026-09-06T03:45:43.085104+00:00"])
    assert newest_run_date(source) == "2026-09-06"


def _spill_an_uncommitted_rewrite(path: Path, spilled, release) -> None:
    """Hold a write transaction open after it has overwritten committed pages.

    A one-page cache forces SQLite to write uncommitted pages into the database
    file itself, originals parked in the rollback journal. Rewriting existing
    rows matters: appended pages lie past the header's page count and a byte
    copy ignores them, but rewritten ones replace committed data in place —
    the scraper's `UPDATE jobs SET company_id = NULL` over every posting.
    """
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    try:
        conn.execute("PRAGMA cache_size = 1")
        conn.execute("BEGIN")
        conn.execute("UPDATE jobs SET started_at = 'uncommitted'")
        spilled.set()
        release.wait(10)
        conn.execute("ROLLBACK")
    finally:
        conn.close()


def test_a_pin_taken_mid_transaction_copies_the_last_commit_not_the_torn_file(tmp_path):
    """2026-10-14: the scraper held `jobs.db` in one transaction for five hours.

    A byte copy taken then holds pages no commit ever produced. The pin must
    wait out the writer and land on what was committed — here the writer
    rolls back, so the copy must contain none of its writes.
    """
    import shutil
    import threading

    source = _db(tmp_path / "jobs.db", ["2026-10-13T03:45:00+00:00"])
    with sqlite3.connect(source) as conn:
        conn.executemany("INSERT INTO jobs (started_at) VALUES (?)", [("x" * 2000,)] * 400)
    spilled, release = threading.Event(), threading.Event()
    writer = threading.Thread(target=_spill_an_uncommitted_rewrite, args=(source, spilled, release))
    writer.start()
    assert spilled.wait(10)

    # The precondition, so this test cannot pass vacuously: a byte copy now
    # really is a database that no commit produced — and SQLite's own
    # integrity check cannot tell.
    torn = tmp_path / "torn.db"
    shutil.copy2(source, torn)
    with sqlite3.connect(torn) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        uncommitted = "SELECT COUNT(*) FROM jobs WHERE started_at = 'uncommitted'"
        assert conn.execute(uncommitted).fetchone()[0] > 0

    target = tmp_path / "copy.db"
    copier = threading.Thread(target=snapshot.copy_consistently, args=(source, target))
    copier.start()
    copier.join(0.3)
    assert copier.is_alive(), "the copy must wait for the writer, not read past it"
    release.set()
    writer.join(10)
    copier.join(10)
    assert not copier.is_alive()

    with sqlite3.connect(target) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 400
        assert conn.execute(uncommitted).fetchone()[0] == 0


def test_the_pin_waits_out_a_lock_longer_than_sqlites_default():
    """sqlite3 gives up after five seconds; the scraper's lock lasted hours."""
    assert snapshot.LOCK_WAIT_SECONDS >= 60 * 60
