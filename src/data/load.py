"""Reproducible loading of a pinned scraper snapshot into DataFrames.

    data/raw/<date>/jobs.db  ->  load_snapshot()  ->  {runs, jobs, observations}

This layer is deliberately dumb. It reads, checks the schema, coerces dtypes,
reports duplicates and validates obvious breakage — and does nothing else.
There is **no ML preprocessing here**: no imputation, no encoding, no scaling,
no label. Those are fitted on a training fold, and anything fitted before the
split is leakage.

Two properties this module must keep:

* **Determinism.** No ``datetime.now()``, no randomness, no set iteration order,
  and every query carries an explicit ``ORDER BY``. Delete ``data/processed/``,
  re-run, and the bytes are identical.
* **Run provenance.** ``runs`` keeps ``status``, ``page_cap`` and
  ``pages_fetched``. A scrape that stopped at its page cap did not observe the
  whole board, so a posting's absence from it is not evidence of removal. Any
  later labelling step needs that distinction and cannot recover it downstream.

Usage::

    python -m src.data.load                        # latest pinned snapshot
    python -m src.data.load --date 2026-09-04
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.snapshot import latest_snapshot

PROCESSED_ROOT = Path("data/processed")

#: Columns each table must provide. Extra columns are allowed and preserved;
#: a missing one is fatal, because every downstream assumption rests on these.
REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "runs": ("id", "source", "started_at", "status"),
    "jobs": (
        "id",
        "source",
        "source_id",
        "title",
        "company",
        "location",
        "remote",
        "salary_min",
        "salary_max",
        "currency",
        "salary_raw",
        "posted_at",
        "url",
        "description",
        "first_seen",
        "last_seen",
        "content_hash",
        "seniority",
    ),
    "job_observations": ("job_id", "run_id"),
}

_TIMESTAMP_COLUMNS = {
    "runs": ("started_at", "finished_at"),
    "jobs": ("posted_at", "first_seen", "last_seen"),
}

_NULLABLE_INT_COLUMNS = {
    "runs": ("rows_parsed", "page_cap", "pages_fetched", "parser_version", "rules_version"),
    "jobs": ("salary_min", "salary_max", "hash_version", "parser_version"),
}

_STRING_COLUMNS = {
    "runs": ("source", "status"),
    "jobs": (
        "source",
        "source_id",
        "title",
        "company",
        "location",
        "currency",
        "salary_raw",
        "url",
        "description",
        "content_hash",
        "seniority",
    ),
}


class SchemaError(ValueError):
    """A snapshot table is missing a column the pipeline depends on."""


@dataclass
class ValidationReport:
    """Findings from :func:`validate`. Warnings describe the data; they are not
    failures. Only structural breakage raises."""

    row_counts: dict[str, int] = field(default_factory=dict)
    duplicate_source_ids: int = 0
    duplicate_content_hashes: int = 0
    duplicate_title_company: int = 0
    orphan_observations: int = 0
    last_seen_before_first_seen: int = 0
    salary_min_above_max: int = 0
    runs_by_status: dict[str, int] = field(default_factory=dict)
    truncated_runs: int = 0

    def format(self) -> str:
        lines = ["row counts:"]
        lines += [f"  {name:<18} {count:>8,}" for name, count in self.row_counts.items()]
        lines.append("duplicates:")
        lines.append(f"  (source, source_id)   {self.duplicate_source_ids:>8,}")
        lines.append(f"  content_hash          {self.duplicate_content_hashes:>8,}")
        lines.append(f"  (title, company)      {self.duplicate_title_company:>8,}")
        lines.append("integrity:")
        lines.append(f"  orphan observations   {self.orphan_observations:>8,}")
        lines.append(f"  last_seen < first_seen{self.last_seen_before_first_seen:>8,}")
        lines.append(f"  salary_min > max      {self.salary_min_above_max:>8,}")
        lines.append("run provenance:")
        for status, count in sorted(self.runs_by_status.items()):
            lines.append(f"  status={status:<14} {count:>8,}")
        lines.append(f"  hit page cap          {self.truncated_runs:>8,}")
        return "\n".join(lines)


def _read_table(conn: sqlite3.Connection, table: str, order_by: str) -> pd.DataFrame:
    """Read one table in a fixed row order, then check its schema."""
    frame = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY {order_by}", conn)
    missing = [c for c in REQUIRED_COLUMNS[table] if c not in frame.columns]
    if missing:
        raise SchemaError(f"table {table!r} is missing required columns: {missing}")
    return frame


def _coerce(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Apply explicit dtypes. Reading SQLite gives everything back as object or
    float; leaving that in place lets a float64 job id silently lose precision
    and makes 'missing' indistinguishable from NaN-the-number."""
    frame = frame.copy()

    for column in _TIMESTAMP_COLUMNS.get(table, ()):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, format="ISO8601")

    for column in _NULLABLE_INT_COLUMNS.get(table, ()):
        if column in frame.columns:
            frame[column] = frame[column].astype("Int64")

    for column in _STRING_COLUMNS.get(table, ()):
        if column in frame.columns:
            frame[column] = frame[column].astype("string")

    if table == "jobs":
        frame["id"] = frame["id"].astype("int64")
        # SQLite stores this as 0/1/NULL. 'boolean' keeps missing distinct from
        # False, which matters: Greenhouse never populates `remote` at all.
        frame["remote"] = frame["remote"].astype("boolean")
    elif table == "runs":
        frame["id"] = frame["id"].astype("int64")
    elif table == "job_observations":
        frame["job_id"] = frame["job_id"].astype("int64")
        frame["run_id"] = frame["run_id"].astype("int64")

    return frame


def load_snapshot(snapshot_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load a pinned snapshot into typed, deterministically ordered frames."""
    snapshot_dir = snapshot_dir or latest_snapshot()
    db_path = snapshot_dir / "jobs.db"
    if not db_path.exists():
        raise FileNotFoundError(f"no jobs.db in snapshot {snapshot_dir}")

    order = {"runs": "id", "jobs": "id", "job_observations": "job_id, run_id"}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        return {
            table: _coerce(_read_table(conn, table, order_by), table)
            for table, order_by in order.items()
        }


def validate(frames: dict[str, pd.DataFrame]) -> ValidationReport:
    """Describe the frames and fail on structural breakage.

    Duplicates and odd values are *reported*, not dropped — deciding what a
    duplicate means here is a modelling judgement, not a loader's business.
    """
    jobs, runs, observations = frames["jobs"], frames["runs"], frames["job_observations"]

    if jobs.empty or runs.empty or observations.empty:
        raise ValueError("snapshot contains an empty table; refusing to proceed")

    report = ValidationReport(
        row_counts={name: len(frame) for name, frame in frames.items()},
        duplicate_source_ids=int(jobs.duplicated(["source", "source_id"]).sum()),
        duplicate_content_hashes=int(jobs.duplicated(["content_hash"]).sum()),
        duplicate_title_company=int(jobs.duplicated(["title", "company"]).sum()),
        orphan_observations=int((~observations["job_id"].isin(jobs["id"])).sum()),
        last_seen_before_first_seen=int((jobs["last_seen"] < jobs["first_seen"]).sum()),
        salary_min_above_max=int((jobs["salary_min"] > jobs["salary_max"]).fillna(False).sum()),
        runs_by_status=runs["status"].value_counts().sort_index().to_dict(),
        truncated_runs=int((runs["pages_fetched"] >= runs["page_cap"]).fillna(False).sum()),
    )

    if report.duplicate_source_ids:
        raise ValueError(
            f"{report.duplicate_source_ids} duplicate (source, source_id) rows: the snapshot "
            "violates its own uniqueness constraint"
        )
    return report


def write_processed(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    """Write frames to Parquet. Deterministic: fixed row order in, index dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in sorted(frames):
        path = out_dir / f"{name}.parquet"
        frames[name].to_parquet(path, index=False, engine="pyarrow", compression="snappy")
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None, help="snapshot date (default: latest pinned)")
    args = parser.parse_args()

    snapshot_dir = Path("data/raw") / args.date if args.date else latest_snapshot()
    frames = load_snapshot(snapshot_dir)
    report = validate(frames)

    out_dir = PROCESSED_ROOT / snapshot_dir.name
    written = write_processed(frames, out_dir)

    print(f"snapshot: {snapshot_dir}")
    print(report.format())
    print("wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
