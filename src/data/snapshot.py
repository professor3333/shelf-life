"""Pin an immutable snapshot of the scraper database.

The scraper keeps running, so "the data" is a moving target: numbers computed on
different days are not comparable. Every experiment is therefore run against a
dated copy of ``jobs.db`` under ``data/raw/<snapshot-date>/``, never against the
live database.

Usage::

    python -m src.data.snapshot                     # pin today's snapshot
    python -m src.data.snapshot --date 2026-09-04   # name it explicitly
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

DEFAULT_SOURCE_DB = Path.home() / "code" / "job-listing-scraper" / "data" / "jobs.db"
RAW_ROOT = Path("data/raw")

#: Tables the snapshot is expected to contain. A snapshot missing any of these
#: is unusable, so we fail at pin time rather than at load time.
REQUIRED_TABLES = ("runs", "jobs", "job_observations", "job_changes")


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file, so a snapshot can be proven unmodified later."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table_counts(db_path: Path) -> dict[str, int]:
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        present = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        missing = [t for t in REQUIRED_TABLES if t not in present]
        if missing:
            raise ValueError(f"snapshot is missing required tables: {missing}")
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in REQUIRED_TABLES}


def pin(source_db: Path = DEFAULT_SOURCE_DB, date: str | None = None) -> Path:
    """Copy ``source_db`` into ``data/raw/<date>/`` and write a manifest.

    Returns the snapshot directory. Refuses to overwrite an existing snapshot:
    a pinned snapshot is immutable by definition, and silently replacing one
    would invalidate every number already computed against it.
    """
    if not source_db.exists():
        raise FileNotFoundError(f"scraper database not found: {source_db}")

    date = date or dt.date.today().isoformat()
    snapshot_dir = RAW_ROOT / date
    target_db = snapshot_dir / "jobs.db"
    if target_db.exists():
        raise FileExistsError(
            f"snapshot already pinned at {target_db}; delete it explicitly to re-pin"
        )

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, target_db)

    manifest = {
        "snapshot_date": date,
        "pinned_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_db": str(source_db),
        "sha256": sha256_of(target_db),
        "size_bytes": target_db.stat().st_size,
        "row_counts": _table_counts(target_db),
    }
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return snapshot_dir


def latest_snapshot(raw_root: Path = RAW_ROOT) -> Path:
    """The most recent pinned snapshot directory."""
    candidates = sorted(p for p in raw_root.glob("*/jobs.db") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            f"no snapshot pinned under {raw_root}; run `python -m src.data.snapshot` first"
        )
    return candidates[-1].parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-db", type=Path, default=DEFAULT_SOURCE_DB)
    parser.add_argument("--date", default=None, help="snapshot name (default: today)")
    args = parser.parse_args()

    snapshot_dir = pin(args.source_db, args.date)
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    print(f"pinned snapshot -> {snapshot_dir}")
    print(f"  sha256: {manifest['sha256']}")
    for table, count in manifest["row_counts"].items():
        print(f"  {table:<18} {count:>8,}")


if __name__ == "__main__":
    main()
