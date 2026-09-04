"""Extract as-of-t fields from the archived Greenhouse API responses.

    data/raw/boards-api.greenhouse.io/<board>/<stamp>.html.gz
        -> one row per (posting, fetch)

The scraper stores each API response before parsing it, so fields its parser
drops are still recoverable for every run since collection began. This module
reads them back.

**Why this matters more than "extra columns".** The `jobs` table holds *current*
state: reading a title from it to build a feature for a row dated 09-01 gives
the title as edited on 09-03, which leaks an edit backwards in time. Each
archived payload, by contrast, is a sealed observation of what the board said at
one instant — the correct source for any feature attached to a prediction point.

Two fields here have no equivalent in the parsed schema at all:

* ``first_published`` — when the employer actually posted. The database's
  ``first_seen`` is when *this project* first looked, which for 1,135 of 1,240
  postings is an artefact of when collection started.
* ``updated_at`` — the employer's own last-edit timestamp, as of that fetch.

Reproducibility: archived files are immutable once written (named by fetch
stamp, never rewritten), so re-running over the same set of stamps yields the
same rows. New fetches append. The manifest records exactly which files were
read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_ARCHIVE_ROOT = (
    Path.home() / "code" / "job-listing-scraper" / "data" / "raw" / "boards-api.greenhouse.io"
)
OUTPUT_ROOT = Path("data/processed/archive")

#: Filenames are the fetch instant: ``2026-09-04T03-46-01Z.html.gz``. The colons
#: of ISO-8601 are not legal in a path, so the scraper writes hyphens.
_STAMP = re.compile(r"^(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z")

#: Only used for provenance. The board token is *not* the source of truth for
#: which board a posting belongs to: one board (Duolingo) serves from its own
#: domain, so the token is absent. `source` is resolved by identity join instead.
_BOARD_TOKEN = re.compile(r"greenhouse\.io/([^/?]+)/jobs/")


@dataclass(frozen=True)
class ArchiveFile:
    path: Path
    board_dir: str
    fetched_at: pd.Timestamp


def parse_stamp(name: str) -> pd.Timestamp | None:
    """Read the fetch instant out of a filename, or None if it is not a payload."""
    match = _STAMP.match(name)
    if not match:
        return None
    date, hour, minute, second = match.groups()
    return pd.Timestamp(f"{date}T{hour}:{minute}:{second}Z")


def discover(archive_root: Path = DEFAULT_ARCHIVE_ROOT) -> list[ArchiveFile]:
    """Every archived payload, in a deterministic order."""
    if not archive_root.exists():
        raise FileNotFoundError(f"archive root not found: {archive_root}")
    found = []
    for path in sorted(archive_root.glob("*/*.html.gz")):
        stamp = parse_stamp(path.name)
        if stamp is not None:
            found.append(ArchiveFile(path=path, board_dir=path.parent.name, fetched_at=stamp))
    return sorted(found, key=lambda f: (f.fetched_at, f.board_dir))


def _joined(items: list[dict] | None, key: str) -> str | None:
    """Sorted, pipe-joined names from a list of Greenhouse sub-objects.

    Sorted so the value does not depend on the order the API happened to return,
    which would otherwise make an unchanged posting look edited.
    """
    if not items:
        return None
    names = sorted(str(item.get(key, "")).strip() for item in items if item.get(key))
    return "|".join(names) or None


def _metadata_map(items: list[dict] | None) -> str | None:
    """Board-specific custom fields, as a stable JSON object keyed by name."""
    if not items:
        return None
    pairs = {str(item.get("name", "")): item.get("value") for item in items if item.get("name")}
    return json.dumps(pairs, sort_keys=True, ensure_ascii=False) if pairs else None


def _to_utc(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce", format="ISO8601")
    return None if pd.isna(stamp) else stamp


def parse_payload(archive_file: ArchiveFile) -> list[dict]:
    """One record per posting in one archived response."""
    with gzip.open(archive_file.path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)

    records = []
    for job in payload.get("jobs", []):
        url = job.get("absolute_url") or ""
        token = _BOARD_TOKEN.search(url)
        content = job.get("content") or ""
        records.append(
            {
                "fetched_at": archive_file.fetched_at,
                "board_dir": archive_file.board_dir,
                "board_token": token.group(1) if token else None,
                "source_id": str(job["id"]),
                "company_name": job.get("company_name"),
                "requisition_id": job.get("requisition_id"),
                "internal_job_id": (
                    str(job["internal_job_id"]) if job.get("internal_job_id") else None
                ),
                "first_published": _to_utc(job.get("first_published")),
                "updated_at": _to_utc(job.get("updated_at")),
                "application_deadline": _to_utc(job.get("application_deadline")),
                "location_name": (job.get("location") or {}).get("name"),
                "departments": _joined(job.get("departments"), "name"),
                "n_departments": len(job.get("departments") or []),
                "offices": _joined(job.get("offices"), "name"),
                "office_locations": _joined(job.get("offices"), "location"),
                "n_offices": len(job.get("offices") or []),
                "metadata_json": _metadata_map(job.get("metadata")),
                "n_metadata": len(job.get("metadata") or []),
                "content_chars": len(content),
                "content": content,
            }
        )
    return records


def load_archive(
    archive_root: Path = DEFAULT_ARCHIVE_ROOT, *, include_content: bool = False
) -> pd.DataFrame:
    """All archived payloads as one frame: a (posting, fetch) panel.

    ``content`` is excluded by default — it is the largest field by an order of
    magnitude and most callers want the structured columns.
    """
    records: list[dict] = []
    for archive_file in discover(archive_root):
        records.extend(parse_payload(archive_file))

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame = frame.sort_values(["fetched_at", "board_dir", "source_id"], kind="stable")
    frame = frame.reset_index(drop=True)
    if not include_content:
        frame = frame.drop(columns=["content"])
    return frame


def attach_source(frame: pd.DataFrame, snapshot_dir: Path | None = None) -> pd.DataFrame:
    """Resolve each posting's `source` by identity join against the snapshot.

    The board token in ``absolute_url`` is not usable for this: Duolingo serves
    from ``careers.duolingo.com`` and carries no token. The `jobs` table is the
    authoritative identity register, and identity is one of the two things it
    may be read for — the other being labels.
    """
    from src.data.load import load_snapshot

    jobs = load_snapshot(snapshot_dir)["jobs"]
    identity = jobs[["source_id", "source"]].drop_duplicates("source_id")
    return frame.merge(identity, on="source_id", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    files = discover(args.archive_root)
    frame = load_archive(args.archive_root, include_content=True)
    frame = attach_source(frame)

    args.out.mkdir(parents=True, exist_ok=True)
    content = frame[["fetched_at", "source_id", "content"]]
    fields = frame.drop(columns=["content"])
    fields.to_parquet(args.out / "archive_fields.parquet", index=False)
    content.to_parquet(args.out / "archive_content.parquet", index=False)

    manifest = {
        "archive_root": str(args.archive_root),
        "files_read": len(files),
        "stamps": sorted({f.fetched_at.isoformat() for f in files}),
        "rows": len(frame),
        "postings": int(frame["source_id"].nunique()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"payload files read : {len(files):,}")
    print(f"(posting, fetch) rows: {len(frame):,}")
    print(f"distinct postings  : {frame['source_id'].nunique():,}")
    print(f"unmatched source   : {int(frame['source'].isna().sum()):,}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
