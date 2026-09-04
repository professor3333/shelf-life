"""Profile a pinned snapshot: the factual half of the data dictionary.

Generates ``reports/data_profile_<snapshot>.md`` — row counts, dtypes, null
rates overall and *per source*, run completeness, and the shape of the
observation panel. Everything here is measured, not judged.

The judgements — what a column means, and whether its value exists at the
moment a prediction would be made — belong in ``docs/data_dictionary.md`` and
are written by hand. This script will scaffold that file with the measured
columns if it does not exist, but never overwrites it.

The report is deterministic: it contains no wall-clock time, only the snapshot
identity, so re-running it on the same snapshot reproduces the same file.

Usage::

    python -m src.data.profile
    python -m src.data.profile --date 2026-09-04
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.load import load_snapshot
from src.data.snapshot import latest_snapshot

REPORTS_ROOT = Path("reports")
DICTIONARY_PATH = Path("docs/data_dictionary.md")

#: Columns whose coverage differs sharply by source. Highlighted separately
#: because that pattern is the central difficulty of this dataset: for several
#: of them "missing" is a near-perfect synonym for "came from this source".
COVERAGE_COLUMNS = ("salary_min", "salary_raw", "seniority", "remote", "location", "posted_at")


def _md_table(frame: pd.DataFrame, floatfmt: str = "{:.1f}") -> str:
    """Render a DataFrame as a GitHub markdown table (no extra dependency).

    Formatting is done column-wise, not row-wise: iterating rows would upcast
    each one to a single common dtype and render every integer as a float.
    """
    display = pd.DataFrame(index=frame.index)
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            display[column] = values.map(lambda v: "" if pd.isna(v) else str(bool(v)))
        elif pd.api.types.is_float_dtype(values):
            display[column] = values.map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        elif pd.api.types.is_integer_dtype(values):
            display[column] = values.map(lambda v: "" if pd.isna(v) else f"{int(v):,}")
        else:
            display[column] = values.map(lambda v: "" if pd.isna(v) else str(v))

    header = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for idx, row in display.iterrows():
        out.append("| " + " | ".join([str(idx), *row.tolist()]) + " |")
    return "\n".join(out)


def column_summary(jobs: pd.DataFrame) -> pd.DataFrame:
    """Dtype, null rate and cardinality for every column of ``jobs``."""
    summary = pd.DataFrame(
        {
            "dtype": jobs.dtypes.astype(str),
            "non_null": jobs.notna().sum(),
            "null_pct": (jobs.isna().mean() * 100).round(1),
            "n_unique": jobs.nunique(dropna=True),
        }
    )
    summary.index.name = "column"
    return summary


def coverage_by_source(jobs: pd.DataFrame, columns=COVERAGE_COLUMNS) -> pd.DataFrame:
    """Percent non-null per source — the missingness fingerprint."""
    present = [c for c in columns if c in jobs.columns]
    table = jobs.groupby("source")[present].apply(lambda g: g.notna().mean() * 100).round(0)
    table.insert(0, "jobs", jobs.groupby("source").size())
    return table.sort_values("jobs", ascending=False)


def run_completeness(runs: pd.DataFrame) -> pd.DataFrame:
    """Per source: how many runs, how many were complete, and whether any run
    stopped at its page cap. A capped run did not observe the whole board."""
    frame = runs.copy()
    frame["hit_cap"] = (frame["pages_fetched"] >= frame["page_cap"]).fillna(False)
    table = frame.groupby("source").agg(
        runs=("id", "size"),
        ok=("status", lambda s: int((s == "ok").sum())),
        partial=("status", lambda s: int((s == "partial").sum())),
        failed=("status", lambda s: int((s == "failed").sum())),
        capped_runs=("hit_cap", "sum"),
        first_run=("started_at", lambda s: s.min().date().isoformat()),
        last_run=("started_at", lambda s: s.max().date().isoformat()),
    )
    return table.sort_values("runs", ascending=False)


def panel_shape(jobs, runs, observations) -> pd.DataFrame:
    """Per source: distinct observation days per job, and how many jobs were
    seen exactly once. A job seen once has no trajectory to learn from."""
    obs = observations.merge(runs[["id", "started_at"]], left_on="run_id", right_on="id").merge(
        jobs[["id", "source"]], left_on="job_id", right_on="id", suffixes=("_run", "_job")
    )
    obs["day"] = obs["started_at"].dt.date
    per_job = obs.groupby(["source", "job_id"])["day"].nunique().rename("days_seen").reset_index()
    table = per_job.groupby("source").agg(
        jobs=("job_id", "size"),
        mean_days_seen=("days_seen", "mean"),
        max_days_seen=("days_seen", "max"),
        seen_once=("days_seen", lambda s: int((s == 1).sum())),
    )
    table["seen_once_pct"] = (table["seen_once"] / table["jobs"] * 100).round(1)
    table["mean_days_seen"] = table["mean_days_seen"].round(2)
    for column in ("jobs", "max_days_seen", "seen_once"):
        table[column] = table[column].astype(int)
    return table.sort_values("jobs", ascending=False)


def build_report(snapshot_dir: Path) -> str:
    frames = load_snapshot(snapshot_dir)
    jobs, runs, observations = frames["jobs"], frames["runs"], frames["job_observations"]
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())

    sections = [
        f"# Data profile — snapshot `{snapshot_dir.name}`",
        "",
        "Generated by `python -m src.data.profile`. Measured facts only; the",
        "interpretation lives in `docs/data_dictionary.md`.",
        "",
        f"- snapshot sha256: `{manifest['sha256']}`",
        f"- jobs: {len(jobs):,} · observations: {len(observations):,} · runs: {len(runs):,}",
        "",
        "## Columns (`jobs`)",
        "",
        _md_table(column_summary(jobs)),
        "",
        "## Coverage by source (% non-null)",
        "",
        "Read this as the missingness fingerprint: where a column is 0 or 100 for a",
        "whole source, 'missing' is a synonym for 'came from that source'.",
        "",
        _md_table(coverage_by_source(jobs), floatfmt="{:.0f}"),
        "",
        "## Run completeness",
        "",
        "`capped_runs` counts runs where `pages_fetched >= page_cap` — the scraper",
        "stopped early, so the board was only partly observed and a posting's absence",
        "from that run is not evidence it was removed.",
        "",
        _md_table(run_completeness(runs)),
        "",
        "## Panel shape",
        "",
        _md_table(panel_shape(jobs, runs, observations)),
        "",
    ]
    return "\n".join(sections)


def scaffold_dictionary(jobs_columns) -> str:
    rows = "\n".join(f"| `{c}` |  |  |  |" for c in jobs_columns)
    return (
        "# Data dictionary\n\n"
        "One row per column of `jobs`. The measured facts (dtype, null rates, "
        "per-source coverage) are regenerated into `reports/` by "
        "`python -m src.data.profile`; this file holds the parts that require a "
        "decision.\n\n"
        "**Available at prediction point?** is the leakage question: at the moment "
        "the prediction would be made — when a posting is first seen — does this "
        "value exist yet? Answer yes / no / partly, and give the reason. A column "
        "answered 'no' cannot be a feature no matter how predictive it looks.\n\n"
        "| column | meaning | available at prediction point? | notes |\n"
        "|---|---|---|---|\n" + rows + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=None)
    args = parser.parse_args()

    snapshot_dir = Path("data/raw") / args.date if args.date else latest_snapshot()
    REPORTS_ROOT.mkdir(exist_ok=True)
    report_path = REPORTS_ROOT / f"data_profile_{snapshot_dir.name}.md"
    report_path.write_text(build_report(snapshot_dir))
    print(f"wrote {report_path}")

    if not DICTIONARY_PATH.exists():
        DICTIONARY_PATH.parent.mkdir(exist_ok=True)
        jobs_columns = load_snapshot(snapshot_dir)["jobs"].columns
        DICTIONARY_PATH.write_text(scaffold_dictionary(jobs_columns))
        print(f"scaffolded {DICTIONARY_PATH} — meanings and leakage calls are yours to fill in")
    else:
        print(f"{DICTIONARY_PATH} exists; left untouched")


if __name__ == "__main__":
    main()
