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

from src import plots
from src.data.load import load_snapshot
from src.data.snapshot import latest_snapshot
from src.models import provenance

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


#: Numeric columns whose *shape* a median and a standard deviation misdescribe.
#: Each is a long tail or a spike, which is the reason the figure exists — see
#: `src/plots.py` for why these five and not every numeric column.
DISTRIBUTION_COLUMNS = ("salary_min", "salary_max", "content_chars", "n_offices", "n_metadata")


def _log_figures(figures: list[Path], snapshot: str, n_jobs: int) -> None:
    """Put the profile's figures in MLflow, with the snapshot they describe.

    A data profile is a run over a pinned snapshot, so it has exactly the
    provenance a model run has — and a missingness grid six weeks old is only
    worth anything if you can say which snapshot it came from.
    """
    from src.models import provenance, tracking

    prov = provenance.collect(
        Path("data/raw") / snapshot / "manifest.json", n_jobs, provenance.REAL
    )
    run = tracking.log_figure_run("data profile", figures, prov, params={"snapshot": snapshot})
    if run:
        print(f"logged figures to MLflow run {run[:8]}")


def _figure_section(jobs: pd.DataFrame, snapshot: str, figures_dir: Path | None) -> list[str]:
    """The two figures for this report, or the reason there are none.

    Written as a section rather than appended at the end because the missingness
    grid belongs *beside* the coverage table it draws — the table is the evidence
    and the figure is how the pattern in it becomes visible at a glance.

    Matplotlib is an optional extra, so its absence is reported rather than
    raised: a data profile is still worth having without pictures, and a report
    that silently dropped them would leave a reader wondering whether the figures
    were omitted or never produced.
    """
    if figures_dir is None:
        return []
    if not plots.available():
        return [
            "_Figures not rendered: matplotlib is not installed. "
            "`pip install -e '.[plots]'` and re-run._",
            "",
        ]

    grid = figures_dir / f"missingness_{snapshot}.png"
    spread = figures_dir / f"distributions_{snapshot}.png"
    plots.missingness_by_source(coverage_by_source(jobs), grid)
    plots.distributions(jobs, DISTRIBUTION_COLUMNS, spread)
    _log_figures([grid, spread], snapshot, len(jobs))
    relative = Path(grid.name).parent
    return [
        f"![Percent present by source and column]({figures_dir.name}/{grid.name})",
        "",
        "Missing is not random here: the blocks of red are whole sources that never",
        'populate a field, so "this column is null" is largely a restatement of which',
        "board the row came from.",
        "",
        f"![Distributions of the long-tailed columns]({figures_dir.name}/{spread.name})",
        "",
        f"_Figures regenerate with the report; `{relative}` is written beside it._",
        "",
    ]


def build_report(snapshot_dir: Path, figures_dir: Path | None = None) -> str:
    frames = load_snapshot(snapshot_dir)
    jobs, runs, observations = frames["jobs"], frames["runs"], frames["job_observations"]
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())

    sections = [
        f"# Data profile — snapshot `{snapshot_dir.name}`",
        "",
        *provenance.header(
            provenance.collect(
                Path("data/raw") / snapshot_dir.name / "manifest.json",
                len(jobs),
                provenance.REAL,
            ),
            "python -m src.data.profile",
            extra={
                "Snapshot sha256": f"`{manifest['sha256'][:12]}…`",
                "Contents": (
                    f"{len(jobs):,} jobs · {len(observations):,} observations · {len(runs):,} runs"
                ),
            },
        ),
        "Measured facts only; the interpretation lives in `docs/data_dictionary.md`.",
        "",
        "## Columns (`jobs`)",
        "",
        _md_table(column_summary(jobs)),
        "",
        "## Coverage by source (% non-null)",
        "",
        *_figure_section(jobs, snapshot_dir.name, figures_dir),
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
    report_path.write_text(build_report(snapshot_dir, figures_dir=plots.FIGURES_DIR))
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
