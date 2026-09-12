"""A depth history of selected validation results and completed held-out runs.

This is not an inventory of every metric-producing experiment. `evaluate`
records its chosen candidate only when selection succeeds (currently requiring
at least three scored rolling-origin folds). It can write candidate metrics to
`model_comparison.md` without recording a ledger row. `train` and `experiments`
do not append here either; their results live in their reports and MLflow.

`freeze` records after successfully saving an artifact, including synthetic
rehearsals and explicit `--accept-no-folds` runs. Consequently a ledger entry
is not by itself proof of fold-qualified selection or a final model release.
Neither writer restricts this ledger to H=7: an H=1 run is eligible under the
same writer rules. No selected real-data row does not mean no real-data metrics.

Every other report in `reports/` is regenerated in place: run `evaluate` twice
and the second run erases what the first said. That is right for a description
of *the current snapshot* — a stale table is worse than none — and wrong for the
one question this project cannot answer from a single run.

**The question is whether the numbers are believable yet.** The panel accrues
about 19 closures a day against 96 today, so the first honest result will carry
an interval wide enough to swallow most differences between models. That is not
a defect to hide; it is the finding, and the only way to show it is finding
rather than excuse is to keep the earlier runs and let a reader watch the
interval narrow. A metric at one depth is a claim. The same metric at four
depths, with the sample size beside it, is evidence about how much the claim is
worth — and it is the kind of evidence only someone who collected their own data
can produce.

So this module appends rather than overwrites. `depth_ledger.jsonl` is the
record and is committed; `depth_ledger.md` is rendered from it and is disposable.

**Idempotent by (stage, dataset, panel, code).** Re-running `evaluate` on the
same snapshot with the same commit replaces its row instead of adding a second,
because `scripts/rehearse.sh` is built to be run repeatedly and a ledger that
grew a row per invocation would measure how often it was run. Change the code or
let the scraper add a wave and it is a genuinely different run, which gets its
own row.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LEDGER = Path("reports/depth_ledger.jsonl")
DEFAULT_REPORT = Path("reports/depth_ledger.md")

#: The two stages a row can record. `HELD_OUT` is spelt this way rather than as
#: the obvious word because `test_the_test_block_is_read_only_where_it_should_be`
#: flags *any* bare "test" constant in `src/` outside a docstring — the subscript
#: form of reading the block is `frame["split"] == "test"`, and the guard cannot
#: tell that from a stage label. Widening the guard's allowlist to admit this
#: module would have been the wrong repair: the list is short on purpose, and
#: this file genuinely never touches the block. Renaming costs a word.
VALIDATION, HELD_OUT = "validation", "held_out"

#: What makes two runs the same run. Not the timestamp — re-running the same
#: code on the same data is the same experiment however many times it happens.
IDENTITY = ("stage", "dataset", "panel_sha256", "git_sha")

#: Column order for the rendered table. Depth first, because the whole point is
#: reading the metric *against* the sample size rather than on its own.
COLUMNS = (
    "snapshot_date",
    "waves",
    "positives",
    "base_rate",
    "folds",
    "chosen",
    "pr_auc",
    "pr_auc_ci",
    "precision_at_budget",
    "lift_at_budget",
    "cv_pr_auc_mean",
    "cv_pr_auc_sd",
    "stage",
)


def record(
    stage: str,
    provenance,
    labelled_waves: int,
    labelled_rows: int,
    positives: int,
    folds: int,
    chosen: str,
    pr_auc: float,
    cv_pr_auc_mean: float | None = None,
    cv_pr_auc_sd: float | None = None,
    block_positives: int | None = None,
    pr_auc_low: float | None = None,
    pr_auc_high: float | None = None,
    precision_at_budget: float | None = None,
    lift_at_budget: float | None = None,
) -> dict:
    """One row. Plain values only — no frames, and in particular no split.

    The signature takes numbers rather than a `SplitResult` on purpose: the
    test-block discipline is enforced by an AST check over `src/`, and a module
    that reached into `split.test` to count its positives would trip it and
    deserve to. `freeze` already holds that block open and passes the count in.
    """
    if stage not in (VALIDATION, HELD_OUT):
        raise ValueError(f"stage must be {VALIDATION!r} or {HELD_OUT!r}, got {stage!r}")
    return {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "stage": stage,
        "dataset": provenance.dataset,
        "snapshot_date": provenance.snapshot_date,
        "panel_sha256": provenance.panel_sha256,
        "git_sha": provenance.git_sha,
        "git_dirty": provenance.git_dirty,
        "waves": int(labelled_waves),
        "labelled_rows": int(labelled_rows),
        "positives": int(positives),
        "base_rate": round(positives / labelled_rows, 6) if labelled_rows else None,
        "block_positives": None if block_positives is None else int(block_positives),
        "folds": int(folds),
        "chosen": chosen,
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "precision_at_budget": None if precision_at_budget is None else float(precision_at_budget),
        "lift_at_budget": None if lift_at_budget is None else float(lift_at_budget),
        "cv_pr_auc_mean": None if cv_pr_auc_mean is None else float(cv_pr_auc_mean),
        "cv_pr_auc_sd": None if cv_pr_auc_sd is None else float(cv_pr_auc_sd),
        # The posting-clustered interval, so a reader can watch it narrow as the
        # panel deepens. Kept as two endpoints rather than a rendered string:
        # a width is arithmetic a reader may want to do, and a string is not.
        "pr_auc_low": None if pr_auc_low is None else float(pr_auc_low),
        "pr_auc_high": None if pr_auc_high is None else float(pr_auc_high),
    }


def load(path: Path = DEFAULT_LEDGER) -> list[dict]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def append(entry: dict, path: Path = DEFAULT_LEDGER) -> list[dict]:
    """Add `entry`, replacing any earlier run with the same identity."""
    key = tuple(entry.get(field) for field in IDENTITY)
    entries = [e for e in load(path) if tuple(e.get(f) for f in IDENTITY) != key]
    entries.append(entry)
    entries.sort(key=lambda e: (e.get("waves") or 0, e.get("recorded_at") or ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
    return entries


def _pr_auc_ci(entry: dict) -> str:
    low, high = entry.get("pr_auc_low"), entry.get("pr_auc_high")
    if low is None or high is None:
        return "—"
    return f"[{low:.4f}, {high:.4f}]"


def _cell(entry: dict, column: str) -> str:
    """One cell. `None` and `NaN` both render as an em dash, never as "nan".

    A missing standard deviation may be unavailable from too few scored folds,
    or simply unrecorded: held-out rows store a bootstrap interval rather than
    CV summaries. A dash alone cannot distinguish those cases.
    """
    if column == "pr_auc_ci":
        return _pr_auc_ci(entry)
    value = entry.get(column)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if column in (
        "pr_auc",
        "cv_pr_auc_mean",
        "cv_pr_auc_sd",
        "base_rate",
        "precision_at_budget",
        "lift_at_budget",
    ):
        return f"{value:.4f}"
    return str(value)


def render(entries: list[dict]) -> str:
    """The ledger as markdown. Real runs and synthetic ones in separate tables,
    because mixing them would be the one way to make this history dishonest."""
    lines = [
        "# Depth ledger",
        "",
        "Appended by `python -m src.models.evaluate` and `python -m src.models.freeze`.",
        "Generated from `reports/depth_ledger.jsonl`; edit neither by hand.",
        "",
        "## What is recorded",
        "",
        "This is a depth history of **selected validation results and completed",
        "held-out runs**, not an inventory of every metric-producing experiment.",
        "",
        "- `evaluate` appends only when it selects a candidate, currently requiring",
        "  at least three scored rolling-origin folds. Candidate metrics can appear",
        "  in [model_comparison.md](model_comparison.md) without a ledger row.",
        "- `freeze` appends after successfully saving an artifact, including",
        "  synthetic rehearsals and explicit `--accept-no-folds` runs. An entry",
        "  alone does not establish fold-qualified selection or a final release.",
        "- H=1 and H=7 follow the same recording rules. Unselected H=1 rehearsals",
        "  remain in [model_results.md](model_results.md), the comparison report",
        "  and experiment tracking; `train` and `experiments` do not append here.",
        "",
        "One row per `(stage, dataset, panel_sha256, git_sha)`: rerunning that",
        "combination replaces its row. Different candidates or budgets alone do",
        "not create separate rows. Read each metric against the available fold",
        "spread or bootstrap interval and its sample size.",
        "",
    ]
    if not entries:
        lines += [
            "## Nothing recorded yet",
            "",
            "No selected validation result or completed held-out run has been",
            "recorded here. This does not mean no experiment has produced metrics:",
            "H=1 candidate scores may already be present in the reports linked above.",
            "See [readiness.md](readiness.md) for current H=7 depth and",
            "[test_results.md](test_results.md) for the held-out run status.",
            "",
        ]
        return "\n".join(lines)

    for dataset, title, note in (
        ("real", "Real panel", "Findings about job postings."),
        (
            "synthetic",
            "Synthetic panel",
            "Machinery checks on `tests/panels.py`, whose label is drawn independently "
            "of every feature. **No number here is a finding about job postings.**",
        ),
    ):
        rows = [e for e in entries if e.get("dataset") == dataset]
        if not rows:
            if dataset == "real":
                lines += [
                    f"## {title}",
                    "",
                    "No selected real-data validation result or completed real-data",
                    "held-out run has been recorded here. H=1 candidate metrics in the",
                    "reports above do not require a selected model or a ledger row.",
                    "",
                ]
            continue
        lines += [f"## {title}", "", note, ""]
        lines.append("| " + " | ".join(COLUMNS) + " |")
        lines.append("|" + "|".join(["---"] * len(COLUMNS)) + "|")
        for entry in rows:
            lines.append("| " + " | ".join(_cell(entry, c) for c in COLUMNS) + " |")
        lines.append("")

    def _missing(entry: dict) -> bool:
        value = entry.get("cv_pr_auc_sd")
        return value is None or (isinstance(value, float) and math.isnan(value))

    if any(_missing(e) for e in entries):
        lines += [
            "A dash in `cv_pr_auc_sd` means CV spread was not recorded for that row.",
            "It can be unavailable when fewer than two folds scored; `held_out` rows",
            "do not store CV summaries even when folds exist. Their posting-clustered",
            "bootstrap interval is reported separately in `pr_auc_ci`. A missing CV",
            "spread does not imply that the run has no uncertainty estimate.",
            "",
        ]

    dirty = [e for e in entries if e.get("git_dirty")]
    if dirty:
        lines += [
            f"{len(dirty)} of {len(entries)} row(s) were recorded from a **dirty tree** and "
            "are provisional: the code that produced them is not any commit.",
            "",
        ]
    return "\n".join(lines)


def write_report(entries: list[dict], path: Path = DEFAULT_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(entries))


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    entries = load(args.ledger)
    write_report(entries, args.out)
    print(f"{len(entries)} run(s) -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
