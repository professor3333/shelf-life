"""Is the evidence one bundle? The reports a freeze is chosen on must agree.

A freeze spends the held-out block on a candidate that was chosen by reading
`model_comparison.md`, checked against `board_fingerprint.md`, and gated by
`readiness.md`. Each of those says which snapshot and which commit produced it.
Nothing, until 2026-09-20, checked that they said the *same* one — and on that
day they did not: `readiness.md` came from a feature branch, `model_comparison.md`
from `main` two commits later, and `test_results.md` from a snapshot two days
older. Harmless while every report said *not ready*. On the day one of them says
*ready*, a bundle assembled from three different pasts is a selection made on
evidence that no single commit produced, and the artifact's provenance line
would be true of the freeze and false of the choice behind it.

`scripts/regenerate_reports.sh` already refuses a dirty tree and writes every
report from one pinned snapshot at one commit. That is a procedure. This module
is the check that the procedure was followed, asked at the step that cannot be
repeated: `freeze` on a real panel refuses unless every report in
`EVIDENCE_REPORTS` names the same snapshot, that snapshot is the one the panel
was built from, all name the same commit, and that commit is `HEAD` or differs
from it only under `reports/` and the README's generated block — because the
runbook commits the regenerated reports *before* freezing, which moves `HEAD` by
exactly one commit of reports.

No override flag. The remedy is `./scripts/regenerate_reports.sh`, which takes
minutes, and a flag that skips a minutes-long step is how a rule becomes a
suggestion.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from src.models import provenance

#: The reports a real freeze is chosen on. All of them are written for the
#: present panel at the build's horizon by `scripts/regenerate_reports.sh`, in
#: one run, so they can and must agree. Deliberately not in the set:
#: `test_results.md` is what the freeze *writes*; `label_check.md` samples live
#: pages and keeps its previous provenance under `--no-net`; the dated
#: `data_profile_<date>.md` files are each about their own snapshot; the `*_h1_*`
#: rehearsal reports are evidence about the machinery, not about the candidate;
#: and `depth_ledger.md`, the cold-start reports and `feature_hypotheses.md`
#: carry no panel provenance by design.
EVIDENCE_REPORTS: tuple[str, ...] = (
    "readiness.md",
    "label_validity.md",
    "cohort_audit.md",
    "baseline_results.md",
    "model_results.md",
    "experiment_log.md",
    "model_comparison.md",
    "board_fingerprint.md",
)

DEFAULT_REPORTS_DIR = Path("reports")

#: Paths a commit may touch between the evidence's commit and `HEAD` without
#: invalidating the evidence: the reports themselves, and the README, whose
#: generated block `regenerate_reports.sh` rewrites in the same run.
REPORTS_ONLY: tuple[str, ...] = ("reports/", "README.md")

_DATA_ROW = re.compile(
    r"^\| Data \| \*\*(?P<dataset>\w+)\*\*(?: · snapshot `(?P<snapshot>[0-9-]+)`)?"
)
_CODE_ROW = re.compile(r"^\| Code \| `(?P<sha>[0-9a-f]{7,40})`")


@dataclass(frozen=True)
class Header:
    """What a report's provenance block says about where it came from."""

    report: str
    dataset: str | None
    snapshot: str | None
    sha: str | None


def parse_header(report: str, text: str) -> Header:
    """Read the `| Data |` and `| Code |` rows `provenance.header` writes.

    Tolerant of a missing row — a report that does not say is reported as
    saying nothing, and `problems` treats nothing as disagreement.
    """
    dataset = snapshot = sha = None
    for line in text.splitlines():
        if (m := _DATA_ROW.match(line)) and dataset is None:
            dataset, snapshot = m["dataset"], m["snapshot"]
        elif (m := _CODE_ROW.match(line)) and sha is None:
            sha = m["sha"]
        if dataset is not None and sha is not None:
            break
    return Header(report, dataset, snapshot, sha)


def read_headers(reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[Header]:
    headers = []
    for name in EVIDENCE_REPORTS:
        path = reports_dir / name
        if not path.exists():
            headers.append(Header(name, None, None, None))
            continue
        headers.append(parse_header(name, path.read_text()))
    return headers


def _head_sha() -> str | None:
    return provenance._git("rev-parse", "HEAD")


def _paths_changed_since(sha: str) -> list[str] | None:
    """Every path that differs between `sha` and `HEAD`, or None if git cannot say."""
    out = provenance._git("diff", "--name-only", sha, "HEAD")
    if out is None:
        return None
    return [p for p in out.splitlines() if p]


def _same_commit(short: str, full: str | None) -> bool:
    return full is not None and full.startswith(short)


def problems(
    headers: Iterable[Header],
    *,
    panel_snapshot: str | None,
    head_sha: str | None,
    changed_since: Callable[[str], list[str] | None],
) -> list[str]:
    """Every reason the bundle is not one bundle. Empty means it is.

    Pure: the git and filesystem reads are injected, so a test can describe a
    bundle in a few lines and the freeze can hand in the real thing.
    """
    headers = list(headers)
    found: list[str] = []

    missing = [h.report for h in headers if h.dataset is None and h.sha is None]
    if missing:
        found.append(f"no provenance header in: {', '.join(missing)}")
    headers = [h for h in headers if h.report not in missing]

    unreal = [h.report for h in headers if h.dataset != provenance.REAL]
    if unreal:
        found.append(f"not about the real panel: {', '.join(unreal)}")

    snapshots = {h.snapshot for h in headers}
    if len(snapshots) > 1:
        found.append(
            "snapshots disagree: "
            + ", ".join(f"{h.report}={h.snapshot}" for h in headers if h.snapshot)
        )
    elif snapshots and panel_snapshot is not None and snapshots != {panel_snapshot}:
        (only,) = snapshots
        found.append(
            f"the evidence is about snapshot {only}; the panel being frozen is from "
            f"{panel_snapshot}"
        )

    shas = {h.sha for h in headers if h.sha}
    if len(shas) > 1:
        found.append(
            "commits disagree: " + ", ".join(f"{h.report}={h.sha}" for h in headers if h.sha)
        )
    elif len(shas) == 1:
        (sha,) = shas
        if head_sha is None:
            found.append("not a git checkout, so the evidence's commit cannot be compared to HEAD")
        elif not _same_commit(sha, head_sha):
            changed = changed_since(sha)
            if changed is None:
                found.append(f"commit {sha} is not in this checkout's history")
            else:
                outside = [p for p in changed if not p.startswith(REPORTS_ONLY)]
                if outside:
                    found.append(
                        f"source changed since the evidence was written at {sha}: "
                        + ", ".join(outside[:8])
                        + (" …" if len(outside) > 8 else "")
                    )
    return found


def check(panel_path: Path, reports_dir: Path = DEFAULT_REPORTS_DIR) -> list[str]:
    """The real thing: this checkout's reports, this checkout's git."""
    return problems(
        read_headers(reports_dir),
        panel_snapshot=provenance._snapshot_date_for(panel_path),
        head_sha=_head_sha(),
        changed_since=_paths_changed_since,
    )


def table(headers: Iterable[Header]) -> str:
    rows = ["| report | data | snapshot | commit |", "|---|---|---|---|"]
    for h in headers:
        rows.append(
            f"| `{h.report}` | {h.dataset or '—'} | {h.snapshot or '—'} | "
            f"{'`' + h.sha + '`' if h.sha else '—'} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=None)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    from src.models.train_baseline import DEFAULT_PANEL

    panel = args.panel or DEFAULT_PANEL
    headers = read_headers(args.reports)
    print(table(headers))
    print()
    found = problems(
        headers,
        panel_snapshot=provenance._snapshot_date_for(panel),
        head_sha=_head_sha(),
        changed_since=_paths_changed_since,
    )
    if found:
        print("the evidence is not one bundle:")
        for line in found:
            print(f"  - {line}")
        print("remedy: ./scripts/regenerate_reports.sh, then commit reports/ as one change.")
        raise SystemExit(5)
    print("one bundle: every report names the same snapshot and the same commit.")


if __name__ == "__main__":  # pragma: no cover
    main()
