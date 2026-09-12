"""The README's one block of moving numbers, generated from the canonical reports.

The README is narrative and should stay that way. Every count, rate, date and
verdict in it that moves with the panel was, until 2026-09-12, typed by hand —
and the panel moves every day, so the README carried a test count from one
week, a snapshot from another, and a reading of the H=1 comparison that the
regenerated table no longer supported. The reports under `reports/` are the
authoritative values; this module reads four of them and writes the README's
"current state" block between two markers, so the only numbers in the README
that can drift are the ones a test holds equal to the reports.

Canonical sources, and what is taken from each:

    reports/readiness.md          snapshot, the two gates, legal cuts today
    reports/model_comparison.md   horizon, the ladder table, the verdict
    reports/test_results.md       whether the held-out block has been opened
    reports/depth_ledger.jsonl    how many real and synthetic runs are kept

Parsed from the reports' markdown rather than recomputed, on purpose: the test
that guards the block runs in CI without the data, and "the README agrees with
the committed reports" is the claim, not "the README agrees with the panel".
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPORTS = Path("reports")
README = Path("README.md")
BEGIN = "<!-- generated: begin — python -m src.data.readme_summary; do not edit by hand -->"
END = "<!-- generated: end -->"

_PIPE_ROW = re.compile(r"^\|(.+)\|$")


def _cells(line: str) -> list[str]:
    match = _PIPE_ROW.match(line.strip())
    return [cell.strip() for cell in match.group(1).split("|")] if match else []


def _table(text: str, first_header: str) -> list[dict[str, str]]:
    """The first markdown table whose header row starts with `first_header`."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        cells = _cells(line)
        if cells and cells[0] == first_header:
            header = cells
            rows = []
            for row in lines[index + 2 :]:
                cells = _cells(row)
                if not cells:
                    break
                rows.append(dict(zip(header, cells, strict=False)))
            return rows
    return []


def _kv(text: str, key: str) -> str | None:
    """A `| key | value |` row's value."""
    for line in text.splitlines():
        cells = _cells(line)
        if len(cells) == 2 and cells[0] == key:
            return cells[1]
    return None


def read(reports: Path = REPORTS) -> dict:
    readiness = (reports / "readiness.md").read_text()
    comparison = (reports / "model_comparison.md").read_text()
    held_out = (reports / "test_results.md").read_text()
    ledger_path = reports / "depth_ledger.jsonl"
    ledger = [
        json.loads(line)
        for line in (ledger_path.read_text().splitlines() if ledger_path.exists() else [])
        if line.strip()
    ]

    data = _kv(readiness, "Data") or ""
    snapshot = re.search(r"snapshot `([^`]+)`", data)
    code = _kv(readiness, "Code") or ""
    sha = re.search(r"`([0-9a-f]{7,})`", code)
    gates = {row["gate"]: row for row in _table(readiness, "gate")}
    cuts = re.search(r"Legal cuts available today: \*\*(\d+)\*\*", readiness)

    ladder = _table(comparison, "model")
    horizon = _kv(comparison, "Horizon") or ""
    verdict = re.search(r"\*\*Chosen: ([^*]+)\*\* — ([^\n]+)", comparison)
    fitted = [
        row for row in ladder if row.get("val_pr_auc", "—") not in ("—", "") and row["model"]
    ]
    best = max(fitted, key=lambda r: float(r["val_pr_auc"]), default=None)
    prior = next((r for r in ladder if r["model"] == "prior"), None)
    scored = [int(r["folds_scored"]) for r in ladder if r.get("folds_scored", "").isdigit()]
    folds = max(scored, default=0)

    return {
        "snapshot": snapshot.group(1) if snapshot else None,
        "sha": sha.group(1) if sha else None,
        "gates": gates,
        "legal_cuts": int(cuts.group(1)) if cuts else None,
        "held_out_opened": "## Nothing below has run" not in held_out,
        "h1_horizon": horizon,
        "candidates": len(ladder),
        "folds_scored": folds,
        "verdict": (verdict.group(1).strip(), verdict.group(2).strip()) if verdict else None,
        "best": best,
        "prior": prior,
        "ledger_real": sum(1 for r in ledger if r.get("dataset") == "real"),
        "ledger_synthetic": sum(1 for r in ledger if r.get("dataset") == "synthetic"),
        "latest_profile": sorted(p.name for p in reports.glob("data_profile_*.md"))[-1:],
    }


def render(state: dict) -> str:
    g = state["gates"]
    split = g.get("a legal three-way split", {})
    folds = g.get("3 rolling-origin folds", {})
    lines = [
        BEGIN,
        f"**Current state, read from the generated reports** — snapshot `{state['snapshot']}`, "
        f"regenerated at `{state['sha']}`:",
        "",
        f"- **H=7 readiness** ([`readiness.md`](reports/readiness.md)): {split.get('have', '?')} "
        f"labelled waves. A legal split needs {split.get('labelled waves needed', '?')} "
        f"(short by {split.get('short by', '?')}, projected {split.get('projected', '?')}); three "
        f"rolling-origin folds need {folds.get('labelled waves needed', '?')} (short by "
        f"{folds.get('short by', '?')}, projected {folds.get('projected', '?')}). Legal cuts "
        f"today: {state['legal_cuts']}.",
        "- **H=7 held-out block** ([`test_results.md`](reports/test_results.md)): "
        + ("**opened** — the result is in the report." if state["held_out_opened"]
           else "**not opened**; the report records the refusal."),
    ]
    if state["verdict"]:
        chosen, reason = state["verdict"]
        best, prior = state["best"], state["prior"]
        best_text = (
            f" Best single validation draw: `{best['model']}` at PR-AUC {best['val_pr_auc']}"
            + (f" against a prior of {prior['val_pr_auc']}" if prior else "")
            + " — one draw with no error bar, which is why it selected nothing."
            if best
            else ""
        )
        lines.append(
            f"- **H=1 rehearsal** ([`model_comparison.md`](reports/model_comparison.md); "
            f"{state['h1_horizon']}): {state['candidates']} candidates, "
            f"{state['folds_scored']} rolling-origin folds scored. Verdict: **{chosen}** — "
            f"{reason}.{best_text}"
        )
    lines.append(
        f"- **Depth ledger** ([`depth_ledger.md`](reports/depth_ledger.md)): "
        f"{state['ledger_real']} real run(s) kept, {state['ledger_synthetic']} synthetic."
    )
    if state["latest_profile"]:
        name = state["latest_profile"][0]
        lines.append(f"- **Latest data profile**: [`{name}`](reports/{name}).")
    lines.append(END)
    return "\n".join(lines)


def splice(readme_text: str, block: str) -> str:
    """The README with its generated block replaced, or inserted if absent."""
    if BEGIN in readme_text and END in readme_text:
        start = readme_text.index(BEGIN)
        end = readme_text.index(END) + len(END)
        return readme_text[:start] + block + readme_text[end:]
    raise ValueError("README has no generated block markers")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=REPORTS)
    parser.add_argument("--readme", type=Path, default=README)
    parser.add_argument("--check", action="store_true", help="exit 1 if the README is behind")
    args = parser.parse_args(argv)
    block = render(read(args.reports))
    current = args.readme.read_text()
    updated = splice(current, block)
    if args.check:
        if updated != current:
            print("README's generated block is behind the reports; run without --check")
            return 1
        print("README's generated block matches the reports")
        return 0
    args.readme.write_text(updated)
    print(f"wrote -> {args.readme} (generated block)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
