"""Does "disappeared from the board" mean what the label needs it to mean?

`docs/problem_definition.md` §4 defines a positive as a posting absent from two
consecutive complete runs and never seen again. That is deliberately *not*
"filled", and the README says so. But a posting can also vanish for reasons that
have nothing to do with hiring at all, and those split into two kinds that
deserve very different treatment.

**Mechanical causes are checkable, and they are the dangerous ones.** An employer
migrating ATS, a board changing its pagination, a posting relisted under a new
id — each would remove postings for reasons unrelated to the outcome, and each
would do it *systematically*. Systematic contamination is what biases a model,
so it is worth measuring rather than conceding. This module measures it.

**Outcome causes are not checkable by anyone.** Filled, cancelled, withdrawn,
expired: all four are "the employer took it down", and a vanished Greenhouse
posting carries no signal about which. Neither the board nor the employer
publishes one. A study reporting "actually filled: 43" would be producing the
strongest-sounding number in the repository from the weakest evidence, so this
module reports that they are indistinguishable and stops.

**Every rate here comes with its control, and that is enforced by the types.**
The first version of this audit found that 4 of 100 closed postings had their
requisition relisted under a new posting id within about a day, which reads as a
4% false-positive rate in the target. It is not: still-open postings are
relisted at *8.3%*, twice as often. The uncontrolled number was not merely
imprecise, it pointed the wrong way. So `compare_relisting` returns both arms or
nothing, and there is no public function in this module that will hand back a
bare rate for someone to interpret alone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.models.train_baseline import _table

DEFAULT_PANEL = Path("data/processed/features/job_days_h1_calendar.parquet")
DEFAULT_REPORT = Path("reports/label_validity.md")

#: How long after a disappearance a reappearance still counts as a relisting.
#: Generous on purpose: the observed relistings all returned inside 1.6 days, and
#: a window too tight would make the audit look clean by construction.
RELIST_WINDOW = pd.Timedelta(days=1.6)

#: The identities a posting can be relisted under. `requisition_id` is the strong
#: one — it is the employer's own key for the role, so a match is the same role
#: by the employer's definition. `title` is weak and is kept precisely because it
#: is weak: on a large board many distinct roles share a title, and its control
#: arm is what shows that.
RELIST_KEYS = ("requisition_id", "title")


@dataclass(frozen=True)
class Comparison:
    """A rate and the rate it must be read against. Never one without the other."""

    key: str
    closed_hits: int
    closed_n: int
    open_hits: int
    open_n: int

    @property
    def closed_rate(self) -> float | None:
        return self.closed_hits / self.closed_n if self.closed_n else None

    @property
    def open_rate(self) -> float | None:
        return self.open_hits / self.open_n if self.open_n else None

    @property
    def standard_error(self) -> float | None:
        """SE of the difference between the two rates, normal approximation."""
        if self.closed_rate is None or self.open_rate is None:
            return None
        p, q = self.closed_rate, self.open_rate
        return (p * (1 - p) / self.closed_n + q * (1 - q) / self.open_n) ** 0.5

    @property
    def verdict(self) -> str:
        """Three states, not two, because `closed_rate > open_rate` is not a finding.

        The first version of this property was that bare comparison, and on the
        real panel it labelled a 12.0%-against-11.0% difference — twelve events
        against eight — as *elevated*, with a recommendation to change the
        labelling rule. One point of separation on tens of events is noise, and a
        module written to stop an uncontrolled rate being over-read had reproduced
        the same error one level up: it controlled the rate and then over-read the
        difference.

        So a difference has to clear twice its standard error before it is called
        anything. That is a crude bar — a normal approximation on small counts —
        and crude is the right shape here, because its job is to refuse a reading
        the data cannot support rather than to price one it can.
        """
        if self.closed_rate is None or self.open_rate is None:
            return "not assessable"
        difference = self.closed_rate - self.open_rate
        if abs(difference) <= 2 * (self.standard_error or 0):
            return "indistinguishable"
        return "elevated" if difference > 0 else "below control"


def _last_sighting(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per posting: the last time we saw it, and what it looked like."""
    return frame.sort_values("t").groupby(["source", "source_id"], sort=False).tail(1)


def _relisted(panel: pd.DataFrame, row, key: str, window: pd.Timedelta) -> bool:
    """Did an equal `key` appear under a *different* posting id, soon after?"""
    value = getattr(row, key)
    if pd.isna(value):
        return False
    later = panel[
        (panel["source"] == row.source)
        & (panel["t"] > row.t)
        & (panel["t"] <= row.t + window)
        & (panel[key] == value)
        & (panel["source_id"] != row.source_id)
    ]
    if key == "title":  # a title alone is not an identity; require the employer too
        later = later[later["company"] == row.company]
    return not later.empty


def compare_relisting(
    panel: pd.DataFrame, key: str, window: pd.Timedelta = RELIST_WINDOW
) -> Comparison:
    """Relisting rate among closed postings, against the rate among survivors.

    The control arm is restricted to the same crawl instants as the closed arm,
    so the two are drawn from comparable board states rather than from different
    weeks of a growing panel.
    """
    labelled = panel[panel["label_observable"]]
    closed = _last_sighting(labelled[labelled["y"] == 1])
    survivors = _last_sighting(labelled[labelled["y"] == 0])
    survivors = survivors[survivors["t"].isin(set(closed["t"]))]

    def count(sample: pd.DataFrame) -> tuple[int, int]:
        usable = sample[sample[key].notna()]
        hits = sum(_relisted(panel, row, key, window) for row in usable.itertuples())
        return hits, len(usable)

    closed_hits, closed_n = count(closed)
    open_hits, open_n = count(survivors)
    return Comparison(key, closed_hits, closed_n, open_hits, open_n)


def board_stability(panel: pd.DataFrame) -> pd.DataFrame:
    """Per source: the board's size range, and its worst one-day fall.

    A migration or a pagination change removes a large fraction of a board at
    once, so it shows here as a cliff rather than as drift. A source whose worst
    day is a few percent did not lose its listings to a systems change.
    """
    sizes = panel.groupby(["source", panel["t"].dt.floor("D")])["source_id"].nunique()
    rows = []
    for source, series in sizes.groupby(level=0):
        series = series.droplevel(0).sort_index()
        # Ignore a source's first observed day: arriving is not a fall.
        drops = (series.diff() / series.shift()).dropna()
        rows.append(
            {
                "source": source,
                "days": len(series),
                "smallest": int(series.min()),
                "largest": int(series.max()),
                "worst_daily_fall": f"{drops.min():.1%}" if len(drops) else "—",
            }
        )
    return pd.DataFrame(rows).sort_values("largest", ascending=False)


def closure_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    """Closures per source per day. A systems change empties a board in one day."""
    labelled = panel[panel["label_observable"]]
    positives = labelled[labelled["y"] == 1]
    if positives.empty:
        return pd.DataFrame()
    table = positives.groupby(["source", positives["t"].dt.floor("D")]).size().unstack(fill_value=0)
    table.columns = [c.date().isoformat() for c in table.columns]
    return table.reset_index()


def _verdict(comparison: Comparison) -> str:
    verdict = comparison.verdict
    if verdict == "not assessable":
        return "not assessable — one arm is empty"

    difference = (comparison.closed_rate or 0) - (comparison.open_rate or 0)
    margin = 2 * (comparison.standard_error or 0)
    arithmetic = f"{difference:+.1%} ± {margin:.1%}"

    if verdict == "indistinguishable":
        return (
            f"**indistinguishable** ({arithmetic}) — the difference does not clear "
            "twice its standard error, so this sample says nothing either way"
        )
    if verdict == "below control":
        return (
            f"**below control** ({arithmetic}) — relisting is *less* common among "
            "postings the label calls closed than among postings that stayed up"
        )
    return (
        f"**elevated** ({arithmetic}) — relisting is more common among closed "
        "postings than among survivors, which is contamination of the target and "
        "wants a labelling rule rather than a caveat"
    )


def render(panel: pd.DataFrame) -> str:
    comparisons = [compare_relisting(panel, key) for key in RELIST_KEYS]
    labelled = panel[panel["label_observable"]]
    positives = int((labelled["y"] == 1).sum())

    lines = [
        "# Label validity",
        "",
        "Generated by `python -m src.data.label_audit`. Regenerate rather than edit.",
        "",
        "A positive is a posting absent from two consecutive complete runs and never",
        "seen again. That is **not** the same as *filled*, and this file is about how",
        "far apart the two are.",
        "",
        "## What can be checked, and what cannot",
        "",
        "A posting can vanish for reasons unrelated to hiring. They split in two:",
        "",
        "- **Mechanical** — an ATS migration, a board changing behaviour, a posting",
        "  relisted under a new id. These would remove postings *systematically*, which",
        "  is the kind of contamination that biases a model, and they leave traces in",
        "  the panel. They are measured below.",
        "- **Outcome** — filled, cancelled, withdrawn, expired. All four are *the",
        "  employer took it down*, and a vanished posting carries no signal about which.",
        "  Neither the board nor the employer publishes one. **These are not",
        "  distinguishable, by this project or by a human reading the dead URL**, and no",
        "  count of them appears in this file for that reason.",
        "",
        f"## Relisting under a new posting id — {positives} closure(s)",
        "",
        "If a posting disappears only to return under a new id, the label has recorded",
        "a closure that did not happen. Every rate is therefore reported against the",
        "same rate among postings that *stayed up*, because a bare rate here cannot be",
        "read at all: the first run of this audit found 4 of 100 closed postings",
        "relisted and took it for a 4% error rate in the target. Survivors relist at a",
        "similar rate, so the 4% measures how these boards behave rather than anything",
        "about closure — and whether the two rates differ is a further question the",
        "counts are not yet large enough to answer.",
        "",
        "| identity | closed | control (still open) | reading |",
        "|---|---|---|---|",
    ]
    for c in comparisons:
        closed = f"{c.closed_hits}/{c.closed_n}" + (
            f" = {c.closed_rate:.1%}" if c.closed_rate is not None else ""
        )
        control = f"{c.open_hits}/{c.open_n}" + (
            f" = {c.open_rate:.1%}" if c.open_rate is not None else ""
        )
        lines.append(f"| `{c.key}` | {closed} | {control} | {_verdict(c)} |")

    stability = board_stability(panel)
    dispersion = closure_dispersion(panel)
    lines += [
        "",
        "`requisition_id` is the employer's own key for a role, so a match is the same",
        "role by their definition. `title` is weak — many distinct roles share one on a",
        "large board — and it is kept *because* it is weak: its control arm is what",
        "demonstrates that a raw title-match rate measures the board's naming habits",
        "rather than anything about the label.",
        "",
        "**What the counts do not support.** These are tens of events, so a difference",
        "is only called anything once it clears twice its standard error. Most will not,",
        "and *indistinguishable* is the honest majority verdict at this depth — it means",
        "the audit cannot see contamination of a few percent, not that there is none.",
        "It sharpens as closures accumulate; the panel adds roughly 19 a day.",
        "",
        "## Board stability",
        "",
        "A migration or a pagination change takes a large fraction of a board at once,",
        "so it appears as a cliff rather than as drift.",
        "",
        _table(stability, list(stability.columns)),
        "",
        "## Closure dispersion",
        "",
        "Closures spread across sources and days are consistent with ordinary hiring.",
        "A systems change would empty one board on one day.",
        "",
    ]
    lines.append(
        _table(dispersion, list(dispersion.columns))
        if not dispersion.empty
        else "_No closures yet._"
    )
    lines += [
        "",
        "## What would strengthen this",
        "",
        "Fetching a sample of closed postings' URLs and recording whether each is",
        "genuinely gone from the board. That validates *scraper fidelity* — did the",
        "posting really leave, or did the crawl miss it — which is checkable and has a",
        "real answer. It does not, and cannot, establish why the posting left.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(panel))
    for key in RELIST_KEYS:
        comparison = compare_relisting(panel, key)
        print(
            f"{key:16} closed {comparison.closed_hits}/{comparison.closed_n}  "
            f"control {comparison.open_hits}/{comparison.open_n}"
        )
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
