"""Can the seven-day model be trained yet, and if not, what exactly is missing?

`scripts/watch_depth.sh` answers this every morning into a gitignored log. That
is the right home for a daily measurement and the wrong one for the answer: the
log is a record of a wait, invisible to anyone who has not run the script, and
the question *is the H=7 model ready* is asked by readers of the repository.

So this writes the same measurement as a committed report — the two gates, the
distance to each, whether the panel is still accruing, and what runs on the day
it clears. It is the readiness answer with a provenance header on it, which is
what makes yesterday's copy obviously yesterday's.

**Accrual is reported before depth, because it changes what the depth means.**
Eighteen waves short is a wait if crawls are arriving and an outage if they are
not, and the two look identical in any report that only counts waves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.split import (
    DEFAULT_TARGET_FOLDS,
    accrual_status,
    crawl_waves,
    feasible_cuts,
    minimum_waves,
    projected_clear,
)
from src.features.assemble import DEFAULT_HORIZON, horizon_banner, panel_path
from src.models import provenance

DEFAULT_REPORT = Path("reports/readiness.md")


def _gate_table(depth: dict, ahead: dict) -> list[str]:
    def when(date) -> str:
        if ahead["stalled"]:
            return "**not projected** — the panel is not accruing"
        return "already open" if date is None else f"{date.date()}"

    return [
        "| gate | labelled waves needed | have | short by | projected |",
        "|---|---|---|---|---|",
        f"| a legal three-way split | {depth['needed']} | {depth['present']} | "
        f"{depth['shortfall']} | {when(ahead['split_clears'])} |",
        f"| {depth['target_folds']} rolling-origin folds | {depth['needed_for_folds']} | "
        f"{depth['present']} | {depth['folds_shortfall']} | {when(ahead['folds_clear'])} |",
    ]


def render(panel: pd.DataFrame, prov, now: pd.Timestamp | None = None) -> str:
    depth = minimum_waves(panel, target_folds=DEFAULT_TARGET_FOLDS)
    ahead = projected_clear(panel, now=now)
    accrual = accrual_status(panel, now=now)
    usable = int(feasible_cuts(panel)["valid"].sum())

    lines = [
        "# Readiness — can the seven-day model be trained yet?",
        "",
        *provenance.header(prov, "python -m src.data.readiness", horizon_banner(panel)),
        "**No.** What follows is how far off, and whether the distance is closing.",
        "",
        "## Is the panel still accruing?",
        "",
    ]

    if accrual["stalled"]:
        lines += [
            f"**No — and this is the finding.** The newest crawl in the panel is "
            f"`{accrual['newest_wave']:%Y-%m-%d %H:%M}Z`, {accrual['age'].days} day(s) old "
            f"against an observed cadence of {accrual['spacing']}. About "
            f"{accrual['waves_missed']} wave(s) have not arrived.",
            "",
            "Every shortfall below is therefore a distance that is **not closing**, and no",
            "date is projected: `newest wave + shortfall × spacing` is arithmetic on an",
            "assumption that has stopped holding, and its answer would be the most",
            "confident-looking line in this file.",
            "",
            "Eighteen waves short is a wait if crawls are arriving and an outage if they are",
            "not. The two are identical in any report that only counts waves, which is why",
            "this section is above the gates rather than below them.",
            "",
            "Check the collector **and the path from it to this panel**. A collector that runs",
            "but whose output no longer reaches the panel is indistinguishable from one that",
            "has stopped, and only one of them is fixed by restarting a scheduler.",
            "",
        ]
    else:
        lines += [
            f"Yes. The newest crawl is `{accrual['newest_wave']:%Y-%m-%d %H:%M}Z`, "
            f"{accrual['age']} old against a cadence of {accrual['spacing']} — within "
            "tolerance, so the projections below are meaningful.",
            "",
        ]

    lines += [
        "## The two gates",
        "",
        "They are not the same day. The first is when a cut is *legal*; the second is when",
        "a model can be **selected** on one, which needs folds to compare across. A legal",
        "split with no folds yields a single number with no spread, and choosing on it is",
        "selection on the validation block.",
        "",
        *_gate_table(depth, ahead),
        "",
        f"At H={DEFAULT_HORIZON} the embargo is {depth['embargo']} against a cadence of "
        f"{depth['spacing']}, so each of the two block boundaries discards "
        f"{depth['burnt_per_boundary']} wave(s), and the newest {depth['blind_tail']} "
        "labelled wave(s) cannot carry a positive at all — a removal needs the posting "
        "absent from two consecutive later runs.",
        "",
        f"Legal cuts available today: **{usable}**. Rolling-origin folds at the deepest "
        f"legal cut: **{depth['folds_available']}**.",
        "",
        "**These are properties of this snapshot, not dates.** They move with the panel: a",
        "widened embargo or a missed crawl pushes them later, never earlier.",
        "",
        "## What runs on the day it clears",
        "",
        "In order, and none of it is written on the day — all of it exists and is tested",
        "against the synthetic panel now:",
        "",
        "1. `./scripts/rehearse.sh` — the ladder, tuning and feature ablations on train and",
        "   validation only. It never calls `freeze`.",
        "2. `python -m src.models.evaluate` — the ten candidates compared across folds, the",
        "   pre-registered rule in `docs/design.md` §14 applied, the operating threshold",
        "   chosen against an alert budget, calibration, per-source and seen/unseen",
        "   breakdowns, and the leave-one-board-out transfer measurement.",
        "3. Read the verdict — and the leave-one-board-out table beneath it. `design.md`",
        "   §4a: the production features name the board at 100% (`board_fingerprint.md`),",
        "   so a held-out board whose score collapses to its base rate says the candidate",
        "   spent that fingerprint, and is a reason not to freeze it. **The model choice is",
        "   a person's decision**, which is why the watch stops here rather than continuing.",
        "4. `python -m src.models.freeze --run <spec>` — fits, fixes the threshold on",
        "   validation, then opens the held-out block **once** and reports the score with",
        "   posting-clustered bootstrap intervals.",
        "5. Release and deploy: `MODEL_TAG`, then `scripts/cold_start.sh` for the",
        "   definitive measurement against a real artifact.",
        "",
        "## What is already known about the answer",
        "",
        "Two things, and both survive whatever the folds say.",
        "",
        "**The label is sound.** [`label_check.md`](label_check.md) verified it against the",
        "boards: 59 of 60 sampled removals are genuinely gone under their own id, against a",
        "control drift of 3.3%. *Removed* is still not *filled* — 12% had their title",
        "relisted under a new id within days.",
        "",
        "**Complexity has not yet earned its place.** On the H=1 rehearsal no fitted rung",
        "clears a hand-written rule by more than fold variance would explain, and step 4 of",
        "the selection rule returns a heuristic when that holds. Shipping a rule a person",
        "could follow unaided is a legitimate outcome of this build, not a failure to",
        "produce a model.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=panel_path())
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    prov = provenance.collect(args.panel, len(panel), provenance.REAL)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(panel, prov))

    status = accrual_status(panel)
    waves = len(crawl_waves(panel[panel["label_observable"]]))
    print(f"labelled waves {waves} · accruing: {'no' if status['stalled'] else 'yes'}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
