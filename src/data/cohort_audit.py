"""The cohort audit: what does age encode, and where would a score come from?

    python -m src.data.cohort_audit                      # H=7, the build's horizon
    python -m src.data.cohort_audit --panel <parquet> --out <md>

A panel that starts on a given morning sees two populations from then on. The
**incumbent** stock — every posting already on the board at its source's first
complete run — has survived for an unknown time before it was ever observed,
so it is old, and old in a way that has already been filtered for staying.
The **incident** flow — postings first listed after collection began — arrives
at age zero and has been filtered for nothing. Any feature that tracks age can
separate the two, and a label that treats them differently will make that
feature look like a model.

`design.md` §3 dissolves left truncation by the unit of analysis: a job-day
row conditions on survival to `t`, so age is a feature rather than a missing
outcome. That is the right formulation *if the label is indifferent to which
population a row came from*. This audit is the check that it is, made
explicitly and repeatably: the same measurements that on 2026-09-11 showed a
"cohort effect" (77 of 77 unseen validation postings positive, `age_only` at
PR-AUC 0.777) and turned out to be a label bug dated before a posting's first
row (`DEBUGGING.md`, 2026-09-11). Had this file existed, the bug's signature
— every incident row positive, no incumbent row positive, age separating the
two perfectly and nothing within either — would have been one table.

**Which rows the product scores is decided, not assumed.** `design.md` §15:
the product ranks a day's whole board, so incumbents and incidents both belong
in the dataset and age is a legitimate input. The first-observation slice —
a posting scored the day it first appears — is the narrower promise the API
also makes, and it is reported here as its own row so that a model which only
works on the stock cannot hide inside the board-wide number.

Reads train and validation only. Nothing here touches the held-out block.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.split import Cuts, SplitTooShallow, best_cuts, feasible_cuts, temporal_split
from src.features.assemble import horizon_banner, panel_path
from src.models import provenance

DEFAULT_REPORT = Path("reports/cohort_audit.md")

INCUMBENT = "incumbent"
INCIDENT = "incident"

#: Quantiles reported for the age distribution. The extremes are included on
#: purpose: an incumbent stock whose *minimum* age is above the incident flow's
#: *maximum* is the left-truncation signature in one line.
AGE_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)


def annotate(panel: pd.DataFrame) -> pd.DataFrame:
    """The cohort columns every table below is cut on. All of them are as-of-`t`.

    ``first_seen_idx``
        The source-relative index of the first complete run that saw the posting.
    ``cohort``
        ``incumbent`` if that index is the source's first run, else ``incident``.
    ``runs_seen``
        How many complete runs had seen the posting up to and including this
        row — `run_index - first_seen_idx + 1`, never the lifetime count, which
        reads the future (`label_audit.lifespan_concentration`).
    ``first_observation``
        This row is an incident posting's first sighting — the day it appeared.
        The rows the narrower product promise, "score it the day it appears", is
        evaluated on. An incumbent's wave-0 row is not one: that is the day the
        *panel* first looked, and the posting had already been up for an unknown
        time (left truncation, `design.md` §3).
    """
    out = panel.copy()
    by_posting = out.groupby(["source", "source_id"])["run_index"]
    out["first_seen_idx"] = by_posting.transform("min")
    board_first = out.groupby("source")["run_index"].transform("min")
    out["cohort"] = np.where(out["first_seen_idx"] == board_first, INCUMBENT, INCIDENT)
    out["runs_seen"] = out["run_index"] - out["first_seen_idx"] + 1
    out["first_observation"] = (out["run_index"] == out["first_seen_idx"]) & (
        out["cohort"] == INCIDENT
    )
    return out


def attach_first_observation(block: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """`block` with a `first_observation` column looked up from the whole panel.

    Joined on (source, source_id, run_index), never on index: `temporal_split`
    renumbers the rows of each block, so an index-aligned lookup against the
    panel picks other rows and raises nothing. The panel is the right thing to
    derive the flag from because first sight is a property of the posting's
    whole history, which a block does not hold.
    """
    keys = ["source", "source_id", "run_index"]
    lookup = annotate(panel)[keys + ["first_observation"]].drop_duplicates(keys)
    flags = block[keys].merge(lookup, on=keys, how="left")["first_observation"]
    return block.assign(first_observation=flags.fillna(False).to_numpy(dtype=bool))


def labelled(panel: pd.DataFrame) -> pd.DataFrame:
    rows = panel[panel["label_observable"]].copy()
    rows["y"] = rows["y"].astype(int)
    return rows


# --- rates ------------------------------------------------------------------


def _rates(frame: pd.DataFrame, by: str | list[str]) -> pd.DataFrame:
    keys = [by] if isinstance(by, str) else by
    table = frame.groupby(keys, observed=True, dropna=False).agg(
        rows=("y", "size"),
        postings=("source_id", "nunique"),
        positives=("y", "sum"),
        rate=("y", "mean"),
        mean_age=("age_days", "mean"),
    )
    return table.reset_index()


def cohort_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Incumbent against incident: the left-truncation check."""
    return _rates(rows, "cohort")


def first_seen_wave_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One line per first-seen run: a collection-era effect shows up as a rate
    that moves with *when the panel first saw the posting*, not with anything
    about the posting."""
    return _rates(rows, "first_seen_idx")


def runs_seen_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Positive rate against how many runs had seen the posting as of `t`."""
    return _rates(rows, "runs_seen")


def source_table(rows: pd.DataFrame) -> pd.DataFrame:
    return _rates(rows, "source")


def source_by_cohort_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Board effect and cohort effect, separated. A board whose incident rate
    matches its incumbent rate has no cohort effect; a cohort whose rate is the
    same on every board has no board effect."""
    return _rates(rows, ["source", "cohort"])


def age_distribution(rows: pd.DataFrame) -> pd.DataFrame:
    """`age_days` quantiles per cohort. What age is encoding is read straight
    off this: if the two ranges do not overlap, age *is* cohort."""
    table = (
        rows.groupby("cohort", observed=True)["age_days"].quantile(list(AGE_QUANTILES)).unstack()
    )
    table.columns = [f"p{int(q * 100)}" for q in AGE_QUANTILES]
    table["n"] = rows.groupby("cohort", observed=True).size()
    return table.reset_index()


# --- what age alone can do, and where ----------------------------------------


def age_rank(block: pd.DataFrame) -> dict[str, float]:
    """PR-AUC of `age_days` as a bare ranker, in both directions, on one block.

    Both directions because the honest question is not "does young mean
    closing" but "does age order the label at all". The larger of the two is
    the ceiling on a monotone age rule; the base rate is what either has to
    clear. NaN where a direction cannot be scored (no positives, or nothing but).
    """
    y = block["y"].to_numpy()
    if not len(y):
        return {"n": 0, "positives": 0, "base_rate": float("nan")} | dict.fromkeys(
            ("young_first", "old_first"), float("nan")
        )
    age = block["age_days"].to_numpy(dtype=float)
    # Rows with no employer date are ranked as oldest in both directions — the
    # same "unknown is old" convention the pipeline's imputer applies.
    fill = np.nanmax(age) if np.isfinite(age).any() else 0.0
    age = np.where(np.isnan(age), fill, age)
    scorable = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "young_first": float(average_precision_score(y, -age)) if scorable else float("nan"),
        "old_first": float(average_precision_score(y, age)) if scorable else float("nan"),
    }


def age_rank_by_slice(val: pd.DataFrame) -> pd.DataFrame:
    """The same ranker on the whole validation block and on each of its slices.

    The slices are the ones that answer *where a score comes from*: the two
    cohorts, the two `seen_in_train` arms, and the first-observation rows. A
    ranker that scores on the whole block and on neither cohort alone is
    scoring the cohort boundary.
    """
    slices = {
        "whole block": val,
        f"{INCUMBENT} only": val[val["cohort"] == INCUMBENT],
        f"{INCIDENT} only": val[val["cohort"] == INCIDENT],
        "seen in training": val[val["seen_in_train"]],
        "unseen in training": val[~val["seen_in_train"]],
        "first observation only": val[val["first_observation"]],
    }
    return pd.DataFrame([{"slice": name, **age_rank(block)} for name, block in slices.items()])


def rolling_age_rank(panel: pd.DataFrame) -> pd.DataFrame:
    """The age ranker on each successive validation block the panel admits.

    One row per distinct `val_end` among the legal cuts, at the deepest
    training window that reaches it. Whether the relationship *persists* as
    waves arrive is a question the single-cut tables cannot ask.
    """
    cuts = feasible_cuts(panel)
    if cuts.empty or not cuts["valid"].any():
        return pd.DataFrame()
    legal = cuts[cuts["valid"]].sort_values(["val_end", "train_end"])
    legal = legal.groupby("val_end", as_index=False).last()
    rows = []
    for record in legal.itertuples(index=False):
        try:
            split = temporal_split(panel, Cuts(record.train_end, record.val_end))
        except SplitTooShallow:
            continue
        val = annotate(split.val) if "cohort" not in split.val else split.val
        val = val.assign(y=val["y"].astype(int))
        rows.append(
            {
                "train_end": record.train_end.date(),
                "val_end": record.val_end.date(),
                **age_rank(val),
                "incident_rows": int((val["cohort"] == INCIDENT).sum()),
                "incident_positives": int(val.loc[val["cohort"] == INCIDENT, "y"].sum()),
                "first_obs_rows": int(val["first_observation"].sum()),
                "first_obs_positives": int(val.loc[val["first_observation"], "y"].sum()),
            }
        )
    return pd.DataFrame(rows)


def validation_block(panel: pd.DataFrame) -> pd.DataFrame | None:
    """The validation block at the cut the ladder uses (`best_cuts`), annotated;
    None if no cut is legal yet. The same block `model_comparison.md` scores on,
    so the two reports describe one set of rows. Train and validation only — the
    held-out block is not read."""
    try:
        split = temporal_split(panel, best_cuts(panel))
    except SplitTooShallow:
        return None
    val = split.val
    val = annotate(val) if "cohort" not in val else val
    return val.assign(y=val["y"].astype(int))


# --- the report --------------------------------------------------------------


def _md(table: pd.DataFrame, rate_cols=("rate", "base_rate")) -> list[str]:
    if table.empty:
        return ["_empty_"]
    shown = table.copy()
    for col in shown.columns:
        if col in rate_cols:
            shown[col] = shown[col].map(lambda v: "—" if pd.isna(v) else f"{v:.1%}")
        elif col in ("young_first", "old_first"):
            shown[col] = shown[col].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        elif col == "mean_age" or col.startswith("p") and col[1:].isdigit():
            shown[col] = shown[col].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    sep = "|" + "---|" * len(shown.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return [header, sep, *body]


def _cohort_verdict(table: pd.DataFrame) -> str:
    by = table.set_index("cohort")
    if INCIDENT not in by.index or INCUMBENT not in by.index:
        return "_Only one cohort is labelled at this depth; nothing to compare._"
    inc, stock = by.loc[INCIDENT], by.loc[INCUMBENT]
    if inc["positives"] == 0 and stock["positives"] == 0:
        return "_No closures in either cohort yet._"
    ratio = inc["rate"] / stock["rate"] if stock["rate"] else float("inf")
    if ratio >= 10 or (inc["rate"] >= 0.9 and stock["rate"] <= 0.1):
        return (
            f"**The label is not indifferent to cohort.** Incident rows close at "
            f"{inc['rate']:.1%} against {stock['rate']:.1%} for the stock, a factor of "
            f"{ratio:.0f}. Read the age tables below before believing any age-shaped "
            "feature: this is the signature of a label that keys on arrival, and on "
            "2026-09-11 it was a bug."
        )
    return (
        f"Incident rows close at {inc['rate']:.1%} against {stock['rate']:.1%} for the "
        f"stock ({int(inc['rows']):,} against {int(stock['rows']):,} rows). Close enough that "
        "the label is not keying on which population a row came from — the condition under which "
        "the job-day formulation makes age a feature rather than an artefact."
    )


def _age_verdict(dist: pd.DataFrame) -> str:
    by = dist.set_index("cohort")
    if INCIDENT not in by.index or INCUMBENT not in by.index:
        return ""
    if by.loc[INCUMBENT, "p0"] > by.loc[INCIDENT, "p100"]:
        return (
            f"**The two age ranges do not overlap**: the youngest incumbent is "
            f"{by.loc[INCUMBENT, 'p0']:.0f} days old and the oldest incident row "
            f"{by.loc[INCIDENT, 'p100']:.0f}. On this panel `age_days` *is* the cohort, "
            "and anything it predicts, cohort predicts."
        )
    return (
        f"The ranges overlap (incumbent p0 {by.loc[INCUMBENT, 'p0']:.0f}, incident p100 "
        f"{by.loc[INCIDENT, 'p100']:.0f} days), so age carries information the cohort "
        "boundary does not."
    )


def _rank_verdict(table: pd.DataFrame) -> str:
    by = table.set_index("slice")
    whole = by.loc["whole block"]
    best_whole = np.nanmax([whole["young_first"], whole["old_first"]])
    if np.isnan(best_whole):
        return "_The validation block cannot be scored (no positives, or nothing but)._"
    lift = best_whole - whole["base_rate"]
    if lift < max(0.01, whole["base_rate"]):
        return (
            f"Age alone does not order the label: its best direction scores "
            f"{best_whole:.3f} against a base rate of {whole['base_rate']:.3f}. Whatever a "
            "model finds, it will not be age."
        )
    parts = []
    for name in (f"{INCUMBENT} only", f"{INCIDENT} only"):
        row = by.loc[name]
        best = np.nanmax([row["young_first"], row["old_first"]])
        parts.append(
            f"{name}: {'—' if np.isnan(best) else f'{best:.3f}'} against {row['base_rate']:.3f}"
        )
    return (
        f"Age orders the label on the whole block ({best_whole:.3f} against "
        f"{whole['base_rate']:.3f}). Within cohorts — {'; '.join(parts)}. If neither cohort "
        "alone shows the lift, the ranker is scoring the cohort boundary, not the posting."
    )


def render(panel: pd.DataFrame, prov: provenance.Provenance | None = None) -> str:
    frame = annotate(panel)
    rows = labelled(frame)
    cohorts = cohort_table(rows)
    dist = age_distribution(rows)
    val = validation_block(frame)

    lines = [
        "# Cohort audit — what does age encode, and where would a score come from?",
        "",
        *provenance.header(prov, "python -m src.data.cohort_audit", horizon_banner(panel)),
        "A panel that starts on a given morning sees two populations: the **incumbent**",
        "stock, already on the board and already filtered for staying, and the **incident**",
        "flow, arriving at age zero and filtered for nothing. Every table here asks whether",
        "the label tells them apart, because a label that does will make any age-shaped",
        "feature look like a model. The product ranks a day's whole board (`design.md` §15),",
        "so both belong in the dataset — provided the label is indifferent to which is which.",
        "",
        "## 1. Incumbent stock against incident flow — left truncation",
        "",
        *_md(cohorts),
        "",
        _cohort_verdict(cohorts),
        "",
        "## 2. First-seen wave — collection-era effects",
        "",
        "A rate that moves with *when the panel first saw the posting* is an effect of",
        "collection, not of the posting. Wave 0 is the incumbent stock.",
        "",
        *_md(first_seen_wave_table(rows)),
        "",
        "## 3. What `age_days` is encoding",
        "",
        *_md(dist),
        "",
        _age_verdict(dist),
        "",
        "## 4. Positive rate against runs seen as of `t`",
        "",
        "Counted up to the row, never over the posting's life — the lifetime count reads",
        "the future and made this table a tautology until 2026-09-11.",
        "",
        *_md(runs_seen_table(rows)),
        "",
        "## 5. Positive rate by source, and by source × cohort",
        "",
        *_md(source_table(rows)),
        "",
        "Board effect and cohort effect, separated. A board whose incident rate matches",
        "its incumbent rate has no cohort effect on it:",
        "",
        *_md(source_by_cohort_table(rows)),
        "",
        "## 6. What age alone can do, and on which rows",
        "",
    ]

    if val is None:
        lines += [
            "_No legal three-way cut exists at this depth, so there is no validation block",
            "to score. Sections 6 and 7 fill in on the day `reports/readiness.md` says the",
            "split is legal._",
            "",
        ]
    else:
        ranks = age_rank_by_slice(val)
        n_first = int(val["first_observation"].sum())
        first_note = (
            [
                "",
                f"_The block holds **no first observations**: it is {val['t'].dt.date.nunique()} "
                "wave(s) wide after the embargo, and no posting first appeared on it. The "
                "narrower promise cannot be evaluated at this cut; section 7 shows which cuts "
                "can._",
            ]
            if n_first == 0
            else []
        )
        lines += [
            f"Validation block at the ladder's cut (`best_cuts`): {len(val):,} rows, "
            f"{int(val['y'].sum())} positives. `age_days` as a bare ranker, both directions,",
            "on the whole block and on each slice. `first observation only` is the narrower",
            "product promise — a posting scored the day it appears — and the row a model that",
            "only works on the stock cannot hide behind.",
            "",
            *_md(ranks),
            *first_note,
            "",
            _rank_verdict(ranks),
            "",
            "## 7. Does it persist? The age ranker on each successive validation block",
            "",
            "One row per distinct `val_end` the panel admits, at the deepest training window",
            "that reaches it.",
            "",
            *_md(rolling_age_rank(frame)),
            "",
        ]

    lines += [
        "## What this file cannot tell you",
        "",
        "Whether a closure is a hire. `reports/label_check.md` verifies that removals are",
        "removals; nothing in the panel says why. And an incident cohort that is small —",
        "it is, at every depth this panel has reached — carries wide intervals on every",
        "rate above; a difference between cohorts that is inside those is not a finding.",
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

    rows = labelled(annotate(panel))
    table = cohort_table(rows).set_index("cohort")
    for cohort in (INCUMBENT, INCIDENT):
        if cohort in table.index:
            r = table.loc[cohort]
            summary = f"{int(r['rows']):6,} rows  {int(r['positives']):4} closures  {r['rate']:.1%}"
            print(f"{cohort:10} {summary}")
    print(f"wrote -> {args.out}")


if __name__ == "__main__":  # pragma: no cover
    main()
