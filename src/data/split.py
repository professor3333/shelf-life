"""Temporal splitting for the job-day panel.

The split is the experiment. Everything downstream — the baseline, the metric,
the threshold — is a statement about the world only insofar as this file
simulates the world correctly, which is why it is a module with tests rather
than three lines in a notebook.

**The scenario being simulated**, in one sentence: *a model
fitted on every posting the board showed us up to some Monday, scoring the
postings that are on the board on a later day it has never seen.*

Three things make that harder here than the usual `train_test_split(shuffle=
False)`:

**1. The cut is on `t`, never on `run_index`.** Run indices are per-source and
they are *not* aligned: python_org's run 0 is 2026-09-01 14:07, which is
greenhouse's run 1, because python_org missed the 2026-08-31 crawl. Cutting on
run index would put one source's Tuesday in the same block as another's Sunday
and call it "the past". `assert_temporal_order` exists to make that unfixable
silently.

**2. A label reads forward, so the blocks need an embargo between them.** A row
at `t` is labelled by whether the posting was absent at the next complete run
*and still absent at the one after* (`docs/problem_definition.md` §4). That
corroboration requirement buys label quality with an extra run of forward reach,
so the label of a training row is a function of data up to roughly `t + H + one
run`. If the test block starts inside that reach, the training labels were
computed from the test period. The embargo — a discarded strip of time between
consecutive blocks, wide enough to cover the reach — is the fix. `H` alone is
too narrow, and `H` alone is what the validation protocol in
`docs/problem_definition.md` §7 originally specified.

**3. Postings appear in more than one block, deliberately.** `docs/design.md` §8
settled this: the job-day unit is the person-period setup used in discrete-time
hazard models, 91% of postings straddle any mid-panel cut, and a grouped split
would throw away the data to defend a rule imported from the IID setting. The
risk that replaces it is memorisation — a posting's rows share a byte-identical
title and description, so a high-capacity model can learn *this* posting rather
than duration dependence. The mitigation is measurement, not exclusion, so this
module labels every evaluation row with `seen_in_train` and the report breaks
every count down by it. A model that scores well on carried-over postings and
badly on unseen ones has memorised.

**What this module will not do.** It will not return a split whose evaluation
blocks contain no positives. On a shallow panel that is the normal outcome, and
returning it would hand Component 8 a test set on which precision, recall and
PR-AUC are all undefined while looking like a perfectly ordinary DataFrame.
`feasible_cuts` reports why, and `SplitTooShallow` carries that report in its
message.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")

#: How many complete runs a label reads *beyond* the horizon. One, because
#: `t_gone` requires absence at a run and at the run after it. Raise this if the
#: corroboration rule in `problem_definition.md` §4 is ever widened.
CORROBORATION_RUNS = 1


class SplitTooShallow(ValueError):
    """The panel cannot yet support the requested split.

    Raised rather than returning a degenerate split, because a test block with
    no positives is not a hard test set — it is an undefined metric.
    """


@dataclass(frozen=True)
class Cuts:
    """The two instants that divide the timeline. Blocks are `t <= train_end`,
    then `t <= val_end`, then the remainder, each pair separated by the embargo.
    """

    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.train_end >= self.val_end:
            raise ValueError(f"train_end {self.train_end} must precede val_end {self.val_end}")


@dataclass(frozen=True)
class SplitResult:
    """A split, plus everything needed to defend it."""

    frame: pd.DataFrame
    cuts: Cuts
    embargo: pd.Timedelta
    n_embargoed: int
    n_unlabelled: int

    @property
    def train(self) -> pd.DataFrame:
        return self.frame[self.frame["split"] == "train"]

    @property
    def val(self) -> pd.DataFrame:
        return self.frame[self.frame["split"] == "val"]

    @property
    def test(self) -> pd.DataFrame:
        """The test block. Opened once, at the end. Reading
        this property during development is the thing the rule forbids; nothing
        in code can stop you, which is why it is written here."""
        return self.frame[self.frame["split"] == "test"]


#: Two run instants closer together than this belong to the same crawl wave.
#: The scheduler fires every source within about ten seconds; the shortest gap
#: between two waves in the 2026-09-04 snapshot is 13.6 hours, so an hour
#: separates them with three orders of magnitude to spare.
WAVE_TOLERANCE = pd.Timedelta(hours=1)


def run_instants(frame: pd.DataFrame) -> pd.Series:
    """Every distinct prediction instant in the frame, ascending."""
    return pd.Series(sorted(pd.unique(frame["t"]))).reset_index(drop=True)


def crawl_waves(frame: pd.DataFrame, tolerance: pd.Timedelta = WAVE_TOLERANCE) -> pd.Series:
    """The last instant of each crawl wave, ascending.

    A *wave* is one sweep of the scheduler across all sources. It matters
    because the six Greenhouse boards are crawled seconds apart — 34 distinct
    values of `t` in the 2026-09-04 snapshot, but only five waves — so the raw
    instants are not the natural candidate set for a cut. A cut placed between
    two of them divides a single sweep, putting gitlab's Sunday crawl in `train`
    and anthropic's Sunday crawl, four seconds later, in `val`. That is
    temporally legal and experimentally meaningless.

    Cutting still happens on `t` and only on `t` — this function decides *where*
    a cut may go, not how rows are compared to it, so the misalignment of
    `run_index` across sources is still not something this module can express.
    """
    instants = run_instants(frame)
    if instants.empty:
        return instants
    new_wave = instants.diff() > tolerance
    return instants.groupby(new_wave.cumsum()).max().reset_index(drop=True)


def assert_cuts_land_between_waves(
    frame: pd.DataFrame, cuts: Cuts, tolerance: pd.Timedelta = WAVE_TOLERANCE
) -> None:
    """A cut may not fall inside a crawl wave."""
    waves = crawl_waves(frame, tolerance)
    for name, cut in (("train_end", cuts.train_end), ("val_end", cuts.val_end)):
        nearest = (waves - cut).abs().min()
        if not (waves == cut).any() and nearest <= tolerance:
            raise ValueError(
                f"{name}={cut} falls inside a crawl wave (nearest wave boundary is {nearest} "
                f"away, within the {tolerance} tolerance). It would divide one sweep of the "
                f"scheduler across two blocks. Use a value from crawl_waves()."
            )


def max_run_gap(frame: pd.DataFrame) -> pd.Timedelta:
    """The widest gap between consecutive runs of the same source.

    Deliberately the worst case, not the median. The embargo has to cover the
    reach of the label for *every* row, and the observed schedule is irregular
    enough for the distinction to matter: gaps in the 2026-09-04 snapshot run
    from 13.6h to 34.4h.
    """
    gaps = [
        pd.Series(sorted(pd.unique(group["t"]))).diff().dropna().max()
        for _, group in frame.groupby("source", sort=False)
        if group["t"].nunique() > 1
    ]
    if not gaps:
        raise ValueError("no source has two distinct run instants; cannot infer run spacing")
    return max(gaps)


def embargo_width(
    frame: pd.DataFrame,
    horizon_days: int,
    corroboration_runs: int = CORROBORATION_RUNS,
) -> pd.Timedelta:
    """How much time must be discarded between two blocks.

    `H` days for the horizon itself, plus one run's worth of reach for the
    corroborating absence. Anything narrower and a training row's label was
    computed from data in the following block.
    """
    if horizon_days < 0:
        raise ValueError(f"horizon_days must be non-negative, got {horizon_days}")
    if corroboration_runs < 0:
        raise ValueError(f"corroboration_runs must be non-negative, got {corroboration_runs}")
    return pd.Timedelta(days=horizon_days) + corroboration_runs * max_run_gap(frame)


def assign_split(frame: pd.DataFrame, cuts: Cuts, embargo: pd.Timedelta) -> pd.Series:
    """Label every row `train`, `val`, `test` or `embargo`, by `t` alone.

    Note what is *not* an input: no random state, no row order, no index. The
    assignment is a pure function of the timestamp, which is what makes the
    split reproducible across runs and across machines.
    """
    t = frame["t"]
    split = pd.Series("embargo", index=frame.index, dtype="object")
    split[t <= cuts.train_end] = "train"
    split[(t > cuts.train_end + embargo) & (t <= cuts.val_end)] = "val"
    split[t > cuts.val_end + embargo] = "test"
    return split


def temporal_split(
    frame: pd.DataFrame,
    cuts: Cuts,
    horizon_days: int | None = None,
    embargo: pd.Timedelta | None = None,
    corroboration_runs: int = CORROBORATION_RUNS,
    require_positives: bool = True,
) -> SplitResult:
    """Cut the panel into train / val / test on calendar time.

    Unlabelled rows — the right-censored ones, whose horizon has not elapsed —
    are dropped first. They are not negatives, and `problem_definition.md` §4
    drops them at labelling time for the same reason.

    `horizon_days` is read from the frame's own `horizon_days` column when not
    given, so the embargo cannot silently disagree with the label it is
    protecting.
    """
    if "t" not in frame or "y" not in frame:
        raise ValueError("frame must carry `t` and `y`; pass the assembled job-day panel")

    assert_cuts_land_between_waves(frame, cuts)

    labelled = frame[frame["label_observable"]].copy()
    n_unlabelled = len(frame) - len(labelled)
    if labelled.empty:
        raise SplitTooShallow("no labelable rows: every row's horizon extends past the last run")

    if embargo is None:
        if horizon_days is None:
            horizons = pd.unique(frame["horizon_days"])
            if len(horizons) != 1:
                raise ValueError(f"frame mixes horizons {horizons}; pass horizon_days explicitly")
            horizon_days = int(horizons[0])
        embargo = embargo_width(frame, horizon_days, corroboration_runs)

    labelled["split"] = assign_split(labelled, cuts, embargo)
    n_embargoed = int((labelled["split"] == "embargo").sum())
    kept = labelled[labelled["split"] != "embargo"].copy()
    kept["seen_in_train"] = _seen_in_train(kept)

    result = SplitResult(
        frame=kept.sort_values(["t", "source", "source_id"], kind="stable").reset_index(drop=True),
        cuts=cuts,
        embargo=embargo,
        n_embargoed=n_embargoed,
        n_unlabelled=n_unlabelled,
    )
    _validate(result, labelled, require_positives=require_positives)
    return result


def _posting_id(frame: pd.DataFrame) -> pd.Series:
    """Identity across sources. `source_id` is unique only within a board."""
    return frame["source"].astype(str) + "|" + frame["source_id"].astype(str)


def _seen_in_train(frame: pd.DataFrame) -> pd.Series:
    """Was this posting also in the training block?

    `docs/design.md` §8: subject overlap is accepted, and this column is the
    price of accepting it. Every evaluation metric downstream is reported both
    overall and split on this flag, because the gap between the two arms is the
    memorisation diagnosis.
    """
    pid = _posting_id(frame)
    trained_on = set(pid[frame["split"] == "train"])
    seen = pid.isin(trained_on)
    seen[frame["split"] == "train"] = False  # meaningless for training rows themselves
    return seen


def _validate(result: SplitResult, labelled: pd.DataFrame, require_positives: bool) -> None:
    """Every invariant of the validation protocol, checked on the real output.

    `labelled` is the panel *before* the cut. The feasibility table in a refusal
    message has to describe what was available, not what survived the embargo —
    a rejected split can easily leave too few run instants to infer the run
    spacing from, and the report would then fail instead of explaining itself.
    """
    assert_temporal_order(result)

    frame = result.frame
    for name in SPLIT_NAMES:
        block = frame[frame["split"] == name]
        if block.empty:
            raise SplitTooShallow(
                f"the {name!r} block is empty under cuts {result.cuts} with an embargo of "
                f"{result.embargo}.\n\n{feasibility_report(labelled)}"
            )
    if not require_positives:
        return
    for name in ("val", "test"):
        block = frame[frame["split"] == name]
        positives = int((block["y"] == 1).sum())
        if positives == 0:
            raise SplitTooShallow(
                f"the {name!r} block has {len(block)} rows and no positives, so precision, "
                f"recall and PR-AUC are all undefined on it. This is a panel-depth problem, "
                f"not a cut that can be moved.\n\n{feasibility_report(labelled)}"
            )


def assert_temporal_order(result: SplitResult) -> None:
    """Train strictly precedes val strictly precedes test, with the embargo held.

    Checked on the assigned rows rather than on the cut instants, because the
    cut is a claim and the rows are the fact.
    """
    frame = result.frame
    bounds = {}
    for name in SPLIT_NAMES:
        block = frame[frame["split"] == name]
        if not block.empty:
            bounds[name] = (block["t"].min(), block["t"].max())

    ordered = [name for name in SPLIT_NAMES if name in bounds]
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if bounds[earlier][1] >= bounds[later][0]:
            raise AssertionError(
                f"{earlier} ends at {bounds[earlier][1]} but {later} starts at {bounds[later][0]}"
            )
        gap = bounds[later][0] - bounds[earlier][1]
        if gap <= result.embargo:
            raise AssertionError(
                f"gap between {earlier} and {later} is {gap}, which does not clear the "
                f"{result.embargo} embargo: {earlier} labels were computed from {later} data"
            )


def feasible_cuts(
    frame: pd.DataFrame,
    horizon_days: int | None = None,
    corroboration_runs: int = CORROBORATION_RUNS,
) -> pd.DataFrame:
    """Every candidate cut, and whether it yields a usable split.

    One row per (train_end, val_end) pair drawn from the observed run instants.
    When nothing is feasible — the ordinary case on a shallow panel — this table
    is the diagnosis, and it will start returning feasible rows on its own as
    the scraper adds depth. It is therefore both a report and a readiness check.
    """
    labelled = frame[frame["label_observable"]]
    instants = crawl_waves(labelled)
    if horizon_days is None:
        horizon_days = int(pd.unique(frame["horizon_days"])[0])
    embargo = embargo_width(frame, horizon_days, corroboration_runs)

    rows = []
    for i in range(len(instants) - 1):
        for j in range(i + 1, len(instants)):
            cuts = Cuts(train_end=instants[i], val_end=instants[j])
            split = assign_split(labelled, cuts, embargo)
            record: dict[str, object] = {
                "train_end": cuts.train_end,
                "val_end": cuts.val_end,
                "embargoed": int((split == "embargo").sum()),
            }
            for name in SPLIT_NAMES:
                block = labelled[split == name]
                record[f"{name}_rows"] = len(block)
                record[f"{name}_pos"] = int((block["y"] == 1).sum())
            record["valid"], record["reason"] = _verdict(record)
            rows.append(record)
    return pd.DataFrame(rows)


def _verdict(record: dict[str, object]) -> tuple[bool, str]:
    for name in SPLIT_NAMES:
        if record[f"{name}_rows"] == 0:
            return False, f"{name} block empty"
    for name in ("val", "test"):
        if record[f"{name}_pos"] == 0:
            return False, f"{name} block has no positives"
    return True, ""


def feasibility_report(frame: pd.DataFrame) -> str:
    """The feasibility table as text, for exception messages."""
    table = feasible_cuts(frame)
    usable = table[table["valid"]]
    header = f"{len(usable)} of {len(table)} candidate cuts are usable."
    return header + "\n" + table.to_string(index=False)


def split_report(result: SplitResult) -> pd.DataFrame:
    """What each block contains — the table that goes in the README.

    Reports `seen_in_train` for the evaluation blocks because `docs/design.md`
    §8 made that breakdown part of the split rather than an afterthought.
    """
    rows = []
    for name in SPLIT_NAMES:
        block = result.frame[result.frame["split"] == name]
        arms = [("all", block)]
        if name != "train":
            arms += [
                ("carried over", block[block["seen_in_train"]]),
                ("unseen", block[~block["seen_in_train"]]),
            ]
        for arm, part in arms:
            positives = int((part["y"] == 1).sum())
            rows.append(
                {
                    "split": name,
                    "postings": arm,
                    "rows": len(part),
                    "distinct_postings": _posting_id(part).nunique() if len(part) else 0,
                    "positives": positives,
                    "positive_rate": round(positives / len(part), 4) if len(part) else float("nan"),
                    "t_min": part["t"].min() if len(part) else pd.NaT,
                    "t_max": part["t"].max() if len(part) else pd.NaT,
                }
            )
    return pd.DataFrame(rows)


def resurrection_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Postings that vanished from the board and came back.

    `t_gone` requires that a posting *never re-appeared*, and that clause reads
    the whole remaining panel rather than a bounded window — so no embargo, of
    any width, fully seals a training label from the evaluation period. This
    function measures how much that costs. Two of 1,240 postings do it in the
    2026-09-04 snapshot, one of them across two consecutive absences, which is
    wide enough to defeat the corroboration guard and flip a label from 1 to 0
    when the panel deepens.

    Recorded rather than fixed, because bounding the resurrection window is a
    change to the label definition and belongs in `docs/design.md`.
    """
    rows = []
    for (source, source_id), group in frame.groupby(["source", "source_id"], sort=False):
        seen = sorted(group["run_index"])
        gaps = [(a, b) for a, b in zip(seen, seen[1:], strict=False) if b - a > 1]
        if gaps:
            rows.append(
                {
                    "source": source,
                    "source_id": source_id,
                    "present_at_runs": seen,
                    "max_absent_streak": max(b - a - 1 for a, b in gaps),
                }
            )
    return pd.DataFrame(
        rows, columns=["source", "source_id", "present_at_runs", "max_absent_streak"]
    )
