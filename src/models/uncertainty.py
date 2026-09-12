"""Confidence intervals for the headline numbers, resampled by posting.

The test block will hold on the order of twenty positives. A bare PR-AUC on
twenty positives is a number whose second decimal place is decoration, and
reporting it without an interval invites exactly the over-reading this project
has spent its whole history refusing. `docs/design.md` §5 makes PR-AUC the
headline; this module is what stops the headline being read as precise.

**The resampling unit is the posting, not the row.** That is the one decision in
this file that matters, and it is a decision about what a repetition of the
experiment would look like. The panel is one row per (posting, crawl) and
averages 6.37 rows per posting, so 8,037 labelled rows are not 8,037 independent
observations — they are roughly 1,262 postings, each contributing a short run of
job-days that share a title, a company, a board and an outcome. A row-level
bootstrap treats each of those rows as its own draw, which is a claim about the
data-generating process that is simply false: the board does not sample
job-days, it accumulates postings and observes them daily.

**On the observed panel the cluster interval is narrower, not wider**, which is
the opposite of what the usual "clustering inflates variance" intuition
predicts. Measured on a synthetic block of 241 rows and 99 postings: row-level
[0.0939, 0.2782], by posting [0.0969, 0.2508]. The reason is specific to this
label — a closing posting contributes exactly one positive row, its last, so
resampling rows lets positives be duplicated and makes the positive count itself
noisy, while resampling postings carries each positive in exactly once. This is
recorded because the direction is surprising and because it is *not* the
argument: the cluster bootstrap is right because it matches the sampling design,
and it would still be right if it came out wider.

**A threshold-dependent metric is measured at the frozen threshold, held
fixed.** Recomputing the alert-budget threshold inside each resample would mix
two different uncertainties — how well the model separates, and where the
operating point happens to land — and the artifact ships one threshold, not a
distribution of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.metrics import (
    DEFAULT_ALERT_BUDGET,
    average_precision,
    brier_score,
    confusion_at,
    expected_calibration_error,
)

#: Resamples. Enough that the percentile endpoints are stable to about the third
#: decimal, which is finer than any difference this project would act on.
DEFAULT_RESAMPLES = 2000

#: Below this many positives in a block, an interval is reported but flagged.
#: Not a refusal — a wide interval is informative — but a 95% percentile
#: interval computed from a handful of events has endpoints determined by one or
#: two rows, and a reader should be told rather than left to infer it.
FRAGILE_POSITIVES = 30


@dataclass(frozen=True)
class Interval:
    """A point estimate and the range the resampling puts around it."""

    metric: str
    point: float
    low: float
    high: float
    resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def format(self) -> str:
        if np.isnan(self.point):
            return "—"
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def posting_ids(block: pd.DataFrame) -> np.ndarray:
    """The clustering key. `source_id` is unique only within a board."""
    return (block["source"].astype(str) + "|" + block["source_id"].astype(str)).to_numpy()


def _cluster_indices(groups: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
    blocks = np.split(order, boundaries[1:])
    return blocks, sorted_groups[boundaries]


def cluster_resample(
    truth: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Draw postings with replacement, recompute `statistic`, return the samples.

    A resample that lands with no positives is discarded rather than scored:
    average precision is undefined there, and a zero would be averaged in as if
    it were a bad score rather than an absent one. With twenty positives in a
    block that happens rarely, but "rarely" is not "never" and the difference
    shows up in the lower endpoint, which is the number a reader leans on.
    """
    blocks, _ = _cluster_indices(groups)
    rng = np.random.default_rng(seed)
    n_clusters = len(blocks)
    samples: list[float] = []
    for _ in range(resamples):
        drawn = rng.integers(0, n_clusters, n_clusters)
        index = np.concatenate([blocks[k] for k in drawn])
        if truth[index].sum() == 0:
            continue
        samples.append(float(statistic(truth[index], scores[index])))
    return np.asarray(samples, dtype=float)


def interval(
    metric: str,
    point: float,
    samples: np.ndarray,
    level: float = 0.95,
) -> Interval:
    """Percentile interval. `nan` endpoints when nothing could be resampled.

    A resample on which the statistic is undefined — precision with nothing
    above the threshold, say — is dropped the way a resample with no positives
    is, rather than poisoning the percentile: one `nan` in the samples made
    `np.percentile` return `nan` for the whole interval, which read as "no
    interval" for a metric that had one. Found 2026-09-12 on a per-board
    table; the headline path had the same exposure.
    """
    samples = samples[~np.isnan(samples)] if samples.size else samples
    if samples.size == 0:
        return Interval(metric, point, float("nan"), float("nan"), 0)
    tail = (1.0 - level) / 2.0
    low, high = np.percentile(samples, [100 * tail, 100 * (1 - tail)])
    return Interval(metric, point, float(low), float(high), int(samples.size))


def _statistics(threshold: float) -> dict[str, Callable[[np.ndarray, np.ndarray], float]]:
    """The metrics that get an interval, and how each is computed on a resample.

    `precision` and `recall` close over the **frozen** threshold rather than
    recomputing one per resample — see the module docstring.
    """
    return {
        "precision_at_budget": lambda t, s: confusion_at(t, s, threshold)["precision"],
        "recall_at_budget": lambda t, s: confusion_at(t, s, threshold)["recall"],
        "lift_at_budget": lambda t, s: _lift(t, s, threshold),
        "pr_auc": average_precision,
        "brier": brier_score,
        "ece": lambda t, s: expected_calibration_error(t, s),
        "precision": lambda t, s: confusion_at(t, s, threshold)["precision"],
        "recall": lambda t, s: confusion_at(t, s, threshold)["recall"],
    }


def _lift(truth: np.ndarray, score: np.ndarray, threshold: float) -> float:
    """Precision at the frozen threshold over the resample's own base rate."""
    base_rate = float(np.mean(truth)) if truth.size else float("nan")
    precision = confusion_at(truth, score, threshold)["precision"]
    return precision / base_rate if base_rate and not np.isnan(precision) else float("nan")


def bootstrap_block(
    block: pd.DataFrame,
    scores,
    threshold: float,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    metrics: Sequence[str] | None = None,
) -> dict[str, Interval]:
    """Every headline metric on one block, with a posting-clustered interval."""
    truth = block["y"].astype(int).to_numpy()
    score = np.asarray(scores, dtype=float)
    groups = posting_ids(block)

    wanted = _statistics(threshold)
    if metrics is not None:
        wanted = {name: wanted[name] for name in metrics}

    out: dict[str, Interval] = {}
    for name, statistic in wanted.items():
        point = float(statistic(truth, score))
        samples = cluster_resample(truth, score, groups, statistic, resamples, seed)
        out[name] = interval(name, point, samples)
    return out


def fragility_note(block: pd.DataFrame) -> str | None:
    """What the reader needs told about the denominator, or `None` if it is fine."""
    positives = int((block["y"].astype(int) == 1).sum())
    postings = len(set(posting_ids(block)))
    if positives >= FRAGILE_POSITIVES:
        return None
    return (
        f"**These intervals rest on {positives} positive(s) across {postings} posting(s).** "
        "At that count a percentile endpoint is determined by one or two rows, so read "
        "the interval as an order of magnitude rather than as a bound — its job here is "
        "to show that the point estimate is not precise, which it does honestly, and not "
        "to say precisely how imprecise it is, which it cannot."
    )


def format_table(intervals: dict[str, Interval], level: float = 0.95) -> list[str]:
    """The intervals as a markdown table."""
    percent = int(round(level * 100))
    lines = [
        f"| metric | estimate | {percent}% interval | width |",
        "|---|---|---|---|",
    ]
    for name, item in intervals.items():
        if np.isnan(item.low):
            lines.append(f"| `{name}` | {item.point:.4f} | — | — |")
        else:
            lines.append(
                f"| `{name}` | {item.point:.4f} | [{item.low:.4f}, {item.high:.4f}] "
                f"| {item.width:.4f} |"
            )
    return lines


#: Resamples per group in a breakdown table. Fewer than the headline's 2,000
#: because a table has several groups and a report has several tables; at
#: this count a percentile endpoint moves by less than the third decimal.
GROUP_RESAMPLES = 500


def evaluate_by_with_evidence(
    frame: pd.DataFrame,
    y_score,
    group: str,
    n_days: int = 1,
    budget_per_day: int = DEFAULT_ALERT_BUDGET,
    resamples: int = GROUP_RESAMPLES,
    seed: int = 0,
) -> pd.DataFrame:
    """`metrics.evaluate_by`, plus what a per-group number needs beside it.

    Per-board and per-cohort positive rates on this panel run from 0% to
    13.5%, and several groups hold a handful of positives. A PR-AUC on five
    events is a number set by which five, and printed beside a real one it
    invites a comparison that cannot be made. So every row of a breakdown
    carries: `n` rows, `postings` — the independent units, since a posting
    contributes about six rows — `positives`, a posting-clustered 95%
    interval on `pr_auc` and on `precision_at_budget`, and `fragile`, true
    under `FRAGILE_POSITIVES` events. The interval is the evidence; the point
    estimate on a fragile row is an order of magnitude, not a bound.
    """
    from src.models.metrics import evaluate_by  # noqa: PLC0415 - metrics imports this module

    table = evaluate_by(frame, y_score, group, n_days=n_days, budget_per_day=budget_per_day)
    scores = pd.Series(np.asarray(y_score, dtype=float), index=frame.index)
    extra = []
    for value, block in frame.groupby(group, sort=True, dropna=False):
        truth = block["y"].astype(int).to_numpy()
        score = scores.loc[block.index].to_numpy()
        groups = posting_ids(block)
        row: dict = {group: value, "postings": int(len(set(groups)))}
        row["fragile"] = bool(truth.sum() < FRAGILE_POSITIVES)
        threshold = float(table.loc[table[group] == value, "threshold"].iloc[0])
        for name, statistic in (
            ("pr_auc", average_precision),
            ("precision_at_budget", _precision_at(threshold)),
        ):
            samples = cluster_resample(truth, score, groups, statistic, resamples, seed)
            band = interval(name, float("nan"), samples)
            row[f"{name}_low"], row[f"{name}_high"] = band.low, band.high
        extra.append(row)
    return table.merge(pd.DataFrame(extra), on=group, how="left")


def _precision_at(threshold: float) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda t, s: confusion_at(t, s, threshold)["precision"]


def evidence_columns(group: str) -> list[str]:
    """The columns a breakdown table prints, in the order a reader needs them."""
    return [
        group,
        "n",
        "postings",
        "positives",
        "base_rate",
        "pr_auc",
        "pr_auc_low",
        "pr_auc_high",
        "precision_at_budget",
        "precision_at_budget_low",
        "precision_at_budget_high",
        "fragile",
    ]
