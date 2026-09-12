"""Tests for the bootstrap intervals on the headline numbers.

The decision under test is the resampling unit. Everything else here is
arithmetic that could be checked by eye; that one is a claim about what a
repetition of the experiment would look like, and getting it wrong produces
intervals that are confidently the wrong width.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.metrics import average_precision
from src.models.uncertainty import (
    FRAGILE_POSITIVES,
    bootstrap_block,
    cluster_resample,
    format_table,
    fragility_note,
    interval,
    posting_ids,
)


def _block(labels: list[int], ids: list[str] | None = None) -> pd.DataFrame:
    ids = ids or [str(i) for i in range(len(labels))]
    return pd.DataFrame(
        {"source": ["greenhouse:acme"] * len(labels), "source_id": ids, "y": labels}
    )


# --- the resampling unit ------------------------------------------------------


def test_rows_of_one_posting_are_drawn_or_dropped_together():
    """The property that makes this a cluster bootstrap. If a posting's six
    job-days can be drawn independently, the block behaves as six observations
    when it is one, and every interval computed from it is the wrong width."""
    labels = [0, 0, 0, 1, 0, 0]
    block = _block(labels, ids=["a", "a", "a", "b", "b", "b"])
    groups = posting_ids(block)
    truth = np.array(labels)
    scores = np.arange(6, dtype=float)

    seen = set()

    def record(t, s):
        # each resample must contain a whole number of each posting's rows
        seen.add(tuple(sorted(s.tolist())))
        return 0.5

    cluster_resample(truth, scores, groups, record, resamples=60, seed=1)

    for drawn in seen:
        counts = pd.Series(drawn).map({0: "a", 1: "a", 2: "a", 3: "b", 4: "b", 5: "b"})
        by_posting = counts.value_counts()
        # every posting present appears as a multiple of its three rows
        assert all(count % 3 == 0 for count in by_posting)


def test_posting_ids_do_not_collide_across_boards():
    """`source_id` is unique only within a board. Two boards reusing an id would
    be merged into one cluster and their rows drawn together, which is a
    silently wrong grouping rather than an error."""
    block = pd.DataFrame(
        {
            "source": ["greenhouse:acme", "greenhouse:other"],
            "source_id": ["1", "1"],
            "y": [0, 1],
        }
    )
    assert len(set(posting_ids(block))) == 2


def test_a_resample_with_no_positives_is_discarded_not_scored():
    """Average precision is undefined with no positives, and a zero would be
    averaged in as a bad score rather than an absent one — which drags the lower
    endpoint, the number a reader leans on."""
    truth = np.array([0, 0, 0, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    groups = np.array(["a", "b", "c", "d"])

    samples = cluster_resample(truth, scores, groups, average_precision, resamples=200, seed=0)
    assert samples.size < 200, "some resamples must have missed the single positive"
    assert not np.isnan(samples).any()
    assert (samples > 0).all()


def test_the_interval_brackets_the_point_estimate():
    """Overlapping scores on purpose. A perfectly separable fixture returns 1.0
    on every resample and a width of exactly zero — which is the right answer
    there and tests nothing, since no data this project sees is separable."""
    rng = np.random.default_rng(3)
    truth = np.array([1] * 20 + [0] * 80)
    scores = rng.uniform(size=100) * 0.6 + truth * 0.25

    point = average_precision(truth, scores)
    groups = np.array([str(i) for i in range(100)])
    samples = cluster_resample(truth, scores, groups, average_precision, resamples=400, seed=0)
    result = interval("pr_auc", point, samples)

    assert result.low <= point <= result.high
    assert result.width > 0
    assert result.resamples == 400


def test_a_perfectly_separable_block_has_no_width():
    """Recorded rather than avoided: a zero-width interval means every resample
    agreed, not that the bootstrap failed."""
    truth = np.array([1] * 20 + [0] * 80)
    scores = np.concatenate([np.linspace(0.9, 0.6, 20), np.linspace(0.4, 0.0, 80)])
    groups = np.array([str(i) for i in range(100)])

    samples = cluster_resample(truth, scores, groups, average_precision, resamples=200, seed=0)
    assert interval("pr_auc", 1.0, samples).width == 0.0


def test_more_data_gives_a_narrower_interval():
    """The property that makes the interval worth reporting across snapshots: it
    should tighten as the panel deepens, without anything being rewritten."""
    rng = np.random.default_rng(0)

    def width(n_postings: int) -> float:
        truth = rng.binomial(1, 0.2, n_postings)
        scores = rng.uniform(size=n_postings) * 0.4 + truth * 0.3
        groups = np.array([str(i) for i in range(n_postings)])
        point = average_precision(truth, scores)
        samples = cluster_resample(truth, scores, groups, average_precision, 400, seed=0)
        return interval("pr_auc", point, samples).width

    assert width(1000) < width(100)


# --- the threshold is the frozen one ------------------------------------------


def test_precision_and_recall_use_the_threshold_passed_in():
    """Not a threshold recomputed per resample. The artifact ships one operating
    point, and an interval that re-derives it would mix separation with where
    the budget happened to fall."""
    block = _block([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])

    strict = bootstrap_block(block, scores, threshold=0.85, resamples=50, metrics=["recall"])
    loose = bootstrap_block(block, scores, threshold=0.05, resamples=50, metrics=["recall"])
    assert strict["recall"].point == pytest.approx(0.5)
    assert loose["recall"].point == pytest.approx(1.0)


# --- what the report says about a thin block ----------------------------------


def test_a_thin_block_is_flagged():
    note = fragility_note(_block([1] * 5 + [0] * 95))
    assert note is not None
    assert "5 positive(s)" in note
    assert "order of magnitude rather than as a bound" in note


def test_a_healthy_block_is_not_flagged():
    assert fragility_note(_block([1] * FRAGILE_POSITIVES + [0] * 100)) is None


def test_the_table_renders_an_absent_interval_as_a_dash_not_nan():
    """A block that could not be resampled has no interval, and printing `nan`
    in a results table makes a fact about the block look like a broken run."""
    empty = interval("pr_auc", 0.3, np.array([]))
    text = "\n".join(format_table({"pr_auc": empty}))
    assert "nan" not in text.lower()
    assert "0.3000" in text


def test_every_headline_metric_gets_an_interval():
    """The five the design document names plus the three product metrics
    (added 2026-09-12: precision@budget, recall@budget, lift), so a metric
    cannot be added to the report without someone deciding whether it needs
    one. The product ones come first: they are the headline."""
    block = _block([1] * 10 + [0] * 40)
    scores = np.linspace(0.9, 0.1, 50)
    result = bootstrap_block(block, scores, threshold=0.5, resamples=100)
    assert list(result)[:3] == ["precision_at_budget", "recall_at_budget", "lift_at_budget"]
    assert set(result) == {
        "precision_at_budget",
        "recall_at_budget",
        "lift_at_budget",
        "pr_auc",
        "brier",
        "ece",
        "precision",
        "recall",
    }
    assert result["lift_at_budget"].point == pytest.approx(
        result["precision_at_budget"].point / 0.2
    )


def test_the_bootstrap_is_deterministic_for_a_given_seed():
    """A confidence interval that moves when nothing changed cannot be compared
    across snapshots, which is the whole point of recording it in the ledger."""
    block = _block([1] * 10 + [0] * 40)
    scores = np.linspace(0.9, 0.1, 50)
    first = bootstrap_block(block, scores, threshold=0.5, resamples=100, seed=7)
    second = bootstrap_block(block, scores, threshold=0.5, resamples=100, seed=7)
    assert first["pr_auc"].low == second["pr_auc"].low
    assert first["pr_auc"].high == second["pr_auc"].high
