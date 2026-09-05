"""Tests for the temporal split.

Every fixture is built by hand. Nothing here reads `data/`, both because the
real panel is not committed and because a split test that depends on today's
scrape is a test that changes its mind overnight.

The four assertions the validation protocol demands — no shuffle, no id in two splits
(as amended by `docs/design.md` §8), strict temporal order, determinism — are
each a test below, plus the ones the data turned out to need.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import (
    Cuts,
    SplitTooShallow,
    assert_temporal_order,
    crawl_waves,
    depth_report,
    embargo_width,
    feasible_cuts,
    max_run_gap,
    minimum_waves,
    resurrection_risk,
    split_report,
    temporal_split,
)

DAY = pd.Timedelta(days=1)
NO_JITTER = pd.Timedelta(0)
WAVE0 = pd.Timestamp("2026-08-31T03:45:00Z")


def _panel(
    presence: dict[str, list[set[str] | None]],
    positives: set[tuple[str, str, int]] = frozenset(),
    unlabelled: set[tuple[str, str, int]] = frozenset(),
    horizon_days: int = 1,
    spacing: pd.Timedelta = DAY,
    jitter: pd.Timedelta = NO_JITTER,
) -> pd.DataFrame:
    """Build a job-day panel.

    `presence` maps a source to what each *wave* saw; `None` means the source
    did not run in that wave, which is how `run_index` comes adrift from
    calendar time. `jitter` offsets each source within a wave, reproducing the
    seconds-apart Greenhouse crawls.
    """
    rows = []
    for ordinal, (source, waves) in enumerate(presence.items()):
        run_index = 0
        for wave, ids in enumerate(waves):
            if ids is None:
                continue
            t = WAVE0 + wave * spacing + ordinal * jitter
            for source_id in sorted(ids):
                key = (source, source_id, wave)
                rows.append(
                    {
                        "source": source,
                        "source_id": source_id,
                        "t": t,
                        "run_index": run_index,
                        "y": pd.NA if key in unlabelled else int(key in positives),
                        "label_observable": key not in unlabelled,
                        "horizon_days": horizon_days,
                    }
                )
            run_index += 1
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("Int8")
    return frame


def _wide(n_waves: int = 9, ids=("a", "b", "c")) -> dict[str, list[set[str]]]:
    """A panel deep enough that a three-way split is actually feasible."""
    return {"board": [set(ids) for _ in range(n_waves)]}


def _feasible_frame() -> pd.DataFrame:
    """Nine waves, with a removal in each of the later blocks so val and test
    both carry a positive."""
    return _panel(
        _wide(),
        positives={("board", "b", 5), ("board", "a", 8), ("board", "c", 5)},
    )


# --- the embargo -----------------------------------------------------------


def test_embargo_is_the_horizon_plus_one_run():
    """A label reads absence at the next run *corroborated at the one after*, so
    the reach is H plus one run, not H. This is the correction to
    problem_definition.md §7, which said H."""
    frame = _panel(_wide(3))
    assert max_run_gap(frame) == DAY
    assert embargo_width(frame, horizon_days=1) == 2 * DAY
    assert embargo_width(frame, horizon_days=7) == 8 * DAY


def test_embargo_uses_the_worst_run_gap_not_the_typical_one():
    """The real schedule slipped to 34.4h once. An embargo sized on the median
    gap would leave that row's label reaching into the next block."""
    frame = _panel({"board": [{"a"}, {"a"}, {"a"}]})
    frame.loc[frame["run_index"] == 2, "t"] += pd.Timedelta(hours=10)
    assert max_run_gap(frame) == DAY + pd.Timedelta(hours=10)
    assert embargo_width(frame, horizon_days=1) == 2 * DAY + pd.Timedelta(hours=10)


def test_embargo_is_read_from_the_frames_own_horizon():
    """The embargo must not be able to disagree with the label it protects."""
    frame = _panel(_wide(3), horizon_days=7)
    cuts = Cuts(train_end=WAVE0, val_end=WAVE0 + DAY)
    with pytest.raises(SplitTooShallow):
        temporal_split(frame, cuts)  # a 7-day horizon cannot fit in a 3-day panel


# --- the four §4.2 assertions ---------------------------------------------


def test_blocks_are_strictly_ordered_in_time_and_clear_the_embargo():
    result = temporal_split(_feasible_frame(), Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    assert result.train["t"].max() < result.val["t"].min()
    assert result.val["t"].max() < result.test["t"].min()
    assert result.val["t"].min() - result.train["t"].max() > result.embargo
    assert result.test["t"].min() - result.val["t"].max() > result.embargo
    assert_temporal_order(result)


def test_assignment_ignores_row_order_entirely():
    """`shuffle=True` is banned; this is the stronger property — the split is a
    pure function of `t`, so no ordering of the input can change it."""
    frame = _feasible_frame()
    cuts = Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY)
    forward = temporal_split(frame, cuts).frame
    reversed_rows = temporal_split(frame.iloc[::-1].reset_index(drop=True), cuts).frame
    pd.testing.assert_frame_equal(forward, reversed_rows)


def test_split_is_deterministic_across_repeated_calls():
    frame = _feasible_frame()
    cuts = Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY)
    first = temporal_split(frame, cuts).frame
    second = temporal_split(frame, cuts).frame
    pd.testing.assert_frame_equal(first, second)


def test_no_row_lands_in_two_blocks():
    result = temporal_split(_feasible_frame(), Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    assert result.frame["split"].isin(["train", "val", "test"]).all()
    counts = len(result.train) + len(result.val) + len(result.test)
    assert counts == len(result.frame)


# --- cutting on `t`, never on `run_index` ---------------------------------


def test_a_source_that_missed_a_wave_is_placed_by_time_not_by_run_index():
    """python_org skipped the 2026-08-31 crawl, so its `run_index` 0 is
    2026-09-01 — the same instant as every Greenhouse board's `run_index` 1. A
    split on run index would put 31 rows of the future in the earliest training
    block. This fixture is that situation in miniature."""
    frame = _panel(
        {
            "board": [{"a"}, {"a"}, {"a"}, {"a"}, {"a"}, {"a"}, {"a"}, {"a"}, {"a"}],
            "latecomer": [None, {"z"}, {"z"}, {"z"}, {"z"}, {"z"}, {"z"}, {"z"}, {"z"}],
        },
        positives={("board", "a", 5), ("board", "a", 8)},
    )
    late = frame[frame["source"] == "latecomer"]
    assert late["run_index"].min() == 0
    assert late["t"].min() == WAVE0 + DAY  # run_index 0, but one day late

    result = temporal_split(frame, Cuts(WAVE0, WAVE0 + 5 * DAY))
    assert "latecomer" not in set(result.train["source"])  # its run 0 is not in the past


def test_a_cut_inside_a_crawl_wave_is_refused():
    """The six Greenhouse boards are crawled seconds apart. A cut between two of
    them divides one sweep of the scheduler across two blocks."""
    frame = _panel(
        {f"board{i}": [{"a"}] * 9 for i in range(3)},
        positives={("board0", "a", 5), ("board0", "a", 8)},
        jitter=pd.Timedelta(seconds=4),
    )
    waves = crawl_waves(frame)
    assert len(waves) == 9  # not 27
    with pytest.raises(ValueError, match="inside a crawl wave"):
        temporal_split(frame, Cuts(WAVE0 + 2 * DAY + pd.Timedelta(seconds=2), waves[5]))


# --- refusing a degenerate split ------------------------------------------


def test_an_evaluation_block_with_no_positives_is_refused():
    """A test set with no positives has undefined precision, recall and PR-AUC.
    Returning it would look like an ordinary DataFrame all the way to the
    metric."""
    frame = _panel(_wide(), positives={("board", "a", 0), ("board", "b", 1)})
    with pytest.raises(SplitTooShallow, match="no positives"):
        temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


def test_an_empty_block_is_refused():
    frame = _panel(_wide(4), positives={("board", "a", 3)})
    with pytest.raises(SplitTooShallow, match="empty"):
        temporal_split(frame, Cuts(WAVE0, WAVE0 + DAY))


def test_the_refusal_carries_the_feasibility_table():
    """The exception has to say what would fix it, because on a shallow panel
    this is the normal outcome and the answer is usually 'wait for depth'."""
    frame = _panel(_wide(), positives={("board", "a", 0)})
    with pytest.raises(SplitTooShallow, match="candidate cuts are usable"):
        temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


def test_require_positives_can_be_switched_off_for_plumbing_tests():
    frame = _panel(_wide(), positives={("board", "a", 0)})
    result = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY), require_positives=False)
    assert len(result.test) > 0


# --- censored rows --------------------------------------------------------


def test_unlabelled_rows_are_dropped_and_counted_not_treated_as_negative():
    frame = _panel(
        _wide(),
        positives={("board", "a", 5), ("board", "a", 8)},
        unlabelled={("board", "c", 8), ("board", "b", 8)},
    )
    result = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    assert result.n_unlabelled == 2
    assert result.frame["y"].notna().all()


# --- the seen/unseen breakdown (design.md §8) -----------------------------


def test_evaluation_rows_are_flagged_by_whether_training_saw_the_posting():
    frame = _panel(
        {
            "board": [
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b"},
                {"a", "b", "newcomer"},
            ]
        },
        positives={("board", "a", 5), ("board", "a", 8)},
    )
    result = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    test_block = result.test
    assert test_block.loc[test_block["source_id"] == "newcomer", "seen_in_train"].eq(False).all()
    assert test_block.loc[test_block["source_id"] == "a", "seen_in_train"].all()
    assert not result.train["seen_in_train"].any()  # meaningless for training rows


def test_the_report_breaks_evaluation_blocks_into_carried_over_and_unseen():
    result = temporal_split(_feasible_frame(), Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    report = split_report(result)
    assert set(report["split"]) == {"train", "val", "test"}
    for name in ("val", "test"):
        arms = set(report.loc[report["split"] == name, "postings"])
        assert arms == {"all", "carried over", "unseen"}
        block = report[(report["split"] == name) & (report["postings"] != "all")]
        assert block["rows"].sum() == int(
            report.loc[(report["split"] == name) & (report["postings"] == "all"), "rows"].iloc[0]
        )


# --- diagnostics ----------------------------------------------------------


def test_feasible_cuts_enumerates_waves_and_marks_the_usable_ones():
    table = feasible_cuts(_feasible_frame())
    assert len(table) == 36  # every ordered pair of the nine waves
    assert table["valid"].any()
    assert set(table.loc[~table["valid"], "reason"]) <= {
        "train block empty",
        "val block empty",
        "test block empty",
        "val block has no positives",
        "test block has no positives",
    }


def test_resurrection_risk_finds_a_posting_that_came_back():
    """`t_gone` requires that a posting never re-appeared, and that clause reads
    the whole remaining panel — so no embargo of any width fully seals a
    training label from the evaluation period. greenhouse:gitlab 8615319002 was
    present at runs [0, 3, 4]: two consecutive absences, enough to satisfy the
    corroboration guard, and then it returned."""
    frame = _panel({"board": [{"a", "b"}, {"b"}, {"b"}, {"a", "b"}, {"a", "b"}]})
    risk = resurrection_risk(frame)
    assert list(risk["source_id"]) == ["a"]
    assert risk["max_absent_streak"].iloc[0] == 2


def test_resurrection_risk_is_empty_when_nothing_comes_back():
    assert resurrection_risk(_panel(_wide(4))).empty


def test_minimum_waves_counts_what_the_embargo_burns():
    """Seven waves for the smallest legal split on an even daily panel at H=1.

    One wave per block, plus everything inside each of the two embargoes. The
    embargo is the horizon plus one run's reach — two days here — and the wave
    sitting exactly on the boundary is discarded too, so each boundary costs
    three waves and the arithmetic is 1 + 2*3 = 7. Written as a test because
    that number is the answer to "how much longer must the scraper run", and a
    wrong one is a wrong plan.
    """
    depth = minimum_waves(_panel(_wide(9)))
    assert depth["burnt_per_boundary"] == 3
    assert depth["needed"] == 7
    assert depth["present"] == 9
    assert depth["shortfall"] == 0


def test_minimum_waves_reports_a_shortfall_on_a_panel_too_short():
    depth = minimum_waves(_panel(_wide(3)))
    assert depth["shortfall"] == depth["needed"] - 3
    assert "more labelled wave(s) needed" in depth_report(_panel(_wide(3)))


def test_a_single_late_crawl_widens_the_embargo_for_the_whole_panel():
    """The widest run gap sets the reach, so one late crawl costs waves forever.

    Not a hypothetical: the 2026-09-01 crawl fired at 14:07 instead of 03:45,
    making the widest gap 34.4h, and every boundary in the panel has paid for it
    since.
    """
    even = minimum_waves(_panel(_wide(9)))
    ids = {"a", "b", "c"}
    skipped = [ids] * 9
    skipped[2] = None  # the source did not run in that wave, so its gap doubles
    late = minimum_waves(_panel({"board": [ids] * 9, "erratic": skipped}))
    assert late["embargo"] > even["embargo"]
    assert late["needed"] > even["needed"]
