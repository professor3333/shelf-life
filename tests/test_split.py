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
    SPLIT_NAMES,
    Cuts,
    SplitTooShallow,
    assert_temporal_order,
    best_cuts,
    crawl_waves,
    depth_report,
    embargo_width,
    feasible_cuts,
    max_run_gap,
    minimum_waves,
    projected_clear,
    resurrection_risk,
    rolling_origin_folds,
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


def _structural(n_waves: int, blind_tail: int = 1) -> pd.DataFrame:
    """A panel whose positives sit where `compute_labels` can actually put them.

    One removal per wave, and **none in the newest `blind_tail` waves**. That is
    not a stylistic choice: a positive needs the posting absent at two
    consecutive later runs, so the newest labelled wave cannot have one, and a
    fixture that gives it one is describing a panel the scraper cannot produce.
    `test_the_newest_labelled_wave_can_never_carry_a_positive` in
    `test_assemble.py` is where that claim is checked against the labeller
    itself rather than assumed here.
    """
    ids = tuple(f"p{index}" for index in range(n_waves + 2))
    presence = {"board": [set(ids) for _ in range(n_waves)]}
    positives = {("board", ids[wave], wave) for wave in range(max(0, n_waves - blind_tail))}
    return _panel(presence, positives=positives)


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
    # A fourth posting, so that dropping the two unlabelled rows still leaves a
    # negative in the held-out block. With only a, b and c the block reduces to
    # a single positive row, which `_validate` now refuses — correctly, since
    # PR-AUC on one class is 1.0 by construction. The fixture, not the rule, was
    # what needed changing: this test is about counting dropped rows.
    frame = _panel(
        _wide(ids=("a", "b", "c", "d")),
        positives={("board", "a", 5), ("board", "a", 8)},
        unlabelled={("board", "c", 8), ("board", "b", 8)},
    )
    result = temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))
    assert result.n_unlabelled == 2
    assert result.frame["y"].notna().all()


def test_an_all_positive_evaluation_block_is_refused():
    """The mirror of the no-positives check, and the one that congratulates you.

    A block with no positives announces itself: precision, recall and PR-AUC are
    undefined and something downstream complains. A block where *every* row is a
    positive is undefined in exactly the same way and reports 1.0 for all three —
    nothing looks broken, and the number is the best one you will ever see.

    Reachable at a long horizon, because a closure is knowable at once while
    survival needs the whole window to elapse. At H=7 on the 2026-09-08 snapshot
    that was 151 rows, every one of them a closure. `compute_labels` now drops
    those cohorts, so this should never fire — which is the point of checking it.
    """
    frame = _panel(
        _wide(),
        positives={("board", "a", 5), ("board", "a", 8), ("board", "b", 8), ("board", "c", 8)},
    )
    with pytest.raises(SplitTooShallow, match="every one"):
        temporal_split(frame, Cuts(WAVE0 + 2 * DAY, WAVE0 + 5 * DAY))


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
    """Eight waves for the smallest legal split on an even daily panel at H=1.

    One wave per block, plus everything inside each of the two embargoes, plus
    the blind wave at the tail. The embargo is the horizon plus one run's reach —
    two days here — and the wave sitting exactly on the boundary is discarded
    too, so each boundary costs three waves: 1 + 2*3 + 1 = 8.

    **This said 7 until 2026-09-07**, counting only the embargo's geometry. The
    missing term is the last one. Written as a test because that number is the
    answer to "how much longer must the scraper run", and a wrong one is a wrong
    plan — a 7 would have had the panel declared ready a day early, on a cut
    `_validate` then rejects for having no positives in the test block.
    """
    depth = minimum_waves(_structural(9))
    assert depth["burnt_per_boundary"] == 3
    assert depth["blind_tail"] == 1
    assert depth["needed"] == 8
    assert depth["present"] == 9
    assert depth["shortfall"] == 0


def test_the_reported_minimum_is_a_depth_that_actually_splits():
    """The regression guard for the 7-versus-8 bug, and the only honest form of
    it: take the number this function reports, build a panel exactly that deep,
    and require `feasible_cuts` to find a cut in it — then require one wave
    fewer to find none. A readiness check that disagrees with the acceptance
    check is worse than no readiness check, because it is trusted."""
    needed = int(minimum_waves(_structural(20))["needed"])
    assert feasible_cuts(_structural(needed))["valid"].any()
    assert not feasible_cuts(_structural(needed - 1))["valid"].any()


def test_the_cut_one_wave_short_is_rejected_for_the_right_reason():
    """Not merely rejected — rejected for *no positives in test*, which is the
    failure the wave count was blind to. An empty block would be a different
    bug with the same symptom."""
    table = feasible_cuts(_structural(7))
    assert "test block has no positives" in set(table["reason"])


def test_minimum_waves_separates_legal_from_evaluable():
    """A legal split is not an evaluable one. At the bare minimum the training
    block is a single wave, `wave_forward_folds` returns nothing from it, and
    every model comparison is then a single number with no error bar."""
    depth = minimum_waves(_structural(8))
    assert depth["shortfall"] == 0
    assert depth["folds_available"] == 0
    # burnt + (target - 1) waves on top of the legal minimum: 8 + 3 + 2
    assert depth["needed_for_folds"] == 13
    assert depth["folds_shortfall"] == 5
    assert minimum_waves(_structural(13))["folds_available"] == 3


def test_wave_spacing_is_measured_on_every_wave_not_the_labelled_ones():
    """Cadence is a fact about the scraper; labelling is a fact about the horizon.

    At a long horizon only the oldest few waves are labelled, so measuring
    spacing on that subset measures it on the smallest sample there is — and the
    error runs the wrong way. Measured at H=7 on 2026-09-09: the two labelled
    waves straddled a late 14:07 crawl and gave a median of 1d10h against a true
    cadence of 1d, so the embargo divided into six waves per boundary instead of
    nine and the panel appeared to need 14 waves when it needs 20. An
    underestimate of the wait, produced exactly when the panel is shallowest.

    Here: ten daily waves, of which only the oldest three are labelled, and one
    of those three runs ten hours late — the shape of the real 2026-09-01 crawl.
    The labelled subset therefore reports a day and a half; the schedule is a
    day, and the schedule is the answer.
    """
    ids = ("a", "b")
    presence = {"board": [set(ids) for _ in range(10)]}
    frame = _panel(presence, positives={("board", "a", 1)}).copy()

    waves = sorted(pd.unique(frame["t"]))
    late = waves[2]
    frame.loc[frame["t"] == late, "t"] = late + pd.Timedelta(hours=10)
    frame["label_observable"] = frame["t"] <= late + pd.Timedelta(hours=10)

    labelled_gaps = crawl_waves(frame[frame["label_observable"]]).diff().dropna().median()
    assert labelled_gaps > pd.Timedelta(days=1), "the fixture must skew the labelled subset"

    depth = minimum_waves(frame)
    assert depth["spacing"] == pytest.approx(DAY, abs=pd.Timedelta(hours=1)), (
        f"spacing {depth['spacing']} was taken from the labelled subset, not the schedule"
    )


def test_projected_clear_dates_the_two_gates_separately():
    """`minimum_waves` says how many waves; a person waiting wants which day.

    Both gates get a date because they fall on different days, and an open gate
    gets `None` rather than today's date — "already open" and "opens now" read
    the same in a log and mean different things.
    """

    # `now` is pinned just after each fixture's newest crawl: these panels are
    # months old by the wall clock, and a stalled panel is deliberately given no
    # date at all, which would otherwise make this a test of the fixture's age.
    def _ahead(waves: int):
        panel = _structural(waves)
        return projected_clear(panel, now=crawl_waves(panel).max() + pd.Timedelta(hours=1))

    eight = _ahead(8)
    assert eight["split_clears"] is None, "the legal split is already possible at 8"
    assert eight["folds_clear"] is not None

    spacing = eight["spacing"]
    newest = eight["newest_labelled_wave"]
    # 13 needed, 8 present: five more waves at the observed spacing.
    assert eight["folds_clear"] == newest + 5 * spacing

    both_open = _ahead(13)
    assert both_open["split_clears"] is None
    assert both_open["folds_clear"] is None
    assert both_open["folds_available"] == 3


def test_projected_clear_never_dates_a_gate_earlier_than_the_shortfall_allows():
    """It is the optimistic bound — every future wave labelled, spacing held —
    so it may be late but must never be early, which is the safe direction for a
    date a freeze is planned around."""
    for waves in range(2, 14):
        panel = _structural(waves)
        depth = minimum_waves(panel)
        # Pinned to just after the panel's own newest crawl. Left to the wall
        # clock these fixtures are months stale, and `projected_clear` now
        # declines to date a panel that has stopped accruing — correctly, but it
        # would make this a test of the fixture's age.
        ahead = projected_clear(panel, now=crawl_waves(panel).max() + pd.Timedelta(hours=1))
        if depth["folds_shortfall"] == 0:
            assert ahead["folds_clear"] is None
        else:
            gap = ahead["folds_clear"] - ahead["newest_labelled_wave"]
            assert gap >= depth["folds_shortfall"] * ahead["spacing"]


def test_depth_report_states_both_waits():
    text = depth_report(_structural(8))
    assert "cannot carry a positive" in text
    assert "rolling-origin fold(s) today" in text


def test_best_cuts_leaves_a_training_window_folds_can_be_cut_from():
    """The defect in `waves[0], waves[len // 2]`, which three modules used.

    It pins `train_end` to the first wave at every depth, so the training block
    is one wave deep on a panel of any size and `wave_forward_folds` returns
    nothing from it. Measured on the real panel: zero folds at 9 waves and zero
    at 17. Every wave the scraper added went to the evaluation blocks.
    """
    from src.models.evaluate import wave_forward_folds

    frame = _structural(15)
    naive = temporal_split(frame, Cuts(*_naive_cuts(frame)))
    chosen = temporal_split(frame, best_cuts(frame))

    assert len(wave_forward_folds(naive.train, naive.embargo)) == 0
    assert len(wave_forward_folds(chosen.train, chosen.embargo)) >= 3


def _naive_cuts(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    waves = crawl_waves(frame[frame["label_observable"]])
    return waves.iloc[0], waves.iloc[len(waves) // 2]


def test_best_cuts_never_recommends_a_cut_the_split_rejects():
    """The property the two formulas could not offer. `best_cuts` selects from
    `feasible_cuts`, so anything it returns has already been checked for three
    non-empty blocks and positives in both evaluation blocks — it cannot hand
    back a cut `temporal_split` then refuses."""
    for depth in range(8, 18):
        frame = _structural(depth)
        temporal_split(frame, best_cuts(frame))  # raises if the cut is not usable


def test_best_cuts_refuses_rather_than_returning_a_degenerate_cut():
    with pytest.raises(SplitTooShallow):
        best_cuts(_structural(7))


def test_best_cuts_grows_every_block_as_the_panel_deepens():
    """Neither greedy rule does this, which is why the objective is proportional.

    Maximising folds deepens training without limit and leaves validation one
    wave wide for ever; maximising the evaluation blocks once the fold target is
    met starves the fit instead. The shares are of the waves that *survive the
    embargo*, which is the part the old 60/20/20 got wrong.
    """
    widths = []
    for depth in range(13, 24):
        frame = _structural(depth)
        result = temporal_split(frame, best_cuts(frame))
        block = result.frame
        widths.append(tuple(block[block["split"] == name]["t"].nunique() for name in SPLIT_NAMES))

    for earlier, later in zip(widths, widths[1:], strict=False):
        assert all(b >= a for a, b in zip(earlier, later, strict=True)), (
            f"a block shrank as the panel deepened: {earlier} -> {later}"
        )
    assert widths[-1][1] > widths[0][1], "validation never grew"
    assert widths[-1][2] > widths[0][2], "test never grew"


def test_rolling_origin_folds_is_the_block_less_one_embargo():
    assert rolling_origin_folds(3, 3) == 0
    assert rolling_origin_folds(4, 3) == 1
    assert rolling_origin_folds(8, 3) == 5


def test_minimum_waves_reports_a_shortfall_on_a_panel_too_short():
    depth = minimum_waves(_panel(_wide(3)))
    assert depth["shortfall"] == depth["needed"] - 3
    assert "more labelled wave(s) needed" in depth_report(_panel(_wide(3)))


def test_a_single_late_crawl_widens_the_embargo_for_the_whole_panel():
    """A gap wide enough moves the wave count, not merely the embargo.

    The threshold matters and is easy to state wrongly. A *missed* crawl doubles
    a source's gap to 48h, pushing the embargo past three times the spacing and
    costing a wave at each boundary — that is the fixture below. The real
    2026-09-01 slip, 03:45 to 14:07, is not that: it widened the embargo from 2d
    to 2d10h and `burnt` stayed at 3, so it cost time and no waves at all.
    """
    even = minimum_waves(_panel(_wide(9)))
    ids = {"a", "b", "c"}
    skipped = [ids] * 9
    skipped[2] = None  # the source did not run in that wave, so its gap doubles
    late = minimum_waves(_panel({"board": [ids] * 9, "erratic": skipped}))
    assert late["embargo"] > even["embargo"]
    assert late["needed"] > even["needed"]


# --- is the wait running down, or has the collector stopped? ----------------


def test_a_panel_whose_newest_crawl_is_stale_is_called_stalled():
    """The failure this ends: a daily watch that prints the same line either way.

    `waves=2` on Tuesday and `waves=2` on Thursday reads as patience and is
    actually an outage. Nothing in the depth arithmetic can tell them apart,
    because it only ever looks at how many waves there are.
    """
    from src.data.split import accrual_status, crawl_waves

    panel = _structural(9)
    newest = crawl_waves(panel).max()

    fresh = accrual_status(panel, now=newest + pd.Timedelta(hours=6))
    assert fresh["stalled"] is False

    stalled = accrual_status(panel, now=newest + pd.Timedelta(days=3))
    assert stalled["stalled"] is True
    assert stalled["waves_missed"] == 3


def test_one_late_crawl_is_a_hiccup_and_two_are_a_stopped_input():
    """The tolerance is two waves of slack, deliberately. A watch that cried stall
    on every late crawl would be muted within a week."""
    from src.data.split import accrual_status, crawl_waves

    panel = _structural(9)
    newest = crawl_waves(panel).max()
    assert accrual_status(panel, now=newest + pd.Timedelta(days=1, hours=12))["stalled"] is False
    assert accrual_status(panel, now=newest + pd.Timedelta(days=2, hours=12))["stalled"] is True


def test_no_date_is_projected_through_a_stall():
    """`newest + shortfall x spacing` is arithmetic on an assumption that stopped
    holding, and its answer would be the most confident-looking line in the report."""
    from src.data.split import crawl_waves

    panel = _structural(9)
    newest = crawl_waves(panel).max()

    running = projected_clear(panel, now=newest + pd.Timedelta(hours=6))
    assert running["stalled"] is False
    assert running["folds_clear"] is not None, "a healthy shallow panel still gets a date"

    stopped = projected_clear(panel, now=newest + pd.Timedelta(days=5))
    assert stopped["stalled"] is True
    assert stopped["split_clears"] is None and stopped["folds_clear"] is None


def test_the_depth_report_says_the_distance_is_not_closing():
    from src.data.split import crawl_waves, depth_report

    panel = _structural(9)
    newest = crawl_waves(panel).max()

    assert "NOT ACCRUING" not in depth_report(panel, now=newest + pd.Timedelta(hours=6))
    stalled = depth_report(panel, now=newest + pd.Timedelta(days=4))
    assert "NOT ACCRUING" in stalled
    assert "not closing" in stalled


def test_the_watch_never_prints_open_for_a_stalled_panel():
    """`folds_clear` is None both when the gate is open and when nothing is
    arriving. Collapsing those would print the most reassuring word available on
    the day the collector died."""
    from pathlib import Path

    script = (Path(__file__).resolve().parents[1] / "scripts" / "watch_depth.sh").read_text()
    assert 'clears = "stalled"' in script, "the watch cannot distinguish stalled from open"
    stalled_branch = script.index('clears = "stalled"')
    open_branch = script.index('clears = "open"')
    assert stalled_branch < open_branch, (
        "the open branch is reachable before the stalled one, so a stalled panel logs clears=open"
    )
    assert "exit 5" in script, "a stall does not get its own exit code"
