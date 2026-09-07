"""Tests for leave-one-board-out evaluation.

The question is whether the model learned about job postings or about seven
Greenhouse boards. The tests that matter here are the ones about *refusing*:
on the panel this project has, most folds cannot be run, and a module that
quietly reported a PR-AUC on five positives would answer the question wrongly
while looking like it had answered it.
"""

from __future__ import annotations

import pandas as pd
import pytest
from panels import make_closing_panel

from src.data.split import best_cuts, temporal_split
from src.models.generalisation import (
    MIN_HELD_OUT_POSITIVES,
    BoardFold,
    leave_one_board_out,
    report_tables,
    verdict,
)
from src.models.train import build_xgboost


def _multi_board(boards: int = 4) -> pd.DataFrame:
    """One synthetic panel per board, stacked. Same rows, different `source`,
    so every board is equally learnable and a transfer gap cannot come from one
    board being intrinsically harder."""
    frames = []
    for index in range(boards):
        frame = make_closing_panel()
        frame = frame.copy()
        frame["source"] = f"greenhouse:board{index}"
        frame["source_id"] = frame["source_id"].astype(str) + f"-b{index}"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def multi_split():
    panel = _multi_board()
    return temporal_split(panel, best_cuts(panel))


# --- the refusals, which are most of the behaviour on this panel -------------


def test_a_board_with_too_few_positives_is_refused_not_scored():
    """The failure this module exists to prevent. On the 2026-09-07 panel
    airtable has 0 closures, python_org 3 and duolingo 5 — a PR-AUC on five
    positives is a number set by which five, and printing it beside a real one
    invites a comparison that cannot be made."""
    panel = _multi_board(3)
    split = temporal_split(panel, best_cuts(panel))
    folds, skipped = leave_one_board_out(split, lambda: build_xgboost(split), min_positives=10_000)
    assert folds == []
    assert {item.board for item in skipped} == set(split.val["source"].unique())
    assert all("positive(s) in validation" in item.reason for item in skipped)


def test_a_board_too_large_to_hold_out_is_refused():
    """Holding out anthropic removes 52% of the training positives, so the
    transfer arm would be fitted on half the data *and* one fewer board, and the
    two effects cannot be told apart in the result."""
    panel = make_closing_panel()  # a single board: removing it removes everything
    split = temporal_split(panel, best_cuts(panel))
    folds, skipped = leave_one_board_out(split, lambda: build_xgboost(split), min_positives=1)

    assert folds == []
    assert len(skipped) == 1
    assert "of training positives" in skipped[0].reason


def test_the_refusal_says_which_guard_it_hit():
    """Two different reasons a fold cannot run, and conflating them would hide
    which constraint is the binding one."""
    panel = make_closing_panel()
    split = temporal_split(panel, best_cuts(panel))
    _, thin = leave_one_board_out(split, lambda: build_xgboost(split), min_positives=10_000)
    _, large = leave_one_board_out(split, lambda: build_xgboost(split), min_positives=1)

    assert "positive(s) in validation" in thin[0].reason
    assert "training positives" in large[0].reason


# --- the fold itself ----------------------------------------------------------


def test_a_runnable_fold_scores_both_arms_on_the_same_rows(multi_split):
    """The control the obvious version of this experiment omits. A bare
    transfer score is uninterpretable — boards have different base rates — so
    each fold is also scored by a model that *did* see the board, on the same
    rows."""
    folds, _ = leave_one_board_out(
        multi_split, lambda: build_xgboost(multi_split), min_positives=1, min_train_share=0.1
    )
    assert folds, "the multi-board fixture must yield at least one runnable fold"

    for fold in folds:
        assert fold.eval_positives > 0
        assert fold.eval_rows > fold.eval_positives
        assert fold.gap == pytest.approx(fold.ceiling_pr_auc - fold.transfer_pr_auc)
        assert 0.0 < fold.train_share < 1.0


def test_the_transfer_arm_never_sees_the_held_out_board(multi_split):
    """The property the whole experiment rests on. Checked by construction:
    holding out a board must leave strictly fewer training positives than the
    ceiling arm has."""
    total = int((multi_split.train["y"] == 1).sum())
    folds, _ = leave_one_board_out(
        multi_split, lambda: build_xgboost(multi_split), min_positives=1, min_train_share=0.1
    )
    for fold in folds:
        assert fold.train_positives < total


def test_every_board_present_in_both_blocks_is_accounted_for(multi_split):
    """Scored or refused — never silently dropped. A board that vanished from
    the report would be a board nobody knew was untested."""
    folds, skipped = leave_one_board_out(
        multi_split, lambda: build_xgboost(multi_split), min_positives=1, min_train_share=0.1
    )
    seen = {fold.board for fold in folds} | {item.board for item in skipped}
    expected = set(multi_split.val["source"]) & set(multi_split.train["source"])
    assert seen == expected


# --- the verdict --------------------------------------------------------------


def _folds(gaps: list[float]) -> list[BoardFold]:
    return [
        BoardFold(
            board=f"b{i}",
            eval_rows=200,
            eval_positives=20,
            train_positives=80,
            train_share=0.8,
            base_rate=0.1,
            transfer_pr_auc=0.2,
            ceiling_pr_auc=0.2 + gap,
        )
        for i, gap in enumerate(gaps)
    ]


def test_no_folds_is_reported_as_unanswered_not_as_transfer_working():
    """The distinction that matters when the panel is too shallow. 'No fold
    could be run' and 'the model transfers' are different statements, and only
    one of them is licensed today."""
    text = verdict([])
    assert "unanswered rather than answered negatively" in text
    assert "transfer looks intact" not in text.lower()


def test_one_fold_is_called_an_observation():
    text = verdict(_folds([0.05]))
    assert "One board only" in text
    assert "observation rather than a pattern" in text


def test_a_gap_inside_the_spread_is_read_as_transfer_intact():
    text = verdict(_folds([0.01, -0.008, 0.004, 0.006]))
    assert "Transfer looks intact" in text
    assert "properties of postings rather than of boards" in text


def test_a_consistent_gap_is_read_as_board_specific_learning():
    text = verdict(_folds([0.09, 0.085, 0.092, 0.088]))
    assert "Board-specific learning is doing work" in text
    assert "Greenhouse caveat" in text


def test_a_negative_gap_is_flagged_rather_than_celebrated():
    """Doing better on unseen boards usually means the two arms differ in some
    way other than the board, and that is worth understanding before reporting."""
    text = verdict(_folds([-0.09, -0.085, -0.092, -0.088]))
    assert "not a result to celebrate" in text


def test_the_tables_carry_the_denominator_beside_every_score():
    scored, refused = report_tables(_folds([0.05, 0.01]), [])
    assert {"eval_rows", "eval_positives", "base_rate", "train_share"} <= set(scored.columns)
    assert refused.empty


def test_the_minimum_is_thin_enough_to_be_stated_honestly():
    """Ten is already too few for a tight interval; the constant exists to
    exclude numbers that are not about the model at all."""
    assert MIN_HELD_OUT_POSITIVES == 10
