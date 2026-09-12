"""A board scored in pages ranks exactly as the same board scored in one call would."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_inference import FIXED_POSTING, FIXED_T

from src.inference import boards as boards_module
from src.inference.boards import BoardError, BoardExpired, BoardStore
from src.inference.predict import MAX_BATCH, Predictor


def _board(n: int) -> list[dict]:
    titles = ("Senior Data Engineer", "Werkstudent Marketing", "Staff Engineer", "Head of Product")
    return [
        {
            **FIXED_POSTING,
            "title": f"{titles[i % len(titles)]} {i // len(titles)}",
            "content_chars": 400 + (i * 37) % 2600,
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def predictor(synthetic_artifact):
    return Predictor.load(synthetic_artifact)


def _upload_and_score(store, predictor, postings, **kwargs):
    board = store.create(Predictor.moment(FIXED_T), **kwargs)
    for start in range(0, len(postings), MAX_BATCH):
        board.add(postings[start : start + MAX_BATCH])
    calls = 0
    while not board.done:
        board.score_next(predictor)
        calls += 1
    return board, calls


def test_a_paged_board_ranks_as_one_call_would(predictor, monkeypatch):
    """The property the whole design rests on: the server's paged scoring plus
    one ordering equals `Predictor.rank` on the whole board with the cap lifted."""
    from src.inference import predict as predict_module

    postings = _board(2 * MAX_BATCH + 100)
    board, calls = _upload_and_score(BoardStore(), predictor, postings)
    assert calls == 3
    paged = board.rank(predictor, budget=20)

    monkeypatch.setattr(predict_module, "MAX_BATCH", len(postings))
    whole = predictor.rank(postings, t=FIXED_T, budget=20)
    assert [p.rank for p in paged.postings] == [p.rank for p in whole.postings]
    assert [p.watch for p in paged.postings] == [p.watch for p in whole.postings]
    assert paged.threshold_applied == pytest.approx(whole.threshold_applied)
    assert paged.threshold_source == whole.threshold_source


def test_a_snapshot_board_derives_context_over_the_whole_board(predictor, monkeypatch):
    """Same-title counts and the board size must see every page, not one."""
    from src.inference import predict as predict_module

    postings = _board(MAX_BATCH + 40)
    postings[0]["title"] = postings[-1]["title"] = "Same Title"  # first page and last page
    board, _ = _upload_and_score(
        BoardStore(), predictor, postings, is_snapshot=True, previous_board_size=300
    )
    ranked = board.rank(predictor, budget=5)
    assert ranked.board_size == len(postings)
    assert ranked.board_context_source == "derived from the snapshot"
    assert board.scored_payloads[0]["n_same_title_on_board"] == 2.0
    assert board.scored_payloads[0]["board_size_at_t"] == float(len(postings))
    assert board.scored_payloads[0]["board_growth"] == float(len(postings) - 300)

    monkeypatch.setattr(predict_module, "MAX_BATCH", len(postings))
    whole = predictor.rank(
        postings, t=FIXED_T, budget=5, board_snapshot=True, previous_board_size=300
    )
    assert [p.probability for p in ranked.postings] == pytest.approx(
        [p.probability for p in whole.postings]
    )


def test_membership_is_sealed_once_scoring_starts(predictor):
    board = BoardStore().create(Predictor.moment(FIXED_T), is_snapshot=True)
    board.add(_board(10))
    board.score_next(predictor)
    with pytest.raises(BoardError, match="can no longer be added"):
        board.add(_board(1))


def test_rank_refuses_until_every_page_is_scored(predictor):
    board = BoardStore().create(Predictor.moment(FIXED_T))
    board.add(_board(MAX_BATCH))
    board.add(_board(10))
    board.score_next(predictor)
    with pytest.raises(BoardError, match="call score until done"):
        board.rank(predictor, budget=3)
    board.score_next(predictor)
    assert board.done and board.rank(predictor, budget=3).budget == 3


def test_the_pages_and_the_board_have_caps(predictor):
    board = BoardStore().create(Predictor.moment(FIXED_T))
    with pytest.raises(BoardError, match="at most"):
        board.add(_board(MAX_BATCH + 1))
    with pytest.raises(BoardError, match="adds nothing"):
        board.add([])
    board.postings = _board(boards_module.MAX_POSTINGS)  # at the cap
    with pytest.raises(BoardError, match="holds at most"):
        board.add(_board(1))


def test_boards_expire_are_evicted_and_say_so():
    store = BoardStore(ttl_seconds=0.0, max_boards=2)
    gone = store.create(Predictor.moment(FIXED_T))
    with pytest.raises(BoardExpired, match="re-upload"):
        store.get(gone.board_id)

    store = BoardStore(ttl_seconds=3600.0, max_boards=2)
    first = store.create(Predictor.moment(FIXED_T))
    store.create(Predictor.moment(FIXED_T))
    store.create(Predictor.moment(FIXED_T))
    assert len(store) == 2
    with pytest.raises(BoardExpired):
        store.get(first.board_id)


def test_an_empty_board_cannot_be_scored(predictor):
    board = BoardStore().create(Predictor.moment(FIXED_T))
    with pytest.raises(BoardError, match="no postings"):
        board.score_next(predictor)


def test_scores_are_plain_floats_and_the_moment_is_utc(predictor):
    board = BoardStore().create(Predictor.moment("2026-09-14T03:45:00"))
    assert board.moment.tzinfo is not None
    board.add(_board(3))
    board.score_next(predictor)
    assert all(isinstance(s, float) and 0.0 <= s <= 1.0 for s in board.scores)
    assert isinstance(np.asarray(board.scores), np.ndarray) and pd.notna(board.scores).all()
