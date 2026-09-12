"""The board-context rule: which pipeline ships as the default public model.

Held here: the rule fires on the comparison it names (imputed against a refit
without, with the fold spread as the noise scale) and not on the one §12
already answered (supplied against without); the refit really lacks the four
columns and is still a full pipeline; the decision travels on the artifact;
and when the rule ships the refit, the transfer gate is run on that pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.inference.artifact import assert_is_full_pipeline
from src.inference.contract import BOARD_CONTEXT
from src.models import board_context
from src.models.experiments import spec_by_name
from src.models.freeze import build_metadata, freeze
from tests.conftest import FROZEN_RUN


def _builder(split):
    spec = spec_by_name(FROZEN_RUN)
    resolved = spec.resolve(split)
    return lambda: spec.build(split, resolved)


def test_the_refit_lacks_exactly_the_four_board_columns_and_is_still_a_pipeline(split):
    full = _builder(split)()
    reduced = board_context.without_board_context_pipeline(full)
    selected = set(full.named_steps["select"].kw_args["columns"])
    kept = set(reduced.named_steps["select"].kw_args["columns"])
    assert selected - kept == set(BOARD_CONTEXT)
    assert kept < selected
    assert [name for name, _ in reduced.steps] == [name for name, _ in full.steps]


def test_the_decision_reads_the_three_regimes_and_names_its_rule(split):
    decision = board_context.decide(split, _builder(split), fold_sd=0.01)
    out = decision.as_dict()
    assert out["ship"] in {board_context.FULL, board_context.WITHOUT}
    assert out["rule"] == board_context.RULE
    for key in ("val_pr_auc_supplied", "val_pr_auc_imputed", "val_pr_auc_without"):
        assert 0.0 <= out[key] <= 1.0
    assert out["board_context_features"] == list(BOARD_CONTEXT)
    assert out["fold_sd"] == pytest.approx(0.01)


def test_the_rule_compares_imputed_against_without_not_supplied_against_without(monkeypatch):
    """A candidate that is excellent with board context and useless without it
    ships the refit — the default caller has no board context."""
    calls = iter([0.9, 0.2, 0.5])  # supplied, imputed, without
    monkeypatch.setattr(board_context, "_pr_auc", lambda model, block: next(calls))
    monkeypatch.setattr(board_context, "fit_on_frame", lambda pipeline, block: pipeline)
    monkeypatch.setattr(
        board_context, "without_board_context_pipeline", lambda pipeline, leaky=False: pipeline
    )
    monkeypatch.setattr(board_context, "without_board_context", lambda block: block)

    class Split:
        train = val = None

    decision = board_context.decide(Split(), lambda: object(), fold_sd=0.05)
    assert decision.ship == board_context.WITHOUT
    assert decision.supplied == 0.9 and decision.imputed == 0.2 and decision.without == 0.5


def test_a_difference_inside_the_fold_spread_keeps_the_full_model(monkeypatch):
    calls = iter([0.6, 0.48, 0.50])
    monkeypatch.setattr(board_context, "_pr_auc", lambda model, block: next(calls))
    monkeypatch.setattr(board_context, "fit_on_frame", lambda pipeline, block: pipeline)
    monkeypatch.setattr(
        board_context, "without_board_context_pipeline", lambda pipeline, leaky=False: pipeline
    )
    monkeypatch.setattr(board_context, "without_board_context", lambda block: block)

    class Split:
        train = val = None

    assert board_context.decide(Split(), lambda: object(), fold_sd=0.05).ship == board_context.FULL
    calls2 = iter([0.6, 0.48, 0.50])
    monkeypatch.setattr(board_context, "_pr_auc", lambda model, block: next(calls2))
    assert (
        board_context.decide(Split(), lambda: object(), fold_sd=0.0).ship == board_context.WITHOUT
    )


def test_a_nan_fold_spread_is_treated_as_zero_not_as_infinite_tolerance(monkeypatch):
    calls = iter([0.6, 0.40, 0.50])
    monkeypatch.setattr(board_context, "_pr_auc", lambda model, block: next(calls))
    monkeypatch.setattr(board_context, "fit_on_frame", lambda pipeline, block: pipeline)
    monkeypatch.setattr(
        board_context, "without_board_context_pipeline", lambda pipeline, leaky=False: pipeline
    )
    monkeypatch.setattr(board_context, "without_board_context", lambda block: block)

    class Split:
        train = val = None

    decision = board_context.decide(Split(), lambda: object(), fold_sd=float("nan"))
    assert decision.ship == board_context.WITHOUT and decision.fold_sd == 0.0


def test_a_frozen_model_carries_the_decision_and_ships_what_it_says(split, panel, tmp_path):
    from src.inference import artifact as artifact_module

    forced = {
        "ship": board_context.WITHOUT,
        "rule": board_context.RULE,
        "val_pr_auc_supplied": 0.5,
        "val_pr_auc_imputed": 0.2,
        "val_pr_auc_without": 0.4,
        "fold_sd": 0.01,
        "board_context_features": list(BOARD_CONTEXT),
        "reason": "forced for the test",
    }
    frozen = freeze(split, FROZEN_RUN, board_context_decision=forced)
    assert frozen.board_context == forced
    assert not set(BOARD_CONTEXT) & set(frozen.features), "the shipped pipeline lacks them"
    assert_is_full_pipeline(frozen.pipeline)
    assert frozen.transfer["verdict"] in {"intact", "board_specific", "reversed", "unmeasured"}

    metadata = build_metadata(frozen, FROZEN_RUN, panel, tmp_path / "p.parquet", "synthetic", 20)
    path = tmp_path / "shelf_life.joblib"
    artifact_module.save(frozen.pipeline, metadata, path)
    loaded = artifact_module.load(path)
    assert loaded.metadata.board_context["ship"] == board_context.WITHOUT
    assert not set(BOARD_CONTEXT) & set(loaded.metadata.features)


def test_the_default_decision_keeps_the_full_model_on_the_fixture(frozen):
    """The session fixture is frozen with no forced decision; on the synthetic
    panel the rule reads the numbers and, whichever way it goes, records them."""
    assert frozen.board_context["rule"] == board_context.RULE
    assert np.isfinite(frozen.board_context["val_pr_auc_imputed"])
