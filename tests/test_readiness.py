"""The readiness report renders the two gates from the split's own arithmetic."""

from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "tests")
from panels import make_panel  # noqa: E402

from src.data import readiness  # noqa: E402


def test_a_shallow_panel_reports_both_gates_and_no_legal_cut():
    panel = make_panel(n_waves=4)
    now = pd.Timestamp(panel["t"].max()) + pd.Timedelta(hours=1)
    text = readiness.render(panel, None, now=now)
    assert "| a legal three-way split |" in text
    assert "| 3 rolling-origin folds |" in text
    assert "Legal cuts available today: **0**" in text
    assert "## What runs on the day it clears" in text
    # Short on depth is a different finding from deep-but-label-starved, and the
    # headline must not describe the second when the first is what happened.
    assert "**No.** The panel is" in text and "short of a legal split" in text
    assert "Depth alone" not in text


def test_a_deep_panel_reports_a_legal_cut():
    panel = make_panel(n_waves=14)
    now = pd.Timestamp(panel["t"].max()) + pd.Timedelta(hours=1)
    text = readiness.render(panel, None, now=now)
    assert "Legal cuts available today: **0**" not in text
    assert "**Yes — the depth gates are open.**" in text
    assert "Model selection still requires review" in text


def test_a_legal_split_without_enough_folds_is_not_ready():
    panel = make_panel(n_waves=10)
    now = pd.Timestamp(panel["t"].max()) + pd.Timedelta(hours=1)
    text = readiness.render(panel, None, now=now)
    assert "Legal cuts available today: **0**" not in text
    assert "**Not yet.** A legal split exists" in text


def test_depth_without_usable_labels_is_not_ready():
    panel = make_panel(n_waves=14, positives_per_wave=0, random_labels=True)
    now = pd.Timestamp(panel["t"].max()) + pd.Timedelta(hours=1)
    text = readiness.render(panel, None, now=now)
    assert "Legal cuts available today: **0**" in text
    assert "**No.** The panel is deep enough, but no usable three-way split exists" in text
    assert "short of a legal split" not in text


def test_a_stalled_panel_is_named_as_stalled_not_projected():
    panel = make_panel(n_waves=6)
    long_after = pd.Timestamp(panel["t"].max()) + pd.Timedelta(days=30)
    text = readiness.render(panel, None, now=long_after)
    assert "NOT ACCRUING" in text or "not accruing" in text.lower()


def test_a_deep_but_stalled_panel_does_not_announce_readiness():
    panel = make_panel(n_waves=14)
    now = pd.Timestamp(panel["t"].max()) + pd.Timedelta(days=30)
    text = readiness.render(panel, None, now=now)
    assert "**No — the panel is not accruing.**" in text
    assert "**Yes" not in text
