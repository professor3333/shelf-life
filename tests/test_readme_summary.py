"""The README's generated block equals what the committed reports say."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import readme_summary

ROOT = Path(__file__).resolve().parent.parent


def test_the_readme_block_matches_the_committed_reports():
    """Run in CI without the data: the claim is that the README agrees with
    the reports in the repository, and a stale block fails here rather than
    being read by a stranger."""
    readme = (ROOT / "README.md").read_text()
    block = readme_summary.render(readme_summary.read(ROOT / "reports"))
    assert readme_summary.splice(readme, block) == readme, (
        "README's generated block is behind reports/: run `python -m src.data.readme_summary`"
    )


def test_the_block_reads_the_canonical_reports_and_names_them():
    state = readme_summary.read(ROOT / "reports")
    assert state["snapshot"] and state["sha"]
    assert set(state["gates"]) >= {"a legal three-way split", "3 rolling-origin folds"}
    block = readme_summary.render(state)
    for report in (
        "readiness.md",
        "test_results.md",
        "model_comparison_h1_calendar.md",
        "depth_ledger.md",
    ):
        assert f"reports/{report}" in block
    assert block.startswith(readme_summary.BEGIN) and block.endswith(readme_summary.END)


def test_a_large_single_draw_is_reported_as_a_draw_not_a_selection(tmp_path):
    """The H=1 table shows a large random-forest number and selects nothing;
    the block must say both, in that order, so the number cannot be read as
    a result."""
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "readiness.md").write_text(
        "| Data | **real** · snapshot `2026-09-12` · x |\n"
        "| Code | `abc1234` on `main` (clean) |\n\n"
        "| gate | labelled waves needed | have | short by | projected |\n|---|---|---|---|---|\n"
        "| a legal three-way split | 22 | 6 | 16 | 2026-09-21 |\n"
        "| 3 rolling-origin folds | 34 | 6 | 28 | 2026-10-03 |\n\n"
        "Legal cuts available today: **0**.\n"
    )
    (reports / "model_comparison_h1_calendar.md").write_text(
        "| Horizon | H=1 (calendar basis) |\n\n"
        "| model | folds_scored | cv_pr_auc_mean | cv_pr_auc_sd | val_pr_auc |\n"
        "|---|---|---|---|---|\n"
        "| prior | 0 | — | — | 0.0052 |\n| random_forest | 0 | — | — | 0.1804 |\n\n"
        "## The verdict\n\n**Chosen: None** — no model was scored on any fold\n"
    )
    (reports / "test_results.md").write_text("## Nothing below has run\n")
    block = readme_summary.render(readme_summary.read(reports))
    assert "**not opened**" in block
    assert "Verdict: **None**" in block
    assert "`random_forest` at PR-AUC 0.1804 against a prior of 0.0052" in block
    assert block.index("Verdict") < block.index("Best single validation draw")
    assert "which is why it selected nothing" in block


def test_a_readme_without_markers_is_refused():
    with pytest.raises(ValueError, match="markers"):
        readme_summary.splice("no markers here", "block")


def test_a_selected_rehearsal_model_is_not_described_as_no_selection():
    state = {
        "snapshot": "2026-09-17",
        "sha": "abc1234",
        "gates": {},
        "legal_cuts": "0",
        "held_out_opened": False,
        "verdict": ("random_forest", "selected by the fold comparison"),
        "best": {"model": "random_forest", "val_pr_auc": "0.0838"},
        "prior": {"val_pr_auc": "0.0203"},
        "h1_horizon": "H=1 (calendar basis)",
        "candidates": 10,
        "folds_scored": 3,
        "ledger_real": 1,
        "ledger_synthetic": 1,
        "latest_profile": [],
    }
    block = readme_summary.render(state)

    assert "Verdict: **random_forest**" in block
    assert "3 rolling-origin folds scored" in block
    assert "Best single validation draw: `random_forest` at PR-AUC 0.0838" in block
    assert "selected nothing" not in block
    assert "**not opened**" in block
