"""Tests for the depth ledger.

The ledger's whole value is that it *accumulates*, so the properties worth
testing are about what happens on the second run and the tenth, not the first.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.models import ledger


@dataclass(frozen=True)
class _Provenance:
    dataset: str = "real"
    snapshot_date: str = "2026-09-13"
    panel_sha256: str = "abc123"
    git_sha: str = "deadbee"
    git_dirty: bool = False


def _entry(**overrides):
    fields = {
        "stage": ledger.VALIDATION,
        "provenance": _Provenance(**overrides.pop("provenance", {})),
        "labelled_waves": 13,
        "labelled_rows": 6874,
        "positives": 96,
        "folds": 3,
        "chosen": "xgboost",
        "pr_auc": 0.31,
        "cv_pr_auc_mean": 0.29,
        "cv_pr_auc_sd": 0.04,
    }
    fields.update(overrides)
    return ledger.record(**fields)


def test_a_rerun_of_the_same_code_on_the_same_data_replaces_its_row(tmp_path):
    """`scripts/rehearse.sh` is built to be run repeatedly, so a ledger that
    appended per invocation would measure how often it was run rather than what
    the pipeline scored."""
    path = tmp_path / "ledger.jsonl"
    ledger.append(_entry(), path)
    entries = ledger.append(_entry(), path)
    assert len(entries) == 1


def test_a_rerun_after_the_scraper_adds_a_wave_is_a_new_row(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append(_entry(), path)
    entries = ledger.append(
        _entry(provenance={"panel_sha256": "def456"}, labelled_waves=14, positives=117), path
    )
    assert len(entries) == 2
    assert [e["positives"] for e in entries] == [96, 117]


def test_a_rerun_after_a_code_change_is_a_new_row(tmp_path):
    """Same data, different pipeline, genuinely different result."""
    path = tmp_path / "ledger.jsonl"
    ledger.append(_entry(), path)
    entries = ledger.append(_entry(provenance={"git_sha": "cafef00"}), path)
    assert len(entries) == 2


def test_validation_and_held_out_rows_coexist(tmp_path):
    path = tmp_path / "ledger.jsonl"
    ledger.append(_entry(), path)
    entries = ledger.append(_entry(stage=ledger.HELD_OUT), path)
    assert {e["stage"] for e in entries} == {ledger.VALIDATION, ledger.HELD_OUT}


def test_rows_are_ordered_by_depth_so_the_curve_reads_downward(tmp_path):
    path = tmp_path / "ledger.jsonl"
    for waves, sha in ((17, "c"), (13, "a"), (15, "b")):
        ledger.append(_entry(provenance={"panel_sha256": sha}, labelled_waves=waves), path)
    assert [e["waves"] for e in ledger.load(path)] == [13, 15, 17]


def test_a_missing_standard_deviation_renders_as_a_dash_not_nan():
    """`summarise_folds` returns NaN when fewer than two folds scored. Printing
    "nan" in a table read as a result makes a fact about the run look like a
    bug."""
    text = ledger.render([_entry(cv_pr_auc_sd=float("nan"))])
    assert "nan" not in text
    assert "fewer than two folds scored" in text


def test_synthetic_rows_are_never_mixed_with_real_ones():
    """The one way this history could become dishonest."""
    text = ledger.render(
        [_entry(), _entry(provenance={"dataset": "synthetic", "panel_sha256": "syn"})]
    )
    assert "## Real panel" in text
    assert "## Synthetic panel" in text
    assert "No number here is a finding about job postings." in text


def test_a_dirty_tree_is_flagged_as_provisional():
    assert "dirty tree" in ledger.render([_entry(provenance={"git_dirty": True})])
    assert "dirty tree" not in ledger.render([_entry()])


def test_an_empty_ledger_renders_without_pretending_to_have_results():
    text = ledger.render([])
    assert "Nothing recorded yet" in text
    assert "No run has produced a metric" not in text
    assert "This does not mean no experiment has produced metrics" in text
    assert "model_comparison.md" in text


def test_synthetic_only_history_does_not_imply_no_real_candidate_metrics():
    text = ledger.render([_entry(provenance={"dataset": "synthetic"})])
    real_section = text.split("## Real panel")[1].split("## Synthetic panel")[0]
    assert "No selected real-data validation result" in real_section
    assert "H=1 candidate metrics" in real_section
    assert "| snapshot_date |" not in real_section


def test_held_out_interval_survives_missing_cv_summary_with_five_folds():
    text = ledger.render(
        [
            _entry(
                stage=ledger.HELD_OUT,
                folds=5,
                cv_pr_auc_mean=None,
                cv_pr_auc_sd=None,
                pr_auc_low=0.10,
                pr_auc_high=0.31,
            )
        ]
    )
    assert "[0.1000, 0.3100]" in text
    assert "do not store CV summaries even when folds exist" in text
    assert "means fewer than two folds scored" not in text
    assert "the run has a metric and no error bar" not in text


def test_the_stage_must_be_one_of_the_two():
    with pytest.raises(ValueError, match="stage must be"):
        _entry(stage="whenever")


def test_every_row_is_one_json_object_per_line(tmp_path):
    """Line-delimited so an append is an append, and a corrupt run costs one
    row rather than the file."""
    path = tmp_path / "ledger.jsonl"
    ledger.append(_entry(), path)
    ledger.append(_entry(provenance={"panel_sha256": "z"}), path)
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["chosen"] == "xgboost" for line in lines)
