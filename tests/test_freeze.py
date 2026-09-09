"""The gate in front of the one irreversible step.

`freeze` opens the held-out block, once, and spends it. Everything upstream is
repeatable: a bad split can be recut, a bad comparison rerun, a bad threshold
rechosen. This cannot, which is why the checks in front of it matter more than
the checks anywhere else in `src/`.

**The gap these tests close.** Until 2026-09-09 the only refusal here was
`SplitTooShallow` — *is a three-way cut legal?* — and legality arrives days
before evaluability. On the real panel it arrived five days early: a legal split
existed with a one-wave training window, no rolling-origin fold, and therefore
no comparison carrying an error bar. Every gate upstream declined in that
window; `scripts/rehearse.sh` refused to go past validation and
`scripts/watch_depth.sh` reported the shortfall. This step waved it through, so
the single unrepeatable action sat behind the weakest check in the pipeline, and
the README told the reader to run it.

Nothing here reads the held-out block. The tests assert on whether `main`
refused and on what it wrote, never on a score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from panels import make_closing_panel

from src.inference import artifact as artifact_module
from src.models import freeze as freeze_module

#: Eleven waves: a legal three-way cut whose training window yields no fold.
#: The real panel's state on 2026-09-09, and the state this gate exists for.
#: Twelve is the first depth this builder cuts a fold at, and fourteen the first
#: with three — see `DEFAULT_TARGET_FOLDS`.
#:
#: Eleven rather than nine, which is also foldless: at nine the training block
#: has no positives at all, so the override test would fail inside the estimator
#: instead of proving the door opens. The gate under test is about fold
#: evidence, and the fixture must not fail for a different reason first.
LEGAL_BUT_FOLDLESS = 11
DEEP_ENOUGH_TO_CHOOSE = 14

RUN = "02-logistic"


def _panel_file(tmp_path: Path, n_waves: int) -> Path:
    path = tmp_path / f"panel_{n_waves}.parquet"
    make_closing_panel(n_waves=n_waves).to_parquet(path)
    return path


def _run_freeze(tmp_path: Path, monkeypatch, n_waves: int, *extra: str) -> dict[str, object]:
    paths = {
        "panel": _panel_file(tmp_path, n_waves),
        "report": tmp_path / "test_results.md",
        "artifact": tmp_path / "shelf_life.joblib",
    }
    # The ledger is a committed record and these are fixture runs. Patching the
    # module constants does NOT redirect it: `append(entry, path=DEFAULT_LEDGER)`
    # binds its default at import. The first draft did exactly that, passed, and
    # wrote two synthetic rows into `reports/depth_ledger.md` under the heading
    # "Real panel". So the functions are patched, and `conftest`'s autouse guard
    # is what catches it if this is ever got wrong again.
    ledger_path = tmp_path / "ledger.jsonl"
    report_path = tmp_path / "depth_ledger.md"
    real_append, real_write = freeze_module.ledger.append, freeze_module.ledger.write_report
    monkeypatch.setattr(
        freeze_module.ledger, "append", lambda entry: real_append(entry, ledger_path)
    )
    monkeypatch.setattr(
        freeze_module.ledger, "write_report", lambda entries: real_write(entries, report_path)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze",
            "--run",
            RUN,
            "--panel",
            str(paths["panel"]),
            "--out",
            str(paths["report"]),
            "--artifact",
            str(paths["artifact"]),
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as exit_info:
        freeze_module.main()
        raise SystemExit(0)  # main returns rather than exiting when it succeeds
    paths["exit_code"] = exit_info.value.code
    return paths


def test_a_legal_split_with_no_folds_is_refused(tmp_path, monkeypatch, capsys):
    """Legality is not evaluability, and freeze declines on the difference.

    What the held-out block buys is a check on a model validation already chose.
    With no fold, no comparison has a spread, so nothing chose one — and there is
    nothing for the check to check.
    """
    paths = _run_freeze(tmp_path, monkeypatch, LEGAL_BUT_FOLDLESS)

    assert paths["exit_code"] == 3, "a refusal that exits 0 reads as success"
    assert not paths["artifact"].exists(), "the test block was spent on an unselected model"
    report = paths["report"].read_text()
    assert "Nothing below has run" in report
    assert "rolling-origin fold" in report
    assert "--accept-no-folds" in report, "a refusal must say how to override it"
    assert "not run:" in capsys.readouterr().out


def test_the_refusal_names_the_other_wait_it_is_not(tmp_path, monkeypatch):
    """Two different waits, two different remedies. Collapsing them into one
    message is how "the panel is too shallow" stops being read once it is half
    true — the split *is* legal here, and a report saying otherwise would send
    the reader to look for a bug in the splitter."""
    paths = _run_freeze(tmp_path, monkeypatch, LEGAL_BUT_FOLDLESS)
    report = paths["report"].read_text()
    assert "A legal three-way split exists" in report
    assert "no honest three-way split exists" not in report.lower()


def test_the_override_exists_works_and_is_recorded(tmp_path, monkeypatch):
    """A gate with no door is a gate someone edits out under pressure, and an
    edit leaves no trace on the artifact. The door is deliberate and loud: the
    fold count travels with the model, so a served probability whose model was
    chosen by nothing can say so."""
    paths = _run_freeze(tmp_path, monkeypatch, LEGAL_BUT_FOLDLESS, "--accept-no-folds")

    assert paths["exit_code"] == 0
    assert paths["artifact"].exists()
    loaded = artifact_module.load(paths["artifact"])
    assert loaded.metadata.selection_folds == 0


def test_a_panel_deep_enough_to_choose_on_is_not_blocked(tmp_path, monkeypatch):
    """The gate has to open on its own, or the fix is a wall.

    This is the day the wait ends, rehearsed on a fixture: enough folds to have
    compared models with a spread, so freeze runs without an override and the
    artifact records how many folds stood behind the choice.
    """
    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE)

    assert paths["exit_code"] == 0
    assert paths["artifact"].exists()
    loaded = artifact_module.load(paths["artifact"])
    assert loaded.metadata.selection_folds >= 3
    assert "Nothing below has run" not in paths["report"].read_text()


def test_an_artifact_frozen_before_the_count_existed_still_loads(tmp_path):
    """`selection_folds` defaults to -1 rather than 0, because "not recorded" and
    "recorded as none" are different claims and only one of them is a warning."""
    metadata = artifact_module.Metadata(
        run_name=RUN,
        question="q",
        params={},
        features=("a",),
        fitted_on="train",
        threshold=0.5,
        budget_per_day=20,
        horizon_days=1,
        horizon_basis="calendar",
        metrics={},
        provenance={},
        dataset="synthetic",
    )
    assert metadata.selection_folds == -1


@pytest.mark.parametrize("n_waves", [LEGAL_BUT_FOLDLESS, DEEP_ENOUGH_TO_CHOOSE])
def test_the_fixture_depths_are_what_this_file_claims(n_waves):
    """These two constants carry the whole file: if `make_closing_panel` or the
    fold rule changes and nine waves starts yielding a fold, every refusal test
    above passes for the wrong reason."""
    from src.data.split import best_cuts, temporal_split
    from src.models.evaluate import wave_forward_folds

    panel = make_closing_panel(n_waves=n_waves)
    split = temporal_split(panel, best_cuts(panel))  # legal at both depths, or this raises
    folds = len(wave_forward_folds(split.train, split.embargo))
    assert folds == 0 if n_waves == LEGAL_BUT_FOLDLESS else folds >= 3
