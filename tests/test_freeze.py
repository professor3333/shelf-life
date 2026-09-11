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

import pandas as pd
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


def _run_freeze(
    tmp_path: Path, monkeypatch, n_waves: int, *extra: str, clean: bool = True
) -> dict[str, object]:
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
    # Capture the FrozenModel `main` builds, so a test can check the report
    # against the object it was rendered from rather than re-deriving it — a
    # re-derivation fits a second model and proves nothing about the first.
    captured = {}
    real_freeze = freeze_module.freeze

    def recording(*args, **kwargs):
        captured["frozen"] = real_freeze(*args, **kwargs)
        return captured["frozen"]

    monkeypatch.setattr(freeze_module, "freeze", recording)
    # The harness runs on whatever tree the developer has; the clean-tree
    # refusal is tested on its own below, not by every other test's luck.
    monkeypatch.setattr(freeze_module.provenance, "worktree_is_clean", lambda: clean)

    with pytest.raises(SystemExit) as exit_info:
        freeze_module.main()
        raise SystemExit(0)  # main returns rather than exiting when it succeeds
    paths["exit_code"] = exit_info.value.code
    paths["frozen"] = captured.get("frozen")
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


# --- the report the one-time run has to produce ----------------------------

#: Every section the held-out report must carry, because the block opens once
#: and a section discovered missing afterwards cannot be added — regenerating it
#: means reading the block a second time, and a test set read twice is a
#: validation set. This list is the acceptance criterion for the final run,
#: pinned while it is still cheap to fix.
REQUIRED_SECTIONS = (
    "## Validation and test, side by side",  # pr_auc, brier, roc_auc, ece, both blocks
    "## How much of this is signal",  # bootstrap intervals on every headline
    "## The operating point, applied as frozen",  # precision/recall at the shipped threshold
    "## Per source",  # is it one board's model
    "## Carried over from training, or not",  # memorisation check
    "## Calibration on test",  # brier, ece, and the binned curve
    "## What this means for the person using it",  # the decision the numbers inform
)


def test_the_held_out_report_carries_every_section_the_run_gets_one_chance_at(
    tmp_path, monkeypatch
):
    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE)
    report = paths["report"].read_text()

    missing = [section for section in REQUIRED_SECTIONS if section not in report]
    assert not missing, "the one-time report would have been written without:\n  " + "\n  ".join(
        missing
    )

    # The calibration section must carry the curve, not only its two scalars.
    calibration = report.split("## Calibration on test")[1]
    assert "bin_low" in calibration and "observed_rate" in calibration, (
        "Brier and ECE say how far the probabilities are from the truth, never which way"
    )


def test_the_reading_of_the_numbers_is_arithmetic_not_assertion(tmp_path, monkeypatch):
    """The interpretation section has to be derived from the confusion matrix
    that is actually shipped, or it is a claim sitting next to the evidence
    rather than a reading of it."""
    from src.models.freeze import what_the_user_gets

    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE)
    frozen = paths["frozen"]
    reading = what_the_user_gets(frozen, budget_per_day=20)
    confusion = frozen.test_at_frozen_threshold
    days = max(1, frozen.test_days)

    assert reading["alerts_per_day"] == pytest.approx(confusion["flagged"] / days)
    assert reading["real_closures_caught_per_day"] == pytest.approx(confusion["tp"] / days)
    assert reading["closures_missed_per_day"] == pytest.approx(confusion["fn"] / days)
    assert reading["share_of_closures_caught"] == pytest.approx(confusion["recall"])

    # The counterfactual is the same reading effort with no model: the budget
    # drawn from a board closing at the block's own rate.
    base_rate = float(frozen.test["base_rate"])
    assert reading["unaided_closures_caught_per_day"] == pytest.approx(
        confusion["flagged"] * base_rate / days
    )
    assert reading["lift_over_unaided"] == pytest.approx(confusion["precision"] / base_rate)


def test_a_model_no_better_than_the_board_reports_a_lift_of_one():
    """The case the section exists to be able to state. A model whose precision
    equals the base rate has sorted nothing, whatever its PR-AUC looks like, and
    the report has to be able to say so in the same words it uses for a success."""
    from src.models.freeze import FrozenModel, what_the_user_gets

    useless = FrozenModel(
        pipeline=None,
        threshold=0.5,
        validation={},
        test={"base_rate": 0.10},
        test_at_frozen_threshold={
            "flagged": 200.0,
            "tp": 20.0,
            "fp": 180.0,
            "fn": 80.0,
            "tn": 720.0,
            "precision": 0.10,
            "recall": 0.20,
        },
        test_intervals={},
        test_fragility=None,
        by_source=pd.DataFrame(),
        by_seen_in_train=pd.DataFrame(),
        calibration={},
        reliability=pd.DataFrame(),
        test_days=10,
        params={},
        features=(),
        fitted_on="train",
    )
    reading = what_the_user_gets(useless, budget_per_day=20)
    assert reading["lift_over_unaided"] == pytest.approx(1.0)
    assert reading["real_closures_caught_per_day"] == pytest.approx(
        reading["unaided_closures_caught_per_day"]
    )


# --- release discipline: a clean tree, and everything the artifact is traceable to


def test_a_real_freeze_refuses_a_dirty_worktree(tmp_path, monkeypatch, capsys):
    """A SHA cannot reproduce a result if uncommitted changes affected the run.
    Reports may say "dirty tree" and be read as provisional; the artifact ships,
    so it may not (`design.md` §16). Its own exit code, and no report written —
    this is not a finding about the data."""
    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE, clean=False)

    assert paths["exit_code"] == 4
    assert not paths["artifact"].exists(), "the test block was spent from a dirty tree"
    assert not paths["report"].exists(), "a refusal about the code is not a report about the data"
    assert "uncommitted changes" in capsys.readouterr().out


def test_the_clean_tree_check_comes_after_the_depth_checks(tmp_path, monkeypatch, capsys):
    """Depth refusals are recorded in the report and are about the data; they
    must not be masked by the code check, which is only asked at the moment the
    block would actually be opened."""
    paths = _run_freeze(tmp_path, monkeypatch, LEGAL_BUT_FOLDLESS, clean=False)
    assert paths["exit_code"] == 3
    assert paths["report"].exists()
    assert "uncommitted" not in capsys.readouterr().out


def test_a_synthetic_freeze_is_a_rehearsal_and_may_run_dirty(tmp_path, monkeypatch):
    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE, "--synthetic", clean=False)
    assert paths["exit_code"] == 0
    assert paths["artifact"].exists()


def test_the_artifact_names_everything_it_is_traceable_to(tmp_path, monkeypatch):
    """Git SHA, panel hash, rules version, horizon, dependency lock, training
    config, seed, and the artifact's own checksum — the list in `design.md` §16."""
    import json

    from src.inference import artifact as artifact_module

    paths = _run_freeze(tmp_path, monkeypatch, DEEP_ENOUGH_TO_CHOOSE)
    assert paths["exit_code"] == 0
    loaded = artifact_module.load(paths["artifact"])
    meta = loaded.metadata

    assert meta.rules_version == 2
    assert meta.seed is not None
    # The lock is committed, so an artifact frozen in this checkout always
    # names it — a None here would mean the hash points at a file nobody has.
    import hashlib

    from src.models import provenance

    assert provenance.LOCK_FILE.exists()
    assert meta.lock_sha256 == hashlib.sha256(provenance.LOCK_FILE.read_bytes()).hexdigest()
    assert "git_sha" in meta.provenance and "panel_sha256" in meta.provenance
    assert meta.horizon_days >= 1 and meta.params and meta.features

    sidecar = json.loads(paths["artifact"].with_suffix(".json").read_text())
    assert sidecar["artifact_sha256"] == hashlib.sha256(paths["artifact"].read_bytes()).hexdigest()
