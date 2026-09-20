"""The evidence bundle: every report a freeze is chosen on names one snapshot and one commit."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models import evidence, provenance
from src.models.evidence import EVIDENCE_REPORTS, Header, parse_header, problems, read_headers

HEAD = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _bundle(sha: str = HEAD[:9], snapshot: str = "2026-10-11", **overrides: Header) -> list[Header]:
    return [
        overrides.get(name.removesuffix(".md"), Header(name, provenance.REAL, snapshot, sha))
        for name in EVIDENCE_REPORTS
    ]


def _check(headers, *, panel="2026-10-11", head=HEAD, changed=lambda sha: []):
    return problems(headers, panel_snapshot=panel, head_sha=head, changed_since=changed)


# --- reading what a report says about itself


def test_the_header_is_read_from_the_rows_provenance_writes():
    prov = provenance.Provenance(
        HEAD, "main", False, provenance.REAL, "p.parquet", "ab" * 32, 10, "2026-10-11"
    )
    text = "\n".join(["# Title", "", *provenance.header(prov, "python -m x"), "body"])
    got = parse_header("readiness.md", text)
    assert got == Header("readiness.md", provenance.REAL, "2026-10-11", HEAD[:9])


def test_a_report_with_no_header_says_nothing(tmp_path: Path):
    (tmp_path / "readiness.md").write_text("# no provenance block here\n")
    got = {h.report: h for h in read_headers(tmp_path)}
    assert got["readiness.md"] == Header("readiness.md", None, None, None)
    assert got["model_comparison.md"] == Header("model_comparison.md", None, None, None), (
        "a missing file is reported, not skipped"
    )


# --- one bundle, or not


def test_a_bundle_from_one_snapshot_at_head_is_one_bundle():
    assert _check(_bundle()) == []


def test_reports_committed_after_they_were_written_are_still_one_bundle():
    """The runbook commits the regenerated reports before freezing, so HEAD is
    one commit past the evidence. That commit touches reports/ and the README's
    generated block, and nothing else — which is exactly what is allowed."""

    def changed(sha):
        return ["reports/readiness.md", "reports/model_comparison.md", "README.md"]

    assert _check(_bundle(sha=OTHER[:9]), changed=changed) == []


def test_source_edited_since_the_evidence_is_not_one_bundle():
    def changed(sha):
        return ["reports/readiness.md", "src/features/preprocessing.py"]

    (only,) = _check(_bundle(sha=OTHER[:9]), changed=changed)
    assert "source changed" in only and "preprocessing.py" in only


def test_a_commit_git_does_not_know_is_not_one_bundle():
    (only,) = _check(_bundle(sha="deadbeef1"), changed=lambda sha: None)
    assert "not in this checkout's history" in only


def test_two_commits_are_two_bundles():
    headers = _bundle(
        model_comparison=Header("model_comparison.md", provenance.REAL, "2026-10-11", OTHER[:9])
    )
    found = _check(headers)
    assert len(found) == 1 and found[0].startswith("commits disagree")
    assert "model_comparison.md=" + OTHER[:9] in found[0]


def test_two_snapshots_are_two_bundles():
    headers = _bundle(readiness=Header("readiness.md", provenance.REAL, "2026-10-10", HEAD[:9]))
    found = _check(headers)
    assert any(f.startswith("snapshots disagree") for f in found)


def test_evidence_about_an_older_snapshot_than_the_panel_is_refused():
    """The panel on disk is what freeze opens; the evidence must be about it,
    not about the snapshot before the collector's most recent wave."""
    (only,) = _check(_bundle(snapshot="2026-10-10"), panel="2026-10-11")
    assert "about snapshot 2026-10-10" in only and "2026-10-11" in only


def test_a_synthetic_report_in_the_bundle_is_refused():
    headers = _bundle(
        readiness=Header("readiness.md", provenance.SYNTHETIC, "2026-10-11", HEAD[:9])
    )
    found = _check(headers)
    assert any("not about the real panel" in f for f in found)


def test_a_missing_report_is_refused_by_name():
    headers = _bundle(board_fingerprint=Header("board_fingerprint.md", None, None, None))
    found = _check(headers)
    assert any("no provenance header" in f and "board_fingerprint.md" in f for f in found)


def test_outside_a_checkout_the_bundle_cannot_be_trusted():
    (only,) = _check(_bundle(), head=None)
    assert "not a git checkout" in only


# --- the set itself


def test_the_set_is_the_h7_evidence_and_not_what_freeze_writes():
    """`test_results.md` is what freeze writes; the dated profiles and the H=1
    rehearsal reports are about other things. The set is the H=7 evidence."""
    assert "test_results.md" not in EVIDENCE_REPORTS
    assert not any("h1" in name for name in EVIDENCE_REPORTS)
    assert {"readiness.md", "model_comparison.md", "board_fingerprint.md"} <= set(EVIDENCE_REPORTS)


def test_the_cli_exit_code_is_its_own(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["evidence", "--reports", str(tmp_path), "--panel", "x.parquet"]
    )
    monkeypatch.setattr(evidence, "_head_sha", lambda: HEAD)
    with pytest.raises(SystemExit) as exit_info:
        evidence.main()
    assert exit_info.value.code == 5
    assert "no provenance header" in capsys.readouterr().out
