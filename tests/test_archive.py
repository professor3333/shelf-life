"""Tests for reading archived Greenhouse payloads.

These build their own gzipped payloads in a temp directory: no scraper archive,
no network, no fixtures to keep in sync.
"""

from __future__ import annotations

import gzip
import json

import pandas as pd
import pytest

from src.data.archive import _joined, discover, load_archive, parse_payload, parse_stamp


def _job(job_id, **overrides):
    job = {
        "id": job_id,
        "internal_job_id": 99,
        "absolute_url": f"https://job-boards.greenhouse.io/acme/jobs/{job_id}",
        "company_name": "Acme",
        "requisition_id": "R-1",
        "first_published": "2026-08-06T20:13:29-04:00",
        "updated_at": "2026-09-02T12:36:22-04:00",
        "application_deadline": None,
        "location": {"name": "Berlin"},
        "departments": [{"id": 2, "name": "Engineering"}, {"id": 1, "name": "Applied AI"}],
        "offices": [{"id": 3, "name": "Berlin, DE", "location": "Berlin, Germany"}],
        "metadata": [{"id": 4, "name": "Featured", "value": None, "value_type": "single_select"}],
        "content": "<p>hello</p>",
    }
    job.update(overrides)
    return job


def _write(root, board, stamp, jobs):
    directory = root / board
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}.html.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"jobs": jobs, "meta": {}}, handle)
    return path


@pytest.fixture
def archive(tmp_path):
    _write(tmp_path, "boardA", "2026-09-03T03-45-01Z", [_job(1), _job(2)])
    _write(tmp_path, "boardA", "2026-09-04T03-46-01Z", [_job(1, content="<p>edited longer</p>")])
    _write(tmp_path, "boardB", "2026-09-04T03-46-05Z", [_job(3)])
    (tmp_path / "boardA" / "notes.txt").write_text("not a payload")
    return tmp_path


def test_parse_stamp_reads_the_fetch_instant():
    assert parse_stamp("2026-09-04T03-46-01Z.html.gz") == pd.Timestamp("2026-09-04T03:46:01Z")


@pytest.mark.parametrize("name", ["notes.txt", "index.html.gz", "2026-09-04.html.gz", ""])
def test_parse_stamp_rejects_non_payload_names(name):
    assert parse_stamp(name) is None


def test_discover_finds_payloads_in_deterministic_order(archive):
    files = discover(archive)
    assert [f.path.name for f in files] == [
        "2026-09-03T03-45-01Z.html.gz",
        "2026-09-04T03-46-01Z.html.gz",
        "2026-09-04T03-46-05Z.html.gz",
    ]
    assert all(f.path.suffixes[-2:] == [".html", ".gz"] for f in files)


def test_discover_raises_on_a_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "nope")


def test_parse_payload_extracts_the_dropped_fields(archive):
    record = parse_payload(discover(archive)[0])[0]
    assert record["source_id"] == "1"
    assert record["requisition_id"] == "R-1"
    assert record["departments"] == "Applied AI|Engineering"
    assert record["n_departments"] == 2
    assert record["offices"] == "Berlin, DE"
    assert record["board_token"] == "acme"
    assert record["content_chars"] == len("<p>hello</p>")


def test_sub_object_names_are_sorted_not_api_ordered():
    """The API's ordering is not stable. Sorting keeps an unchanged posting from
    looking edited between two fetches."""
    forward = [{"name": "Zeta"}, {"name": "Alpha"}]
    reverse = [{"name": "Alpha"}, {"name": "Zeta"}]
    assert _joined(forward, "name") == _joined(reverse, "name") == "Alpha|Zeta"
    assert _joined([], "name") is None
    assert _joined(None, "name") is None


def test_timestamps_are_converted_to_utc(archive):
    record = parse_payload(discover(archive)[0])[0]
    assert record["first_published"] == pd.Timestamp("2026-08-07T00:13:29Z")
    assert record["updated_at"] == pd.Timestamp("2026-09-02T16:36:22Z")
    assert record["application_deadline"] is None


def test_load_archive_is_a_posting_fetch_panel(archive):
    frame = load_archive(archive)
    assert len(frame) == 4, "2 postings on day one, 1 on day two, 1 on board B"
    assert frame["source_id"].nunique() == 3
    assert "content" not in frame.columns
    assert frame["fetched_at"].is_monotonic_increasing


def test_content_is_opt_in_and_captures_edits(archive):
    frame = load_archive(archive, include_content=True)
    posting_one = frame[frame["source_id"] == "1"].sort_values("fetched_at")
    assert posting_one["content"].tolist() == ["<p>hello</p>", "<p>edited longer</p>"]
    assert posting_one["content_chars"].is_monotonic_increasing


def test_empty_archive_returns_an_empty_frame(tmp_path):
    (tmp_path / "boardA").mkdir()
    assert load_archive(tmp_path).empty


def test_missing_optional_fields_do_not_crash(tmp_path):
    _write(
        tmp_path,
        "b",
        "2026-09-04T03-46-01Z",
        [{"id": 7, "absolute_url": "https://careers.duolingo.com/jobs/7?gh_jid=7"}],
    )
    record = load_archive(tmp_path).iloc[0]
    assert record["source_id"] == "7"
    assert record["board_token"] is None, "custom domains carry no board token"
    assert record["n_departments"] == 0
    assert pd.isna(record["requisition_id"])
