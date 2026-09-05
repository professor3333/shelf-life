"""Fetching a released artifact, and refusing one that is not what was published.

The Docker build runs this code with no network fixture and no second chance, so
the failure paths matter more than the happy one: a truncated download, a tag
that has no release, an asset the checksums do not cover. Each of those has to
fail the *build* rather than produce an image that boots and serves a file
nobody published.

Nothing here touches the network. `fetch_artifact` takes its opener as an
argument for exactly that reason — the alternative is a test that passes when
GitHub is up.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from src.inference.fetch import (
    ASSETS,
    CHECKSUM_FILE,
    FetchError,
    asset_url,
    fetch_artifact,
    parse_checksums,
    sha256_of,
    verify,
    write_checksums,
)

TAG = "artifact-2026-09-07"
REPO = "someone/shelf-life"


def make_release(contents: dict[str, bytes], *, corrupt: str | None = None):
    """A fake release: an opener over in-memory assets, with optional rot.

    `corrupt` names an asset whose bytes are served altered *after* the checksums
    were computed — which is what a truncated download or a stale cache looks
    like from the client's side.
    """
    digests = "\n".join(f"{sha256_of(data)}  {name}" for name, data in contents.items()) + "\n"
    served = dict(contents)
    if corrupt is not None:
        served[corrupt] = served[corrupt] + b"tampered"

    def opener(url: str) -> bytes:
        name = url.rsplit("/", 1)[-1]
        if name == CHECKSUM_FILE:
            return digests.encode()
        if name not in served:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return served[name]

    return opener


@pytest.fixture
def assets() -> dict[str, bytes]:
    return {
        "shelf_life.joblib": b"\x80\x05 a pickled pipeline",
        "shelf_life.json": b'{"threshold": 0.42}',
    }


def test_the_asset_url_needs_no_api_call_and_therefore_no_token():
    url = asset_url(REPO, TAG, "shelf_life.joblib")
    assert url == f"https://github.com/{REPO}/releases/download/{TAG}/shelf_life.joblib"
    assert "api.github.com" not in url


def test_a_release_is_fetched_and_written_with_its_checksums(tmp_path: Path, assets):
    written = fetch_artifact(TAG, into=tmp_path / "models", repo=REPO, opener=make_release(assets))

    assert [p.name for p in written] == list(ASSETS)
    for name, data in assets.items():
        assert (tmp_path / "models" / name).read_bytes() == data
    # The checksums are kept beside the artifact so the image can be audited
    # after the fact without going back to the release.
    assert (tmp_path / "models" / CHECKSUM_FILE).exists()


def test_a_file_that_does_not_match_its_published_checksum_is_refused(tmp_path: Path, assets):
    opener = make_release(assets, corrupt="shelf_life.joblib")

    with pytest.raises(FetchError, match="does not match the checksum"):
        fetch_artifact(TAG, into=tmp_path / "models", repo=REPO, opener=opener)


def test_nothing_is_written_when_any_file_fails_verification(tmp_path: Path, assets):
    """A half-populated `models/` is worse than an empty one.

    The next build step loads `models/shelf_life.joblib`. If a failed fetch left
    a previous good copy of it beside a corrupt sidecar, that step would succeed
    and the image would ship a threshold from one release and a model from
    another.
    """
    into = tmp_path / "models"
    opener = make_release(assets, corrupt="shelf_life.json")

    with pytest.raises(FetchError):
        fetch_artifact(TAG, into=into, repo=REPO, opener=opener)

    assert not into.exists() or list(into.iterdir()) == []


def test_an_asset_with_no_checksum_entry_is_an_error_not_a_pass(assets):
    """Adding an asset must not silently opt it out of verification."""
    with pytest.raises(FetchError, match="no entry for"):
        verify("shelf_life.joblib", b"anything", {"shelf_life.json": "abc"})


def test_a_tag_with_no_release_says_how_to_make_one(tmp_path: Path):
    def opener(url: str) -> bytes:
        raise HTTPError(url, 404, "Not Found", {}, None)

    with pytest.raises(FetchError, match="gh release create"):
        fetch_artifact(TAG, into=tmp_path, repo=REPO, opener=opener)


def test_an_unreachable_network_fails_the_build_rather_than_the_container(tmp_path: Path):
    def opener(url: str) -> bytes:
        raise URLError("name resolution failed")

    with pytest.raises(FetchError, match="cannot reach the release"):
        fetch_artifact(TAG, into=tmp_path, repo=REPO, opener=opener)


def test_checksums_round_trip_through_the_coreutils_format(tmp_path: Path, assets):
    for name, data in assets.items():
        (tmp_path / name).write_bytes(data)

    written = write_checksums(tmp_path)
    digests = parse_checksums(written.read_text())

    assert digests == {name: sha256_of(data) for name, data in assets.items()}
    # Two spaces, so `sha256sum -c SHA256SUMS` reads it unmodified.
    assert all("  " in line for line in written.read_text().splitlines())


def test_checksumming_an_artifact_that_was_never_frozen_says_so(tmp_path: Path):
    with pytest.raises(FetchError, match="freeze"):
        write_checksums(tmp_path)


def test_the_image_cannot_bake_a_laptops_artifact():
    """`models/` must be in `.dockerignore`, or `COPY . .` reintroduces §7a's problem.

    With it ignored there is exactly one way a model enters an image — a
    verified fetch from a tag — and the question "which model is serving?" has a
    checkable answer. Without it, an image built on a developer's machine
    silently bakes whatever was in `models/` at the time.
    """
    ignored = Path(".dockerignore").read_text().splitlines()
    assert "models/" in [line.strip() for line in ignored]
