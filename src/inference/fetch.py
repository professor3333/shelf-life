"""Fetch a frozen artifact from a tagged release, and prove it is that one.

    release tag  ──>  shelf_life.joblib + shelf_life.json  ──>  a verified image

**Why the image cannot simply copy `models/`.** That directory is derived
output, like `data/processed/`, so it is not committed — which means an image
built from a clean clone has no model in it, and an image built on my laptop has
whatever happened to be sitting there that afternoon. The first is useless and
the second is worse, because it answers "which model is serving?" with "trust
me". `docs/design.md` §7a settles it: `freeze` writes the artifact, the artifact
is attached to a git tag, and the build fetches it by tag.

**What is verified, and what that is worth.** The release carries a
`SHA256SUMS` file written at the same moment as the artifact, and every download
is checked against it. That proves the bytes arrived intact and are the bytes
that were published under that tag. It is *not* a signature: anyone who can
rewrite the release can rewrite the checksums with it. The threat being handled
is a truncated download, a cached proxy and the wrong tag — not an adversary
with a token.

The stronger check is the one the Dockerfile makes immediately afterwards, by
loading the artifact: `assert_is_full_pipeline` proves the file is a fitted
end-to-end pipeline in *the environment that will serve it*, which no checksum
can do.

Stdlib only, on purpose — this runs during a Docker build, before there is any
reason to trust that the project's dependencies installed correctly.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

#: The files that make up a release. The sidecar is not optional: a probability
#: whose threshold and provenance live somewhere else is a number, not a
#: decision.
ASSETS: tuple[str, ...] = ("shelf_life.joblib", "shelf_life.json")

CHECKSUM_FILE = "SHA256SUMS"

DEFAULT_REPO = "professor3333/shelf-life"

#: Injected in tests. A callable taking a URL and returning bytes.
Opener = Callable[[str], bytes]


class FetchError(RuntimeError):
    """The artifact could not be fetched, or is not the one that was published."""


def asset_url(repo: str, tag: str, name: str) -> str:
    """Where a release asset lives. No API call, and therefore no token."""
    return f"https://github.com/{repo}/releases/download/{tag}/{name}"


def _read(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:  # noqa: S310 - https URL built above
        return response.read()


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_checksums(directory: Path, names: Iterable[str] = ASSETS) -> Path:
    """Write `SHA256SUMS` next to the artifact, in the format `sha256sum -c` reads.

    Run this immediately after `freeze`, before the release is created, so that
    the checksums describe the file that is actually uploaded rather than one
    regenerated later from a panel that has since grown.
    """
    lines = []
    for name in names:
        path = directory / name
        if not path.exists():
            raise FetchError(
                f"nothing to checksum at {path}; run `python -m src.models.freeze` first"
            )
        lines.append(f"{sha256_of(path.read_bytes())}  {name}")
    destination = directory / CHECKSUM_FILE
    destination.write_text("\n".join(lines) + "\n")
    return destination


def parse_checksums(text: str) -> dict[str, str]:
    """`<hex>  <name>` per line, the coreutils format. Blank lines ignored."""
    digests: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if not name:
            raise FetchError(f"malformed checksum line: {line!r}")
        digests[name.strip()] = digest.strip()
    return digests


def verify(name: str, data: bytes, digests: dict[str, str]) -> None:
    """Refuse a file the release did not publish, or published differently.

    A missing entry is an error rather than a pass. "No checksum was recorded for
    this file" and "this file matches its checksum" must never take the same
    branch, or adding an asset silently opts it out of verification.
    """
    if name not in digests:
        raise FetchError(f"{CHECKSUM_FILE} has no entry for {name}; it lists {sorted(digests)}")
    actual = sha256_of(data)
    if actual != digests[name]:
        raise FetchError(
            f"{name} does not match the checksum published with the release.\n"
            f"  published: {digests[name]}\n"
            f"  received:  {actual}\n"
            "Refusing to build an image around a file the release did not publish."
        )


def fetch_artifact(
    tag: str,
    into: Path,
    repo: str = DEFAULT_REPO,
    opener: Opener = _read,
    names: Iterable[str] = ASSETS,
) -> list[Path]:
    """Download a release's artifact into `into`, verified. Returns what it wrote.

    Nothing is written until every file has been downloaded *and* checked, so a
    failure leaves no half-populated `models/` for a later step to mistake for a
    successful fetch.
    """
    names = tuple(names)
    try:
        checksums = opener(asset_url(repo, tag, CHECKSUM_FILE))
    except HTTPError as error:
        if error.code == 404:
            raise FetchError(
                f"release {tag!r} in {repo} has no {CHECKSUM_FILE}. Releases are created with\n"
                f"  python -m src.inference.fetch --checksums models\n"
                f"  gh release create {tag} models/shelf_life.joblib models/shelf_life.json "
                f"models/{CHECKSUM_FILE}"
            ) from error
        raise FetchError(f"fetching {CHECKSUM_FILE} for {tag}: HTTP {error.code}") from error
    except URLError as error:
        raise FetchError(f"cannot reach the release for {tag}: {error.reason}") from error

    digests = parse_checksums(checksums.decode())

    downloaded: dict[str, bytes] = {}
    for name in names:
        try:
            data = opener(asset_url(repo, tag, name))
        except HTTPError as error:
            raise FetchError(f"release {tag} is missing {name}: HTTP {error.code}") from error
        except URLError as error:
            raise FetchError(f"cannot download {name} from {tag}: {error.reason}") from error
        verify(name, data, digests)
        downloaded[name] = data

    into.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in downloaded.items():
        path = into / name
        path.write_bytes(data)
        written.append(path)
    (into / CHECKSUM_FILE).write_bytes(checksums)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checksums", metavar="DIR", type=Path, help="write SHA256SUMS for a release and exit"
    )
    parser.add_argument("--tag", help="the release tag to fetch")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--into", type=Path, default=Path("models"))
    args = parser.parse_args(argv)

    try:
        if args.checksums is not None:
            path = write_checksums(args.checksums)
            print(f"wrote {path}")
            print(path.read_text().rstrip())
            return 0
        if not args.tag:
            parser.error("one of --checksums or --tag is required")
        written = fetch_artifact(args.tag, into=args.into, repo=args.repo)
        for path in written:
            print(f"fetched {path} ({path.stat().st_size:,} bytes, checksum verified)")
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
