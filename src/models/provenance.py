"""What a run needs to be reproducible: the code version and the data version.

Every run logs its params and its metrics, and then two things that are easy to
leave out: **the dataset version and the git SHA**. Those two matter more than
they look, because they are the ones that cannot be recovered afterwards. A metric with no
dataset version is not comparable to anything — the scraper adds a wave a day,
so "PR-AUC 0.31" without a snapshot is a number about an unknown quantity of
data. A metric with no commit is not reproducible — the pipeline that produced
it has since been edited.

**`dirty` is logged, not prevented.** Refusing to run on a dirty tree would be
the wrong trade: most runs happen mid-edit, and a driver that will not run until
you commit is a driver you stop using. Recording it means a run tagged
`git_dirty=true` is understood as provisional, which is the honest status.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

#: Written into every run so a reader can tell a real result from a rehearsal
#: on the synthetic panel. Not cosmetic: the synthetic runs exist precisely to
#: exercise machinery that the real panel is still too shallow to drive, and a
#: history that mixed the two without saying so would be worse than no history.
REAL, SYNTHETIC = "real", "synthetic"


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):  # pragma: no cover
        return None


@dataclass(frozen=True)
class Provenance:
    """Everything needed to say *which code, on which data*."""

    git_sha: str | None
    git_branch: str | None
    git_dirty: bool
    dataset: str
    panel_path: str
    panel_sha256: str | None
    panel_rows: int
    snapshot_date: str | None

    def as_tags(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Content hash of a file. Same rule as `src/data/snapshot.py`."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_date(raw_root: Path = Path("data/raw")) -> str | None:
    """The date of the most recent pinned snapshot, if one exists.

    Read from the manifest rather than the directory name, because the
    directory name is a label a human typed and the manifest is what
    `src/data/snapshot.py` actually wrote.
    """
    manifests = sorted(raw_root.glob("*/manifest.json"))
    if not manifests:
        return None
    try:
        return str(json.loads(manifests[-1].read_text())["snapshot_date"])
    except (OSError, ValueError, KeyError):  # pragma: no cover
        return None


def collect(panel_path: Path, n_rows: int, dataset: str = REAL) -> Provenance:
    """Describe the current code and data.

    The panel is hashed rather than named, because `data/processed/` is
    regenerable and its filename is stable across regenerations: two runs
    against `job_days_h1_calendar.parquet` a week apart are runs against
    different data with the same path. The hash is what distinguishes them.
    """
    return Provenance(
        git_sha=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_dirty=bool(_git("status", "--porcelain")),
        dataset=dataset,
        panel_path=str(panel_path),
        panel_sha256=sha256_of(panel_path) if panel_path.exists() else None,
        panel_rows=int(n_rows),
        snapshot_date=_snapshot_date(),
    )
