"""One file on disk holding the whole model: pipeline, threshold, provenance.

The roadmap's warning, restated because it is the entire reason this module
exists:

    ✗  model.joblib containing the fitted XGBoost booster
    ✓  model.joblib containing derive -> select -> preprocess -> booster

Persisting the estimator alone guarantees the serving code re-implements the
feature logic, and a second implementation is a slow leak: it agrees with the
first one on the day it is written and stops agreeing the day either changes.
So what is saved is the `Pipeline` object itself, and `load` refuses anything
that is not one — `assert_is_full_pipeline` checks the steps by name and checks
that the fitted parts are fitted, which turns the roadmap's "REVIEW that the
saved object really is the full pipeline" from an instruction into an assertion.

**The metadata is not decoration.** A probability means nothing without the
threshold it is compared against, the horizon it refers to, and the data it was
fitted on; a served model whose artifact cannot say which git commit built it is
a model nobody can reproduce. All of it travels in the same file, and a copy is
written beside it as JSON so a human — or a deployment log — can read it without
loading pickled Python.

**Portability, stated plainly.** joblib pickles, so the library versions that
wrote the file are recorded and checked on load. A mismatch warns rather than
raises: it is usually harmless and occasionally the explanation for a number
that moved.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import sklearn
import xgboost
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError, check_is_fitted

DEFAULT_ARTIFACT = Path("models/shelf_life.joblib")

#: Steps every served pipeline must have, in this order. `company_volume` is
#: optional — it appears only when board identity is admitted — so membership is
#: checked rather than equality.
REQUIRED_STEPS: tuple[str, ...] = ("derive", "select", "preprocess", "model")


class ArtifactError(ValueError):
    """The file on disk is not a servable model."""


@dataclass(frozen=True)
class Metadata:
    """Everything needed to interpret a probability this pipeline produces."""

    run_name: str
    question: str
    params: dict
    features: tuple[str, ...]
    fitted_on: str
    threshold: float
    budget_per_day: int
    horizon_days: int
    horizon_basis: str
    metrics: dict[str, float]
    provenance: dict[str, str]
    dataset: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    versions: dict[str, str] = field(
        default_factory=lambda: {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        }
    )

    def as_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)


@dataclass(frozen=True)
class Artifact:
    pipeline: Pipeline
    metadata: Metadata


def assert_is_full_pipeline(obj: object) -> Pipeline:
    """Refuse anything that is not the fitted, end-to-end pipeline.

    Three separate claims, because each fails differently. That it is a
    `Pipeline` at all. That it carries the feature-building steps, so nothing
    downstream has to rebuild them. And that the learned step is fitted — an
    unfitted transformer raises at the first request rather than at load, which
    is the difference between a deploy that fails and a deploy that succeeds and
    then fails on a stranger.
    """
    if not isinstance(obj, Pipeline):
        raise ArtifactError(
            f"artifact holds a {type(obj).__name__}, not a Pipeline. Saving a bare "
            "estimator forces the serving path to re-implement the feature logic, "
            "which is training/serving skew."
        )
    missing = [name for name in REQUIRED_STEPS if name not in obj.named_steps]
    if missing:
        raise ArtifactError(f"pipeline is missing step(s) {missing}; has {list(obj.named_steps)}")
    try:
        check_is_fitted(obj.named_steps["preprocess"])
    except NotFittedError as error:
        raise ArtifactError("pipeline's preprocessor is not fitted") from error
    return obj


def save(pipeline: Pipeline, metadata: Metadata, path: Path = DEFAULT_ARTIFACT) -> Path:
    """Write the pipeline and its metadata, plus a readable JSON sidecar."""
    assert_is_full_pipeline(pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": asdict(metadata)}, path)
    path.with_suffix(".json").write_text(metadata.as_json() + "\n")
    return path


def load(path: Path = DEFAULT_ARTIFACT) -> Artifact:
    """Read an artifact back, checking it is what it claims to be."""
    if not Path(path).exists():
        raise ArtifactError(f"no artifact at {path}. Build one with `python -m src.models.freeze`.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ArtifactError(f"{path} is not a shelf-life artifact")

    metadata = Metadata(**bundle["metadata"])
    for library, recorded in metadata.versions.items():
        current = {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        }[library]
        if current != recorded:
            warnings.warn(
                f"artifact was written with {library} {recorded}, loading under {current}; "
                "predictions may differ from the recorded metrics",
                RuntimeWarning,
                stacklevel=2,
            )
    return Artifact(assert_is_full_pipeline(bundle["pipeline"]), metadata)
