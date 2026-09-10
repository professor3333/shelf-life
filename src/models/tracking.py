"""One place that writes a run to MLflow, for every experiment that produces one.

**Why this is its own module.** The tracking helpers used to live in
`src/models/experiments.py`, beside the eight scripted runs they were written
for. `src/models/train.py` — which runs the feature ablations, the overfit
sweep, the board-context folds and the serve-time regime — could not reach them:
`experiments.py` imports `_xgb_parameters` from `train.py`, so importing back
the other way is a cycle. The result was that the eight runs a person invokes
deliberately were tracked and the four families the training entry point runs
every time were not, which is the wrong way round if anything.

So the generic half moved here, where both entry points can use it and neither
owns it. `log_run` stays in `experiments.py` because it is specific to a
`RunSpec`; everything it does is done through `log_variant` below.

**Every experiment logs the same six things**, because a run missing any of them
cannot be compared with one that has them:

* the **feature subset** — as an artifact, never a param, since MLflow truncates
  long param values and a truncated feature list is worse than none: it reads as
  complete
* the **split** — both cut instants, the embargo, and the block sizes
* the **dataset version** — path and sha256 of the panel, via `Provenance`
* the **parameters** that distinguish this variant from its siblings
* the **metrics**
* the **code version** — git sha, branch and whether the tree was dirty

**Tracking is optional and its absence is loud.** MLflow pulls in a server, a
database layer and a web UI, and `pyproject.toml` keeps it out of the base
install precisely so a stranger can run the ladder without any of that. A
missing MLflow therefore cannot be an error — but an untracked run must not be
silent either, or "every experiment is logged" quietly stops being true. The
callers print what happened and write it into the report.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import pandas as pd

from src.data.split import SplitResult
from src.models import provenance

DEFAULT_EXPERIMENT = "shelf-life"
#: SQLite rather than the `./mlruns` directory. MLflow 3 put the filesystem
#: store into maintenance mode and refuses it without an opt-out environment
#: variable, and the database backend is what `mlflow ui` expects anyway:
#:
#:     mlflow ui --backend-store-uri sqlite:///mlflow.db
#:
#: The file is gitignored. It is a cache of runs that `replay` can rebuild, not
#: a source artifact — the same reason `data/processed/` is not committed.
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _mlflow():
    """Imported lazily so the rest of `src/` does not depend on it.

    MLflow pulls in a large dependency tree and the modelling code has no need
    of it; a stranger who wants to run the ladder should not have to install a
    tracking server to do so.
    """
    import mlflow

    return mlflow


def available() -> bool:
    """Is MLflow importable? Asked before logging, so the answer can be reported."""
    try:
        _mlflow()
    except ImportError:
        return False
    return True


def start(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str = DEFAULT_TRACKING_URI):
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow


def split_params(split: SplitResult) -> dict[str, object]:
    """The split, in the form a later reader can check a run against.

    Both cut instants rather than one, because a run is only comparable with
    another if the *same* rows were available to fit and the same rows were
    scored — and the embargo width decides how much of the timeline neither
    block got. The block sizes are recorded beside them so a run whose cut moved
    under it is visible as a change in `n_train` rather than only as a different
    number.

    The third block is deliberately not counted here. `tests/test_evaluate.py`
    parses `src/` to keep the list of modules that touch it at two, and a
    tracking helper is not one of them.
    """
    return {
        "train_end": split.cuts.train_end.isoformat(),
        "val_end": split.cuts.val_end.isoformat(),
        "embargo": str(split.embargo),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_embargoed": int(split.n_embargoed),
        "n_unlabelled": int(split.n_unlabelled),
    }


def pipeline_features(model) -> tuple[str, ...] | None:
    """The columns a built pipeline selects, or `None` if it selects none.

    Read off the object rather than recomputed from the registry, for the reason
    `freeze.spec_features` gives: a model may restrict the set, and a field
    listing features the model never saw is worse than no field. Reading it needs
    no fit — `build_pipeline` fixes the column list when it constructs the step.

    `None` is a real answer, not a failure. The ladder's rule baselines are not
    pipelines at all: they apply a stated rule to the frame and never build a
    feature matrix, so "which columns did it fit on" has no answer for them and
    inventing the full registry as one would be a lie in the same slot.
    """
    steps = getattr(model, "named_steps", None)
    if not steps or "select" not in steps:
        return None
    return tuple(steps["select"].kw_args["columns"])


def _clean_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    """Numbers MLflow will accept: floats, and nothing that is not one.

    A `NaN` is dropped rather than logged as `nan`. A metric that could not be
    computed is absent, which a reader can see; a metric present and unreadable
    is worse, because it looks like a measurement.
    """
    clean: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        if pd.isna(value):
            continue
        clean[key] = float(value)
    return clean


@contextmanager
def family(
    mlflow,
    run_name: str,
    *,
    prov: provenance.Provenance,
    params: Mapping[str, object] | None = None,
    tags: Mapping[str, object] | None = None,
) -> Iterator[object]:
    """A parent run holding one family of variants.

    The families this project runs — ablations, the overfit sweep, the
    board-context folds — are each a set of refits that only mean anything
    beside one another. Logged flat they would be a dozen sibling runs with no
    marker of which sweep they came from; nested, the parent carries what they
    share (the split, the dataset, the code) and each child carries only what
    makes it different.
    """
    with mlflow.start_run(run_name=run_name) as active:
        mlflow.log_params({key: value for key, value in (params or {}).items()})
        mlflow.set_tags(
            {"family": run_name, **{k: str(v) for k, v in (tags or {}).items()}, **prov.as_tags()}
        )
        yield active


def log_variant(
    mlflow,
    *,
    run_name: str,
    prov: provenance.Provenance,
    params: Mapping[str, object] | None = None,
    metrics: Mapping[str, object] | None = None,
    tags: Mapping[str, object] | None = None,
    features: Sequence[str] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    nested: bool = False,
) -> str:
    """Write one run and return its id.

    `features` becomes `features.json` and `tables` become CSV artifacts, for the
    reason given in the module docstring: a param long enough to be truncated is
    a param that lies about being complete.
    """
    with mlflow.start_run(run_name=run_name, nested=nested) as active:
        mlflow.log_params({key: value for key, value in (params or {}).items()})
        mlflow.log_metrics(_clean_metrics(metrics or {}))
        mlflow.set_tags({**{k: str(v) for k, v in (tags or {}).items()}, **prov.as_tags()})
        if features is not None:
            mlflow.log_dict({"features": list(features)}, "features.json")
        for name, table in (tables or {}).items():
            if table is not None and not table.empty:
                mlflow.log_text(table.to_csv(index=False), f"{name}.csv")
        return active.info.run_id
