"""Fixtures shared by the inference and API tests.

Both need the same thing — a frozen artifact to score against — and freezing one
costs a fit, so it is done once per session here rather than once per module in
each file.

**The artifact is frozen from the synthetic panel.** `tests/panels.py` gives the
general reason (a test that reads today's scrape changes its mind overnight) and
there is a particular one: the real panel is still too shallow to split three
ways, so no real artifact exists to test against. When one does, these fixtures
point at it and nothing else in either file changes.
"""

from __future__ import annotations

import pytest

from src.data.split import crawl_waves, temporal_split
from src.inference import artifact as artifact_module
from src.models.experiments import default_cuts
from src.models.freeze import build_metadata, freeze
from src.models.metrics import DEFAULT_ALERT_BUDGET
from tests.panels import make_closing_panel

#: The run frozen for every test that needs an artifact. Named once so the
#: inference test and the API test are provably scoring the same model.
#:
#: **Logistic rather than boosted, and the reason is a bug this repository
#: shipped.** The pin was written against `05-xgboost_engineered` and passed on
#: the machine that wrote it and nowhere else: CI on Linux/x86 returned 0.0603
#: for the payload macOS/arm64 scored 0.0435. XGBoost's tree construction is not
#: bit-reproducible across platforms, and on a small panel with a noisy label a
#: gradient difference in the last decimal place is enough to choose a different
#: split — after which the two models are genuinely different models, not the
#: same model with rounding error.
#:
#: A convex fit has no such fork in it. `LogisticRegression` reaches the same
#: optimum from the same data on any machine, so a number pinned against it
#: means "the features changed" rather than "the runner changed". Boosting is
#: still exercised end to end by `test_a_boosted_artifact_also_freezes_and_serves`;
#: what it no longer does is carry a constant that only one laptop can verify.
#: See `DEBUGGING.md`.
FROZEN_RUN = "02-logistic"

#: The boosted spec, frozen in one test to prove the packaging is estimator-
#: agnostic. Its probability is deliberately not pinned.
BOOSTED_RUN = "05-xgboost_engineered"


@pytest.fixture(scope="session")
def panel():
    return make_closing_panel()


@pytest.fixture(scope="session")
def split(panel):
    return temporal_split(panel, default_cuts(crawl_waves(panel[panel["label_observable"]])))


@pytest.fixture(scope="session")
def frozen(split):
    return freeze(split, FROZEN_RUN)


@pytest.fixture(scope="session")
def synthetic_artifact(tmp_path_factory, frozen, panel):
    """A saved artifact on disk, which is what both the loader and the API need."""
    metadata = build_metadata(
        frozen,
        FROZEN_RUN,
        panel,
        artifact_module.DEFAULT_ARTIFACT,
        "synthetic",
        DEFAULT_ALERT_BUDGET,
    )
    path = tmp_path_factory.mktemp("artifact") / "shelf_life.joblib"
    return artifact_module.save(frozen.pipeline, metadata, path)
