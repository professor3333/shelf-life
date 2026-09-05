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
FROZEN_RUN = "05-xgboost_engineered"


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
