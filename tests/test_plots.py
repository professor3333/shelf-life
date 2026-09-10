"""Tests for the diagnostic figures.

A figure cannot be asserted to be *legible*, so these check the things that can
be: that each one is produced, that it is a real PNG rather than an empty file,
that the marks the figure's argument depends on are actually drawn, and that the
plotting library stays out of the serving image.

The last is the one worth reading. `docs/design.md` §7e sets a cold-start budget
that is already 36% spent before a model loads, and the Dockerfile installs
`.[api]` — so a plotting library in the base dependencies would ship to a
0.1-CPU instance and pay for itself on every cold start, forever, to draw
nothing.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import plots

pytest.importorskip("matplotlib", reason="diagnostic figures are an optional extra")

ROOT = Path(__file__).resolve().parents[1]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
ACCENT_HEX = plots.ACCENT


def _is_a_real_png(path: Path, least_bytes: int = 5_000) -> bool:
    return (
        path.exists() and path.read_bytes()[:8] == PNG_MAGIC and path.stat().st_size > least_bytes
    )


# --- the boundary the extra exists to defend --------------------------------


def test_matplotlib_is_not_installed_into_the_serving_image():
    """The reason `plots` is an extra rather than a dependency.

    The Dockerfile installs `.[api]`, so anything in `[project].dependencies`
    reaches the free instance. A plotting library there would draw nothing and
    cost cold-start seconds out of a budget the project treats as a stop rule.
    """
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    base = " ".join(pyproject["project"]["dependencies"])
    assert "matplotlib" not in base, (
        "matplotlib is in the base dependencies, so `pip install .[api]` ships it to "
        "the serving image. Keep it in the `plots` extra."
    )
    api = " ".join(pyproject["project"]["optional-dependencies"]["api"])
    assert "matplotlib" not in api


def test_nothing_on_the_serving_path_imports_the_plotting_module():
    """`src/plots.py` may be imported lazily by report writers, never by inference."""
    for path in sorted((ROOT / "src" / "inference").glob("*.py")) + sorted(
        (ROOT / "api").glob("*.py")
    ):
        body = path.read_text()
        assert "src.plots" not in body and "from src import plots" not in body, (
            f"{path.relative_to(ROOT)} imports the plotting module, which drags matplotlib "
            "into the container"
        )


# --- each figure is produced, and is a figure -------------------------------


def test_the_missingness_grid_is_drawn_from_the_coverage_table(tmp_path):
    coverage = pd.DataFrame(
        {"jobs": [3531, 629], "salary_min": [9.0, 87.0], "remote": [100.0, 0.0]},
        index=pd.Index(["arbeitnow", "greenhouse:anthropic"], name="source"),
    )
    path = plots.missingness_by_source(coverage, tmp_path / "missingness.png")
    assert _is_a_real_png(path)


def test_the_distribution_figure_refuses_columns_it_does_not_have(tmp_path):
    """Silently drawing nothing would leave an empty figure that looks like a finding."""
    frame = pd.DataFrame({"age_days": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="none of the requested columns"):
        plots.distributions(frame, ["not_a_column"], tmp_path / "x.png")


def test_the_distribution_figure_survives_an_all_null_column(tmp_path):
    """`salary_min` is null for whole sources; a profile must not die on one."""
    frame = pd.DataFrame({"age_days": [1.0, 2.0, 3.0, 90.0], "salary_min": [None] * 4})
    path = plots.distributions(frame, ["age_days", "salary_min"], tmp_path / "d.png")
    assert _is_a_real_png(path)


def test_the_precision_recall_figure_draws_the_baseline(tmp_path):
    """The baseline is the argument of the figure, not an ornament.

    Read without it a PR curve invites comparison against 0.5, which is the ROC
    intuition; on a rare positive the line to clear is the base rate. The base
    rate is asserted to reach the rendered legend, because a baseline that is
    computed and not drawn is exactly the failure this checks for.
    """
    rng = np.random.default_rng(0)
    recall = np.linspace(0, 1, 50)
    curves = {"model": (recall, np.clip(0.3 - 0.2 * recall + rng.normal(0, 0.01, 50), 0, 1))}
    path = plots.precision_recall(curves, 0.0815, tmp_path / "pr.png")
    assert _is_a_real_png(path)

    # The dashed baseline is the only thing in this figure drawn in ACCENT, so
    # its pixels are proof the line reached the canvas rather than only the
    # argument list. A baseline computed and not drawn is the failure here.
    import matplotlib.image as image

    pixels = image.imread(path)[:, :, :3]
    accent = np.array([int(ACCENT_HEX[i : i + 2], 16) / 255 for i in (1, 3, 5)])
    close_enough = (np.abs(pixels - accent) < 0.08).all(axis=-1)
    assert close_enough.sum() > 50, (
        "the base-rate line is not in the rendered figure, so the curve is being shown "
        "without the only thing that makes its height meaningful"
    )


def test_the_calibration_figure_draws_one_panel_per_model(tmp_path):
    from src.models.metrics import reliability_curve

    rng = np.random.default_rng(1)
    truth = rng.binomial(1, 0.08, 500)
    reliabilities = {
        name: reliability_curve(truth, rng.uniform(0, 1, 500)) for name in ("prior", "xgboost")
    }
    path = plots.calibration(reliabilities, tmp_path / "cal.png", eces={"prior": 0.01})
    assert _is_a_real_png(path)


def test_the_complexity_figure_names_the_widest_gap(tmp_path):
    """The annotation is the finding; a picture nobody can read the point of is decoration."""
    sweep = pd.DataFrame(
        {
            "setting": ["stump", "moderate", "deep"],
            "train_pr_auc": [0.10, 0.99, 1.00],
            "val_pr_auc": [0.09, 0.12, 0.10],
        }
    )
    path = plots.complexity_gap(sweep, tmp_path / "gap.png")
    assert _is_a_real_png(path)


def test_figures_carry_no_library_version_in_their_bytes(tmp_path):
    """An unrelated matplotlib upgrade must not make every report look edited."""
    sweep = pd.DataFrame(
        {"setting": ["a", "b"], "train_pr_auc": [0.5, 0.9], "val_pr_auc": [0.4, 0.4]}
    )
    path = plots.complexity_gap(sweep, tmp_path / "gap.png")
    assert b"matplotlib version" not in path.read_bytes()
