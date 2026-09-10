"""Diagnostic figures. Not decoration — each one answers a question a table asks badly.

**Why these five and not a gallery.** A plot earns its place here only where the
shape of the thing is the finding and a column of numbers hides it:

* **missingness by source** — the central fact about this data is that missing
  is not random, it is a fingerprint of which board a row came from
  (`docs/design.md`, and the table in `src/data/profile.py:coverage_by_source`).
  Read as numbers that is eight rows to compare by eye; read as a grid the
  blocks of "never" are immediate.
* **distributions** — a median and a standard deviation describe a bell. Nothing
  here is a bell: `age_days` is a long tail with a spike at the panel edge, and
  the summary statistics of a long tail are a description of a distribution that
  is not present.
* **precision–recall, against the base rate** — the headline metric is PR-AUC on
  a rare positive, and the only honest way to read one is beside the line a
  constant predictor draws. A PR curve with no baseline invites the reader to
  compare the area against 0.5, which is the ROC intuition and wrong here.
* **calibration** — a reliability diagram is the one diagnostic whose failure
  mode is invisible in every scalar. Brier and ECE both average, so a model that
  is confidently wrong at the top of its range and compensating at the bottom
  scores like a model that is right everywhere.
* **train against validation across complexity** — the overfit sweep exists to
  open a gap and close it, and a gap is a distance between two lines. As a table
  it is two columns the reader has to subtract in their head, one row at a time.

**Matplotlib is an optional extra**, like MLflow and for the same reason: the
Dockerfile installs `.[api]`, so anything in the base dependencies is shipped to
a 0.1-CPU instance whose cold start is already 36% spent before a model loads.
A plotting library has no business in the serving image. Callers ask
`available()` and report the answer rather than failing.

**Figures are written to `reports/figures/` and committed**, because the reports
that reference them are committed and a report whose images are missing is worse
than one with none. They are not byte-reproducible across machines — font
rasterisation differs — and nothing here pretends otherwise; the *data* behind
each one is in the table beside it, which is the artifact to diff.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

#: Where figures go. Beside the reports that reference them, not inside a
#: build directory, because the reader of `reports/*.md` is the audience.
FIGURES_DIR = Path("reports/figures")

#: One ink colour and one accent, both legible in grey-scale and distinguishable
#: under the common colour-vision deficiencies. Two is enough: every figure here
#: compares at most two series, and a palette wide enough to need a legend is a
#: sign the figure is doing two jobs.
INK = "#1f2a36"
ACCENT = "#c2453a"
MUTED = "#9aa5b1"

#: Colour *and* dash per series, so a figure survives being printed in grey and
#: read by someone who cannot separate the two hues. Four is the most any figure
#: here draws; a fifth series would mean the figure is doing two jobs.
_SERIES: tuple[tuple[str, str], ...] = (
    (INK, "-"),
    ("#3f7fa6", "--"),
    (MUTED, "-."),
    ("#6b5b95", ":"),
)


def available() -> bool:
    """Is matplotlib importable? Asked before plotting, so the answer can be reported."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        return False
    return True


def _pyplot():
    """Matplotlib in headless mode, imported lazily.

    `Agg` is selected before `pyplot` is imported: choosing a backend afterwards
    is a no-op on some installs and an error on others, and the failure mode —
    a figure that renders on a laptop and hangs in CI waiting for a display —
    is the kind that only appears where it is expensive.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@contextmanager
def _figure(path: Path, size: tuple[float, float] = (7.0, 4.5)) -> Iterator[tuple]:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=size, dpi=140)
    try:
        yield figure, axes
        figure.tight_layout()
        # `Software` carries the matplotlib version, which would make every
        # figure's bytes change on an unrelated upgrade and every report look
        # edited. The data behind the figure is what should show a diff.
        figure.savefig(path, format="png", metadata={"Software": None})
    finally:
        plt.close(figure)


def _annotate(axes, text: str) -> None:
    """The finding, written on the figure, so it survives being pasted elsewhere."""
    axes.text(
        0.5,
        -0.28,
        text,
        transform=axes.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=INK,
        wrap=True,
    )


def missingness_by_source(coverage: pd.DataFrame, path: Path) -> Path:
    """The missingness fingerprint, as a grid rather than eight rows of percentages.

    `coverage` is `src/data/profile.py:coverage_by_source` — percent *non-null*
    per source per column, with a `jobs` count in front. The count is dropped
    here: it is a row label, not a measurement on the same scale as the rest, and
    a colour ramp shared between "3,531 jobs" and "9 percent" is meaningless.
    """
    grid = coverage.drop(columns=["jobs"], errors="ignore")
    with _figure(path, size=(1.4 * max(len(grid.columns), 3) + 3, 0.5 * len(grid) + 2.6)) as (
        figure,
        axes,
    ):
        image = axes.imshow(grid.to_numpy(dtype=float), cmap="RdYlGn", vmin=0, vmax=100)
        axes.set_xticks(range(len(grid.columns)), grid.columns, rotation=30, ha="right")
        axes.set_yticks(range(len(grid.index)), grid.index)
        for row in range(len(grid.index)):
            for column in range(len(grid.columns)):
                value = grid.to_numpy(dtype=float)[row, column]
                axes.text(
                    column,
                    row,
                    "—" if np.isnan(value) else f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=INK if 25 < value < 85 else "white",
                )
        axes.set_title("Percent present, by source and column", loc="left", fontsize=11)
        figure.colorbar(image, ax=axes, shrink=0.8, label="% non-null")
    return path


def distributions(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> Path:
    """Histograms for the columns whose shape a summary statistic misdescribes.

    Log-scaled counts, because every one of these is a long tail and a linear
    count axis renders the tail as an empty strip — which is exactly the part
    worth seeing.
    """
    present = [name for name in columns if name in frame.columns]
    if not present:
        raise ValueError("none of the requested columns are in the frame")

    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    width = min(len(present), 3)
    height = (len(present) + width - 1) // width
    figure, grid = plt.subplots(height, width, figsize=(4.2 * width, 3.0 * height), dpi=140)
    axes_list = np.atleast_1d(np.asarray(grid)).ravel()
    try:
        for axes, name in zip(axes_list, present, strict=False):
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            axes.hist(values, bins=40, color=INK)
            if len(values):
                # A column null for whole sources is ordinary here, and a log
                # scale on an empty axes warns and renders nothing useful. The
                # panel stays, labelled, because "no values" is the finding.
                axes.set_yscale("log")
            else:
                axes.text(
                    0.5,
                    0.5,
                    "no values",
                    transform=axes.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=MUTED,
                )
            axes.set_title(name, fontsize=10, loc="left")
            axes.set_ylabel("rows (log)", fontsize=8)
            if len(values):
                axes.axvline(values.median(), color=ACCENT, linewidth=1.2)
                span = float(values.max() - values.min())
                on_the_right = span > 0 and (values.median() - values.min()) / span > 0.55
                axes.text(
                    0.03 if on_the_right else 0.97,
                    0.92,
                    f"median {values.median():,.1f}\nmissing {frame[name].isna().mean():.0%}",
                    transform=axes.transAxes,
                    ha="left" if on_the_right else "right",
                    va="top",
                    fontsize=8,
                    color=ACCENT,
                )
        for axes in axes_list[len(present) :]:
            axes.set_visible(False)
        figure.suptitle("Distributions, log counts — the red line is the median", fontsize=11)
        figure.tight_layout()
        figure.savefig(path, format="png", metadata={"Software": None})
    finally:
        plt.close(figure)
    return path


def precision_recall(
    curves: dict[str, tuple[Sequence[float], Sequence[float]]],
    base_rate: float,
    path: Path,
    title: str = "Precision–recall against the base rate",
) -> Path:
    """PR curves with the no-skill line drawn in.

    **The baseline is the point.** A PR curve read without it invites comparison
    against 0.5, which is the ROC intuition; on a rare positive a constant
    predictor scores the base rate, so that is the horizontal line every curve
    has to sit above to mean anything. At a base rate near 0.08 the difference
    between "good" and "nothing" is a couple of centimetres of vertical space,
    and hiding the line makes the same picture look like a result.
    """
    with _figure(path) as (_figure_handle, axes):
        ceiling = base_rate * 4
        for index, (name, (recall, precision)) in enumerate(curves.items()):
            colour, dashes = _SERIES[index % len(_SERIES)]
            axes.step(
                recall,
                precision,
                where="post",
                linewidth=1.6 if index == 0 else 1.2,
                color=colour,
                linestyle=dashes,
                label=name,
            )
            # The scale comes from the part of the curve that carries evidence.
            # Precision below 5% recall is computed on a handful of rows and
            # spikes to 1.0 on a single lucky one; letting that set the axis
            # squashes every curve into the bottom eighth of the figure and hides
            # the comparison the plot exists to make.
            recall_array, precision_array = np.asarray(recall), np.asarray(precision)
            settled = precision_array[recall_array >= 0.05]
            if settled.size:
                ceiling = max(ceiling, float(settled.max()) * 1.2)
        axes.axhline(
            base_rate,
            color=ACCENT,
            linestyle="--",
            linewidth=1.4,
            label=f"constant predictor ({base_rate:.4f})",
        )
        axes.set_xlabel("recall")
        axes.set_ylabel("precision")
        axes.set_ylim(0, min(1.0, ceiling))
        axes.set_title(title, loc="left", fontsize=11)
        axes.legend(fontsize=8, frameon=False)
        _annotate(
            axes,
            "A curve that does not sit above the dashed line has not beaten predicting "
            "the base rate for every posting.",
        )
    return path


def calibration(
    reliabilities: dict[str, pd.DataFrame], path: Path, eces: dict[str, float] | None = None
) -> Path:
    """Reliability diagrams: predicted probability against what actually happened.

    Marker area is the number of rows in the bin, which is the whole reason this
    is a plot and not a line. A bin holding four postings and a bin holding four
    thousand sit at the same height in a table and are not the same evidence; the
    eye corrects for that automatically once the dots are sized.

    Small multiples rather than one crowded axes, because calibration is a
    property of a single model and overlaying three of them invites reading a
    difference between series that is really a difference between bin
    populations.
    """
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(reliabilities)
    figure, grid = plt.subplots(
        1, len(names), figsize=(4.1 * len(names), 4.4), dpi=140, squeeze=False
    )
    try:
        for axes, name in zip(grid[0], names, strict=True):
            usable = reliabilities[name].dropna(subset=["mean_predicted", "observed_rate"])
            usable = usable[usable["n"] > 0]
            axes.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2)
            if not usable.empty:
                sizes = 24 + 376 * (usable["n"] / usable["n"].max())
                axes.scatter(
                    usable["mean_predicted"],
                    usable["observed_rate"],
                    s=sizes,
                    color=INK,
                    alpha=0.75,
                    edgecolor="white",
                    zorder=3,
                )
                # Deliberately no line through the points. Bins differ in weight
                # by three orders of magnitude here, and a line joining a bin of
                # 4,000 postings to one of two draws a slope between them that
                # reads as a trend.
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 1)
            axes.set_aspect("equal")
            title = name if eces is None or name not in eces else f"{name} — ECE {eces[name]:.4f}"
            axes.set_title(title, fontsize=10, loc="left")
            axes.set_xlabel("mean predicted", fontsize=9)
        grid[0][0].set_ylabel("observed rate", fontsize=9)
        figure.suptitle(
            "Calibration — dot area is rows in the bin; dashed line is perfect", fontsize=11
        )
        figure.tight_layout()
        figure.savefig(path, format="png", metadata={"Software": None})
    finally:
        plt.close(figure)
    return path


def complexity_gap(
    sweep: pd.DataFrame,
    path: Path,
    label_column: str = "setting",
    train_column: str = "train_pr_auc",
    validation_column: str = "val_pr_auc",
) -> Path:
    """Train against validation across the overfit sweep, in the sweep's own order.

    The order is the argument. `OVERFIT_SWEEP` walks from a model that cannot
    overfit to one that badly does, then closes the gap one knob at a time, so
    the x-axis is a deliberate sequence rather than a category — and the shaded
    band between the two lines is the overfitting, which is the quantity the
    whole experiment is about.
    """
    positions = np.arange(len(sweep))
    train = sweep[train_column].to_numpy(dtype=float)
    validation = sweep[validation_column].to_numpy(dtype=float)
    with _figure(path, size=(7.6, 4.6)) as (_figure_handle, axes):
        axes.fill_between(positions, validation, train, color=ACCENT, alpha=0.15, label="gap")
        axes.plot(positions, train, marker="o", color=INK, linewidth=1.6, label="train")
        axes.plot(
            positions,
            validation,
            marker="o",
            color=ACCENT,
            linewidth=1.6,
            label="validation",
        )
        axes.set_xticks(positions, sweep[label_column], rotation=20, ha="right", fontsize=8)
        axes.set_ylabel("PR-AUC")
        axes.set_ylim(0, 1.02)
        axes.set_title("Train against validation, across model complexity", loc="left", fontsize=11)
        axes.legend(fontsize=8, frameon=False)
        widest = int(np.nanargmax(train - validation)) if len(sweep) else 0
        _annotate(
            axes,
            "The shaded band is the overfitting. Widest at "
            f"'{sweep[label_column].iloc[widest]}': "
            f"{train[widest] - validation[widest]:.4f} PR-AUC.",
        )
    return path
