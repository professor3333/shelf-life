"""Assemble the (posting, complete-run observation) panel with as-of-t features.

    snapshots/*.csv  +  archived payloads  +  runs  ->  one row per job-day

One row is a *job-day*: one posting as seen by one complete run. A posting
present for five complete runs contributes five rows, each with its own features
and its own label. The reasoning is in ``docs/problem_definition.md`` §2.

**Everything here is assembled at or before `t`.** `t` is the start instant of a
complete run. Tabular fields come from that run's snapshot CSV and structured
detail from that fetch's archived payload — both sealed observations of one
moment. The current-state ``jobs`` table is read only for identity, and the
``runs`` table only to decide which runs are complete and to place them in time.

Three features read other rows, and each is computed over a window that **ends
at `t`**: board size, board growth, and how many postings on the board share a
title or a requisition id. A window that runs past `t` turns each of them from a
legitimate feature into a leak, which is why they are built here rather than
left to a later step that has the whole frame in scope.

Usage::

    python -m src.features.assemble --horizon 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.archive import load_archive
from src.data.clean import parse_salary
from src.data.load import load_snapshot

DEFAULT_SNAPSHOT_CSV_DIR = Path.home() / "code" / "job-listing-scraper" / "snapshots"
OUTPUT_ROOT = Path("data/processed/features")

#: Tolerance for matching an archived fetch to the run that produced it. The two
#: clocks are the same process a moment apart; observed spread is -1s to +5s.
FETCH_MATCH_TOLERANCE = pd.Timedelta("15min")


def complete_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Runs that observed the whole board, in time order, per source.

    A *complete* run is ``status == 'ok'`` at the current ``rules_version``.
    Both halves matter. A ``partial`` run stopped at its page cap and did not see
    the whole board, so a posting's absence from it is not evidence of removal —
    the defect recorded in ``DEBUGGING.md``. And runs are only comparable within
    a ``rules_version``, so an earlier epoch cannot be used to establish that a
    posting later disappeared.
    """
    current = runs["rules_version"].max()
    selected = runs[(runs["status"] == "ok") & (runs["rules_version"] == current)].copy()
    selected = selected.sort_values(["source", "started_at"], kind="stable")
    selected["run_index"] = selected.groupby("source").cumcount()
    return selected[["id", "source", "started_at", "run_index"]].rename(
        columns={"id": "run_id", "started_at": "t"}
    )


def load_snapshot_rows(csv_dir: Path = DEFAULT_SNAPSHOT_CSV_DIR) -> pd.DataFrame:
    """Every per-run snapshot CSV, concatenated.

    One file per run, one row per posting that run saw, never rewritten — so
    these are as-of-t by construction. ``observed_at`` is the run's start
    instant and joins exactly to ``runs.started_at``.
    """
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no snapshot CSVs under {csv_dir}")
    frames = [pd.read_csv(path) for path in files]
    rows = pd.concat(frames, ignore_index=True)
    rows["observed_at"] = pd.to_datetime(rows["observed_at"], utc=True, format="ISO8601")
    rows["source_id"] = rows["source_id"].astype("string")
    return rows.sort_values(["observed_at", "source", "source_id"], kind="stable")


def build_observations(snapshot_rows: pd.DataFrame, runs_complete: pd.DataFrame) -> pd.DataFrame:
    """The job-day panel: one row per (posting, complete run that saw it)."""
    panel = snapshot_rows.merge(
        runs_complete,
        left_on=["source", "observed_at"],
        right_on=["source", "t"],
        how="inner",
    )
    return panel.drop(columns=["observed_at"])


def compute_labels(
    panel: pd.DataFrame,
    runs_complete: pd.DataFrame,
    horizon_days: int,
    basis: str = "calendar",
):
    """Attach ``y`` per ``docs/problem_definition.md`` §4.

    This is a transcription of the rule stated there, and the three branches map
    one-to-one onto its table:

    ``t_gone(j)`` is the start of the earliest complete run in which *j* was
    absent, where *j* was also absent from the next complete run and never
    re-appeared. Undefined if no such pair exists.

    * ``t_gone`` defined and ``<= t + H``            -> 1
    * *j* observed by a complete run at or after ``t + H`` -> 0
    * otherwise                                       -> dropped

    The third branch is the one that matters. A row whose horizon reaches past
    the last complete run has **not** survived — we have not looked yet. Filling
    those with 0 would teach the model that recent postings last forever.

    ``basis`` decides how ``t + H`` is compared. **Calendar, decided 2026-09-04**
    (`docs/design.md` §10); ``"instant"`` is kept so the comparison stays
    reproducible.

    ``"calendar"``
        Compares dates, not instants. On the observed schedule this is exactly
        "was the posting absent at the next complete run", which is the finest
        distinction a once-daily panel can draw.
    ``"instant"``
        Literal arithmetic. Rejected because removals here are
        *interval-censored* — we know a posting vanished somewhere in
        ``(t_last_seen, t_first_absent]``, never when — so a continuous-time
        horizon is not identifiable from this panel at H=1, and in practice the
        label became a function of cron jitter. Measured: run 0 → run 1 is
        34.4h because a run fired at 14:07 instead of 03:45, so all 19 removals
        detected at run 1 fall outside a strict 1-day horizon and are discarded
        — **0 positives in 1,116 rows at run 0**. Run 2 → run 3 is 23.9993h, so
        run 2's 12 positives survive **by 2.6 seconds**, and the +27.0s drift at
        run 3 → run 4 drops 13 further rows as unobservable. The instant
        positive rate across run indices is 0.00% / 1.77% / 1.06% / 0.00%
        against calendar's 1.67% / 1.75% / 1.23% / 0.00% — and ``run_index``,
        ``t_dow`` and ``age_days`` are all features, so that is label noise
        correlated with the model's own inputs.
    """
    if basis not in {"instant", "calendar"}:
        raise ValueError(f"basis must be 'instant' or 'calendar', got {basis!r}")
    horizon = pd.Timedelta(days=horizon_days)
    seen = {
        (source, source_id): set(group["run_index"])
        for (source, source_id), group in panel.groupby(["source", "source_id"], sort=False)
    }
    run_times = {
        source: group.sort_values("run_index")["t"].tolist()
        for source, group in runs_complete.groupby("source", sort=False)
    }

    t_gone: dict[tuple[str, str], pd.Timestamp] = {}
    for (source, source_id), present in seen.items():
        times = run_times[source]
        last_present = max(present)
        for index in range(len(times) - 1):
            if index in present or index + 1 in present:
                continue
            if index > last_present:  # never re-appeared after this gap
                t_gone[(source, source_id)] = times[index]
                break

    def within(moment, deadline):
        """Did `moment` fall at or before the horizon? Date-wise or instant-wise."""
        if basis == "calendar":
            return moment.date() <= deadline.date()
        return moment <= deadline

    def at_or_after(moment, deadline):
        if basis == "calendar":
            return moment.date() >= deadline.date()
        return moment >= deadline

    labels, keep = [], []
    for row in panel.itertuples(index=False):
        key = (row.source, row.source_id)
        deadline = row.t + horizon
        gone = t_gone.get(key)
        if gone is not None and within(gone, deadline):
            labels.append(1)
            keep.append(True)
        elif any(
            at_or_after(times, deadline)
            for index, times in enumerate(run_times[row.source])
            if index in seen[key]
        ):
            labels.append(0)
            keep.append(True)
        else:
            labels.append(pd.NA)
            keep.append(False)

    out = panel.copy()
    out["y"] = pd.array(labels, dtype="Int8")
    out["label_observable"] = keep
    return out


def _attach_archive(panel: pd.DataFrame, archive: pd.DataFrame) -> pd.DataFrame:
    """Join each job-day to the archived payload fetched for that same run."""
    if archive.empty:
        return panel
    # pandas exposes two string dtypes that differ only in their NA sentinel;
    # a CSV and a Parquet round-trip can land on different ones, and merge_asof
    # refuses to join across them. Normalise the key on both sides.
    left = panel.sort_values("t", kind="stable").astype({"source_id": "string"})
    right = archive.sort_values("fetched_at", kind="stable").astype({"source_id": "string"})
    columns = [
        "fetched_at",
        "source_id",
        "first_published",
        "updated_at",
        "departments",
        "n_departments",
        "offices",
        "n_offices",
        "requisition_id",
        "n_metadata",
        "content_chars",
    ]
    merged = pd.merge_asof(
        left,
        right[columns],
        left_on="t",
        right_on="fetched_at",
        by="source_id",
        direction="nearest",
        tolerance=FETCH_MATCH_TOLERANCE,
    )
    return merged.drop(columns=["fetched_at"])


def _board_context(panel: pd.DataFrame) -> pd.DataFrame:
    """Features that read other rows — each windowed to end at `t`.

    Every one of these is computed *within* a single run, so it describes the
    board as it stood at `t` and cannot see a later one. A window that ran past
    `t` would make each of these a leak while looking identical in the frame.
    """
    out = panel.copy()
    by_run = out.groupby(["source", "run_index"], sort=False)
    out["board_size_at_t"] = by_run["source_id"].transform("size")
    out["n_same_title_on_board"] = out.groupby(
        ["source", "run_index", "title"], sort=False, dropna=False
    )["source_id"].transform("size")
    out["n_same_req_on_board"] = out.groupby(
        ["source", "run_index", "requisition_id"], sort=False, dropna=False
    )["source_id"].transform("size")
    out.loc[out["requisition_id"].isna(), "n_same_req_on_board"] = pd.NA

    sizes = out[["source", "run_index", "board_size_at_t"]].drop_duplicates()
    sizes = sizes.sort_values(["source", "run_index"])
    sizes["board_growth"] = sizes.groupby("source")["board_size_at_t"].diff()
    out = out.merge(
        sizes[["source", "run_index", "board_growth"]], on=["source", "run_index"], how="left"
    )

    # How many complete runs have seen this posting up to and including t.
    out = out.sort_values(["source", "source_id", "run_index"], kind="stable")
    out["n_complete_runs_observed"] = out.groupby(["source", "source_id"]).cumcount() + 1
    return out


def row_local_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Row-local features. `age_days` is measured from the employer's own
    publication instant, never from `first_seen` — which records when this
    project started looking, not when the posting appeared.

    **Public, and called from the serving path.** Every column here is a
    function of one row and nothing else, so it can be computed for a single
    posting at serve time — and `src/inference/contract.py` computes it by
    calling *this* function rather than by restating the arithmetic. A second
    implementation of `age_days` that rounded differently would be
    training/serving skew: the model would be scored on a column that no longer
    means what it meant when it was fitted, and nothing would raise.
    """
    out = panel.copy()
    day = pd.Timedelta(days=1)

    out["age_days"] = (out["t"] - out["first_published"]) / day
    out["days_since_update"] = (out["t"] - out["updated_at"]) / day
    out["t_dow"] = out["t"].dt.dayofweek
    out["posted_dow"] = out["first_published"].dt.dayofweek
    out["posted_month"] = out["first_published"].dt.month

    parsed = out["salary_raw"].map(parse_salary)
    out["salary_stated"] = parsed.map(lambda p: p.present).astype("boolean")
    out["salary_parsed"] = parsed.map(lambda p: p.ok).astype("boolean")
    out["salary_min_clean"] = parsed.map(lambda p: p.minimum).astype("Float64")
    out["salary_max_clean"] = parsed.map(lambda p: p.maximum).astype("Float64")
    out["salary_period"] = parsed.map(lambda p: p.period).astype("string")
    out["salary_currency_clean"] = parsed.map(lambda p: p.currency).astype("string")
    return out


def assemble(
    horizon_days: int = 7,
    basis: str = "calendar",
    snapshot_dir: Path | None = None,
    csv_dir: Path = DEFAULT_SNAPSHOT_CSV_DIR,
    archive: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the full job-day frame: features as of `t`, label from after it."""
    frames = load_snapshot(snapshot_dir)
    runs_complete = complete_runs(frames["runs"])
    rows = load_snapshot_rows(csv_dir)

    panel = build_observations(rows, runs_complete)
    panel = compute_labels(panel, runs_complete, horizon_days, basis=basis)
    panel = _attach_archive(panel, load_archive() if archive is None else archive)
    panel = _board_context(panel)
    panel = row_local_features(panel)

    panel["horizon_days"] = horizon_days
    panel["horizon_basis"] = basis
    return panel.sort_values(["t", "source", "source_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--basis", choices=["instant", "calendar"], default="calendar")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    frame = assemble(horizon_days=args.horizon, basis=args.basis)
    labelled = frame[frame["label_observable"]]

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"job_days_h{args.horizon}_{args.basis}.parquet"
    frame.to_parquet(path, index=False)

    print(f"horizon            : {args.horizon} days ({args.basis} basis)")
    print(f"job-day rows       : {len(frame):,}")
    print(f"  labelable        : {len(labelled):,}")
    print(f"  dropped (censored): {len(frame) - len(labelled):,}")
    if len(labelled):
        positives = int((labelled["y"] == 1).sum())
        negatives = len(labelled) - positives
        print(f"  positives        : {positives:,}  ({positives / len(labelled):.2%})")
        print(f"  negatives        : {negatives:,}")
        if positives == 0 or negatives == 0:
            print(
                "\nWARNING: only one class is observable at this horizon, so this"
                " frame\ncannot train or evaluate anything. It is not an empty"
                " result — it looks like data.\nA row is labelled 0 only when the"
                f" posting was seen at or after t + {args.horizon}d, and the"
                " panel is not yet that deep, so every surviving row is a"
                " positive.\nWait for panel depth, or shorten the horizon."
            )
    else:
        print("\nWARNING: no row has an observable outcome at this horizon.")
    print(f"distinct postings  : {frame['source_id'].nunique():,}")
    print(f"wrote -> {path}")


if __name__ == "__main__":
    main()
