#!/usr/bin/env bash
# Run the model ladder on the real panel, on validation only, as soon as the
# panel is deep enough to be split at all — and refuse until then.
#
#     ./scripts/rehearse.sh            # snapshot, assemble, gate, then the ladder
#     ./scripts/rehearse.sh --check    # report the depth and run nothing
#
# **Why this stops short of `freeze`.** The test block opens once
# (`CLAUDE.md` §4.2, `docs/design.md`). A legal split arrives days before one
# deep enough to cut rolling-origin folds from, so the first split this script
# can run on is real but has no error bars — exactly the split you do *not*
# want to spend the test set on. So it runs `train_baseline`, `train`,
# `experiments` and `evaluate`, all of which read train and validation only,
# and stops. `freeze` stays a deliberate act with a `--run` argument naming the
# model a human chose.
#
# What that buys is a rehearsal. Every module below has so far only run against
# `tests/panels.py`, whose columns are synthesised. The first contact with the
# real panel is where a dtype, a missing category or an empty group shows up,
# and it is much better to meet that on a split whose numbers do not matter yet
# than on the one afternoon the test block is available.
#
# Exit codes: 0 ran the ladder · 3 declined, the panel is too shallow ·
# anything else, a step failed.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
HORIZON="${HORIZON:-1}"
BASIS="${BASIS:-calendar}"
PANEL="data/processed/features/job_days_h${HORIZON}_${BASIS}.parquet"
CHECK_ONLY=0

for arg in "$@"; do
  case "$arg" in
    --check) CHECK_ONLY=1 ;;
    *) echo "usage: $0 [--check]" >&2; exit 2 ;;
  esac
done

if [ ! -x "${PYTHON}" ]; then
  echo "no interpreter at ${PYTHON} — set PYTHON=... or create the venv" >&2
  exit 2
fi

# A pinned snapshot is immutable by design, so `snapshot` refuses to re-pin. For
# a script meant to be run repeatedly until the panel clears, "already pinned"
# is the ordinary case rather than a failure, which is what --skip-existing
# says. The check is delegated rather than done here with a date and a `test -f`:
# the snapshot is named after the newest run *inside* the database, not after
# the wall clock, so this script cannot know the name without asking.
echo "== pinning a dated snapshot"
"${PYTHON}" -m src.data.snapshot --skip-existing

echo
echo "== assembling the job-day panel (H=${HORIZON}, ${BASIS})"
"${PYTHON}" -m src.features.assemble --horizon "${HORIZON}" --basis "${BASIS}"

echo
echo "== label validity audit"
# Runs before the gate on purpose: it needs closures, not a three-way split, so
# it has an answer on a panel far too shallow to model. Whether "disappeared"
# means what the label needs it to mean is worth knowing before any model is
# fitted to it, not after.
"${PYTHON}" -m src.data.label_audit

echo
echo "== depth gate"
# `minimum_waves` is the readiness check and `feasible_cuts` is the acceptance
# check; DEBUGGING.md 2026-09-07 is what happens when they disagree, so this
# asks both and believes the pessimistic one.
set +e
"${PYTHON}" - "${PANEL}" <<'PY'
import sys
import pandas as pd
from src.data.split import depth_report, feasible_cuts, minimum_waves

panel = pd.read_parquet(sys.argv[1])
print(depth_report(panel))
depth = minimum_waves(panel)
usable = int(feasible_cuts(panel)["valid"].sum())
print(f"\nusable cuts today: {usable}")
if depth["folds_shortfall"] > 0:
    print(
        f"note: {depth['folds_available']} rolling-origin fold(s) available, so any "
        f"comparison below carries no error bar. That needs "
        f"{depth['needed_for_folds']} labelled wave(s)."
    )
sys.exit(0 if usable > 0 else 3)
PY
GATE=$?
set -e

if [ "${GATE}" -eq 3 ]; then
  echo
  echo "declined: the panel cannot yet be cut three ways. Nothing was run."
  exit 3
elif [ "${GATE}" -ne 0 ]; then
  exit "${GATE}"
fi

if [ "${CHECK_ONLY}" -eq 1 ]; then
  echo
  echo "--check: the split is legal and nothing was run."
  exit 0
fi

echo
echo "== the ladder, on validation only"
"${PYTHON}" -m src.models.train_baseline
"${PYTHON}" -m src.models.train
"${PYTHON}" -m src.models.experiments
"${PYTHON}" -m src.models.evaluate

echo
echo "ran on validation only. The test block is untouched: freeze is a separate,"
echo "deliberate step, and it opens the test set once."
