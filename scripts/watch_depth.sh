#!/usr/bin/env bash
# Measure the panel every day, and say which day the model can be chosen.
#
#     ./scripts/watch_depth.sh          # pin, assemble, measure, log
#     ./scripts/watch_depth.sh --quiet  # same, but print only when the gate moves
#
# **Why this exists.** The build is not waiting on code, it is waiting on crawl
# waves, and a wait nobody measures becomes a wait nobody ends. Left to a person,
# "is it deep enough yet?" gets asked on the days you remember and skipped on the
# days you don't, and the answer arrives late by however long the gap was. This
# runs it every morning after the scraper, appends one line per day, and prints
# the projected date so the next check is a decision rather than a habit.
#
# **What it will not do.** It never runs `freeze`. `freeze` opens the test block,
# once, and spends it: an automated job that opens a test set on a schedule has
# removed the only thing that makes the number mean anything. When the fold gate
# clears, this script runs the *rehearsal* — train and validation only — and then
# stops and says a human owes it a decision. The decision is which model, made on
# the fold evidence that has just become available for the first time.
#
# Exit codes: 0 the fold gate is open · 3 still accruing · 4 no legal split yet ·
# anything else, a step failed.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
# H=7 is the task (`docs/design.md` §2, decided 2026-09-04); H=1 is a pipeline
# smoke test and that document says so in as many words. Defaulting to 1 here
# meant every number this script produced described the smoke test, under
# headings that named the build. `HORIZON=1 ./scripts/...` still asks for it.
HORIZON="${HORIZON:-7}"
BASIS="${BASIS:-calendar}"
PANEL="data/processed/features/job_days_h${HORIZON}_${BASIS}.parquet"

# Gitignored, like everything else under data/. It is a record of a wait, not a
# repository artifact, and committing a line a day would be a changelog of
# nothing happening.
LOG="data/depth_watch.log"

QUIET=0
for arg in "$@"; do
  case "$arg" in
    --quiet) QUIET=1 ;;
    *) echo "usage: $0 [--quiet]" >&2; exit 2 ;;
  esac
done

if [ ! -x "${PYTHON}" ]; then
  echo "no interpreter at ${PYTHON} — set PYTHON=... or create the venv" >&2
  exit 2
fi

# Both are no-ops on a day the scraper has not run yet: `snapshot` refuses to
# re-pin a name it already holds, and `assemble` rewrites an identical panel.
# That is deliberate — this script has to be safe to run twice, because a
# scheduled job and an impatient person will eventually both run it in the same
# hour, and the alternative to being idempotent is being wrong on that day.
"${PYTHON}" -m src.data.snapshot --skip-existing >/dev/null
"${PYTHON}" -m src.features.assemble --horizon "${HORIZON}" --basis "${BASIS}" >/dev/null

set +e
MEASURED=$("${PYTHON}" - "${PANEL}" <<'PY'
import sys
import pandas as pd
from src.data.split import (
    SplitTooShallow,
    best_cuts,
    crawl_waves,
    depth_report,
    feasible_cuts,
    minimum_waves,
    projected_clear,
    temporal_split,
)

panel = pd.read_parquet(sys.argv[1])
labelled = panel[panel["label_observable"]]
depth = minimum_waves(panel)
ahead = projected_clear(panel)
usable = int(feasible_cuts(panel)["valid"].sum())

# Validation only. The held-out block's positive count is tempting here — it is
# what decides whether a freeze is worth spending it on — and it is exactly what
# a daily job must not fetch: `TEST_BLOCK_READERS` names two modules and this is
# neither, and a peek that lives in `scripts/` would evade the AST guard rather
# than satisfy it. `usable` already carries the only part needed to schedule a
# freeze, because a cut is not valid unless every evaluation block has positives.
val_positives = -1
try:
    val_positives = int((temporal_split(panel, best_cuts(panel)).val["y"] == 1).sum())
except SplitTooShallow:
    pass

print("REPORT", depth_report(panel), sep="\t")
print("WAVES", len(crawl_waves(labelled)), sep="\t")
print("FOLDS", depth["folds_available"], sep="\t")
print("SHORTFALL", depth["folds_shortfall"], sep="\t")
print("USABLE_CUTS", usable, sep="\t")
print("VAL_POS", val_positives, sep="\t")
print("CLEARS", "open" if ahead["folds_clear"] is None else ahead["folds_clear"].date(), sep="\t")
PY
)
STATUS=$?
set -e
[ "${STATUS}" -eq 0 ] || { echo "${MEASURED}" >&2; exit "${STATUS}"; }

field() { printf '%s\n' "${MEASURED}" | awk -F'\t' -v k="$1" '$1==k{print $2; exit}'; }

WAVES=$(field WAVES)
FOLDS=$(field FOLDS)
SHORTFALL=$(field SHORTFALL)
USABLE=$(field USABLE_CUTS)
VAL_POS=$(field VAL_POS)
CLEARS=$(field CLEARS)
STAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

mkdir -p "$(dirname "${LOG}")"
printf '%s\twaves=%s\tfolds=%s\tshortfall=%s\tcuts=%s\tval_pos=%s\tclears=%s\n' \
  "${STAMP}" "${WAVES}" "${FOLDS}" "${SHORTFALL}" "${USABLE}" "${VAL_POS}" "${CLEARS}" \
  >> "${LOG}"

# --quiet exists for the scheduled caller: a daily job that prints a paragraph
# on every one of the days nothing changed trains its reader to ignore it, and
# then the one morning it says something is the morning it gets ignored.
CHANGED=1
if [ "${QUIET}" -eq 1 ] && [ "${FOLDS}" -eq 0 ]; then
  PREVIOUS=$(tail -n 2 "${LOG}" | head -n 1 | awk -F'\t' '{print $2}')
  [ "${PREVIOUS}" = "waves=${WAVES}" ] && CHANGED=0
fi

if [ "${CHANGED}" -eq 1 ]; then
  echo "== panel depth at ${STAMP}"
  field REPORT
  echo
  echo "  labelled waves : ${WAVES}"
  echo "  folds available: ${FOLDS} (want 3)"
  if [ "${VAL_POS}" -lt 0 ]; then
    # -1 is the "no split exists" sentinel, and printing it reads as a count.
    echo "  positives      : n/a — no cut to count them in yet"
  else
    echo "  positives      : ${VAL_POS} in validation"
  fi
fi

if [ "${USABLE}" -eq 0 ]; then
  [ "${CHANGED}" -eq 1 ] && echo "  no legal three-way cut yet."
  exit 4
fi

# The gate is the *target* fold count, not the first fold. `SHORTFALL` counts
# waves to the target rather than to one, and the two are days apart: one fold
# gives a mean with no spread at all, and two give a standard deviation over two
# numbers, which is not a standard deviation. Three is the fewest that can show a
# model losing a fold it was expected to win, which is the whole reason to wait.
#
# The first draft branched on `FOLDS -eq 0` and would have announced "the fold
# gate is OPEN" on a day the shortfall was still positive, then pointed at
# `freeze` — contradicting the two-gate table in the README from a script whose
# only job is to say which gate we are behind.
if [ "${SHORTFALL}" -gt 0 ]; then
  if [ "${CHANGED}" -eq 1 ]; then
    echo
    echo "Still accruing: ${SHORTFALL} more labelled wave(s), projected ${CLEARS}."
    if [ "${FOLDS}" -eq 0 ]; then
      echo "A model chosen today would be chosen on a single number with no spread."
    else
      echo "${FOLDS} fold(s) is a mean without a believable spread — not yet a comparison."
    fi
  fi
  exit 3
fi

echo
echo "== the fold gate is OPEN: ${FOLDS} rolling-origin fold(s)"
echo "Running the rehearsal — train and validation only."
echo
./scripts/rehearse.sh

cat <<'MESSAGE'

================================================================================
A decision is now owed, and this script will not make it.

`reports/model_comparison.md` has, for the first time, a cv_pr_auc_mean with a
standard deviation beside it. Read the verdict there: whether the leader is
separated from the runner-up, or inside one standard deviation and therefore
tied — in which case the simpler model is the answer.

Then, and only then:

    .venv/bin/python -m src.models.freeze --run <the model you chose>

That opens the test block. Once. Nothing downstream of it may change afterwards,
including the threshold, so be sure the choice above is settled first.
================================================================================
MESSAGE
