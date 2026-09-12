#!/usr/bin/env bash
# Regenerate every generated report from a clean source tree, in order.
#
#     ./scripts/regenerate_reports.sh            # all of them, on the pinned snapshot
#     ./scripts/regenerate_reports.sh --no-net   # skip the one that asks the boards
#
# Every report names the commit that produced it and whether the tree was
# clean. On 2026-09-12 every committed report said `dirty tree`: each had been
# regenerated mid-edit, or after another report in the same batch, and none
# could be reproduced from the commit it named alone. That is the provenance
# standard the freeze enforces for the artifact, and the diagnostic reports
# are the evidence the artifact is read against, so they get the same one.
#
# The order is the pipeline's: pin and build, then the audits that read the
# panel, then the H=7 gate (which declines until the panel is deep enough and
# writes the refusal), then the H=1 rehearsal — the ladder, the comparison,
# the fingerprint — which is the machinery exercised on real data. Nothing
# here opens the held-out block: `freeze` at H=7 refuses on depth, and at
# H=1 the rehearsal stops short of it by design.
#
# Refuses if anything outside `reports/` is uncommitted: a report regenerated
# from edited source names a commit that did not produce it. The reports
# themselves are the run's output and do not count (`provenance.source_is_dirty`).
#
# The day the gate clears:  merge → clean main → CI green → this script →
# commit the reports → freeze. In that order, so the evidence is in history
# before the result that is read against it.

set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
NET=1
for arg in "$@"; do
  case "$arg" in
    --no-net) NET=0 ;;
    *) echo "usage: $0 [--no-net]" >&2; exit 2 ;;
  esac
done

if [ -n "$(git status --porcelain -- . ':!reports')" ]; then
  echo "refusing: the source tree has uncommitted changes outside reports/." >&2
  echo "A report regenerated now would name a commit that did not produce it. Commit first." >&2
  git status --short -- . ':!reports' >&2
  exit 3
fi
echo "== source tree clean at $(git rev-parse --short=9 HEAD) on $(git rev-parse --abbrev-ref HEAD)"

echo "== pin and build"
"${PYTHON}" -m src.data.snapshot --skip-existing
"${PYTHON}" -m src.data.load
"${PYTHON}" -m src.data.archive
"${PYTHON}" -m src.data.profile
"${PYTHON}" -m src.features.assemble --horizon 7
"${PYTHON}" -m src.features.assemble --horizon 1

echo "== the audits, at H=7"
"${PYTHON}" -m src.data.readiness
"${PYTHON}" -m src.data.label_audit
"${PYTHON}" -m src.data.cohort_audit
if [ "${NET}" -eq 1 ]; then
  "${PYTHON}" -m src.data.label_check
else
  echo "   (label_check skipped: --no-net; its report keeps its previous provenance)"
fi

echo "== the H=7 gate: the refusal, or the result"
# `freeze` needs a run name even to refuse; the refusal is the report. The
# fingerprint at H=7 declines the same way until there is a legal cut, and
# its H=1 finding lives in board_fingerprint_h1_calendar.md meanwhile.
"${PYTHON}" -m src.models.freeze --run 05-xgboost_engineered || true
"${PYTHON}" -m src.models.board_fingerprint --panel data/processed/features/job_days_h7_calendar.parquet || true

echo "== the H=1 rehearsal: the machinery on real data"
HORIZON=1 ./scripts/rehearse.sh || true
"${PYTHON}" -m src.data.cohort_audit --panel data/processed/features/job_days_h1_calendar.parquet \
  --out reports/cohort_audit_h1.md

echo "== the ledger"
"${PYTHON}" -m src.models.ledger

echo "== the README's generated block"
"${PYTHON}" -m src.data.readme_summary

echo
echo "regenerated. Review, then commit reports/ as one change:"
git status --short -- reports
