#!/usr/bin/env bash
# How long does the first request wait when nothing is running?
#
#     ./scripts/cold_start.sh https://shelf-life-xxxxx.europe-west1.run.app
#
# `docs/design.md` §7e promises this number and says why: a cold start nobody has
# timed is a surprise being saved up for whoever is being shown the link. CI
# cannot produce it — the verify workflow runs right after a rebuild, when the
# service is warm. The only honest measurement is taken after the service has
# genuinely idled down, which is what the wait below is for.
#
# What is being measured: DNS and TLS, then the platform starting a spun-down
# container, then Python importing scikit-learn and XGBoost and unpickling the
# artifact **on 0.1 of a CPU**, then one request. That last clause is why the
# estimate is worth so little and the measurement so much. The warm request
# afterwards is printed beside it because the *difference* is the cold-start
# cost; the absolute number alone hides how much of it is just scoring a row.
#
# Preconditions, and the run is worthless without them:
#   * nothing is keeping the service awake — there is deliberately no keep-warm
#     cron (§7e explains the 750-hour arithmetic that rules one out)
#   * no other traffic during the wait — a browser tab left open on the UI, or a
#     second person clicking the link, warms the container and turns this into a
#     measurement of a warm start.

set -euo pipefail

BASE_URL="${1:-${SHELF_LIFE_API:-}}"
if [ -z "${BASE_URL}" ]; then
  echo "usage: $0 <base-url> [idle-minutes]   (or set SHELF_LIFE_API)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

# Render spins a free service down after 15 idle minutes. 16 is the default here
# so the wait clears that rather than racing it.
IDLE_MINUTES="${2:-16}"

#: **The deployment acceptance criterion**, in seconds, and this script exits
#: non-zero when the cold measurement exceeds it.
#:
#: It is enforced here rather than written down somewhere because of what the
#: alternative looks like at the moment it matters. A disappointing cold start
#: arrives with an obvious fix already in reach — raise the client timeout until
#: the number stops being a problem — and that fix changes nothing except who
#: finds out. `docs/design.md` §7e is explicit that past this line the answer is
#: to reassess the architecture, not to widen the window it is measured through.
#:
#: The same 90 is the UI's HTTP timeout, and `tests/test_deploy.py` asserts the
#: two cannot drift apart, so raising the timeout to accommodate a slow service
#: fails CI rather than quietly succeeding.
STOP_RULE_SECONDS="${STOP_RULE_SECONDS:-90}"

#: The most recent measurement, in seconds. `timed` writes it; the caller copies
#: it out immediately, because every later call overwrites it — and the value the
#: acceptance criterion is applied to must be the COLD one, not the warm request
#: that happens to have run last.
LAST_SECONDS=""
COLD_SECONDS=""

timed() {  # timed <label> <curl args...> -> seconds, and fails loudly on a non-2xx
  local label="$1"; shift
  local result
  result=$(curl -fsS -o /dev/null -w '%{time_total} %{http_code}' --max-time 180 "$@") \
    || { echo "  ${label}: FAILED (the service did not answer within 180s)" >&2; return 1; }
  printf '  %-28s %6.2f s   HTTP %s\n' "${label}" "${result% *}" "${result#* }"
  LAST_SECONDS="${result% *}"
}

echo "Waiting ${IDLE_MINUTES} minutes for ${BASE_URL} to scale to zero."
echo "Send it nothing during the wait — including opening the UI."
if [ "${IDLE_MINUTES}" != "0" ]; then
  sleep $((IDLE_MINUTES * 60))
fi

echo
echo "Cold — the first request after ${IDLE_MINUTES} idle minutes:"
timed "/health (cold)" "${BASE_URL}/health"
COLD_SECONDS="${LAST_SECONDS}"

# Which of the two measurements is this? The answer is not a flag the caller
# passes, because a caller who has to remember which kind of run this is will
# eventually record a baseline as if it were the definitive one.
#
#   BASELINE   — the no-artifact image. Process start, interpreter, and the
#                scikit-learn and XGBoost imports. A genuine measurement of a
#                real instance, and a LOWER BOUND: it never touches joblib, never
#                unpickles a pipeline or a booster, and /predict answers 503
#                without reaching the model.
#   DEFINITIVE — the same image with a released artifact in it. Adds the load
#                path the baseline omits, and is the only measurement that can
#                accept the architecture.
#
# `docs/design.md` §7e: the architecture is not accepted until the DEFINITIVE
# measurement is within the criterion.
MODEL_LOADED=$(curl -fsS --max-time 60 "${BASE_URL}/health" \
  | python3 -c 'import json,sys; print("yes" if json.load(sys.stdin).get("model_loaded") else "no")')

payload='{"title": "Senior Data Engineer", "location": "Berlin",
          "content_chars": 1400, "first_published": "2026-08-20T00:00:00Z",
          "as_of": "2026-09-14T03:45:00Z"}'

PREDICT_SECONDS=""
if [ "${MODEL_LOADED}" = "yes" ]; then
  echo
  echo "First prediction — the load path the baseline cannot reach:"
  timed "/predict (first)" -X POST "${BASE_URL}/predict" \
    -H 'content-type: application/json' -d "${payload}"
  PREDICT_SECONDS="${LAST_SECONDS}"
  timed "/predict (warm)" -X POST "${BASE_URL}/predict" \
    -H 'content-type: application/json' -d "${payload}"
else
  echo
  echo "No model loaded, so /predict is a 503 and is not timed."
fi

# --- the acceptance criterion, applied ---------------------------------------
#
# Applied to the SLOWEST request, not to the wake alone. The UI's timeout is per
# request and guards both, so a /health that wakes in 40 s followed by a
# /predict that takes 100 s is a failure even though the wake looked fine.
verdict=$(COLD="${COLD_SECONDS:-0}" PRED="${PREDICT_SECONDS:-0}" STOP="${STOP_RULE_SECONDS}" \
  python3 -c '
import os, sys

cold = float(os.environ["COLD"])
predict = float(os.environ["PRED"] or 0.0)
stop = float(os.environ["STOP"])
worst = max(cold, predict)
print(f"{cold:.2f} {predict:.2f} {worst:.2f} {stop:.0f}")
sys.exit(1 if worst > stop else 0)
') && within=0 || within=1
set -- ${verdict}
cold_s="$1"; predict_s="$2"; worst_s="$3"; stop_s="$4"

echo
if [ "${within}" -ne 0 ]; then
  echo "STOP: ${worst_s} s exceeds the ${stop_s} s acceptance criterion." >&2
  echo "  cold /health ${cold_s} s   first /predict ${predict_s} s" >&2
  echo >&2
  echo "docs/design.md §7e: past this line the decision is reassessed, not tuned" >&2
  echo "around. Specifically NOT the available shortcut — raising the UI timeout" >&2
  echo "until the number stops looking bad changes only who finds out, and" >&2
  echo "tests/test_deploy.py fails if you try it." >&2
  echo >&2
  echo "The questions §7e says to ask instead:" >&2
  echo "  * is a live API needed for this demo at all, or would a static page" >&2
  echo "    with pre-computed examples show the same work?" >&2
  echo "  * does the image have to carry XGBoost, when import-and-unpickle is" >&2
  echo "    what the tenth of a CPU is actually spending its time on?" >&2
  exit 1
fi

if [ "${MODEL_LOADED}" != "yes" ]; then
  echo "BASELINE recorded: ${cold_s} s, within the ${stop_s} s criterion."
  echo
  echo "This does NOT accept the deployment architecture, and the number is a"
  echo "lower bound rather than a result. The image carries no artifact, so this"
  echo "run never unpickled a pipeline, never loaded a booster, and never entered"
  echo "the code path a prediction uses. Whatever the real load costs on 0.1 of a"
  echo "CPU is missing from the figure above."
  echo
  echo "Re-run this against an image built from a real release. Until that second"
  echo "measurement comes back within the criterion, docs/design.md §7e says the"
  echo "architecture is provisional."
  echo
  echo "And do not read this figure as reassurance. The inference runs one way"
  echo "only: a baseline OVER the criterion would have condemned the definitive"
  echo "measurement outright, because the definitive one can only be slower. A"
  echo "baseline within it implies nothing at all, because the work it skipped —"
  echo "unpickling a pipeline and a booster on a tenth of a CPU — is not bounded"
  echo "by anything this run measured. The value of a baseline is that it can end"
  echo "the question early, never that it can settle it."
  exit 0
fi

echo "ACCEPTED: ${worst_s} s is within the ${stop_s} s acceptance criterion."
echo "  cold /health ${cold_s} s   first /predict ${predict_s} s"
echo
echo "This is the definitive measurement — a real artifact, loaded and used."
echo "Put both figures in the README beside the caveat, not instead of it:"
echo "  \"the first request after idle takes N seconds; subsequent ones take M ms\""
