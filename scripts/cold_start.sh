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

payload='{"title": "Senior Data Engineer", "location": "Berlin",
          "content_chars": 1400, "first_published": "2026-08-20T00:00:00Z",
          "as_of": "2026-09-14T03:45:00Z"}'

echo
echo "Warm — the same instance, immediately afterwards:"
timed "/predict (warm)" -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' -d "${payload}"
timed "/predict (warm, again)" -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' -d "${payload}"

echo
echo "Put the cold figure in the README beside the caveat, not instead of it:"
echo "  \"the first request after idle takes N seconds; subsequent ones take M ms\""

# --- the acceptance criterion, applied ---------------------------------------
verdict=$(COLD="${COLD_SECONDS:-0}" STOP="${STOP_RULE_SECONDS}" python3 -c '
import os, sys

cold, stop = float(os.environ["COLD"]), float(os.environ["STOP"])
print(f"{cold:.2f} {stop:.0f}")
sys.exit(1 if cold > stop else 0)
') && passed=0 || passed=1
set -- ${verdict}
echo
if [ "${passed}" -eq 0 ]; then
  echo "PASS: cold start ${1} s is within the ${2} s acceptance criterion."
  exit 0
fi

echo "STOP: cold start ${1} s exceeds the ${2} s acceptance criterion." >&2
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
