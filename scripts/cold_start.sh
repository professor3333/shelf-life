#!/usr/bin/env bash
# How long does the first request wait when nothing is running?
#
#     ./scripts/cold_start.sh https://shelf-life-xxxxx.europe-west1.run.app
#
# `docs/design.md` §7e promises this number and says why: a cold start nobody has
# timed is a surprise being saved up for whoever is being shown the link. The
# deploy workflow cannot produce it — deploying a revision starts an instance to
# verify it serves, so the first request after a deploy finds a warm one. The
# only honest measurement is taken after the service has genuinely idled down,
# which is what the wait below is for.
#
# What is being measured: DNS and TLS, then Cloud Run pulling a ~1 GB image onto
# a cold instance, then Python importing scikit-learn and XGBoost and unpickling
# the artifact, then one request. The warm request afterwards is printed beside
# it because the *difference* is the cold-start cost; the absolute number alone
# hides how much of it is just the model scoring a row.
#
# Preconditions, and the run is worthless without them:
#   * --min-instances is 0 (otherwise there is nothing cold to measure)
#   * no other traffic during the wait — an uptime check, a browser tab left
#     open on the Streamlit UI, or a second person clicking the link all warm
#     the instance and turn this into a measurement of a warm start.

set -euo pipefail

BASE_URL="${1:-${SHELF_LIFE_API:-}}"
if [ -z "${BASE_URL}" ]; then
  echo "usage: $0 <base-url> [idle-minutes]   (or set SHELF_LIFE_API)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

# Cloud Run keeps an idle instance alive for roughly 15 minutes. 20 is the
# default here so the wait clears that with margin rather than racing it.
IDLE_MINUTES="${2:-20}"

timed() {  # timed <label> <curl args...> -> seconds, and fails loudly on a non-2xx
  local label="$1"; shift
  local result
  result=$(curl -fsS -o /dev/null -w '%{time_total} %{http_code}' --max-time 120 "$@") \
    || { echo "  ${label}: FAILED (the service did not answer within 120s)" >&2; return 1; }
  printf '  %-28s %6.2f s   HTTP %s\n' "${label}" "${result% *}" "${result#* }"
}

echo "Waiting ${IDLE_MINUTES} minutes for ${BASE_URL} to scale to zero."
echo "Send it nothing during the wait — including opening the UI."
if [ "${IDLE_MINUTES}" != "0" ]; then
  sleep $((IDLE_MINUTES * 60))
fi

echo
echo "Cold — the first request after ${IDLE_MINUTES} idle minutes:"
timed "/health (cold)" "${BASE_URL}/health"

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
echo
echo "If N is much past 20 seconds, docs/design.md §7e says that argues for a"
echo "small always-on instance instead — a decision to record there, with the"
echo "number that forced it, rather than a knob to quietly turn."
