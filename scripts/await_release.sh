#!/usr/bin/env bash
# Wait until a deployed service is serving a particular release.
#
#     ./scripts/await_release.sh https://shelf-life.onrender.com artifact-2026-09-08
#
# The platform rebuilds on a push, which means there is a window — minutes, on a
# free tier building a 1 GB image — where the URL is up, healthy, and answering
# from the *previous* model. A smoke test run inside that window passes and
# proves nothing, which is the specific way a deploy check becomes decorative.
#
# So: poll /health until `artifact_tag` matches, then hand over to smoke.sh.
# Two states are distinguished on purpose. "Not answering yet" is expected while
# the old container is replaced; "answering with the wrong release" is expected
# while the build finishes. Neither is an error until the deadline.

set -euo pipefail

BASE_URL="${1:-}"
EXPECTED_TAG="${2:-}"
DEADLINE_MINUTES="${3:-20}"

if [ -z "${BASE_URL}" ] || [ -z "${EXPECTED_TAG}" ]; then
  echo "usage: $0 <base-url> <expected-release-tag> [deadline-minutes]" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

deadline=$(( $(date +%s) + DEADLINE_MINUTES * 60 ))
attempt=0

echo "Waiting for ${BASE_URL} to serve ${EXPECTED_TAG} (up to ${DEADLINE_MINUTES} minutes)."

while [ "$(date +%s)" -lt "${deadline}" ]; do
  attempt=$(( attempt + 1 ))
  body=$(curl -fsS --max-time 120 "${BASE_URL}/health" 2>/dev/null || true)

  if [ -z "${body}" ]; then
    echo "  [${attempt}] no answer yet — the container is starting or being replaced"
  else
    actual=$(printf '%s' "${body}" | python3 -c \
      'import json,sys; print(json.load(sys.stdin).get("artifact_tag") or "")' 2>/dev/null || true)
    if [ "${actual}" = "${EXPECTED_TAG}" ]; then
      echo "  [${attempt}] serving ${EXPECTED_TAG}"
      exit 0
    fi
    echo "  [${attempt}] serving ${actual:-(no release)} — still waiting for ${EXPECTED_TAG}"
  fi
  sleep 20
done

echo "TIMED OUT after ${DEADLINE_MINUTES} minutes: ${BASE_URL} never reported ${EXPECTED_TAG}." >&2
echo "Check the platform's build log — a failed build leaves the previous revision serving," >&2
echo "which is the correct behaviour and the reason this timed out instead of passing." >&2
exit 1
