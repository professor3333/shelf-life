#!/usr/bin/env bash
# Is the deployed UI actually there, running, and able to reach the API?
#
#     ./scripts/smoke_ui.sh https://shelf-life-xxxx.streamlit.app [deadline-minutes]
#
# The suite renders `app/streamlit_app.py` through Streamlit's own test runtime,
# so a green CI says the script runs. It says nothing about the copy on
# Streamlit Community Cloud: whether the app still exists, whether the host
# resolved `requirements.txt`, whether the script booted there, whether the
# `SHELF_LIFE_API` secret is set, or whether the host can reach the API at all.
# Each of those breaks a stranger's visit while every unit test stays green,
# and on 2026-09-11 one of them was broken — the secret was not set, and the
# public UI had been telling visitors it "cannot reach the API at
# http://localhost:8000" for as long as nothing had looked.
#
# Three checks, over plain HTTP, and what each proves:
#
#   1. GET /  ->  200, the host page.   The app exists. Community Cloud answers
#      404 for an app that does not, and bootstraps an anonymous session with a
#      303 for one that does — hence the cookie jar, without which every
#      request looks like the first.
#   2. GET /api/v2/app/status  ->  RUNNING.   The host's own view of the app.
#      INSTALLER_ERROR is a `requirements.txt` the host could not resolve;
#      USER_SCRIPT_ERROR is a script that crashed on boot; IS_SHUTDOWN is the
#      free tier's sleep after inactivity, which this script wakes the way a
#      visitor's click would and then waits out.
#   3. GET /~/+/_stcore/health  ->  ok.   The Streamlit server behind the host
#      page is answering. A 400 here is what an asleep app returns.
#
# What HTTP cannot see is the rendered page — Streamlit draws it over a
# WebSocket — so the fourth check, the one that catches the secret and the
# network path to the API, needs a browser: `scripts/smoke_ui_browser.py`.
#
# Exit codes: 0 all three passed · 1 a check failed · 2 usage.

set -euo pipefail

BASE_URL="${1:-${SHELF_LIFE_UI:-}}"
if [ -z "${BASE_URL}" ]; then
  echo "usage: $0 <ui-url> [deadline-minutes]   (or set SHELF_LIFE_UI)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"
DEADLINE_MINUTES="${2:-6}"

fail() { echo "UI SMOKE FAIL: $*" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
JAR="${WORK}/cookies"

# --- 1. the app exists -----------------------------------------------------------
code=$(curl -sS -L -c "${JAR}" -b "${JAR}" -o "${WORK}/index.html" -w '%{http_code}' \
  --max-time 60 "${BASE_URL}/") || fail "${BASE_URL} did not answer"
[ "${code}" = "200" ] || fail "GET / returned ${code} — 404 means no app at this address"
grep -q 'id="root"' "${WORK}/index.html" || fail "GET / is not the Community Cloud host page"
echo "ok  GET /            200, host page"

# --- 2. the host says RUNNING, waking it if it is asleep -------------------------
# The numbers are Community Cloud's AppStatusModel, read from its own bundle on
# 2026-09-11; the names are what a failure should say.
status_name() {
  python3 -c '
import sys
names = {0: "UNKNOWN", 1: "CREATING", 2: "CREATED", 3: "UPDATING", 4: "INSTALLING",
         5: "RUNNING", 6: "RESTARTING", 7: "REBOOTING", 8: "DELETING", 9: "DELETED",
         10: "USER_ERROR", 11: "PLATFORM_ERROR", 12: "IS_SHUTDOWN", 13: "INSTALLER_ERROR",
         14: "USER_SCRIPT_ERROR", 15: "POTENTIAL_MINER_DETECTED"}
print(names.get(int(sys.argv[1]), f"status {sys.argv[1]}"))' "$1"
}
app_status() {  # -> "<number> <streamlit version>"
  curl -sS -c "${JAR}" -b "${JAR}" --max-time 60 -H 'accept: application/json' \
    "${BASE_URL}/api/v2/app/status" \
  | python3 -c 'import json,sys; b=json.load(sys.stdin); print(b.get("status"), b.get("streamlitVersion") or "?")'
}

deadline=$(( $(date +%s) + DEADLINE_MINUTES * 60 ))
woken=0
while :; do
  read -r status version < <(app_status) || fail "/api/v2/app/status did not answer"
  name=$(status_name "${status}")
  case "${name}" in
    RUNNING)
      echo "ok  app status       RUNNING (Streamlit ${version})"
      break ;;
    IS_SHUTDOWN)
      if [ "${woken}" -eq 0 ]; then
        echo "    app status       IS_SHUTDOWN — asleep after inactivity; waking it, as a visitor's click would"
        curl -sS -X POST -c "${JAR}" -b "${JAR}" --max-time 60 -o /dev/null \
          "${BASE_URL}/api/v2/app/resume" || true
        woken=1
      fi ;;
    USER_ERROR|PLATFORM_ERROR|INSTALLER_ERROR|USER_SCRIPT_ERROR|DELETED|DELETING|POTENTIAL_MINER_DETECTED)
      fail "app status ${name} — the host could not run the app (INSTALLER_ERROR: requirements.txt; USER_SCRIPT_ERROR: the script crashed on boot)" ;;
    *)
      echo "    app status       ${name} — waiting" ;;
  esac
  [ "$(date +%s)" -lt "${deadline}" ] || fail "app status still ${name} after ${DEADLINE_MINUTES} minutes"
  sleep 15
done

# --- 3. the Streamlit server behind the host page answers -------------------------
for attempt in 1 2 3 4 5 6 7 8; do
  body=$(curl -sS -c "${JAR}" -b "${JAR}" --max-time 60 "${BASE_URL}/~/+/_stcore/health" || true)
  [ "${body}" = "ok" ] && break
  sleep 10
done
[ "${body}" = "ok" ] || fail "/~/+/_stcore/health answered '${body:-nothing}', not 'ok' — the Streamlit server is not up behind the host page"
echo "ok  _stcore/health   ok"

echo "UI SMOKE PASS (HTTP): ${BASE_URL}"
echo "The rendered page — and whether it reached the API — needs a browser:"
echo "  uv run --no-project --with playwright python scripts/smoke_ui_browser.py ${BASE_URL}"
