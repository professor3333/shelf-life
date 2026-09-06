#!/usr/bin/env bash
# Is the thing that just deployed actually serving predictions?
#
#     ./scripts/smoke.sh https://shelf-life-xxxxx.europe-west1.run.app
#
# `gcloud run deploy` succeeding means a container started and its port opened.
# It does not mean a model loaded: the image builds and boots happily with no
# artifact at all, answers /health with `model_loaded: false`, and returns 503
# from /predict. That is deliberate behaviour (see the Dockerfile) and it is
# exactly the deploy this script exists to fail.
#
# What it does not check: the *number*. The exact probability for a pinned
# posting is asserted in `tests/test_api.py` against the artifact in the
# repository, and the deployed artifact is a released one that may legitimately
# differ. Here the questions are: is a model loaded, is the response shaped like
# a decision rather than a bare score, and does bad input still get a 4xx.

set -euo pipefail

BASE_URL="${1:-${SHELF_LIFE_API:-}}"
if [ -z "${BASE_URL}" ]; then
  echo "usage: $0 <base-url>   (or set SHELF_LIFE_API)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# Somewhere to keep the response bodies while they are inspected. A temporary
# directory rather than a fixed path, so two runs cannot read each other's
# leftovers and pass on a stale body.
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

# --- /health, and the two facts it keeps separate ---------------------------
curl -fsS --max-time 60 "${BASE_URL}/health" > "${WORK}/health.json" || fail "/health did not answer"
python3 - "${WORK}/health.json" <<'PY' || fail "/health reported no usable model"
import json, os, sys

body = json.load(open(sys.argv[1]))
if not body.get("model_loaded"):
    sys.exit(
        "/health says model_loaded: false — the container is up and empty.\n"
        f"  status: {body.get('status')}\n"
        f"  detail: {body.get('detail')}\n"
        "The image was built without ARTIFACT_TAG, or the fetch was skipped."
    )
print(f"  model     {body.get('model')}")
print(f"  dataset   {body.get('dataset')}")
print(f"  fitted on {body.get('fitted_on')}")
print(f"  horizon   {body.get('horizon_days')} days")
print(f"  threshold {body.get('threshold')}")

# The guard that matters most, and the only one here that is about honesty
# rather than plumbing. Every component in this project is exercised on a
# synthetic panel whose label is drawn independently of every feature — so a
# model fitted on it returns a number that means nothing. That is fine on a
# laptop and unacceptable behind a public URL, where nobody can see which panel
# they are being answered from. Set ALLOW_SYNTHETIC=1 to smoke-test the
# rehearsal deliberately.
if body.get("dataset") == "synthetic" and os.environ.get("ALLOW_SYNTHETIC") != "1":
    sys.exit(
        "the loaded model was fitted on the SYNTHETIC panel, whose label is noise.\n"
        "Refusing to pass a public deployment that serves a placeholder as a prediction.\n"
        "Freeze against the real panel first, or re-run with ALLOW_SYNTHETIC=1 if this\n"
        "is a deliberate rehearsal."
    )
PY
echo "ok  /health"

# --- /predict, on a posting a caller could actually describe -----------------
curl -fsS --max-time 60 -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' \
  -d '{
        "title": "Senior Data Engineer",
        "location": "Berlin",
        "salary_raw": "120000 - 160000 USD",
        "departments": "Eng",
        "offices": "HQ",
        "n_offices": 1,
        "n_metadata": 3,
        "content_chars": 1400,
        "first_published": "2026-08-20T00:00:00Z",
        "updated_at": "2026-09-01T00:00:00Z",
        "as_of": "2026-09-14T03:45:00Z"
      }' > "${WORK}/predict.json" || fail "/predict did not answer 2xx"
python3 - "${WORK}/predict.json" <<'PY' || fail "/predict returned an unusable body"
import json, sys

body = json.load(open(sys.argv[1]))
p = body.get("probability")
if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
    sys.exit(f"probability is not a probability: {p!r}")

# A bare score is not an answer, so the fields that turn it into one are part of
# the contract and are checked as such: the operating point it was compared
# against, what "closing" means, which side of the threshold it fell, whether
# the board features carried information, and which panel the model saw.
for field in ("threshold", "horizon_days", "closing_soon", "board_context_supplied",
              "model", "dataset"):
    if body.get(field) is None:
        sys.exit(f"response has no {field}: a score without its context is not a decision")
verdict = "closing soon" if body["closing_soon"] else "not closing soon"
print(f"  probability {p:.4f} vs threshold {body['threshold']:.4f} -> {verdict}")
print(f"  board context supplied: {body['board_context_supplied']}")
PY
echo "ok  /predict"

# --- bad input is a 4xx, never a 500 ----------------------------------------
code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 30 -X POST "${BASE_URL}/predict" \
  -H 'content-type: application/json' -d '{"title": "", "salary": "nonsense"}')
[ "${code}" = "422" ] || fail "malformed payload returned ${code}, expected 422"
echo "ok  malformed payload -> 422"

# --- the audit, reachable without cloning the repository ---------------------
curl -fsS --max-time 30 "${BASE_URL}/contract" > "${WORK}/contract.json" || fail "/contract did not answer"
python3 - "${WORK}/contract.json" <<'PY' || fail "/contract is not usable"
import json, sys

rows = json.load(open(sys.argv[1]))
if not rows:
    sys.exit("/contract is empty")
print(f"  {len(rows)} fields described")
PY
echo "ok  /contract"

echo "SMOKE PASS: ${BASE_URL}"
