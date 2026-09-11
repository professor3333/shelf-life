#!/usr/bin/env bash
# Is the thing that just deployed actually serving predictions?
#
#     ./scripts/smoke.sh https://shelf-life-xxxx.onrender.com [expected-release-tag]
#
# A platform reporting "deployed" means a container started and its port opened.
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
  echo "usage: $0 <base-url> [expected-release-tag]   (or set SHELF_LIFE_API)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

# Optional second argument: the release tag this deployment is supposed to be
# serving. Without it the script proves a model is answering; with it, that the
# *expected* one is. Those differ exactly when a deploy silently did not happen,
# which is the failure a smoke test run straight after a deploy is most likely
# to meet and least likely to notice.
EXPECTED_TAG="${2:-}"

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

# Somewhere to keep the response bodies while they are inspected. A temporary
# directory rather than a fixed path, so two runs cannot read each other's
# leftovers and pass on a stale body.
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT

# --- /health, and the two facts it keeps separate ---------------------------
curl -fsS --max-time 60 "${BASE_URL}/health" > "${WORK}/health.json" || fail "/health did not answer"
EXPECTED_TAG="${EXPECTED_TAG}" python3 - "${WORK}/health.json" <<'PY' || fail "/health did not report a usable model at the expected release"
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
print(f"  release   {body.get('artifact_tag') or '(none — not built from a release)'}")
print(f"  dataset   {body.get('dataset')}")

expected = os.environ.get("EXPECTED_TAG") or ""
actual = body.get("artifact_tag") or ""
if expected and actual != expected:
    sys.exit(
        f"this deployment is serving release {actual or '(none)'}, not {expected}.\n"
        "The push either has not finished building yet, or the build did not pick up\n"
        "MODEL_TAG. Either way the URL is answering from the previous model."
    )
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
# against, the horizon, which side of the threshold it fell, what the number
# predicts in words, whether the board features carried information, and which
# panel the model saw. These names are the API's (`api/schemas.py`
# `PredictionResponse`) and `tests/test_deploy.py` asserts they stay so — this
# list said `closing_soon` for two days after the API renamed it, and the first
# real release would have failed here, at the last link, on a field name.
for field in ("threshold", "horizon_days", "removal_flagged", "predicts",
              "board_context_supplied", "model", "dataset"):
    if body.get(field) is None:
        sys.exit(f"response has no {field}: a score without its context is not a decision")
verdict = "flagged for removal" if body["removal_flagged"] else "not flagged"
print(f"  probability {p:.4f} vs threshold {body['threshold']:.4f} -> {verdict}")
print(f"  predicts: {body['predicts']}")
print(f"  board context supplied: {body['board_context_supplied']}")
PY
echo "ok  /predict"

# --- /rank, the shape the operating point was designed for -------------------
#
# Give the service a board, get back the postings most likely to be removed,
# under a budget. This is the call an operator would actually make (README,
# `docs/design.md` §15), so a deployment where /predict answers and /rank does
# not is a deployment of the wrong endpoint. Checked, on five postings with a
# budget of two: 200, five back, every score a probability, ranks a permutation
# ordered by score, exactly two watched and they are ranks 1 and 2, the applied
# threshold is the second score, the same batch ranks the same way twice, and
# the first posting — the one /predict just scored — gets the same probability
# through /rank as it did alone. The unit tests pin all of this against the
# repository's artifact; this pins it against the one that is serving.
rank_body() {
  cat <<'JSON'
{
  "budget": 2,
  "as_of": "2026-09-14T03:45:00Z",
  "postings": [
    {"title": "Senior Data Engineer", "location": "Berlin",
     "salary_raw": "120000 - 160000 USD", "departments": "Eng", "offices": "HQ",
     "n_offices": 1, "n_metadata": 3, "content_chars": 1400,
     "first_published": "2026-08-20T00:00:00Z", "updated_at": "2026-09-01T00:00:00Z"},
    {"title": "Werkstudent Marketing (m/w/d)", "location": "München",
     "content_chars": 600, "first_published": "2026-09-01T00:00:00Z"},
    {"title": "Staff Software Engineer, Infrastructure", "location": "Remote",
     "salary_raw": "200000 - 260000 USD", "content_chars": 2800,
     "first_published": "2026-08-10T00:00:00Z"},
    {"title": "Customer Support Specialist", "location": "Dublin",
     "content_chars": 900, "first_published": "2026-09-05T00:00:00Z"},
    {"title": "Head of Product", "location": "London",
     "departments": "Product", "n_metadata": 5, "content_chars": 2100,
     "first_published": "2026-07-28T00:00:00Z"}
  ]
}
JSON
}
for pass in 1 2; do
  rank_body | curl -fsS --max-time 60 -X POST "${BASE_URL}/rank" \
    -H 'content-type: application/json' -d @- > "${WORK}/rank${pass}.json" \
    || fail "/rank did not answer 2xx (pass ${pass})"
done
python3 - "${WORK}/rank1.json" "${WORK}/rank2.json" "${WORK}/predict.json" <<'PY' || fail "/rank returned an unusable ranking"
import json, sys

first, second, single = (json.load(open(path)) for path in sys.argv[1:])
postings = first.get("postings")
if not isinstance(postings, list) or len(postings) != 5:
    sys.exit(f"sent 5 postings, got back {len(postings) if isinstance(postings, list) else postings!r}")

for item in postings:
    p = item.get("probability")
    if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
        sys.exit(f"a probability is not a probability: {p!r}")

ranks = sorted(item["rank"] for item in postings)
if ranks != list(range(1, 6)):
    sys.exit(f"ranks are not a permutation of 1..5: {ranks}")

by_rank = sorted(postings, key=lambda item: item["rank"])
scores = [item["probability"] for item in by_rank]
if any(a < b for a, b in zip(scores, scores[1:])):
    sys.exit(f"scores are not descending in rank order: {scores}")

watched = [item["rank"] for item in postings if item["watch"]]
if sorted(watched) != [1, 2]:
    sys.exit(f"budget 2 should watch ranks 1 and 2 and nothing else; watched {sorted(watched)}")
if first.get("budget") != 2 or first.get("threshold_source") != "batch_budget":
    sys.exit(f"budget {first.get('budget')!r}, source {first.get('threshold_source')!r}; expected 2 / batch_budget")
if abs(first["threshold_applied"] - scores[1]) > 1e-9:
    sys.exit(f"threshold_applied {first['threshold_applied']} is not the budget-th score {scores[1]}")

if second.get("postings") != postings:
    sys.exit("the same batch ranked differently on a second call — the ranking is not deterministic")

alone = single.get("probability")
through_rank = postings[0]["probability"]
if abs(alone - through_rank) > 1e-9:
    sys.exit(f"posting 1 scores {alone} alone and {through_rank} in the batch — two code paths, two models")

print(f"  5 postings, budget 2, source {first['threshold_source']}, threshold {first['threshold_applied']:.4f}")
print("  ranks: " + ", ".join(f"#{item['rank']} {item['probability']:.4f}{' *' if item['watch'] else ''}" for item in by_rank))
print("  deterministic across two calls; posting 1 matches /predict")
PY
echo "ok  /rank"

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
