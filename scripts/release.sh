#!/usr/bin/env bash
# The production chain, as one command, run as far as it can be run from here.
#
#     ./scripts/release.sh --run 05-xgboost_engineered    # the real thing
#     ./scripts/release.sh --rehearse                      # same chain, synthetic artifact
#
#     selected run -> frozen artifact -> SHA256SUMS -> GitHub release
#         -> image built FROM that release (fetch, verify, strict load)
#         -> container up -> await_release -> smoke
#         -> [stops] -> MODEL_TAG commit -> Render rebuild -> CI verifies
#
# Every link up to the bracket runs here, against the real GitHub release and
# the real Dockerfile, so that the only thing left for the platform is the
# same build on a different machine. **It stops before `MODEL_TAG`** on
# purpose: deploying is a commit (`docs/design.md` §7f), and the commit that
# points the public URL at a model is made by a person reading the smoke
# output above it, not by a script that got that far.
#
# `--rehearse` runs the identical chain on the synthetic panel, publishes a
# *prerelease* whose notes say so, and passes `ALLOW_SYNTHETIC=1` to the smoke
# test — which would otherwise refuse the artifact, as it should. Until
# 2026-09-11 no release had ever been created, so `fetch`, the checksum
# verification and the build-time load had only ever met a test double; the
# first rehearsal is what found that the smoke test was asking for a response
# field the API had renamed two days earlier.
#
# Prerequisites: `gh` authenticated, Docker running, the locked environment
# (`uv sync --locked`). Exit codes: 0 chain complete to the bracket · 2 usage
# or prerequisite · 3 the freeze declined (depth, folds, or a dirty tree —
# its own message says which) · anything else, the link that printed last.

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
RUN=""
REHEARSE=0
TAG=""
PORT="${PORT:-8765}"

usage() {
  echo "usage: $0 --run <spec> | --rehearse   [--tag <release-tag>]" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN="${2:-}"; shift 2 ;;
    --rehearse) REHEARSE=1; shift ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    *) usage ;;
  esac
done
if [ "${REHEARSE}" -eq 1 ] && [ -n "${RUN}" ]; then usage; fi
if [ "${REHEARSE}" -eq 0 ] && [ -z "${RUN}" ]; then usage; fi

# --- prerequisites, before anything irreversible ------------------------------
[ -x "${PYTHON}" ] || { echo "no interpreter at ${PYTHON} — run \`uv sync --locked\`" >&2; exit 2; }
command -v gh >/dev/null || { echo "gh is not installed" >&2; exit 2; }
gh auth status >/dev/null 2>&1 || { echo "gh is not authenticated" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "Docker is not running" >&2; exit 2; }
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

if [ "${REHEARSE}" -eq 1 ]; then
  RUN="05-xgboost_engineered"
  DIR="models/rehearsal"
  TAG="${TAG:-artifact-rehearsal-$(date -u +%Y-%m-%d)}"
  FREEZE_ARGS=(--synthetic --artifact "${DIR}/shelf_life.joblib" --out "${DIR}/test_results.md")
  SMOKE_ENV=(ALLOW_SYNTHETIC=1)
  RELEASE_ARGS=(--prerelease --title "Rehearsal ${TAG} — synthetic, do not serve")
  NOTES="A rehearsal of the release chain on the synthetic panel (\`tests/panels.py\`), whose
label is drawn independently of every feature. The artifact loads and answers, and the
number it answers with means nothing. \`scripts/smoke.sh\` refuses it without
\`ALLOW_SYNTHETIC=1\`; \`MODEL_TAG\` must never name it."
else
  DIR="models"
  TAG="${TAG:-artifact-$(date -u +%Y-%m-%d)}"
  FREEZE_ARGS=()
  SMOKE_ENV=()
  RELEASE_ARGS=(--title "Model ${TAG}")
  NOTES="Run \`${RUN}\`, frozen by \`python -m src.models.freeze\` at $(git rev-parse --short HEAD).
Numbers: \`reports/test_results.md\` at that commit. Provenance — panel sha256, snapshot
date, rules version, lock sha256, seed — in \`shelf_life.json\`."
fi

if gh release view "${TAG}" --repo "${REPO}" >/dev/null 2>&1; then
  echo "release ${TAG} already exists — pass --tag to name another, or delete it:" >&2
  echo "  gh release delete ${TAG} --yes --cleanup-tag" >&2
  exit 2
fi

# --- 1. freeze ------------------------------------------------------------------
# On a real panel `freeze` refuses on its own: too shallow (3), no folds (3),
# dirty tree (4). Each is the right answer and this script has nothing to add.
echo "== 1/6 freeze ${RUN}"
mkdir -p "${DIR}"
if ! "${PYTHON}" -m src.models.freeze --run "${RUN}" ${FREEZE_ARGS[@]+"${FREEZE_ARGS[@]}"}; then
  echo "freeze declined; nothing was released." >&2
  exit 3
fi

# --- 2. checksums, written now, beside the file that is about to be published ---
echo "== 2/6 checksums"
"${PYTHON}" -m src.inference.fetch --checksums "${DIR}"

# --- 3. the release -------------------------------------------------------------
echo "== 3/6 release ${TAG} on ${REPO}"
gh release create "${TAG}" \
  "${DIR}/shelf_life.joblib" "${DIR}/shelf_life.json" "${DIR}/SHA256SUMS" \
  --repo "${REPO}" "${RELEASE_ARGS[@]}" --notes "${NOTES}"

# --- 4. the image, from that release and nothing local ---------------------------
# `models/` is in .dockerignore, so the only way the artifact gets in is the
# fetch the Dockerfile makes — against GitHub, verified against SHA256SUMS,
# then loaded under the locked environment with the version check strict.
echo "== 4/6 image from ${TAG}"
IMAGE="shelf-life:${TAG}"
docker image rm -f "${IMAGE}" >/dev/null 2>&1 || true
docker build --build-arg ARTIFACT_TAG="${TAG}" -t "${IMAGE}" . 2>&1 | grep -E "artifact (tag|ok)|libraries:|lock:|verified|ERROR|error:" || true
docker image inspect "${IMAGE}" >/dev/null 2>&1 || { echo "the image did not build" >&2; exit 4; }

# --- 5. the container, and the two checks CI would make against the URL ---------
echo "== 5/6 container on :${PORT}"
NAME="shelf-life-release-$$"
docker run -d --rm -p "${PORT}:8000" --name "${NAME}" "${IMAGE}" >/dev/null
trap 'docker stop "${NAME}" >/dev/null 2>&1 || true' EXIT
./scripts/await_release.sh "http://localhost:${PORT}" "${TAG}" 2
echo "== 6/6 smoke"
env ${SMOKE_ENV[@]+"${SMOKE_ENV[@]}"} ./scripts/smoke.sh "http://localhost:${PORT}" "${TAG}"

# --- what is left, and why it is not done here ---------------------------------
echo
if [ "${REHEARSE}" -eq 1 ]; then
  cat <<MSG
Rehearsal complete: ${TAG} is a prerelease of a synthetic model, the image built from it
by fetch + checksum + strict load, and the container passed await_release and smoke with
ALLOW_SYNTHETIC=1. MODEL_TAG is untouched and the public URL still serves no model.

To remove the rehearsal release:   gh release delete ${TAG} --yes --cleanup-tag
MSG
else
  cat <<MSG
Everything up to the deploy has run and passed. The deploy is a commit, and it is yours:

    printf '%s\n' "${TAG}" > MODEL_TAG
    git add MODEL_TAG reports/test_results.md
    git commit -m "deploy: serve ${TAG}"
    git push

Render rebuilds from the push — the same Dockerfile, the same fetch, the same release —
and .github/workflows/verify-deployment.yml waits for the URL to report ${TAG}, then
runs the smoke test you just watched, against the public URL, without ALLOW_SYNTHETIC.
Then: ./scripts/cold_start.sh <url>, and the README's deployment section.
MSG
fi
