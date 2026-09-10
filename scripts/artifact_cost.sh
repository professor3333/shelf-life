#!/usr/bin/env bash
# What does loading the model add to a cold start? Measured before there is a
# model to freeze.
#
#     ./scripts/artifact_cost.sh                # 3 repeats of each arm
#     ./scripts/artifact_cost.sh --repeats 1    # quicker, and noisier
#     ./scripts/artifact_cost.sh --rebuild      # rebuild the image first
#
# **The gap this fills.** `scripts/cold_start.sh` owns the acceptance criterion,
# and it can only settle it against a real released artifact on the real
# instance — which cannot happen until the panel clears and a model is frozen.
# Until then the only measured figure is the 32.65 s baseline of 2026-09-06,
# taken from the no-artifact image, and `docs/design.md` §7e is explicit that
# such a baseline can fail but cannot pass: reading a fitted pipeline off disk
# and unpickling it on a tenth of a CPU is precisely what the no-artifact image
# never does, and precisely what the criterion exists to bound.
#
# That leaves the largest unknown in the deployment sitting on the critical path
# of a day that happens once. Freeze day opens the test block, spends it, and
# ships. Discovering there that the artifact costs another 60 s is discovering
# it at the worst possible moment, because the documented answer to a blown
# criterion is to reassess the architecture — and by then the architecture is
# already serving the one model this project gets to freeze.
#
# So this measures the missing term now, using the artifact already sitting in
# `models/`, in a container throttled to the free instance's shape.
#
# **What it measures.** One image, started twice: once with no artifact, once
# with `models/` mounted read-only, both under `--cpus 0.1 --memory 512m`, timed
# from `docker run` to the first `/health` that answers. The model is loaded in
# the app's lifespan rather than on first request, so that first answer is
# already on the far side of the unpickle. **The difference between the two arms
# is the number worth having**; each arm is printed only so the difference can
# be audited.
#
# **What it does not measure, and this matters more than what it does.**
#
#   * The absolute figures are not Render's and must never be quoted as if they
#     were. Docker's `--cpus 0.1` is a hard CFS quota; a free instance's "0.1
#     CPU" is a share of a busier machine that bursts differently. The local
#     no-artifact arm runs several times slower than the 32.65 s the real
#     instance actually took, which is the clearest possible evidence that the
#     two scales are not the same scale.
#   * **This does not accept the architecture, and cannot.** `cold_start.sh`
#     against a real release still does that, still on freeze day. What this can
#     do is fail early, which is the same asymmetry the baseline has and the same
#     reason it is worth running.
#   * The artifact in `models/` is fitted on the synthetic fixture. It exercises
#     the real load path with a real pipeline of the real shape, but the frozen
#     model's size is not known until it exists, so the difference below is an
#     estimate of a term, not the term.
#
# **Why the difference survives all of that.** The two arms differ by one thing.
# Both pay the same interpreter start and the same scikit-learn and XGBoost
# imports; only one also reads and unpickles a pipeline. And adding a local
# difference to the measured remote baseline is pessimistic by construction —
# the local CPU is the slower of the two, so the artifact's real cost on the
# instance is smaller than the figure measured here, not larger.
#
# Exit codes: 0 the projection is within the criterion · 5 it is not, and the
# architecture needs the conversation now rather than on freeze day · 2 a
# precondition failed.

set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-shelf-life:cost}"
PORT="${PORT:-18000}"
REPEATS=3
REBUILD=0

#: The acceptance criterion, in seconds. The same 90 as `scripts/cold_start.sh`
#: and the UI's HTTP timeout; `tests/test_deploy.py` asserts all three agree, so
#: the number cannot be softened here to make a projection land.
STOP_RULE_SECONDS="${STOP_RULE_SECONDS:-90}"

#: The measured cold baseline of the real instance, 2026-09-06, no-artifact
#: image, 16 minutes of enforced idle. `docs/design.md` §7e records how it was
#: taken. The projection adds the local difference to THIS, not to a local arm.
REMOTE_BASELINE_SECONDS="${REMOTE_BASELINE_SECONDS:-32.65}"

#: How long a single arm may take before something is considered wrong. Generous
#: on purpose: a tenth of a CPU importing XGBoost is slow enough that an
#: impatient limit here would report the throttle as a failure.
ARM_TIMEOUT_SECONDS="${ARM_TIMEOUT_SECONDS:-400}"

while [ $# -gt 0 ]; do
  case "$1" in
    --repeats) REPEATS="${2:?--repeats needs a number}"; shift 2 ;;
    --rebuild) REBUILD=1; shift ;;
    *) echo "usage: $0 [--repeats N] [--rebuild]" >&2; exit 2 ;;
  esac
done

command -v docker >/dev/null 2>&1 || { echo "docker is not on PATH" >&2; exit 2; }
docker info >/dev/null 2>&1 || { echo "the docker daemon is not running" >&2; exit 2; }

ARTIFACT="models/shelf_life.joblib"
if [ ! -f "${ARTIFACT}" ]; then
  echo "no artifact at ${ARTIFACT} — there is nothing to measure the cost of." >&2
  echo "Build one with \`python -m src.models.freeze\`, or check out a tree that has one." >&2
  exit 2
fi

if [ "${REBUILD}" = "1" ] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Building ${IMAGE} (no artifact tag — the model arrives by mount, not by release)."
  docker build -t "${IMAGE}" . >/dev/null
fi

#: Start the image, wait for the first `/health` that answers, print the seconds.
#: Writes the elapsed time to LAST_SECONDS, and refuses to report a number if the
#: arm did not end up in the state it was supposed to be measuring.
LAST_SECONDS=""
arm() {  # arm <want_model_loaded: yes|no> [docker run args...]
  local want="$1"; shift
  local cid start now body loaded
  cid=$(docker run -d --rm --cpus 0.1 --memory 512m -p "${PORT}:8000" "$@" "${IMAGE}")
  # shellcheck disable=SC2064
  trap "docker kill ${cid} >/dev/null 2>&1 || true" RETURN
  start=$(date +%s.%N)
  while :; do
    body=$(curl -s -m 5 "http://localhost:${PORT}/health" 2>/dev/null || true)
    [ -n "${body}" ] && break
    now=$(date +%s.%N)
    if awk -v a="${now}" -v b="${start}" -v t="${ARM_TIMEOUT_SECONDS}" 'BEGIN{exit !(a-b>t)}'; then
      echo "  the container did not answer /health within ${ARM_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    sleep 0.25
  done
  now=$(date +%s.%N)
  LAST_SECONDS=$(awk -v a="${now}" -v b="${start}" 'BEGIN{printf "%.2f", a-b}')

  case "${body}" in
    *'"model_loaded":true'*)  loaded=yes ;;
    *'"model_loaded":false'*) loaded=no ;;
    *) echo "  /health did not report model_loaded at all: ${body}" >&2; return 1 ;;
  esac
  if [ "${loaded}" != "${want}" ]; then
    # Without this the mount silently failing would be measured as "the artifact
    # is free", which is the one wrong answer this script could produce that
    # nobody would question.
    echo "  arm wanted model_loaded=${want} and got ${loaded} — refusing to time it" >&2
    return 1
  fi
}

#: Median of the numbers on stdin. Median rather than mean because a throttled
#: container occasionally loses a whole scheduling slice, and one such arm should
#: not drag the figure the projection is built on.
median() { sort -n | awk '{v[NR]=$1} END{ if (NR%2) printf "%.2f", v[(NR+1)/2]; else printf "%.2f", (v[NR/2]+v[NR/2+1])/2 }'; }

echo "Cold-start cost of loading the artifact — ${REPEATS} repeat(s) per arm,"
echo "--cpus 0.1 --memory 512m, image ${IMAGE}, artifact $(wc -c < "${ARTIFACT}" | tr -d ' ') bytes."
echo "These absolute numbers are NOT the instance's. Only the difference is portable."
echo

without=""
with=""
for i in $(seq 1 "${REPEATS}"); do
  printf 'run %d/%d  no artifact   ... ' "${i}" "${REPEATS}"
  arm no || exit 1
  printf '%6s s\n' "${LAST_SECONDS}"
  without="${without}${LAST_SECONDS}"$'\n'

  printf 'run %d/%d  with artifact ... ' "${i}" "${REPEATS}"
  arm yes -v "${PWD}/models:/app/models:ro" || exit 1
  printf '%6s s\n' "${LAST_SECONDS}"
  with="${with}${LAST_SECONDS}"$'\n'
done

MED_WITHOUT=$(printf '%s' "${without}" | median)
MED_WITH=$(printf '%s' "${with}" | median)
DIFFERENCE=$(awk -v a="${MED_WITH}" -v b="${MED_WITHOUT}" 'BEGIN{printf "%.2f", a-b}')
PROJECTION=$(awk -v a="${REMOTE_BASELINE_SECONDS}" -v d="${DIFFERENCE}" 'BEGIN{printf "%.2f", a+d}')
HEADROOM=$(awk -v c="${STOP_RULE_SECONDS}" -v p="${PROJECTION}" 'BEGIN{printf "%.2f", c-p}')

echo
printf '  %-42s %8s s\n' "median, no artifact (local, throttled)" "${MED_WITHOUT}"
printf '  %-42s %8s s\n' "median, with artifact (local, throttled)" "${MED_WITH}"
printf '  %-42s %8s s\n' "what the artifact costs (the difference)" "${DIFFERENCE}"
echo
printf '  %-42s %8s s\n' "measured remote baseline, 2026-09-06" "${REMOTE_BASELINE_SECONDS}"
printf '  %-42s %8s s\n' "pessimistic projection (baseline + diff)" "${PROJECTION}"
printf '  %-42s %8s s\n' "the criterion" "${STOP_RULE_SECONDS}"
printf '  %-42s %8s s\n' "headroom" "${HEADROOM}"
echo

if awk -v p="${PROJECTION}" -v c="${STOP_RULE_SECONDS}" 'BEGIN{exit !(p>c)}'; then
  echo "OVER: the projection exceeds the criterion. This is the early failure the"
  echo "script exists to produce — the architecture needs reassessing now, while"
  echo "the test block is still unspent, and not by widening the timeout."
  exit 5
fi

echo "WITHIN the criterion, and this does NOT accept the architecture."
echo "The projection is pessimistic — the local CPU is the slower of the two — so"
echo "the real cost of the artifact should come in under the difference above."
echo "The measurement that decides is \`scripts/cold_start.sh\` against a real"
echo "release, on freeze day. What this run buys is the knowledge that freeze day"
echo "is unlikely to be the day the hosting decision falls over."
