#!/usr/bin/env bash
# How long does the first request wait when nothing is running?
#
#     ./scripts/cold_start.sh https://shelf-life-xxxx.onrender.com          # 3 cycles
#     REPEATS=1 ./scripts/cold_start.sh <url>                               # one, noisier
#     ./scripts/cold_start.sh <url> 20                                      # 20 idle minutes
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
# **Repeated, because one sample is an anecdote.** Each cycle waits the full
# idle period, so three cycles cost about fifty minutes — and that is the
# price of a number with a spread. The criterion is applied to the WORST
# request of the WORST cycle, not the median: the stranger who gets the slow
# one does not experience the median.
#
# **Decomposed, because a number without a cause cannot be acted on.** Beside
# the outside timing the script prints what `/health` says the process cost
# itself: `ready_after_seconds` (process start to model ready), `load_seconds`
# (the unpickle alone) and `rss_mb` (peak memory, against the instance's
# 512 MB). Platform wake is the remainder. `api/runtime.py`.
#
# **Recorded, not remembered.** Every cycle is written to a report — the
# definitive run to `reports/cold_start.md`, a baseline to
# `reports/cold_start_baseline.md` — so the acceptance is a committed table,
# not a sentence in a README quoting one figure.
#
# Preconditions, and the run is worthless without them:
#   * nothing is keeping the service awake — there is deliberately no keep-warm
#     cron (§7e explains the 750-hour arithmetic that rules one out)
#   * no other traffic during the wait — a browser tab left open on the UI, or a
#     second person clicking the link, warms the container and turns this into a
#     measurement of a warm start.

set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${1:-${SHELF_LIFE_API:-}}"
if [ -z "${BASE_URL}" ]; then
  echo "usage: $0 <base-url> [idle-minutes]   (or set SHELF_LIFE_API; REPEATS=n, OUT=path)" >&2
  exit 2
fi
BASE_URL="${BASE_URL%/}"

# Render spins a free service down after 15 idle minutes. 16 is the default here
# so the wait clears that rather than racing it.
IDLE_MINUTES="${2:-16}"
REPEATS="${REPEATS:-3}"

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

WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT
CYCLES="${WORK}/cycles.jsonl"
: > "${CYCLES}"

#: The most recent measurement, in seconds. `timed` writes it; the caller copies
#: it out immediately, because every later call overwrites it.
LAST_SECONDS=""

timed() {  # timed <label> <curl args...> -> seconds, and fails loudly on a non-2xx
  local label="$1"; shift
  local result
  result=$(curl -fsS -o /dev/null -w '%{time_total} %{http_code}' --max-time 180 "$@") \
    || { echo "  ${label}: FAILED (the service did not answer within 180s)" >&2; return 1; }
  printf '  %-28s %6.2f s   HTTP %s\n' "${label}" "${result% *}" "${result#* }"
  LAST_SECONDS="${result% *}"
}

health_field() {  # health_field <json-file> <field> -> value or empty
  python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"
}

predict_payload='{"title": "Senior Data Engineer", "location": "Berlin",
          "content_chars": 1400, "first_published": "2026-08-20T00:00:00Z",
          "as_of": "2026-09-14T03:45:00Z"}'
# /rank is the shape the operating point was designed for (README), and its
# first call is timed separately: a batch is not one posting three times.
rank_payload='{"budget": 1, "as_of": "2026-09-14T03:45:00Z", "postings": [
          {"title": "Senior Data Engineer", "location": "Berlin", "content_chars": 1400,
           "first_published": "2026-08-20T00:00:00Z"},
          {"title": "Werkstudent Marketing", "location": "Munich", "content_chars": 600,
           "first_published": "2026-09-01T00:00:00Z"},
          {"title": "Staff Software Engineer, Infrastructure", "location": "Remote",
           "content_chars": 2800, "first_published": "2026-08-10T00:00:00Z"}]}'

# Which of the two measurements is this? The answer is not a flag the caller
# passes, because a caller who has to remember which kind of run this is will
# eventually record a baseline as if it were the definitive one.
#
#   BASELINE   — the no-artifact image. Process start, interpreter, and the
#                scikit-learn and XGBoost imports. A genuine measurement of a
#                real instance, and a LOWER BOUND: it never touches joblib, never
#                unpickles a pipeline or a booster, and /predict answers 503
#                without reaching the model.
#   REHEARSAL  — an image carrying a model fitted on the SYNTHETIC panel. The
#                load path runs for real — same libraries, an artifact of the
#                same shape — so the unpickle cost is measured; but the
#                criterion is about the artifact that ships, and this is not
#                it. Measures, records, accepts nothing.
#   DEFINITIVE — the same image with a released artifact fitted on the real
#                panel. Adds the load path the baseline omits, and is the only
#                measurement that can accept the architecture.
#
# `docs/design.md` §7e: the architecture is not accepted until the DEFINITIVE
# measurement is within the criterion. Asked once, up front, so the wait is not
# spent on a run whose kind is unknown; asked again after every cold request,
# because a deploy landing mid-run would otherwise mix the two.
curl -fsS --max-time 180 "${BASE_URL}/health" > "${WORK}/kind.json"
MODEL_LOADED=$(curl -fsS --max-time 180 "${BASE_URL}/health" \
  | python3 -c 'import json,sys; print("yes" if json.load(sys.stdin).get("model_loaded") else "no")')
DATASET=$(health_field "${WORK}/kind.json" dataset)
# The process that is answering now. A cold cycle is one answered by a
# *different* process — the platform started a new container — and
# `ready_after_seconds` is fixed for the life of a process, so a cycle whose
# value matches the previous one never went cold: something kept the service
# awake through the wait (a fresh deploy, a browser tab, another poller) and
# the number it produced is a warm request wearing a cold label. Measured
# 2026-09-11: 0.34 s "cold" right after a rebuild. Such cycles are kept in the
# report, marked, and excluded from the verdict.
PREVIOUS_PROCESS=$(health_field "${WORK}/kind.json" ready_after_seconds)
if [ "${MODEL_LOADED}" != "yes" ]; then KIND="BASELINE"
elif [ "${DATASET}" != "real" ]; then KIND="REHEARSAL"
else KIND="DEFINITIVE"; fi
echo "Measuring ${BASE_URL}: ${KIND} (model_loaded=${MODEL_LOADED}, dataset=${DATASET:-none})"
echo "${REPEATS} cycle(s) of ${IDLE_MINUTES} idle minutes each; criterion ${STOP_RULE_SECONDS} s."

for cycle in $(seq 1 "${REPEATS}"); do
  echo
  echo "== cycle ${cycle}/${REPEATS}: waiting ${IDLE_MINUTES} minutes for the service to scale to zero."
  echo "   Send it nothing during the wait — including opening the UI."
  if [ "${IDLE_MINUTES}" != "0" ]; then
    sleep $((IDLE_MINUTES * 60))
  fi

  echo "Cold — the first request after ${IDLE_MINUTES} idle minutes:"
  timed "/health (cold)" "${BASE_URL}/health"
  cold="${LAST_SECONDS}"
  curl -fsS --max-time 60 "${BASE_URL}/health" > "${WORK}/health.json"
  loaded_now=$([ "$(health_field "${WORK}/health.json" model_loaded)" = "True" ] && echo yes || echo no)
  if [ "${loaded_now}" != "${MODEL_LOADED}" ]; then
    echo "the service changed kind mid-run (model_loaded flipped) — a deploy landed. Start over." >&2
    exit 1
  fi
  timed "/health (warm)" "${BASE_URL}/health"

  predict_first=""; predict_warm=""; rank_first=""; rank_warm=""
  if [ "${MODEL_LOADED}" = "yes" ]; then
    echo "First prediction — the load path the baseline cannot reach:"
    timed "/predict (first)" -X POST "${BASE_URL}/predict" \
      -H 'content-type: application/json' -d "${predict_payload}"
    predict_first="${LAST_SECONDS}"
    timed "/predict (warm)" -X POST "${BASE_URL}/predict" \
      -H 'content-type: application/json' -d "${predict_payload}"
    predict_warm="${LAST_SECONDS}"
    timed "/rank (first, 3 postings)" -X POST "${BASE_URL}/rank" \
      -H 'content-type: application/json' -d "${rank_payload}"
    rank_first="${LAST_SECONDS}"
    timed "/rank (warm, 3 postings)" -X POST "${BASE_URL}/rank" \
      -H 'content-type: application/json' -d "${rank_payload}"
    rank_warm="${LAST_SECONDS}"
  else
    echo "No model loaded, so /predict and /rank answer 503 and are not timed."
  fi

  ready=$(health_field "${WORK}/health.json" ready_after_seconds)
  load=$(health_field "${WORK}/health.json" load_seconds)
  rss=$(health_field "${WORK}/health.json" rss_mb)
  went_cold=yes
  if [ -n "${ready}" ] && [ "${ready}" = "${PREVIOUS_PROCESS}" ]; then
    went_cold=no
    echo "  NOT COLD: the same process answered as before the wait — the service never spun down." >&2
    echo "  Something kept it awake (a deploy just landed? a tab open on the UI?). Cycle excluded." >&2
  fi
  PREVIOUS_PROCESS="${ready}"
  printf '  %-28s ready after %.2f s, of which unpickle %.2f s; peak RSS %.0f MB\n' \
    "inside the process:" "${ready:-0}" "${load:-0}" "${rss:-0}"

  python3 - "${CYCLES}" "${cycle}" "${cold}" "${predict_first}" "${predict_warm}" \
      "${rank_first}" "${rank_warm}" "${ready}" "${load}" "${rss}" "${WORK}/health.json" "${went_cold}" <<'PY'
import json, sys
path, cycle, cold, pf, pw, rf, rw, ready, load, rss, health, went_cold = sys.argv[1:]
f = lambda v: float(v) if v else None
h = json.load(open(health))
row = {"cycle": int(cycle), "cold_health": f(cold), "predict_first": f(pf),
       "predict_warm": f(pw), "rank_first": f(rf), "rank_warm": f(rw),
       "ready_after": f(ready), "load": f(load), "rss_mb": f(rss),
       "artifact_tag": h.get("artifact_tag"), "model": h.get("model"),
       "dataset": h.get("dataset"), "model_loaded": bool(h.get("model_loaded")),
       "cold": went_cold == "yes"}
open(path, "a").write(json.dumps(row) + "\n")
PY
done

# --- the report, then the criterion --------------------------------------------
#
# Applied to the SLOWEST request of the SLOWEST cycle, not to the wake alone
# and not to a median. The UI's timeout is per request and guards every
# request, so a /health that wakes in 40 s followed by a /predict that takes
# 100 s is a failure even though the wake looked fine.
case "${KIND}" in
  DEFINITIVE) OUT="${OUT:-reports/cold_start.md}" ;;
  REHEARSAL)  OUT="${OUT:-reports/cold_start_rehearsal.md}" ;;
  *)          OUT="${OUT:-reports/cold_start_baseline.md}" ;;
esac
# The script's own version goes in the report's `Code` row — the report is
# about the service, but the way it was measured is this checkout.
GIT_SHA=$(git rev-parse --short=9 HEAD 2>/dev/null || echo "unknown")
GIT_STATE=$([ -z "$(git status --porcelain 2>/dev/null)" ] && echo clean || echo "dirty tree")
verdict=$(python3 - "${CYCLES}" "${OUT}" "${KIND}" "${BASE_URL}" "${IDLE_MINUTES}" "${STOP_RULE_SECONDS}" "${GIT_SHA}" "${GIT_STATE}" <<'PY'
import json, statistics, sys
from datetime import date

cycles_path, out, kind, url, idle, stop, sha, state = sys.argv[1:]
stop = float(stop)
rows = [json.loads(line) for line in open(cycles_path) if line.strip()]
cold_rows = [r for r in rows if r.get("cold", True)]
if not cold_rows:
    print("NO COLD CYCLE: the service never spun down during any wait; nothing here is a cold start",
          file=sys.stderr)
    sys.exit(2)

def worst_of(row):
    return max(v for v in (row["cold_health"], row["predict_first"], row["rank_first"]) if v is not None)

worst = max(worst_of(r) for r in cold_rows)
median_cold = statistics.median(r["cold_health"] for r in cold_rows)
first = rows[0]

def cell(v, unit=" s"):
    return "—" if v is None else f"{v:.2f}{unit}"

lines = [
    f"# Cold start — {kind.lower()} measurement",
    "",
    "| | |",
    "|---|---|",
    "| Generated by | `scripts/cold_start.sh` |",
    f"| Data | the live service `{url}` · release `{first['artifact_tag'] or '(none — no model)'}`"
    f" · model `{first['model'] or '—'}` on the `{first['dataset'] or '—'}` panel |",
    f"| Code | `{sha}` ({state}) — the checkout that measured, not the image measured |",
    f"| Measured on | {date.today().isoformat()} |",
    f"| Cycles | {len(rows)}, each after {idle} idle minutes; {len(cold_rows)} went cold"
    f"{'' if len(cold_rows) == len(rows) else ' — the rest are marked and excluded'} |",
    f"| Criterion | {stop:.0f} s, applied to the slowest request of the slowest cycle |",
    "",
    "_Regenerate rather than edit._",
    "",
]
if kind == "BASELINE":
    lines += [
        "**This is a lower bound and accepts nothing.** The image carries no artifact,",
        "so no cycle here unpickled a pipeline or a booster, and `/predict` and `/rank`",
        "answered 503 without reaching the model. `docs/design.md` §7e: a baseline over",
        "the criterion condemns the architecture; a baseline within it implies nothing.",
        "",
    ]
elif kind == "REHEARSAL":
    lines += [
        "**This is a rehearsal and accepts nothing.** The model loaded here was fitted on",
        "the synthetic panel, so the load path ran for real — the unpickle column below is a",
        "measurement, not an estimate — but the criterion is about the artifact that ships,",
        "and this is not it. `docs/design.md` §7e.",
        "",
    ]
else:
    lines += [
        "**This is the definitive measurement** — a real released artifact, loaded on the",
        "real instance, and used by `/predict` and `/rank`. The criterion in `docs/design.md`",
        "§7e is applied to the worst request of the worst cycle below.",
        "",
    ]
lines += [
    "## Every cycle",
    "",
    "Outside timings are what a visitor waits; the last three columns are what the",
    "process says it cost itself (`/health`), so the platform wake is the difference",
    "between the cold `/health` and *ready after*.",
    "",
    "| cycle | cold `/health` | first `/predict` | warm `/predict` | first `/rank` (3) | warm `/rank` (3) | ready after | of which unpickle | peak RSS |",
    "|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    label = str(r["cycle"]) if r.get("cold", True) else f"{r['cycle']} — **not cold**, excluded"
    lines.append(
        f"| {label} | {cell(r['cold_health'])} | {cell(r['predict_first'])} | "
        f"{cell(r['predict_warm'])} | {cell(r['rank_first'])} | {cell(r['rank_warm'])} | "
        f"{cell(r['ready_after'])} | {cell(r['load'])} | {cell(r['rss_mb'], ' MB')} |"
    )
lines += [
    "",
    "## Verdict",
    "",
    f"- Worst request, worst cycle: **{worst:.2f} s** against {stop:.0f} s",
    f"- Median cold `/health`: {median_cold:.2f} s over {len(cold_rows)} cold cycle(s)",
]
if len(cold_rows) < len(rows):
    lines.append(
        f"- {len(rows) - len(cold_rows)} cycle(s) never went cold — the same process answered "
        "before and after the wait — and are not counted. A cold start is a new process."
    )
if worst > stop:
    lines.append(f"- **STOP** — over the criterion. §7e: reassess the architecture, do not raise the timeout.")
elif kind == "BASELINE":
    lines.append("- Within the criterion, and **that accepts nothing** — the unpickle is not in this number.")
elif kind == "REHEARSAL":
    lines.append("- Within the criterion, and **that accepts nothing** — the artifact is synthetic, not the one that ships.")
else:
    lines.append("- **ACCEPTED** — within the criterion with a real artifact, loaded and used, repeatedly.")
lines.append("")
open(out, "w").write("\n".join(lines))
print(f"{worst:.2f} {median_cold:.2f} {stop:.0f} {len(cold_rows)}")
sys.exit(1 if worst > stop else 0)
PY
) && within=0 || within=$?
if [ "${within}" -eq 2 ]; then
  echo "no cycle went cold: the service was kept awake through every wait. Nothing measured." >&2
  exit 2
fi
set -- ${verdict}
worst_s="$1"; median_s="$2"; stop_s="$3"; n="$4"

echo
echo "wrote -> ${OUT}"
if [ "${within}" -ne 0 ]; then
  echo "STOP: ${worst_s} s exceeds the ${stop_s} s acceptance criterion (worst of ${n} cycles)." >&2
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
  echo "The report above says which term to look at first." >&2
  exit 1
fi

if [ "${MODEL_LOADED}" != "yes" ]; then
  echo "BASELINE recorded: worst ${worst_s} s, median cold ${median_s} s, within the ${stop_s} s criterion."
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

if [ "${KIND}" = "REHEARSAL" ]; then
  echo "REHEARSAL recorded: worst ${worst_s} s, median cold ${median_s} s, within the ${stop_s} s criterion."
  echo
  echo "This does NOT accept the deployment architecture. The load path ran — the"
  echo "unpickle column in ${OUT} is a real measurement of a real artifact of the"
  echo "same shape — but the model was fitted on the synthetic panel, and the"
  echo "criterion is about the one that ships. Re-run against the real release."
  exit 0
fi

echo "ACCEPTED: worst ${worst_s} s over ${n} cycles is within the ${stop_s} s acceptance criterion."
echo "  median cold /health ${median_s} s"
echo
echo "This is the definitive measurement — a real artifact, loaded and used,"
echo "repeatedly. Commit ${OUT}; the README links to it rather than quoting it."
