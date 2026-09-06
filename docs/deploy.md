# Deploying

> The decisions are in `docs/design.md` §7 — why Render, why Streamlit Community
> Cloud, what was verified on 2026-09-06 and what that overturned. This file is
> the runbook: the commands, in order, and what each one is for.

**Everything here is free and stays free.** No payment method, no billing
account, no card on either platform. That is a hard constraint, not a
preference, and §7b explains why Render was chosen partly *for its failure
mode*: with no card on file it suspends a service that exceeds a limit rather
than billing for the overage, because there is nothing to bill.

```
python -m src.models.freeze        the model becomes a file
        │
        ▼
gh release create artifact-<date>  the file becomes a version
        │
        ▼
echo <tag> > MODEL_TAG ; git push  the version becomes the deployed one
        │
        ▼
Render rebuilds, by itself         no deploy credential anywhere in CI
        │
        ▼
Verify deployment workflow         await_release.sh, then smoke.sh
        │
        ▼
scripts/cold_start.sh              the number the README owes a visitor
```

Two things must be true before any of it matters, and the workflow checks both
and exits **green** if either is missing, because a check that fails on every
push before the service exists is a check people learn to ignore:

1. The repository variable `SHELF_LIFE_API` names the deployed API.
2. `MODEL_TAG` names a release. Until the panel clears its depth gate there is
   no model to release, the file is deliberately empty, and the deployed service
   is deliberately model-less.

---

## 1. One-time setup

### The API — a Render web service

No CLI, no key, no local Docker. Render reads `render.yaml` from the repository.

1. Sign up at <https://render.com> with the GitHub account. **Do not add a
   payment method.** Its absence is what makes the $0 guarantee mechanical
   rather than aspirational.
2. **New → Blueprint**, pick this repository. Render finds `render.yaml` and
   proposes one free web service named `shelf-life`.
3. Apply. The first build takes a while — it installs scikit-learn and XGBoost
   into a ~1 GB image on a shared builder.
4. Copy the service URL (`https://shelf-life-<something>.onrender.com`) and tell
   the repository about it:

   ```bash
   gh variable set SHELF_LIFE_API --body "https://shelf-life-xxxx.onrender.com"
   ```

The service is public and unauthenticated by design: the point of the build is
that a stranger can POST to it.

**What you get, and it is worth knowing before the demo rather than during it:**
0.1 CPU, 512 MB of memory, a spin-down after 15 idle minutes, and about a minute
to come back. Render's own documentation says of these instances: *"Do not use
them for production applications."* That is the correct expectation. This is a
portfolio demonstration and the failure mode is a slow first request.

### The UI — Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with the same GitHub account. No card.
2. **Create app → deploy from a repo**, with:

   | Field | Value |
   |---|---|
   | Repository | `professor3333/shelf-life` |
   | Branch | `main` |
   | Main file path | `app/streamlit_app.py` |

3. In **Advanced settings → Secrets**, point the UI at the API:

   ```toml
   SHELF_LIFE_API = "https://shelf-life-xxxx.onrender.com"
   ```

Dependencies come from `requirements.txt` at the repository root, which installs
Streamlit, requests and pandas — **and nothing that could load a model**. That
is deliberate: Community Cloud checks out the whole repository, so `app/` sits
next to `src/` and could import it. The dependency list and `tests/test_app.py`
are what stop that. `docs/design.md` §7c records this as a downgrade from the
previous arrangement, where the UI was deployed without `src/` present at all.

Without the secret the UI defaults to `http://localhost:8000` and shows the
error saying so.

---

## 2. Release a model

The image contains no model until a release exists. `models/` is derived output
and is not committed (`docs/design.md` §7a), so the artifact reaches the build
the only way that leaves "which model is serving?" answerable: as a version.

```bash
python -m src.models.freeze --run <spec>              # opens the test block, once
python -m src.inference.fetch --checksums models      # SHA256SUMS beside the artifact

TAG="artifact-$(date -u +%Y-%m-%d)"
gh release create "${TAG}" \
  models/shelf_life.joblib models/shelf_life.json models/SHA256SUMS \
  --title "Model ${TAG}" \
  --notes "Fitted on <panel snapshot>. Numbers: reports/test_results.md"
```

Write the checksums **immediately after** freezing and before uploading, so they
describe the file that is actually published rather than one regenerated later
from a panel that has since grown.

---

## 3. Deploy

Deploying is a commit. There is no deploy command, no CLI, and no credential.

```bash
printf '%s\n' "${TAG}" > MODEL_TAG
git add MODEL_TAG && git commit -m "deploy: serve ${TAG}"
git push
```

Render rebuilds on the push. The build reads `MODEL_TAG`, downloads that
release's assets, **verifies them against the `SHA256SUMS` published with the
release**, and then loads the artifact to prove it is a fitted end-to-end
pipeline in the environment that will serve it. A bad artifact fails the build
rather than a stranger's first request, and a failed build leaves the previous
revision serving.

`MODEL_TAG` may carry `#` comments; the first non-comment, non-blank line is the
tag. An effectively empty file means "no model", and the image built from it
starts, reports `model_loaded: false`, and answers 503 — deliberately, because a
container that refuses to boot over a missing file turns a one-line diagnosis
into a log-reading exercise.

### What the CI workflow then does

`.github/workflows/verify-deployment.yml` does **not** deploy. It waits for the
URL to report the release this commit named, then smoke-tests it:

```bash
./scripts/await_release.sh "$SHELF_LIFE_API" "$TAG" 20
./scripts/smoke.sh         "$SHELF_LIFE_API" "$TAG"
```

Both are runnable by hand against any URL. The wait exists because the build
takes minutes, during which the service is up, healthy, and answering **from the
previous model** — a smoke test run inside that window passes and proves nothing.

The smoke test fails the deployment unless a model is loaded, the response
carries the threshold it was compared against, a malformed payload still gets a
422, the release serving is the expected one, and the model was **not** fitted on
the synthetic panel. `ALLOW_SYNTHETIC=1` overrides the last check for a
deliberate rehearsal.

### Which model is actually serving

```bash
curl -s "$SHELF_LIFE_API/health" | python3 -m json.tool
```

`artifact_tag` is the release the running image was built from; `model` is the
experiment that was frozen; `dataset` says whether that run saw the real panel or
the synthetic one. Three different questions, three fields.

---

## 4. Measure the cold start, and publish it

```bash
./scripts/cold_start.sh "$SHELF_LIFE_API" 16
```

16 minutes of idle, because Render spins down after 15. The CI workflow cannot
produce this number — it runs right after a rebuild, when the service is warm —
and `docs/design.md` §7e is explicit that a cold start nobody has timed is a
surprise being saved up for whoever is being shown the link.

Two multipliers make the estimate untrustworthy until it is measured: the
container's start is import-and-unpickle, which is pure CPU, and the free
instance has **0.1** of one.

**The script applies the acceptance criterion and exits non-zero past 90
seconds.** That is a stop rule, not a warning: §7e says the architecture gets
reassessed rather than tuned around, and the tuning knob — widening the UI's
timeout until the slow service stops timing out — is blocked by
`tests/test_deploy.py`, which fails if `app/client.py`'s timeout is raised above
the criterion. Both numbers can be changed together, and that is a decision with
a diff and a design-doc entry.

**Two measurements, and only the second one accepts anything.**

```
now                                    when the 7-wave gate clears
 │                                      │
 ├─ deploy the no-artifact image        ├─ MODEL_TAG names a release
 ├─ /health reports model_loaded: false ├─ the image is rebuilt with it
 └─ ./scripts/cold_start.sh  ──────────>└─ ./scripts/cold_start.sh
        BASELINE — a lower bound               DEFINITIVE — the acceptance
        (never unpickles anything)              measurement
```

Take the baseline **now**, before the panel clears. It needs no model: starting
the process and importing scikit-learn and XGBoost is expensive whether or not an
artifact loads, and it is a real measurement of a real instance, so it can
already fail — a baseline over the criterion can only get worse once a model is
added, and learning that today costs nothing.

What it cannot do is pass. That image never opens joblib, never unpickles a
pipeline or a booster, and `/predict` answers 503 without reaching the model, so
the load cost on 0.1 of a CPU is precisely what is missing from the figure. The
script says so: it asks `/health` which kind of deployment it is talking to and
labels the run `BASELINE` or `ACCEPTED` accordingly, rather than trusting whoever
ran it to remember.

**The deployment architecture is not accepted until the definitive measurement
comes back within the criterion.** Until then it is provisional.

---

## 5. When it goes wrong

| Symptom | What it means | What to do |
|---|---|---|
| Workflow notice: "No SHELF_LIFE_API variable set" | the service does not exist yet, or was never recorded | §1, then `gh variable set` |
| Workflow notice: "MODEL_TAG names no release" | working as designed; no model has been frozen | nothing, until the panel clears |
| `await_release.sh` times out | the build failed, so the old revision is still serving | read Render's build log |
| Build fails in `src.inference.fetch` | the published bytes do not match the published checksums | re-cut the release; do not force it |
| Build fails in `assert_is_full_pipeline` | the artifact is not a fitted pipeline in the serving environment — usually a booster written by a different XGBoost than the image installs | re-freeze in an environment matching the image |
| Smoke: "model_loaded: false" | `MODEL_TAG` was empty when the image was built | check the build log's `artifact tag:` line |
| Smoke: "fitted on the SYNTHETIC panel" | working as designed | freeze against the real panel first |
| Smoke: malformed payload returned 500 | a validation gap; bad input must be a 4xx | fix `api/schemas.py`, add the case to `tests/test_api.py` |
| First request hangs, then works | the free tier spun down; this is the 15-minute idle behaviour | expected — and the UI says so on screen |
| Service suspended until next month | the 750 monthly instance-hours ran out | wait, or stop pinging it; **no charge can occur** |

### Rolling back

A rollback is a commit, which is the upside of putting the tag in the repository:

```bash
printf '%s\n' "artifact-2026-09-07" > MODEL_TAG    # the previous release
git commit -am "deploy: roll back to artifact-2026-09-07" && git push
```

Render rebuilds and the old model is serving again. Slower than a traffic split,
and it leaves the reason in the history where the next person can read it.

### Turning it off

Delete the service in Render's dashboard and the app in Streamlit Community
Cloud. Neither leaves anything behind that accrues, because neither could
charge for it in the first place.
