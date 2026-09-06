# Deploying

> The decisions are in `docs/design.md` §7 — *why* Cloud Run, why a Hugging Face
> Space, why the model arrives by release tag. This file is the runbook: the
> commands, in order, and what each one is for.

```
python -m src.models.freeze          the model becomes a file
        │
        ▼
gh release create artifact-<date>    the file becomes a version
        │
        ▼
.github/workflows/deploy.yml         the version becomes an image, then a URL
        │
        ▼
scripts/smoke.sh                     the URL becomes a service that answers
        │
        ▼
scripts/cold_start.sh                the service becomes a number in the README
```

Four things must be true before any of it runs, and the workflow's preflight
checks the first two so a misconfiguration fails in ten seconds rather than in
four minutes:

1. The repository has the six deploy variables set (§1).
2. A release tag exists carrying the three artifact assets (§2).
3. The artifact was fitted on the **real** panel, not the synthetic one. The
   smoke test refuses a deploy serving a model whose `dataset` is `synthetic`,
   because a placeholder behind a public URL is indistinguishable from a
   prediction to everyone except the person who built it.
4. `main` is green.

---

## 1. One-time Google Cloud setup

Done once, by hand, from a machine with `gcloud` installed and authenticated
(`gcloud auth login`). Nothing here is secret — a project id is not a
credential — so all of it ends up in repository **variables**, which are
readable in the workflow log and therefore debuggable.

Set the shell variables once and paste the rest:

```bash
export PROJECT_ID=shelf-life-prod          # yours; must have billing enabled
export REGION=europe-west1                 # near you, and near nothing else
export REPO=containers                     # the Artifact Registry repository
export SERVICE=shelf-life
export GH_REPO=professor3333/shelf-life
```

**A billing account is required even though the usage is free.** Cloud Run's
always-free allowance is far above anything a portfolio endpoint will see, but
the project will not accept a deploy without a card on file, and Artifact
Registry storage for a ~1 GB image sits just above the 0.5 GB free tier —
pennies a month, not zero. `docs/design.md` §7b says so at greater length. Check
the current terms rather than trusting that paragraph: free-tier allowances move.

```bash
gcloud config set project "${PROJECT_ID}"

gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  iamcredentials.googleapis.com

# Somewhere to push the image.
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker --location="${REGION}" \
  --description="Container images for shelf-life"
```

### The identity GitHub Actions deploys as

The workflow authenticates **without a key**. GitHub mints a short-lived OIDC
token for each run and Google trades it for a short-lived access token, scoped
to one service account. The alternative — a service-account JSON key pasted into
a repository secret — is a permanent credential living somewhere that gets
copied, and it stays valid long after the laptop that made it is gone.

```bash
gcloud iam service-accounts create github-deployer \
  --display-name="GitHub Actions deployer"

DEPLOYER="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"

# Exactly three permissions: push an image, deploy a revision, and act as the
# service's own runtime identity. Not Editor.
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${DEPLOYER}" --role="${role}"
done

# The pool, and the provider that trusts GitHub's OIDC issuer.
gcloud iam workload-identity-pools create github \
  --location=global --display-name="GitHub"

gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global --workload-identity-pool=github \
  --display-name="GitHub provider" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == '${GH_REPO}'"
```

**The `--attribute-condition` is the load-bearing line.** Without it the
provider trusts *every* repository on GitHub, and any workflow anywhere could
request a token for this service account. With it, only this repository can.

```bash
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
POOL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github"

# Let workflows from this repository impersonate the deployer.
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER}" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/${POOL}/attribute.repository/${GH_REPO}"

echo "${POOL}/providers/github-provider"   # ← the value of GCP_WORKLOAD_IDENTITY_PROVIDER
```

### Tell the repository where all that is

```bash
gh variable set GCP_PROJECT_ID  --body "${PROJECT_ID}"
gh variable set GCP_REGION      --body "${REGION}"
gh variable set GCP_ARTIFACT_REPO --body "${REPO}"
gh variable set GCP_SERVICE     --body "${SERVICE}"
gh variable set GCP_SERVICE_ACCOUNT --body "${DEPLOYER}"
gh variable set GCP_WORKLOAD_IDENTITY_PROVIDER \
  --body "${POOL}/providers/github-provider"

gh variable list
```

---

## 2. Release the artifact

The image does not contain a model until a release does. `models/` is derived
output and is not committed (`docs/design.md` §7a), so the artifact reaches the
build the only way that leaves "which model is serving?" answerable: as a
version.

```bash
python -m src.models.freeze --run <spec>              # opens the test block, once
python -m src.inference.fetch --checksums models      # SHA256SUMS beside the artifact

TAG="artifact-$(date -u +%Y-%m-%d)"
gh release create "${TAG}" \
  models/shelf_life.joblib models/shelf_life.json models/SHA256SUMS \
  --title "Model ${TAG}" \
  --notes "Fitted on <panel snapshot>. Validation and test numbers: reports/test_results.md"
```

Write the checksums **immediately after** freezing and before uploading, so they
describe the file that is actually published rather than one regenerated later
from a panel that has since grown.

---

## 3. Deploy

Pushing the tag is the deploy:

```bash
git push origin "${TAG}"
gh run watch
```

Or re-deploy an existing artifact — after an API fix, say — without cutting a
new model version:

```bash
gh workflow run deploy.yml -f tag=artifact-2026-09-08
```

What the workflow does, in the order it does it: checks the six variables,
checks the release carries all three assets, authenticates keylessly, builds the
image with `--build-arg ARTIFACT_TAG` (which fetches and checksum-verifies the
artifact *during the build*, so a bad artifact fails the build rather than the
stranger's first request), pushes it tagged with both the artifact tag and the
commit SHA, deploys with `--cpu-boost` and `--min-instances=0`, and finally runs
`scripts/smoke.sh` against the live URL.

**The deploy is not the test.** A revision that answers 503 to every caller is a
successful `gcloud run deploy`: the image boots happily with no artifact and
reports `model_loaded: false` on purpose. The smoke step is what turns that into
a failed workflow.

### Verifying by hand

```bash
./scripts/smoke.sh "$(gcloud run services describe "${SERVICE}" \
    --region "${REGION}" --format 'value(status.url)')"
```

### Which model is actually serving

```bash
curl -s "${URL}/health" | python3 -m json.tool     # run name, dataset, threshold
gcloud run services describe "${SERVICE}" --region "${REGION}" \
  --format 'value(spec.template.spec.containers[0].image)'   # → :artifact-<date>
```

Those two must agree. The image tag says which release the artifact came from;
`/health` says which run is loaded in the process that is answering.

---

## 4. Measure the cold start, and publish it

```bash
./scripts/cold_start.sh "${URL}"        # waits 20 minutes, then times one request
```

The deploy cannot measure this: deploying a revision starts an instance to check
it serves, so the first request afterwards finds a warm one. Only a genuinely
idle service gives the number, and the number belongs in the README beside the
caveat — `docs/design.md` §7e, which also says what a figure much past ~20
seconds would argue for.

---

## 5. The UI

The Streamlit form deploys separately, to a Hugging Face Space, and reaches the
model the only way it is allowed to: over HTTP. `deploy/space/` holds the two
files a Space needs that this repository does not.

```bash
hf auth login                                   # or: huggingface-cli login
hf repo create shelf-life --repo-type space --space_sdk streamlit

git clone https://huggingface.co/spaces/<you>/shelf-life /tmp/shelf-life-space
cp -r app /tmp/shelf-life-space/app
cp deploy/space/README.md deploy/space/requirements.txt /tmp/shelf-life-space/

cd /tmp/shelf-life-space && git add -A && git commit -m "UI" && git push
```

Then set one Space secret, `SHELF_LIFE_API`, to the Cloud Run URL (Settings →
Variables and secrets). Without it the UI defaults to `http://localhost:8000`
and shows the error that says so.

Only `app/` is copied. `src/` is not, and that is the deployment enforcing what
a test already asserts: the UI holds no model, so it cannot drift into serving a
different one.

---

## 6. When it goes wrong

| Symptom | What it means | What to do |
|---|---|---|
| Preflight: "Missing repository variables" | §1 was never run, or a name is misspelt | `gh variable list` |
| Preflight: "release has no SHA256SUMS" | the release exists but the assets were not attached | redo §2's `gh release create` |
| Build fails in `src.inference.fetch` | the published bytes do not match the published checksums | re-cut the release; do not force it |
| Build fails in `assert_is_full_pipeline` | the artifact is not a fitted end-to-end pipeline in the serving environment — usually a booster written by a different XGBoost than the image installs | re-freeze in an environment matching the image |
| Smoke: "model_loaded: false" | the image was built without `ARTIFACT_TAG` | check the build step's `--build-arg` |
| Smoke: "fitted on the SYNTHETIC panel" | working as designed | freeze against the real panel first |
| Smoke: malformed payload returned 500 | a validation gap; bad input must be a 4xx | fix `api/schemas.py`, add the case to `tests/test_api.py` |

### Rolling back

Revisions are immutable, so the previous model is still there:

```bash
gcloud run revisions list --service "${SERVICE}" --region "${REGION}"
gcloud run services update-traffic "${SERVICE}" --region "${REGION}" \
  --to-revisions "${SERVICE}-00003-abc=100"
```

That is a traffic change, not a rebuild: it takes seconds and needs no release.

### Turning it off

```bash
gcloud run services delete "${SERVICE}" --region "${REGION}"
gcloud artifacts repositories delete "${REPO}" --location "${REGION}"
```

Deleting the service stops the compute; deleting the repository stops the
storage charge, which is the only line that accrues while nothing is running.
