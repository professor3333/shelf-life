# Design decisions

The decisions this project is built on, each with a date, the reasoning, and
what would change my mind. Where a decision is not yet made, it says so rather
than pretending a default is a choice.

The learning problem itself is defined in
[`problem_definition.md`](problem_definition.md); this file records the
decisions *around* it — what is fixed, what is still open, and why.

**Status key.** **DECIDED** — settled, with reasons. **OPEN** — not settled;
the entry states the options and what evidence would resolve it.

---

## 1. The prediction target — **DECIDED 2026-09-04**

Will a posting be removed from the board within 7 days of an observation of it.

Full statement in [`problem_definition.md`](problem_definition.md) §1–§4. In
brief: one row is a (posting, complete-run observation) pair; the label is
absence from two consecutive complete runs with no later reappearance; rows
whose horizon has not elapsed are dropped, never labelled 0.

*Rejected:* total lifetime regression — the panel observes neither end of a life
for 1,135 of 1,240 postings. Salary-band prediction — trains on a non-random 25%
whose coverage is confounded with source, which is a better second lesson.
Seniority classification — labels come from the title, which is where the
features come from.

**Would change my mind:** evidence that removal is dominated by board
housekeeping rather than hiring activity, which would make the target real but
uninteresting.

---

## 2. The horizon `H` — **DECIDED 2026-09-04**

**H = 7 days**, with H = 1 retained as a pipeline smoke test only.

Chosen against the measured hazard, not by taste. Across complete runs at
`rules_version = 2`, 77 disappearances in 4,549 job-day transitions — **1.69%
per day**:

| Run date | Present in previous complete run | Absent | Rate |
|---|---|---|---|
| 2026-09-01 | 1,104 | 20 | 1.81% |
| 2026-09-02 | 1,143 | 21 | 1.84% |
| 2026-09-03 | 1,143 | 14 | 1.22% |
| 2026-09-04 | 1,159 | 22 | 1.90% |

| Horizon | Implied positive rate | Comment |
|---|---|---|
| 1 day | 1.7% | labelable today, but dominated by ±1-run timing noise |
| **7 days** | **≈ 11.3%** | matches the decision it feeds — "apply this week or not" |
| 14 days | ≈ 21.3% | better balanced; costs another week of censoring at each end |

**Caveat on the 11.3%.** It is `1 - (1 - 0.0169)^7`, which assumes the daily
hazard is constant in age. §7 of the problem definition asserts the opposite —
that duration dependence carries most of the signal — so the two cannot both be
exactly right. 11.3% is a planning estimate, to be replaced with the measured
7-day rate once the first cohort settles on or about 2026-09-08.

**Would change my mind:** a measured 7-day rate far from 11%, or a hazard curve
steep enough in the first week that a 7-day window averages away the signal.

---

## 3. Censoring and left truncation — **DECIDED 2026-09-04**

**Right censoring: excluded, never zeroed.** A row whose horizon extends past
the last complete run has not survived; we have not looked yet. Filling those
with 0 biases every estimate toward "postings last forever".

**Left truncation: dissolved by the unit of analysis.** 1,135 of 1,240 postings
were already on the board at their source's first complete run, at a mean age of
82.6 days (max 861). Conditioning on survival-to-observation makes age a feature
instead of a missing outcome — the standard discrete-time hazard formulation.

**Age is measured from `first_published`, not `first_seen`.** `first_seen` is
when this project first looked, which for 1,135 postings is an artefact of when
collection started; a model given it learns the scraper's start date.

---

## 4. Is `source` a feature? — **OPEN**

Not decided. It cannot be decided until the deployment story is, and the
deployment story is genuinely ambiguous here.

**The case against.** `source` is the strongest signal in the data and much of
its strength is instrumental rather than about jobs: missingness fingerprints
the source almost perfectly (`remote` is populated for 100% of arbeitnow and 0%
of every Greenhouse board), and per-board hazard varies. If the intended use is
"score a posting from a board we have never scraped", a model leaning on
`source` has learned nothing transferable.

**The case for.** Every labelled row comes from a `greenhouse:*` board, so
within the trainable population `source` is really *which employer's board* —
closer to a company covariate than an instrument. And the primary use in §8 is
ranking tonight's postings from boards already being collected, where the board
identity is known at prediction time and is a legitimate input.

**What resolves it:** state the deployment scenario first. If it is "rank
postings from the boards I already collect", `source` is admissible and should
be reported with and without. If it is "generalise to a new board", `source` is
excluded and per-source metrics become the headline, not a breakdown.

**Provisional handling until decided:** train both, report per-source metrics
either way. §7's acceptance bar compares against a per-board hazard baseline
precisely so that a model which has only learned the board is visible as such.

**Widened 2026-09-04, by the column audit.** The question as posed above — "is
`source` a feature?" — cannot be answered one column at a time, because
excluding `source` alone excludes nothing:

- **`company` is a lossless re-encoding of it.** Each of the six Greenhouse
  boards has exactly one company, and all 31 companies in the frame map to
  exactly one source.
- **`url` carries it in the domain.**
- **Archive-derived missingness carries it.** `first_published`, `updated_at`,
  `departments`, `requisition_id`, `n_metadata` and `content_chars` are null on
  exactly the 127 python_org rows and present on every Greenhouse row, because
  the archive covers Greenhouse only. Any missingness indicator over them
  reconstructs part of `source` for free.

So the decision is **"is board identity a feature?"**, and whichever way it goes
it has to be applied to four columns and one missingness pattern together. See
[`leakage_audit.md`](leakage_audit.md).

---

## 5. The metric and the cost asymmetry — **DECIDED 2026-09-04**

**PR-AUC (average precision) primary; Brier score and a reliability curve
co-primary.** Precision@20/day as the operational read. ROC-AUC reported for
comparability but not decisive. **Accuracy is not reported** — at an 11.3%
positive rate, always predicting "stays" scores 88.7%.

The output is consumed as a probability, so ranking well while calibrated badly
is a failure of the actual use, not a technicality — hence Brier alongside
PR-AUC rather than after it.

**Cost asymmetry:** a false "closing soon" costs a rushed application, measured
in hours. A false "stays open" costs a job never applied to, which is
unrecoverable. The second is worse, so the operating point leans to recall and
the threshold is chosen against a fixed alert budget rather than at 0.5.

---

## 6. Dataset snapshot policy — **DECIDED 2026-09-04**

The scraper keeps running, so "the data" is a moving target and numbers taken on
different days are not comparable.

- **The database is pinned per experiment**: `python -m src.data.snapshot` copies
  `jobs.db` to `data/raw/<date>/` with a sha256 manifest and row counts.
  Re-pinning an existing date is refused, because replacing a snapshot
  invalidates every number already computed against it.
- **Every result cites its snapshot date.** A metric without one is not
  comparable to anything.
- **The raw archive is not pinned, and does not need to be.** Archived payloads
  are immutable once written — named by fetch stamp, never rewritten — so
  re-running over the same stamps is reproducible by construction. The manifest
  written by `python -m src.data.archive` records exactly which files were read.
- **Derived output is disposable.** Deleting `data/processed/` and re-running
  reproduces byte-identical Parquet; this is asserted in the test suite, not
  checked by eye.
- **Nothing under `data/` is committed.** It is regenerable from a snapshot, and
  postings are employer content.

---

## 7. Where it deploys — **REVISED 2026-09-06** (supersedes 2026-09-05)

**The API on a Render free web service. The Streamlit UI on Streamlit Community
Cloud. The frozen artifact still ships as a GitHub release asset.**

The 2026-09-05 entry decided Cloud Run plus a Hugging Face Space. It is
superseded on two counts, and only one of them is a change of mind.

**The constraint changed.** No card, no billing account, genuinely $0 — not
"free tier" but *free*. Cloud Run fails that on its own terms: §7b below said
so a day ago, in the paragraph beginning "Not literally free, and worth saying
so." A decision that already names its own disqualifier is not overturned by
this constraint so much as read properly under it.

**And the facts were wrong.** Verifying rather than assuming — which the old
entry told itself to do, in the line "Verify the terms before deploying" — turned
up that **Hugging Face Spaces running on compute are not free.** The Hub's own
documentation:

> Static Spaces are free for everyone. Gradio and Docker Spaces run on compute
> and require a paid plan to create: PRO for personal accounts, Team or
> Enterprise for organizations.

The hardware table still lists **CPU Basic — 2 vCPU, 16 GB — FREE**, and that is
the trap: the *hardware* costs nothing per hour, while the *right to create a
Space that runs any* requires a subscription. Both readings fit the pricing
page; only one fits the docs. The old §7c chose the wrong one and never checked,
which is the same failure as §7d's 139.6 MiB reading — a number that looked
right, taken once, believed.

So the UI's home decided yesterday does not exist on a free account either, and
Streamlit is no longer even a Spaces SDK: it is a Docker template, and therefore
paid.

### What is actually free, verified 2026-09-06

| Option | $0? | Card? | Fits the API? | Verdict |
|---|---|---|---|---|
| **Render free web service** | yes, 750 instance-h/month | **no card** | 512 MB, 0.1 CPU, Docker | **chosen for the API** |
| **Streamlit Community Cloud** | yes | **no card** | UI only, ~1 GB, GitHub-connected | **chosen for the UI** |
| HF Docker Space | no — PRO required to create | — | would fit technically | rejected: not free |
| HF Gradio on ZeroGPU | yes, 2 per free account | no | **Gradio SDK only** | rejected: cannot host FastAPI |
| HF Static Space | yes | no | no server-side execution | rejected for the API; a UI fallback |
| Google Cloud Run | no — billing account required | **yes** | fits well | rejected: excluded by the constraint |
| Fly.io | no — free allowance withdrawn | yes | fits well | rejected: excluded by the constraint |
| Koyeb | free plan exists | **unclear** — reports of card-on-file for human verification | 512 MB | rejected: cannot verify "no card" |
| Northflank | free plan exists | **card required** | fits | rejected |
| Railway | trial credit, not a free tier | yes | fits | rejected |
| Vercel / Netlify functions | yes | no | bundle limits far below scikit-learn + XGBoost | rejected: wrong shape |

**The requirement doing the most work is "no risk of accidental charges,"** and
it is not satisfied by a generous allowance. It is satisfied by a platform that
*cannot* bill me, because no payment instrument exists for it to charge. Render
is chosen partly because its documented behaviour when a limit is hit is exactly
that: *"If you haven't added a payment method, Render instead suspends all of
your Free services."* Suspension is the correct failure mode here. A dead demo
is recoverable; a surprise invoice on a portfolio project is not.

### 7a. Where the served model comes from

`/models/` is gitignored — it is derived output, like `data/processed/`, and §6
says derived output is not committed. The Dockerfile picks the artifact up with
`COPY . .`, from the working directory. So **the image can only be built on a
machine that has run `python -m src.models.freeze`**, and an image built by CI
from a clean clone boots happily and answers `/health` with
`model_loaded: false`. That is correct behaviour and a useless deployment.

**Decision:** `freeze` writes the artifact, the artifact is attached to a git
tag as a release asset, and the image build fetches it by tag.

Three reasons. The artifact already carries its own provenance — git SHA, panel
sha256, `fitted_on`, and the threshold — so a release asset is self-describing
rather than a loose binary. A tag makes the deployed model *a version* instead
of a file that happened to be on a laptop that afternoon. And it leaves the
"derived output is not committed" rule intact.

*Rejected:* committing the `.joblib` — binary, derived, and stale within a week
of the scraper running. *Rejected:* building the image locally and pushing it —
it works, and the answer to "what is serving right now?" becomes "trust me".

**Amended 2026-09-06:** the decision above is unchanged — the artifact is a
release asset and the build fetches it by tag. What changed is where the *tag*
comes from: a committed `MODEL_TAG` file rather than a build argument passed by
CI, because the platform now builds the image and CI no longer can pass one.
§7f has the reasoning.

### 7b. The API — a Render free web service

Nothing about the container changes: `$PORT` is honoured, the process is
non-root, and `/health` reports "port open" and "model loaded" as separate
facts. Render builds from the `Dockerfile` in the repository, and — usefully —
**translates a service's environment variables into Docker build arguments**, so
`ARTIFACT_TAG` reaches the build exactly as it does locally and §7a survives
untouched.

What the free instance actually is, and both numbers matter:

| | |
|---|---|
| Memory | **512 MB** |
| CPU | **0.1 vCPU** |
| Included | 750 instance-hours per month per workspace |
| Idle behaviour | spins down after **15 minutes** without inbound traffic |
| Wake | about **one minute**, serving a loading page meanwhile |
| Disk | none persistent; filesystem changes are lost on spin-down |
| Build | 500 build-minutes per month, 100 GB bandwidth |
| Payment method | **not required** |

**512 MB is thin but measured.** §7d put the container at 377 MiB resident with
the model loaded, flat across repeated requests — about 135 MB of headroom. That
was measured on arm64 serving the synthetic artifact, so it is an estimate for
this instance rather than a reading of it, and it is the first thing to
re-measure once something real is deployed.

**0.1 CPU is the number that hurts, and the old entry never considered it.**
Every previous paragraph about cold starts reasoned about *memory* and image
pull. But the expensive part of this container's start is importing
scikit-learn and XGBoost and unpickling the artifact, and that is pure CPU. The
2.06 s measured in §7d was on a full core; at one tenth of a core the arithmetic
is not encouraging, and Render's own documented wake time is "about one minute."
**This is an unmeasured number that a demo depends on**, which is precisely the
species of claim §7e exists to forbid. It gets measured before the link is given
to anyone, and if it lands somewhere absurd the fallback in §7c applies.

Render says of these instances: *"Do not use them for production applications."*
Quoted rather than hidden, because it is the correct expectation to set. This is
a portfolio demonstration, the failure mode is a slow first request, and the
alternative that removes it costs money.

*Rejected — Google Cloud Run:* the best technical fit and still the answer if
the constraint ever relaxes. Scale-to-zero, startup CPU boost, `--min-instances`
as a knob. It needs a billing account with a card, which is now disqualifying on
its own, independent of whether a charge would ever arrive.

*Rejected — Hugging Face Docker Space:* would have been ideal — 2 vCPU and 16 GB
would erase both of the concerns above, and it would have put the API and UI on
one platform. Creating a Space that runs on compute requires PRO. The hardware is
free; permission to use it is not.

*Rejected — Koyeb:* a real free plan with scale-to-zero, but the card question
could not be settled: Koyeb states it may request a card when it cannot
otherwise verify a human. "Probably no card" does not satisfy a requirement
whose whole point is certainty. Worth revisiting if that is ever verified.

*Rejected — Fly.io, Northflank, Railway:* the free allowance is gone, a card is
required, and a trial credit is not a free tier, respectively.

*Rejected — function-shaped hosts:* unchanged from yesterday, and now doubly so.
Vercel and Netlify cap a Python bundle far below what scikit-learn and XGBoost
weigh, before the fitted pipeline is even considered.

### 7c. The UI — Streamlit Community Cloud

Free, no card, roughly 1 GB of memory, deployed straight from this GitHub
repository, and — the reason it wins over every alternative — **it runs the
Streamlit app that already exists, unchanged.** Every other free option for the
UI required rewriting it.

It sleeps after **12 hours** without traffic and shows a wake-up page to the next
visitor, who can start it. That is a far better idle story than the API's 15
minutes, which produces a small irony worth naming: *the form will usually be
awake while the service behind it is asleep.* §7e is about what to do with that.

*Rejected — a Hugging Face Space:* yesterday's answer, and it is gone rather
than outvoted. Streamlit is no longer a Spaces SDK; it is a Docker template, and
Docker Spaces require PRO.

*Rejected — a Hugging Face Static Space, or GitHub Pages:* genuinely free, never
sleeps, and would mean rewriting the form as HTML and JavaScript and adding CORS
to the API. It stays the fallback if Streamlit Community Cloud changes terms,
and the cost of taking it is a rewrite plus losing Streamlit from the skill list
this build was supposed to produce.

*Rejected — a Gradio Space on ZeroGPU:* free for up to two Spaces on a personal
account in good standing, and the only free compute Hugging Face offers. It is
Gradio-only and GPU-oriented; using a GPU allocation to host a form that calls a
CPU model over HTTP is off-label enough to risk being flagged, and it would also
mean a rewrite.

*Rejected — putting the model inside the Streamlit app:* one deployment, no cold
start between two hosts, 1 GB of memory, and it would work. It is rejected
because it deletes the boundary this project is partly *about*: `app/` must not
import `src/`, the UI must reach the model over HTTP, and a UI that loads the
artifact directly is a second copy of the model wearing the same name. The
constraint is $0, not "$0 at any architectural price."

**What the boundary costs now that both halves live in one repository.**
Yesterday the separation was enforced by physics: `src/` was not deployed to the
Space, so the UI could not import it if it tried. Streamlit Community Cloud
checks out the whole repository, so that is no longer true. Two things hold the
line instead — `tests/test_app.py`, which parses the UI for imports of `src/` and
fails, and the dependency list the UI installs, which contains no scikit-learn,
no XGBoost and no joblib, so an import added in a hurry fails at load rather than
succeeding quietly. **That is weaker than physics and it is worth writing down
as a downgrade** rather than pretending the test was always the point.

### 7d. What it measures, on 2026-09-05

Taken from the image built at this commit, serving the synthetic artifact on
arm64. Every figure here replaces an estimate, and one of them changed a
decision above.

| | |
|---|---|
| Image | 1.08 GB |
| Resident memory, model loaded, idle | 370.8 MiB |
| Resident after 31 predictions | 376.9 MiB — flat, no growth |
| Container start → `/health` 200 | 2.06 s |
| Container start → first `/predict` | 2.12 s |

**What these are not.** The 2-second start is a *local* container on a warm
page cache with no image pull, **on a full core** — a lower bound, not a
prediction. The real cold start includes fetching a 1.08 GB image onto a cold
instance and then doing that same import-and-unpickle work on **0.1 vCPU**, and
it is the number §7e promises to measure and publish at deploy.

**Re-read under the 2026-09-06 constraint, these numbers say something
different.** They were taken to answer "does it fit in memory?", and the answer
— 377 MiB against 512 MB — still holds. They were never taken to answer "how
long does this take on a tenth of a core?", which is now the question that
decides whether the demo is pleasant, and none of the figures above bear on it.
A measurement kept past the question it was taken for is a number looking for a
claim to support.

**One measurement was nearly wrong in the honest direction.** A first
`docker stats --no-stream` taken immediately after startup read 139.6 MiB — the
sample races the process reaching steady state — and 139.6 MiB is a number that
would have made Render look comfortable. Three samples with a wait between them
give 371 MiB. Recorded in `DEBUGGING.md`: a single sample of a process that is
still starting is not a measurement of that process.

---

### 7e. What happens when it sleeps — **REVISED 2026-09-06**

Under the old plan this had a knob. Under this one it does not, and that is the
price of the constraint: the API spins down after **15 minutes** idle and takes
**about a minute** to come back, and there is no `--min-instances`, no startup
CPU boost, and no paid escape hatch. So the plan is about *managing* a cold start
rather than shortening one.

1. **Measure it, on the deployed service, before showing anyone.** Unchanged and
   more important than before, because the estimate now has two unmeasured
   multipliers in it — 0.1 CPU and a platform wake. `scripts/cold_start.sh`
   already does this; only the idle wait changes, from 20 minutes to 16.
2. **The UI wakes the API while the form is being filled in.** This is the one
   real mitigation the architecture makes free. Streamlit Community Cloud sleeps
   after 12 hours and the API after 15 minutes, so in practice a visitor arrives
   at a *live* form in front of a *sleeping* service. Firing `GET /health` when
   the page loads spends the user's form-filling time on the spin-up instead of
   making them wait for it afterwards. It costs one request and turns the
   asymmetry between the two sleep timers from a problem into the fix.
3. **Say so on screen**, and raise the client timeout. 30 seconds was chosen
   against a platform that promised a fast start; against "about one minute" it
   is a timeout that fires on the normal case. It goes to 90 seconds, with a
   "waking the service, this takes up to a minute on the free tier" message —
   the honest sentence, not a spinner.
4. **Still no keep-warm cron, and now the arithmetic says so too.** Render
   includes 750 instance-hours a month; a 31-day month is 744. Pinging every ten
   minutes to stay always-on would consume essentially the entire allowance to
   hide a delay, leaving nothing for a second service and no margin for a long
   month — and when the allowance runs out, free services are suspended until the
   month turns. That trade is bad: it converts a one-minute wait into a
   multi-day outage risk.
5. **For a live demo, warm it by hand.** Open the URL a minute before. That is
   the whole procedure, it costs nothing, and it is the honest replacement for
   the `--min-instances=1` line this entry used to carry.

**Verify the terms before deploying.** Free-tier allowances change often and
every figure above is as understood on 2026-09-06 — a day on which two figures
believed the previous afternoon turned out to be wrong. None should be trusted
without checking on the day.

### The acceptance criterion — **ADDED 2026-09-06**

Promoted from "would change my mind" to a **stop rule**, because the two are not
the same thing and this one needed to be the second:

> **If the measured cold start on the free instance exceeds 90 seconds, stop and
> reassess the architecture. Do not raise the timeout.**

The value of stating it now is that it is stated *before the number exists*. A
threshold chosen after seeing the measurement is not a threshold, it is a
description — and the specific way this one would have been quietly abandoned is
obvious enough to name: a disappointing cold start arrives with its own fix
already in reach, because `app/client.py` has a timeout and widening it makes the
symptom go away. It changes nothing except who finds out. The stranger still
waits; they just wait without a CI job objecting.

So the rule is enforced rather than recorded. `scripts/cold_start.sh` exits
non-zero past 90 seconds and prints the questions to ask instead, and
`tests/test_deploy.py::test_the_ui_timeout_never_exceeds_the_cold_start_stop_rule`
fails if the client timeout is raised above the criterion. Raising *both*
together still works, which is correct: renegotiating the criterion should cost a
diff and an entry here, not an afternoon's convenience.

**What "reassess" means concretely**, so that the rule cannot be satisfied by
staring at it:

- Does this demonstration need a live API at all, or would a static page over
  pre-computed examples show the same engineering? That trades interactivity for
  a page that is never asleep, and it is the honest option rather than the
  defeated one.
- Does the served image have to carry XGBoost? Import-and-unpickle is what the
  tenth of a CPU is spending its time on, and a convex model on the same
  pipeline would start far faster — at a cost in PR-AUC that the model comparison
  can price exactly, which makes it a measurable trade rather than a guess.
- Is a different free host's instance meaningfully less starved? §7's table is
  dated; the row that matters is CPU, and that is not the column any of them
  advertise.

**The invariant this must not violate**, stated because the pressure runs the
other way when a demo is slow: **no deployment optimisation may modify
`src/data`, `src/features` or `src/models`.** The hosting layer changed twice in
two days and the ML system did not change at all, which is the property that
makes the modelling numbers still mean what the reports say they mean. The
regression guard is not the diff — it is the pinned-prediction test, which fails
if the pipeline's output moves for any reason at all. Serving may be made faster;
the model may not be made *different* in the process without that being its own
decision, with its own entry, and its own re-measured numbers.

**Would change my mind about the platform** (as opposed to the criterion): a
resident-memory reading above ~450 MiB on the deployed instance, which would make
512 MB a coin flip rather than a fit; or Render requiring a payment method, which
would end the option outright and promote the Koyeb question from "unverified" to
"worth an afternoon".

---

### 7f. The deploy path, and the gate on it — **REVISED 2026-09-06**

Written first against Cloud Run, then rewritten the same day when the platform
changed. Most of it survived, which is the useful part of the story: the
decisions that were about *the shape of a deploy* outlived the decision about
where it lands, and the ones that were about a particular vendor did not.

**What survived, unchanged.**

*A deploy is not green because it deployed.* The image boots with no artifact on
purpose (§7a), reports `model_loaded: false`, and answers 503. That is a
successful deploy of a useless service on any platform, so the last step is a
smoke test against the live URL and the deploy fails if it fails.

*The gate: the smoke test refuses a model whose `dataset` is `synthetic`.* Every
component here is exercised on a synthetic panel whose label is drawn
independently of every feature — the right thing to build against, and a
placeholder whose predictions are noise. On a laptop that is obvious. Behind a
public URL it is invisible. `ALLOW_SYNTHETIC=1` overrides it for a deliberate
rehearsal, which is a different act from forgetting.

*The cold start is not measured by the deploy*, because deploying starts an
instance to verify it serves. Only an idle service gives the number.

**What changed, and why.**

*Keyless authentication is gone, because there is nothing to authenticate to.*
Workload Identity Federation was the right answer to "how does CI get a Google
credential without storing one." Render and Streamlit Community Cloud both build
from the GitHub repository they are connected to, so the honest answer is now
that CI holds no deployment credential at all. That is strictly better and it is
not to my credit — the constraint removed the problem.

*The tag moves from a build argument into the repository.* Under the old design
the workflow built the image itself and passed `--build-arg ARTIFACT_TAG`. Render
builds the image, so the tag has to reach it some other way. Render does
translate a service's environment variables into build arguments, which would
work — but setting one from CI needs a Render API key, which reintroduces the
credential that the previous paragraph just celebrated losing.

**Decision: a one-line `MODEL_TAG` file, committed.** The Dockerfile reads it
when no `ARTIFACT_TAG` build argument is supplied, so local builds keep their
override and the deployed build needs no configuration at all. Releasing a model
becomes: cut the release, write the tag into `MODEL_TAG`, push. Render redeploys
on the push, and **"which model is serving?" becomes a question answerable from
git history alone** — better than the build-argument version, which answered it
from a dashboard.

*Rejected — the Render API from CI:* one more key, one more thing to rotate, to
set a value that wants to be version-controlled anyway.

*The trigger changes shape.* Pushing an `artifact-*` tag no longer *performs* the
deploy — the platform does, on the push to `main` that carries the new
`MODEL_TAG`. What remains for GitHub Actions is the half that is still worth
automating: wait for the new revision to answer, then smoke-test it, and fail
loudly if a placeholder or a model-less container reached the URL. A workflow
that verifies someone else's deploy is a smaller thing than one that performs the
deploy, and it is the correct size for the job that is left.

**Would change my mind:** if Render's free tier disappears or grows a card
requirement, this whole entry reopens and §7c's static-page fallback is the first
thing to price. If the measured cold start makes the demo unusable, the question
stops being "where does the API live" and becomes "does this demo need a live API
at all", which is a genuinely different design.

---

## 8. The split — **DECIDED 2026-09-04**

> Accepted 2026-09-04: keep the job-day unit, soften §2 of the problem
> definition, record the departure from the one-id-one-split rule as
> deliberate, and make the seen/unseen breakdown part of the split rather
> than an afterthought. The reasoning that led here is kept below.

This is the decision that determines what Component 6 builds, and it is not yet
made.

**The contradiction.** `problem_definition.md` §2 says any split placing some of
a posting's rows in train and others in test is leaking. §7's protocol — train
`t <= T_cut`, test `t > T_cut + H` — does not prevent that, and **1,131 of 1,240
postings (91%) straddle a mid-panel cut**. It is structural, not incidental:
mean age is 82.6 days against a 1.69% daily hazard, so the median posting
outlives any weekly cut. The H-day gap fixes *label-window* overlap; it does
nothing about *subject* overlap.

**It also departs from a project rule.** The build's own requirement — "no job id
may appear in two splits" — assumes one row per posting. The job-day unit makes
that requirement either impossible or ruinous.

**The decision.** Keep the job-day unit and amend §2.
Subject overlap across time is standard and correct in discrete-time hazard
models: it is the person-period setup used throughout survival analysis, and
excluding it would discard 91% of the data to defend a principle imported from
the IID setting, where it is true.

But the real risk then needs naming, because it is not the one §2 describes: a
posting's rows share a byte-identical title, company and description, so a
high-capacity model can memorise *this posting survives* rather than learning
duration dependence, and that memorisation crosses the cut. The mitigation is
not a group split — it is **reporting the metric separately for postings unseen
in training and postings carried over**. A large gap between the two is the
diagnosis.

**Three things follow, and are now in force:** §2's absolute
sentence is softened, the departure from the one-id-one-split rule is recorded
here as deliberate, and Component 6 implements the seen/unseen breakdown as part
of the split, not as an afterthought.

**Would change my mind:** a deployment story of "score postings from employers
we have never seen", which would make employer-level generalisation the thing
being measured and a grouped split the honest test.

---

## 9. Consequences for the feature set — **DECIDED 2026-09-04**

Two corrections that follow from §6.2 of the problem definition, found while
building the data dictionary:

- **`seniority` cannot be used as stored.** It is listed as an allowed feature,
  but it is absent from the snapshot CSVs and exists only in the current-state
  `jobs` table, so reading it would leak later edits backwards. It is derived
  from the title by a row-local rule, so it is re-derived from the as-of-t title
  instead.
- **Repost counts from `requisition_id` need a window.** The field itself is
  as-of-t and safe. "How many postings share this requisition" reads other rows,
  so like `board_hazard_prior` it is legitimate only when the window ends at `t`.

---

## 10. How the horizon is compared — **DECIDED 2026-09-04**

**The decision.** `t_gone(j) <= t + H` and `observed at or after t + H` are both
evaluated on **calendar dates**, not on instants. `basis="calendar"` is the
default in `src/features/assemble.py`; `basis="instant"` stays implemented so
the comparison remains reproducible rather than a claim in prose.

**Why this was a question at all.** The panel looks once a day at a time that
drifts, and the complete-run schedule is visibly irregular:

```
run 0   2026-08-31 03:45:35
run 1   2026-09-01 14:07:12    +34.3601h   ← fired late
run 2   2026-09-02 03:45:38    +13.6407h
run 3   2026-09-03 03:45:36    +23.9993h   ← 2.6s under 24h
run 4   2026-09-04 03:46:03    +24.0075h   ← 27.0s over 24h
```

**The reason instant arithmetic is wrong here, stated once.** A removal is
*interval-censored*: we learn that a posting vanished somewhere in
`(t_last_seen, t_first_absent]` and never learn when. Nothing in this panel can
resolve an event more finely than the gap between two runs, so at H=1 a horizon
expressed in continuous time is **not identifiable from the data**. Both bases
are therefore proxies for the one question the panel can answer — *was it
absent at the next complete run?* — and calendar comparison is exactly
equivalent to that question on this schedule, while instant is a lossy
approximation of it whose loss is governed by cron jitter.

**What the loss looks like, measured** on the 2026-09-04 snapshot, which is
what this decision was made against. The two never disagree on a row they
both label. They differ in 49 rows that instant discards as unobservable and
calendar labels, 21 of them positive:

| | `y=0` | `y=1` | dropped |
|---|---|---|---|
| calendar | 4,474 | 53 | 1,187 |
| instant | 4,446 | 32 | 1,236 |

| run | instant positive rate | calendar |
|---|---|---|
| 0 | 0 / 1,116 = **0.00%** | 19 / 1,135 = 1.67% |
| 1 | 20 / 1,129 = 1.77% | 20 / 1,144 = 1.75% |
| 2 | 12 / 1,127 = 1.06% | 14 / 1,142 = 1.23% |
| 3 | 0 / 1,106 = 0.00% | 0 / 1,106 = 0.00% |

Run 0 has no positives under instant because the next run came 34.4h later, so
nineteen removals the scraper observed as fast as it physically could fall
outside a one-day horizon and are thrown away. Run 2's twelve positives then
survive **by 2.6 seconds** — the margin by which that gap undershot 24h — and
the +27.0s drift at run 3 costs 13 further rows. This is the point: the instant
base rate varies across run indices for reasons that are entirely cron jitter,
and `run_index`, `t_dow` and `age_days` are all features. That makes it label
noise correlated with the model's own inputs — a signal the model can learn
that does not exist in the world — rather than a bias that could be bounded and
reported.

**The cost accepted.** Calendar's effective window is 24–48h of wall clock
depending on where `t` sits in the day. That is harmless *on this schedule*,
because the only candidate event times are the run instants and exactly one of
them lands inside the window; it would stop being harmless if two complete runs
ever landed on the same date. The claim the label supports is therefore "gone
by tomorrow's check", not "gone within 24 hours", and the README must say so.

**Rejected alternative: define H in runs, not days** — "absent at the next
complete run", stated directly. It is what the panel measures and it is immune
to any schedule change rather than merely to jitter. Rejected because it costs
the wall-clock product claim the user in §5 actually acts on, and because H=7
would become "seven runs", whose meaning drifts with the scrape cadence in
exactly the way a horizon should not.

**Run 3 is structurally zero under both bases**, and that is the two-run
corroboration rule, not the comparison: a removal cannot be confirmed at the
last complete run because there is no run after it to corroborate. The 53
dropped rows at run 3 are that rule working. It leaves the most recent
observable run a pure-negative block, which is a fact the temporal split in
Component 6 has to handle rather than discover.

**Would change my mind:** a scrape cadence of more than one complete run per
day, which would break calendar's equivalence to "the next run" and force the
run-indexed definition; or a deployment story that promises a wall-clock
guarantee ("gone within 24 hours") strongly enough that the 24–48h smear
becomes a misrepresentation rather than a caveat.


---

## 11. The resurrection window — **OPEN**

`t_gone` requires that a posting *never re-appeared*. That clause reads the whole
remaining panel rather than a bounded window, with two consequences:

- **A label is never final.** It can flip as depth accrues.
  `greenhouse:gitlab 8615319002` was present at runs [0, 3, 4] — two consecutive
  absences, enough to satisfy the two-run corroboration guard — and then it came
  back, flipping its run-0 label from 1 to 0 between panel depth 3 and depth 5.
- **No embargo of any width fully seals a training label from the evaluation
  period**, because the reach is unbounded by construction. §10's embargo covers
  the horizon and the corroborating run; it cannot cover this.

**Scale:** 2 of 1,240 postings (0.16%) in the 2026-09-04 snapshot, one of them
with a two-run absence. Measured by `src/data/split.py:resurrection_risk`.

**The options.** Bound it — *"did not reappear within K runs"* — which makes the
reach finite, the embargo computable, and a label final at a known time; the
cost is that a posting returning at K+1 is mislabelled as removed. Or leave it
unbounded and report the residual as a known defect.

**Leaning toward bounding it**, on the deployment argument rather than the
statistical one: an unbounded clause means a training label is never final, and
a label that cannot be computed at a known time cannot be recomputed for
retraining. Not decided, because K is a real choice and 0.16% is small enough
that it is not yet urgent.

**Would change my mind:** a resurrection rate that grows with panel depth. Two
postings over four days may simply be the visible edge of something a longer
panel will show properly, and K should be chosen against that distribution
rather than against two cases.
---

## 12. Board context at serve time — **OPEN, defaulting to imputation**

Four features describe the board rather than the posting: `board_size_at_t`,
`board_growth`, `n_same_title_on_board`, `n_same_req_on_board`. Each was
computed inside a single crawl, so each is honestly as-of-`t` and none of them
is a leak. **But a stranger holding one job posting cannot supply any of them**,
and Component 13's rule is that a field which cannot appear in the request
cannot be a feature.

**The decision, dated 2026-09-05:** accept them when supplied and impute them
when not, and say which happened. `src/inference/contract.py` marks the four as
board-origin, `POST /predict` accepts them as optional, and every response
carries `board_context_supplied` so a caller can tell the two regimes apart.

**What this costs, stated rather than hidden.** For a caller who supplies
nothing, the four arrive as nulls and the training fold's imputers fill them
with constants. The features are then inert, and the served model is *not quite*
the model that was validated — its board columns carry no information at all.
That is a training/serving mismatch of a mild kind: not a wrong value, but a
constant where a variable was expected.

**The alternative** is to drop the four from the fitted model, so that the
validated model and the served model are the same object for every caller. It
costs whatever the four features are worth, and nothing yet says what that is —
the leave-one-out ablation in `reports/model_results.md` has not run, because
the panel cannot be split. Two of the four are also the most plausible
mechanisms in the whole feature set: a posting duplicated across a large board
is a different animal from a lone requisition.

**Would change my mind:** an ablation showing the four are worth little, in
which case dropping them buys a cleaner serving story for nearly nothing. Or a
deployment story that changes the caller — a board owner scoring their own
requisitions has all four, and for them the imputation branch never runs.

---

## 13. What the frozen artifact is fitted on — **DECIDED 2026-09-05**

**The training block only**, by default (`--fit train` in
`src/models/freeze.py`).

The alternative, refitting on train + validation before shipping, is the more
common practice and has the better deployment argument: more data, and more
*recent* data, which on a panel where the board turns over daily is not a small
thing.

It was rejected for one property. Fitted on train alone, the object in
`models/shelf_life.joblib` is the same object the validation number describes
and the same object the test number describes — so the README can say "this
model scores X" without a footnote about which of three fits produced which
figure. Refitting makes the shipped model a fourth thing, measured only by
inheritance from its siblings.

The flag stays, because the argument the other way is real and the panel is
still short. What is not negotiable is that whichever is chosen is recorded in
the artifact's metadata (`fitted_on`) rather than remembered.

**The threshold ships inside the artifact.** It is chosen on validation at the
alert budget from §5, and a probability without the threshold it is compared
against is not a decision. Shipping them in separate files is how the two come
to disagree.

**Would change my mind:** a validation block large enough that discarding it
from the fit is measurably expensive. On the current panel it is one crawl wave.
