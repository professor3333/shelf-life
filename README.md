# shelf-life

[![CI](https://github.com/professor3333/shelf-life/actions/workflows/ci.yml/badge.svg)](https://github.com/professor3333/shelf-life/actions/workflows/ci.yml)

A job posting has a shelf life: it sits on a board until it is pulled. This
project predicts how long that takes, from a panel of postings collected daily
by my own scraper, and serves the prediction over HTTP so a single posting can
be scored at the moment it first appears.

The label is **removed from the board**, which is not the same thing as
**filled**. The name of the project is chosen not to claim otherwise, and that
distinction is repeated everywhere a number appears — including on the screen of
the UI.

> **Status, stated plainly.** The full system is built: ingestion, labelling,
> the leakage audit, the temporal split, the model ladder, experiment tracking,
> the frozen-artifact packaging, the API, the container and the UI. **No model
> has been fitted on the real panel yet**, because the panel is not deep enough
> to cut an honest three-way split, and every report in `reports/` records that
> refusal rather than a number. [Why, and when it clears](#why-there-are-no-real-numbers-yet).

---

## Architecture

Two systems that share exactly one object: the fitted pipeline.

```
OFFLINE — runs on my machine, on a pinned snapshot
──────────────────────────────────────────────────────────────────────
  jobs.db          src/data/       src/features/      src/models/
  (scraper)   ──>  snapshot   ──>  assemble      ──>  ladder, tuning,
                   load            derive             evaluation
                   clean           preprocessing            │
                   archive              │                   │
                        │               │                   ▼
                        └── split ──────┴────────────>  freeze.py
                                                            │
                                                            ▼
                                              models/shelf_life.joblib
                                        (ColumnTransformer + estimator,
                                         threshold, metrics, provenance)
                                                            │
──────────────────────────────────────────────────────────────────────
ONLINE — runs anywhere                                      │
                                                            ▼
   app/streamlit_app.py  ──HTTP──>  api/main.py  ──>  the same object
        (a form)                    POST /predict      loaded once at
                                    GET  /health       startup
```

Three properties that shape everything else:

**The serving path uses the same fitted object as training.** Not a
re-implementation of the feature logic. A second implementation agrees on the
day it is written and drifts silently afterwards — that is training/serving
skew, which is leakage wearing a production hat. `api/` is forbidden by test
from importing `src.features`, `src.models` or `src.data`.

**The UI calls the API, never the model.** `app/` is forbidden by test from
importing `src/` at all. The alternative works fine on a laptop and means the
deployed UI and the deployed API are scoring with two different artifacts.

**Everything learned from data lives inside one scikit-learn `Pipeline`**, and
there is exactly one supported way to fit it — `fit_on_training_fold`, which
takes the split object and can only reach the training block. Fitting a scaler
on the whole frame is the most common way to fake a good score, and the way to
prevent it is to make the wrong call inexpressible rather than merely discouraged.

---

## The problem

**Who would use this, and what changes because of it.** Someone who tracks a
board — a recruiter watching competitor postings, a candidate deciding what to
apply to first, an analyst measuring hiring activity — and can only act on a
handful of postings a day. The prediction orders a list: *these are the ones
about to disappear.*

That user is why the metric and the threshold are what they are. They have a
budget of about **20 postings a day** they will actually look at, so the model
is scored on the quality of a short list, not on the accuracy of a verdict about
every row.

### What is predicted, exactly

> Will this posting be **absent from the board** within `H` days of an
> observation of it, judged from information available at the moment of that
> observation?

One row is a **job-day**: one posting as seen by one complete crawl. A posting
present for five complete crawls contributes five rows, each with its own
features and its own label. (This is the person-period setup used in
discrete-time hazard models; `docs/design.md` §8 records why it was chosen over
one row per posting.)

The label is positive when the posting is absent from a complete run and still
absent from the run after it, and never reappears. Two consecutive absences
rather than one, because a single missed crawl is as likely to be a hiccup as a
removal.

**Rows whose outcome is not yet observable are dropped, not labelled zero.** A
posting first seen yesterday has not had time to close; calling that a negative
teaches the model that recent means open, which is a labelling bug that produces
a beautiful score. 1,185 of 6,878 job-days are dropped for this reason.

### The prediction point

```
              Posting observed by a crawl at time t
                              │
                  INFORMATION AVAILABLE AT t          ← features
                              │
        ──────────────────────┼──────────────────────  the prediction point
                              │
                           MODEL
                              │
                        PREDICTION
                              │
                   What happened after t             ← the label, or leakage.
                                                       There is no third
                                                       category.
```

Every feature is a fact sealed at or before `t`: the tabular fields come from
that run's own snapshot CSV, the structured detail from that fetch's archived
payload, and the three board-context features from a window that ends at `t`.
`docs/leakage_audit.md` gives a verdict for all 44 panel columns, and
`src/features/preprocessing.py` **is** that document as data — a column with no
verdict raises rather than being silently modelled or silently dropped.

---

## The data

Collected by my own scraper (a previous project) into a SQLite database, and
copied here as an immutable dated snapshot before anything reads it. The scraper
keeps running, so "the data" is a moving target; numbers computed on different
days are not comparable unless the snapshot is pinned.

**Snapshot 2026-09-05** — 5,297 postings, 20,447 observations, 95 crawl runs,
window 2026-08-29 → 2026-09-05.

The job-day panel built from it, at `H = 1`:

| | |
|---|---|
| job-days | 6,878 |
| labelled (outcome observable) | 5,693 |
| dropped (right-censored) | 1,185 |
| positives | 75 (**1.32%**) |
| distinct postings | 1,260 |
| complete crawl waves | 6 (5 of them labelled) |

Per source, on the labelled rows:

| source | rows | postings | positives | rate |
|---|---|---|---|---|
| greenhouse:anthropic | 2,884 | 629 | 39 | 1.35% |
| greenhouse:gitlab | 1,120 | 249 | 18 | 1.61% |
| greenhouse:figma | 802 | 167 | 10 | 1.25% |
| greenhouse:duolingo | 431 | 92 | 2 | 0.46% |
| greenhouse:discord | 249 | 54 | 5 | 2.01% |
| python_org | 127 | 33 | 1 | 0.79% |
| greenhouse:airtable | 80 | 16 | 0 | 0.00% |

### What is broken in it, and what was done

**The largest source is excluded, and that is the most important fact here.**
arbeitnow is 3,531 of the 5,297 postings — and every one of its crawls in the
current rules epoch stopped at its page cap without observing the whole board.
A posting's absence from a partial crawl is not evidence of removal; it may
simply have fallen past page 8. Treating those absences as closures would have
manufactured thousands of false positives, and the label would have been
measuring pagination. So `complete_runs` admits only crawls that finished, which
leaves the six Greenhouse boards and python_org. The defect is recorded in
`DEBUGGING.md`; it cost three quarters of the dataset and it was the right
trade.

**Missingness is a fingerprint of the source, not noise.** `remote` is never
populated on Greenhouse and always populated on arbeitnow. `first_published`,
`departments` and `content_chars` are null on exactly the 127 python_org rows.
So an "is this missing?" indicator on any of them reconstructs board identity
for free — which is a decision the project has *not* yet made (`docs/design.md`
§4), so no such indicator exists. Each column's fill and the reason for it are
recorded next to the code that applies it, in `src/features/preprocessing.py`.

**Salaries are text in mixed formats and currencies** — `"90.000 € bis
130.000 €"`, `"$150k-$200k"`, and nothing at all on three quarters of postings.
`src/data/clean.py` parses what it can; `salary_stated` carries the fact of
absence as its own feature, because a posting that declines to state pay is
telling you something.

**Two postings came back from the dead.** The label requires that a posting
never reappeared, which reads the whole remaining panel — so a label can flip as
depth accrues. Two of 1,240 postings did exactly that. It is measured
(`resurrection_risk`), reported, and still open (`docs/design.md` §11).

---

## Leakage: what was found

The panel is unusually rich in traps, because half its columns are written
*after* the outcome. Three worth naming:

**`n_observations_total` and `days_on_board_total`.** A count of how many times
a posting was seen accumulates *because* the posting stayed open. It is the
label with a numeric face. These live in `src/features/leaky.py`, deliberately
kept and deliberately quarantined: run `06-xgboost_leaky` admits them and run
`07-xgboost_leak_removed` takes them away again, so the experiment history
contains a *measured* leak rather than a description of one.

What it is worth, measured on a fixture whose label is drawn independently of
every feature — so the honest answer is known to be the base rate, 0.10:

| run | validation PR-AUC |
|---|---|
| `06-xgboost_leaky` | **0.8385** |
| `07-xgboost_leak_removed` | **0.1915** |

**The leak was worth +0.6471 PR-AUC, and every point of it was a lie.** A model
that could not possibly know anything scored 0.84 because one column counted how
many times the posting had been seen. That pair is the run that changes your
mind about a feature, and it cannot be reconstructed after the fact — which is
why the leaky columns are kept in the repository rather than deleted.

**Any statistic fitted before the split.** `CompanyVolumeEncoder` counts a
company's postings — computed over the whole frame, each row's encoding would
depend on rows in the validation and test blocks. It is a fitted transformer
learned on the training fold only, and `learned_statistics()` exposes the fill
values so "was this fitted on the training fold?" is an assertion rather than an
assurance.

**`t_dow`, `n_complete_runs_observed`, `source_id`.** Excluded as axis proxies.
With five crawl waves, day-of-week nearly identifies the wave; `source_id` is
monotonic in creation order (ρ = 0.917 against `first_published`); the
observation count is constant at serve time and encodes left truncation.

---

## Method

### The split

**Temporal, with an embargo.** Train on the earliest window, validate on the
middle, test on the most recent — cut on calendar time, never on run index,
because run indices are per-source and not aligned.

The scenario it simulates, in one sentence: *a model fitted on every posting the
board showed us up to some Monday, scoring the postings that are on the board on
a later day it has never seen.*

The embargo is the part most temporal splits omit. A row's label reads forward —
`H` days plus one run for the corroborating absence — so without a discarded
strip between blocks, the training labels were computed from the validation
period. On this data the embargo is **2 days 10 hours**: one day of horizon plus
the widest observed gap between consecutive runs, 34.4 hours, caused by a single
crawl that fired at 14:07 instead of 03:45.

Postings deliberately appear in more than one block: 91% straddle any mid-panel
cut, and a grouped split would discard most of the data to defend a rule
imported from the IID setting. The risk that replaces it is memorisation, so
every evaluation row is tagged `seen_in_train` and every metric is reported both
overall and split on that flag. A model that scores well on carried-over
postings and badly on unseen ones has memorised.

The test block is opened **once**, in `src/models/freeze.py`, after the pipeline
and threshold are frozen. `tests/test_evaluate.py` parses `src/` and fails if the
test split is read anywhere but there and in the property that defines it.

### The metric

**PR-AUC, with precision and recall at a chosen threshold.**

*Not accuracy.* At a base rate of 1.32%, predicting "stays open" for every
posting scores **98.7%** and has told you nothing.

*Not ROC-AUC.* With rare positives it flatters. The false-positive rate divides
by the true-negative count — 5,618 of them against 75 positives — so a model can
raise a great many false alarms without visibly moving the x-axis. Precision
divides those same false alarms by the number of rows *flagged*, where they
cannot hide. At this base rate the ROC curve describes a decision nobody makes.

Brier score and expected calibration error are reported alongside, because a
probability that is not calibrated is a score wearing a percent sign — and
because a constant predictor is perfectly calibrated and completely useless,
calibration is never reported on its own.

### The threshold

Not 0.5. The operating point is set by the **alert budget**: 20 postings per
prediction day, the number a person will actually read. The threshold is the
score at which exactly that many rows are flagged, chosen on the validation
block and then frozen inside the artifact — a probability shipped without the
threshold it is compared against is not a decision.

The cost asymmetry behind it: a false positive costs a few seconds of attention
on a posting that was not going anywhere. A false negative costs a missed
posting entirely. Recall is worth more than precision here, but only up to the
budget — a list of 500 alerts nobody reads has perfect recall and zero value.

---

## Why there are no real numbers yet

An honest three-way split needs seven labelled crawl waves. The panel has five.

```
minimum waves = 1 + 2 × (floor(embargo ÷ spacing) + 1)
              = 1 + 2 × (floor(2d10h ÷ 1d) + 1)
              = 7
```

Each of the two block boundaries discards every wave inside the embargo — three
of them — and the wave landing exactly on a boundary goes too. A wave is also
not labelled the day it is crawled: negatives need a later run to confirm
survival, positives need two.

So `src/models/freeze.py` refuses, and `reports/test_results.md` records the
refusal and the shortfall instead of a number. This is the designed behaviour,
not a bug: a test block with no positives is not a hard test set, it is an
undefined metric, and a model frozen against one would produce a README number
that means nothing. `python -m src.models.freeze --run 05-xgboost_engineered`
prints the shortfall today and writes the artifact on the day it clears.

**What is verified in the meantime.** Every component is exercised end to end on
a synthetic panel whose label is drawn *independently of every feature*, so the
correct answer is known: nothing should beat the base rate. The comparison
machinery agrees — it reports the random forest leading the logistic regression
by 0.06 PR-AUC across seven rolling-origin folds and still returns the verdict
*"inside one standard deviation, so treat them as tied."* That refusal, on a
label that is pure noise, is the machinery working.

**The one real number that exists** is the constant-predictor reference: PR-AUC
**0.0132** on 5,693 labelled rows, which is the base rate. Every model must beat
it, and none has been asked to yet.

---

## What is here

Only features that exist:

- **Reproducible ingestion** — pin a dated snapshot of the scraper DB, load it
  with schema validation and dtype coercion, recover as-of-`t` fields from
  archived API payloads. Delete `data/processed/`, re-run, get identical output.
- **A labelled job-day panel** with the censoring rule in one place, in code.
- **A leakage audit** covering all 44 panel columns, enforced by a test that
  refuses a frame carrying a column no verdict has been written for.
- **A leakage-safe pipeline** — imputation, encoding and scaling inside a
  `ColumnTransformer` with exactly one supported fit path. Unseen categories at
  serve time encode as zeros rather than raising.
- **A model ladder** — constant → logistic → decision tree → random forest →
  XGBoost — compared with fold variance and *paired* fold differences, not by
  subtracting two averages.
- **Hypothesis ablation** — each engineered feature was a hypothesis before it
  was a column, and is removed one at a time to see what it was worth.
- **A deliberate overfit** — depth up and regularisation off until train and
  validation separate, then closed again one knob at a time.
- **MLflow tracking** — params, metrics, dataset hash and git SHA per run, with
  a replay path that reproduces a run from what was logged rather than from a
  fresh search.
- **A frozen artifact** — the whole fitted pipeline plus threshold, metrics and
  provenance in one file, with a load-time check that it really is the pipeline.
- **A FastAPI service** — `POST /predict`, `GET /health`, `GET /contract`.
- **A Streamlit UI** over that service, over HTTP.
- **A Docker image** — non-root, `$PORT`-aware, health-checked.

Not here, and deliberately: deep learning, NLP beyond title-derived features,
retraining pipelines, drift monitoring, authentication. Each would be a
different project.

---

## Tech stack

Python 3.12 · pandas · numpy · scikit-learn (`Pipeline`, `ColumnTransformer`) ·
XGBoost · MLflow · FastAPI · pydantic · Streamlit · Docker · pytest · ruff ·
SQLite (read-only, upstream) · Parquet.

---

## Project structure

```
shelf-life/
├── src/
│   ├── data/         snapshot.py load.py clean.py archive.py profile.py split.py
│   ├── features/     assemble.py derive.py preprocessing.py leaky.py
│   ├── models/       metrics.py baselines.py train_baseline.py train.py
│   │                 evaluate.py experiments.py freeze.py provenance.py
│   └── inference/    contract.py artifact.py predict.py
├── api/              main.py schemas.py
├── app/              streamlit_app.py client.py
├── tests/            261 tests, no network, no data files
├── docs/             problem_definition.md design.md leakage_audit.md
│                     data_dictionary.md
├── reports/          generated: profile, baselines, model results, comparison,
│                     experiment log, test results, feature hypotheses
├── data/             not committed — snapshots and derived frames
├── models/           not committed — the frozen artifact
├── Dockerfile
└── pyproject.toml
```

`docs/` holds decisions and audits, written by hand. `reports/` holds generated
output — regenerate rather than edit; each file names the command that writes it.

---

## Requirements

- Python 3.11 or newer (3.12 is what CI runs)
- Docker, to build the image
- **For the data pipeline only:** the scraper's SQLite database. Without it, the
  ingestion commands have nothing to read; everything else — tests, API, UI —
  runs without it.

---

## Installation

```bash
git clone https://github.com/professor3333/shelf-life.git
cd shelf-life
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,ui]"
```

The extras are separable on purpose: `api` for the service, `ui` for the form,
`tracking` for MLflow, `dev` for everything plus the test tooling. Running the
model ladder should not require installing a web framework.

---

## Usage

Every command below was run against this repository before it was written down.

### The offline pipeline

```bash
python -m src.data.snapshot                    # pin a dated copy of the scraper DB
python -m src.data.load                        # validate it into Parquet
python -m src.data.archive                     # recover as-of-t fields from payloads
python -m src.data.profile                     # write reports/data_profile_<date>.md
python -m src.features.assemble --horizon 1    # build the job-day panel
```

### Models

```bash
python -m src.models.train_baseline            # the ladder and the reference number
python -m src.models.train                     # engineered features, XGBoost, ablation
python -m src.models.evaluate                  # comparison, threshold, calibration
python -m src.models.experiments --synthetic   # replay the run history into MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

While the panel is too shallow, the first three print the refusal and write it
into their reports. That is the expected output today, not a failure.

### Freezing a model

```bash
python -m src.models.freeze --run 05-xgboost_engineered
```

`--run` is required and names one of the specs in `src/models/experiments.py`
(`01-prior`, `02-logistic`, `03-random_forest`, `04-xgboost_panel_native`,
`05-xgboost_engineered`, `07-xgboost_leak_removed`, `08-xgboost_tuned`). There is
no default, because choosing the model that ships is a decision recorded in
`docs/design.md`, not a constant in a file. The deliberately leaky run cannot be
frozen at all.

Add `--synthetic` to freeze against the test fixture instead — which is how the
API and UI can be exercised before the real panel is deep enough.

### The service

```bash
uvicorn api.main:app --reload          # http://localhost:8000/docs
```

```bash
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"title": "Senior Data Engineer",
       "location": "Berlin",
       "salary_raw": "120000 - 160000 USD",
       "content_chars": 1400,
       "first_published": "2026-08-20T00:00:00Z"}'
```

```json
{
  "probability": 0.010016298852860928,
  "threshold": 0.3962169587612152,
  "closing_soon": false,
  "horizon_days": 1,
  "board_context_supplied": false,
  "model": "05-xgboost_engineered",
  "dataset": "synthetic",
  "t": "2026-09-05T12:25:40.957128+00:00"
}
```

`t` is the prediction instant — the moment `age_days` is measured from. It
defaults to now, and can be pinned by sending `as_of`, which is what makes a
prediction reproducible.

Every response says which threshold it was compared against, what horizon
"closing" refers to, whether board-level context was supplied — and whether the
loaded model was fitted on the real panel or the synthetic fixture. A service
serving a rehearsal must not look like a service serving a model.

`GET /health` reports the loaded model and stays 200 even when there is none:
up-but-empty is a different fact from unreachable, and `/predict` answers 503
while that is true. `GET /contract` publishes the field-by-field audit — what a
caller may send, and whether somebody holding one posting could know it.

### The UI

```bash
SHELF_LIFE_API=http://localhost:8000 streamlit run app/streamlit_app.py
```

A form, a probability, the threshold it was compared against, and the caveat on
screen rather than in a footnote.

### Docker

```bash
python -m src.models.freeze --run 05-xgboost_engineered --synthetic  # or the real run
docker build -t shelf-life .
docker run --rm -p 8000:8000 shelf-life
```

The image builds without an artifact. The container then starts, `/health`
answers 200 with `model_loaded: false`, and `/predict` returns 503 naming the
command that fixes it — a container that refuses to boot over a missing file
turns a one-line diagnosis into a log-reading exercise.

---

## Data sources, schema and storage

**Source.** A SQLite database produced by my own scraper, covering seven job
boards. This repository never fetches anything: it reads a dated copy of that
file and nothing else.

**Schema.** `runs` (one row per crawl, with `status`, `page_cap`,
`pages_fetched` and `rules_version` — the provenance that decides whether an
absence is evidence), `jobs` (current state per posting), `job_observations`
(one row per posting per crawl — the panel), `job_changes` (recorded field
edits, all of them after first sight, and therefore not features). Column
meanings are in `docs/data_dictionary.md`.

**Storage.** `data/raw/<date>/jobs.db` is an immutable pinned snapshot with a
SHA-256 manifest; `data/processed/` holds derived Parquet. **Neither is
committed.** Both are regenerable from the source database, the snapshot is
several hundred megabytes, and derived frames are output rather than input.
`models/` and `mlflow.db` are excluded for the same reason: they are rebuilt by
the commands above, and a run's provenance records the panel's hash so a metric
can be traced to the data that produced it.

## Collection, politeness and what is not collected

Data collection happens upstream, in the scraper, and this project inherits its
posture rather than restating it:

- `robots.txt` is checked in code before the first request to a host, on every
  run, and a disallowed path is not fetched.
- Per-host rate limiting with backoff on 429 and 5xx responses.
- An honest, identifying User-Agent. It is not a browser and does not pretend to
  be one.
- Public job-posting content only. **No personal data, no candidate data, no
  authenticated pages, no content behind a login.**
- Sources are public job boards and public Greenhouse job-board APIs, used as
  they are published.

Nothing collected is redistributed here: the database, the snapshots and the
derived frames are all local and untracked. What this repository publishes is
code, decisions and aggregate numbers.

---

## Testing

```bash
pytest                 # 261 tests
ruff check .
ruff format --check .
```

**The tests need no network and no data files.** Fixtures are built in code, the
API is exercised through an in-process transport, and the Streamlit UI is
rendered by Streamlit's own test runtime — so a green CI run means the same
thing a green local run means.

Three of them carry more weight than the rest:

- `test_a_fixed_posting_scores_the_same_number_forever` pins a probability for a
  fixed payload at a fixed instant. Every other test asserts a property, and
  properties survive a change in the feature logic. A pinned number does not:
  change how `age_days` rounds and it moves. `tests/test_api.py` asserts the same
  constant survives the HTTP round trip.
- `test_the_test_block_is_read_only_where_it_should_be` parses `src/` and fails
  if the test split is read anywhere but the two places entitled to.
- `test_serve_time_row_local_features_match_the_training_frame` builds the same
  posting down both the training path and the serving path and compares them
  column by column.

---

## Development

Branch per unit of work, PR per feature, and the suite green before either.
`ruff` is the only style authority (line length 100).

Generated files under `reports/` name the command that writes them; regenerate
rather than edit. `docs/design.md` records every decision with a date, the
reasoning, and what would change my mind — including the four still open.

---

## Deployment

**Not yet deployed**, by choice. The service, the container and the UI are
built and tested; what would go on the public URL today is a model fitted on a
synthetic fixture, and a link a stranger can hit should return a number that
means something. The deploy happens when the panel clears the depth gate and a
real artifact exists.

The free-tier cost is known in advance: the image is 1.08 GB, and a cold
container has to start Python, import XGBoost and unpickle a booster before it
answers — so the first request after a sleep is slow, and the UI's HTTP timeout
is set at 30 seconds for exactly that reason.

---

## Known failure modes and caveats

1. **"Closed" is not "filled."** The label is disappearance from the board. A
   posting can be pulled, expire, be reposted, or be moved to another system.
   Every claim this project makes is about disappearance.
2. **It is a Greenhouse model.** arbeitnow — 74% of the collected postings — is
   excluded because its crawls never observed a whole board. Whatever is learned
   here is learned from six Greenhouse boards and python_org, and the per-source
   breakdown is mandatory reporting for exactly that reason.
3. **The panel is short and the positives are few.** 75 positives across 5,693
   labelled rows. Differences of a few points between models will be inside the
   noise, which is why fold variance is reported and paired differences are used
   rather than differences of averages.
4. **Board context is missing at serve time.** Four features describe the board
   rather than the posting, and a caller holding one job ad cannot supply them.
   They are imputed when absent, which makes them inert for that caller — the
   response says so, and `docs/design.md` §12 records the open decision.
5. **Left truncation.** Postings already on the board when collection started
   had been open for an unknown time. `age_days` is measured from the employer's
   own publication instant where the archive provides one, never from when this
   project first looked.
6. **A label is never final.** "Never reappeared" reads the whole remaining
   panel, so a posting returning after two absences flips its earlier label. Two
   of 1,240 have. Measured, reported, and open.
7. **No real evaluation has happened yet.** Everything above describes a system
   that is built and verified on synthetic data. Until the depth gate clears,
   treat every capability claim as *tested*, and no accuracy claim as *made*.

---

## License

MIT.
