# Shelf-life — the ML problem definition

**Written 2026-09-04, against the panel as it stood that morning.** No model
exists yet, and §9 explains why one cannot yet be trained. This document is the
thing that has to be right *before* there is code, because every mistake below
is a mistake that a validation score will happily hide.

Scope note: this is a document about the *learning problem*. The dataset it
draws on is collected by the companion scraper project, whose `schema.md`
defines the tables and guard rules referenced below and whose `questions.md`
holds the questions this one descends from (last row of the second table).
Those documents live in that repository; every rule this document depends on
is restated here in full, so nothing below requires them to be read.

---

## 1. Problem statement

**The question, in English:** *this posting is on the board right now — is it
about to come down?*

That question is worth asking because a job board shows you a list of things
that all look equally available, and they are not. Some of those postings will
be closed within the week. Some have been sitting there for 860 days and will
still be there next spring. A board sorted by "new" cannot tell you which is
which, and neither can any single scrape — the information lives in the
*difference* between scrapes, which is the thing this project has been
collecting since 2026-08-29 and which nobody can go back and collect later.

**The question, as supervised learning:**

> Given everything observable about a posting at the moment it is observed on
> the board, estimate the probability that it will have been removed from that
> board within the next **H = 7 days**.

Three consequences follow immediately from that phrasing, and they are the
whole of §2, §3 and §4:

- It is a *conditional* question — asked about a posting that is open *now*.
  That makes the unit of analysis an observation, not a posting.
- It is asked at a specific instant, which fixes what may be used as input.
- It is answered by the future, which fixes what the label is and how much of
  the data can currently carry one.

**Why this and not "predict total lifetime".** Total lifetime — days from
posting to removal — is the more natural-sounding target and it is the wrong
one for this panel. The panel is six days old and the average posting on it was
already 83 days old when first observed. Regressing on total lifetime would
require observing both ends of a life; this dataset routinely observes neither.
§2 shows how the conditional form dissolves that problem rather than working
around it.

---

## 2. The unit of analysis — what one row is

> **One row = one (posting, complete-run observation) pair.** A *job-day*.

Not one row per posting. If posting `4571182008` was seen by the 08-31, 09-01,
09-02, 09-03 and 09-04 runs of `greenhouse:anthropic`, that posting contributes
**five rows**, each with its own features (the posting was one day older each
time, the board was a different size each time) and its own label.

### Why this is the right unit, and it is not a modelling flourish

Three properties of the real data force it.

**Left truncation.** Of the 1,240 postings ever seen by a complete run, **1,135
were already on the board at that source's first complete run**. We did not
watch them arrive. Their ages at first sight, from `posted_at`:

| Age when first observed | Postings |
|---|---|
| under 1 day | 14 |
| 1–6 days | 195 |
| 7–29 days | 308 |
| 30–89 days | 358 |
| 90–364 days | 289 |
| 365 days or more | 43 |

Mean 82.7 days, max 860.2. A per-posting model of total lifetime has to
explain what to do with a row whose lifetime is "at least 860 days, unknown
above that". A job-day model does not have the problem: it conditions on the
posting having survived to the moment of observation, and *age becomes a
feature instead of a missing outcome*. That is not a workaround. It is the
standard discrete-time hazard formulation, and the reason it is standard is
exactly this.

**Right censoring.** Every posting still on the board today has an unknown
lifetime. Under the job-day framing, only the last H days of rows are censored,
and they are dropped by a rule stated in §4 — not silently coerced to zero.

**Sample size.** 1,240 postings have been seen by a complete run — 1,165 of
them were still on the board on 2026-09-04 — times one observation per day. Four daily transitions are already recorded: **4,549
job-day pairs**, versus 1,240 postings. The unit that produces more data is
also the correct one, which is a pleasant accident and not the argument.

### The cost of this unit, stated

Rows are **not independent**. Five rows from one posting share a title, a
company and most of a description. This has two hard consequences that appear
again in §7:

- Any split that puts some of a posting's rows in train and others in test is
  leaking. **Split by time, and never randomly by row.**
- The effective sample size is not the row count. It is closer to the number of
  *removal events*, which is roughly 1.7% of rows. That is what bounds model
  complexity, and §9 does the arithmetic.

---

## 3. The prediction point

**t = the start timestamp of a complete run that observed the posting.**
In the data that is `runs.started_at` for a run with `status='ok'` and the
current `rules_version` — for example `2026-09-01T03:45:53Z`.

```
   posting first_published                    ← may be up to 860 days before t
              │
              │        ... unobserved by us ...
              │
              ▼
   ┌──────────────────────────────────────┐
   │  runs 08-31, 09-01  (complete)       │   INFORMATION AVAILABLE
   │  posting present, fields as recorded │   — everything at or before t
   └──────────────────────────────────────┘
              │
              ▼
        ●  t = 2026-09-01T03:45:53Z   ← THE PREDICTION POINT
              │
              ▼
           MODEL  ──────────────▶  P(removed by t + 7d)
              │
              ▼
   ┌──────────────────────────────────────┐
   │  runs 09-02 … 09-08                  │   THE FUTURE
   │  presence/absence in these runs      │   — the label, and nothing else
   └──────────────────────────────────────┘
```

Everything above the dot may be used as a feature. Everything below it may only
be used to compute the label. The line is not a guideline; §6 lists the
specific columns in this repository that sit below it and look like they sit
above it.

### Where "as of t" values actually come from

This matters more here than in most projects, because **`data/jobs.db`'s `jobs`
table is current-state, not as-of-t.** Reading `jobs.title` to build a feature
for a row dated 09-01 gives you the title as edited on 09-03. That is a leak,
it is silent, and it would improve your validation score.

| Source | As-of-t? | Use for |
|---|---|---|
| `snapshots/<stamp>.csv` | **yes, by construction** — one file per run, one row per posting that run saw, never rewritten | every tabular feature |
| `data/raw/<host>/<stamp>.html.gz` | **yes** | `description`, and the Greenhouse fields not yet in the schema (§5) |
| `job_changes` filtered to `observed_at <= t` | yes | edit-history features |
| `jobs` table | **no — current state** | labels and identity only |

`snapshots/` begins 2026-08-31, which is also when complete runs at
`rules_version = 2` begin. The two windows coincide, so nothing is lost.
`description` is deliberately absent from the snapshot CSVs (employer
copyright, see `snapshots/README.md`), so any text feature over the body has to
be built from the raw archive. That is the archive doing the job it was kept
for.

---

## 4. Target definition

### The population

**Complete-observation sources only:** the six Greenhouse boards and
`python_org`. **1,240 postings** have been seen by a complete run; 1,165 of
them were still on the board on 2026-09-04. The first number is the
labelable population, the second is the current board size used in §9.

`arbeitnow` (3,531 postings, 75% of the database) is **excluded from labelled
rows entirely.** Not down-weighted — excluded. On a paginated board that
reorders mid-crawl, a posting missing from a run may have been removed or may
have moved to a page already fetched, and the two are indistinguishable. Labels
built from it would be wrong at an unknown rate, in an unknown direction, and
would look exactly like labels that were right. A dataset's worst failure mode
is a label that is confidently wrong, and this is one.

### The label

Let `t` be a prediction point, `H = 7` days, and let a *complete run* mean
`status = 'ok'` at the current `rules_version`.

Define `t_gone(j)` — the removal time of posting `j`:

> the start time of the earliest complete run in which `j` was absent, such
> that `j` was also absent from the next complete run and never re-appeared.
> Undefined if no such pair exists.

That is the project's `ABSENCE_CORROBORATION = 2` rule (the scraper's
`schema.md`, "Guard 2"), imported unchanged. It is imported rather than relaxed because a
label is a stronger claim than a report line, not a weaker one.

Then, for row `(j, t)`:

| Condition | `y` |
|---|---|
| `t_gone(j)` is defined and `t_gone(j) <= t + H` | **1** — removed within the horizon |
| `j` was observed by a complete run at or after `t + H` | **0** — survived the horizon |
| otherwise | **row is dropped** |

The third line is the one that has to be written down. A row whose horizon
extends past the last complete run has **not** survived; we simply have not
looked yet. Filling those with 0 is the single most common way this exact
problem is got wrong, and it biases every estimate toward "postings last
forever" — the direction that would flatter a model built by someone hoping
postings last forever.

### Horizon

**H = 7 days, primary.** Chosen against the measured hazard, not by taste:

| Horizon | Implied positive rate | Comment |
|---|---|---|
| 1 day | 1.7% | labelable today, but severely imbalanced and dominated by ±1-run timing noise |
| **7 days** | **≈ 11.3%** | matches the decision it feeds ("apply this week or not") |
| 14 days | ≈ 21.3% | more balanced; costs a further week of censored rows at each end |

Derived from the observed daily disappearance rate across complete runs —
77 disappearances in 4,549 job-day transitions, 1.69% per day:

| Run date | Present in previous complete run | Absent in this one | Rate |
|---|---|---|---|
| 2026-09-01 | 1,104 | 20 | 1.81% |
| 2026-09-02 | 1,143 | 21 | 1.84% |
| 2026-09-03 | 1,143 | 14 | 1.22% |
| 2026-09-04 | 1,159 | 22 | 1.90% |

H=1 is retained as a **pipeline smoke test only** — it is the one horizon with
labels today, so it can prove the feature-assembly code runs end to end before
there is anything to learn from.

### Two known imprecisions, both accepted

**Interval censoring.** Runs are daily, so a removal is only located to within
the ~24 hours between two runs. At a 7-day horizon a ±1-day boundary error
affects only postings removed within a day of `t + H`; at H=1 it is most of the
signal, which is the second reason H=1 is not the primary horizon.

**Reappearance.** A posting can vanish and return. Across all complete runs
this happened to **2 postings out of 1,240** (0.16%). Small enough to accept,
large enough that `t_gone` is defined with "never re-appeared" in it rather
than being defined on the first absence.

### One operational rule that follows

Bumping `runs.rules_version` **discards the label baseline** — runs are only
compared within a version (the scraper's `schema.md`, "Guard 3"), so a bump
restarts the clock on every horizon and puts a discontinuity in the label
series. Before this project, that was a reporting inconvenience. Once labels
are being accumulated for a model, it is a decision with a training-set cost
attached, and it should be taken deliberately and recorded here.

---

## 5. Features allowed

Everything in this section is knowable strictly at or before `t`, and every
entry names where the as-of-t value comes from.

### 5.1 Posting attributes, as recorded by the run at t

Source: that run's `snapshots/<stamp>.csv` row.

| Feature | Notes |
|---|---|
| `title` | text; and derived seniority/keyword flags — a *derived* feature, so its own derivation code must not look at the label |
| `company` / board token | the strongest available grouping variable |
| `location` | as posted, unnormalised |
| `remote` | three-state; `NULL` is "not stated" and must stay distinguishable from `False` |
| `salary_min`, `salary_max`, `currency` | present for 890 of 1,207 Greenhouse postings (74%) |
| `salary_raw` | and `salary_raw IS NOT NULL` as its own indicator — "the posting stated pay at all" is plausibly more predictive than the amount |
| `posted_at` | the origin, used below |

### 5.2 Time, as of t

| Feature | Definition |
|---|---|
| **`age_days`** | `t − posted_at`. The single most important feature in the problem, and the one that carries the left-truncation correction |
| `posted_at` day-of-week, month | |
| `t` day-of-week | boards move on weekdays; four daily transitions is not enough to see it yet |

**Do not use `t − first_seen` as age.** `first_seen` is when *we* first saw it,
which for 1,135 of 1,240 postings is an artefact of when this project started.
It is legal (it is ≤ t) but it is not age, and a model given it will learn the
scraper's start date.

### 5.3 Board context, as of t

Computed only over complete runs with `started_at <= t`.

| Feature | Definition |
|---|---|
| `board_size_at_t` | postings on that board in run `t` |
| `board_hazard_prior` | that board's disappearance rate over prior runs — a leaky-looking feature that is legitimate *if and only if* the window ends at `t` |
| `n_same_title_on_board` | duplicate-title postings are a repost/ghost signal |
| `board_growth` | `board_size_at_t` minus board size at the previous run |

### 5.4 The posting's own history, up to t

Source: `job_changes` filtered to `observed_at <= t`.

| Feature | Definition |
|---|---|
| `n_edits_before_t` | employer edits, excluding rows attributable to our own parser (`old_parser_version != new_parser_version`, and **all rows before 2026-09-04**, which are unattributable — see `schema.md`) |
| `days_since_last_edit` | |
| `salary_edited`, `title_edited` | flags |
| `n_complete_runs_observed` | usable, but strongly confounded with "how long ago the panel started" — include only alongside `age_days` |

### 5.5 Available in the raw archive, not yet in the schema

The Greenhouse payload carries fields the parser currently drops. Measured on
1,133 postings from 2026-09-04:

| Field | Coverage | Why it is worth extracting |
|---|---|---|
| `departments` | 100% | a real category label, better than inferring one from the title |
| `requisition_id` | 100% | identifies a *req*, so it can link a repost to the posting it replaces |
| `offices` | 93% | structured location, where `location` is free text |
| `metadata` | 85% | board-specific custom fields |
| `description` | 100% | the text feature; already stored, but gitignored and DB-only |
| `application_deadline` | **0%** | populated on no board here; do not plan around it |

These are recoverable for **every run since collection began**, because the raw
response was archived before parsing. Adding them later costs a backfill script
and no lost history. This is the project's day-one rule — archive the raw response
before parsing it — paying out precisely as advertised.

---

## 6. Features forbidden

Three classes. The first is obvious, the second is where this project's schema
will actually catch someone out, and the third is subtle.

### 6.1 The label in disguise

| Do not use | Why |
|---|---|
| `jobs.last_seen` | it **is** the outcome |
| `julianday(last_seen) - julianday(first_seen)` | the outcome, minus a constant |
| any `job_observations` row whose run is after `t` | the outcome, one join away |
| "is the posting in the DB today" | the outcome, phrased as a filter |

A model with `last_seen` in it scores near-perfectly and knows nothing. If a
first model scores above ~0.7 PR-AUC on this problem, the first hypothesis is
this table, not success.

### 6.2 Current state read as past state

| Do not use | Why |
|---|---|
| `jobs.title`, `.salary_*`, `.location`, `.remote`, `.description` for a row where `t` is in the past | the `jobs` table holds the **latest** value; a posting edited after `t` leaks its edit backwards |
| `jobs.content_hash` | current-state, same reason |
| `job_changes` rows with `observed_at > t` | the future's edits |
| `board_hazard_prior` over a window that extends past `t` | this is the leak that hides inside an otherwise-correct feature |

The mechanical defence: **assemble every feature from `snapshots/` and the raw
archive, and let the `jobs` table supply only identity and labels.** A feature
builder that never opens the `jobs` table cannot commit this class of error.

### 6.3 Facts about the instrument, not the job

| Do not use | Why |
|---|---|
| `parser_version` | it encodes when *our code* was deployed, which correlates with calendar time, which correlates with which rows are censored. A model can learn the deploy schedule and appear to predict removals |
| `rules_version` | same, and it also partitions the label definition itself |
| `runs.rows_parsed`, `page_cap`, `pages_fetched` | measurements of our crawl. `questions.md` already records `rows_parsed` reading as hiring activity when it was a `--pages` setting changing |
| `first_seen` as a proxy for age | see §5.2 |
| row order / `jobs.id` | a surrogate key that is monotonic in insertion time |

### 6.4 One narrow exception, made explicit

Text featurisation (a TF-IDF vocabulary, say) fitted on the **excluded**
`arbeitnow` rows is permitted, because those rows carry no labels and cannot
transmit one. It is permitted **only** if the fitting window also ends at the
train/test cut — a vocabulary fitted on the whole corpus still carries
information about which words appear in the future.

---

## 7. Success metric

### The base rate is the thing to beat

At H=7 the positive rate is ≈11.3%. A constant predictor achieves average
precision ≈ 0.113 and ROC-AUC exactly 0.5. Accuracy is meaningless here —
always predicting "stays" scores 88.7% — and is not reported.

### Metrics, in the order they decide anything

| Metric | Role | Why |
|---|---|---|
| **PR-AUC (average precision)** | primary | the positive class is the rare, interesting one; PR-AUC is sensitive to it in a way ROC-AUC is not at 11% |
| **Brier score + reliability curve** | co-primary | the output is used as a probability (§8), so "30%" has to mean 30%. A model that ranks well and is calibrated badly fails the actual use |
| Precision@20/day | operational | mirrors the real decision: a shortlist, not a full ranking |
| ROC-AUC | reported, not decisive | comparable to other people's numbers; insensitive to the imbalance that matters here |

### Baselines, in ascending order of seriousness

A model must beat **all three** to have earned anything:

1. **Constant base rate.** AP ≈ 0.113. Beating this proves only that features exist.
2. **`age_days` alone**, logistic. This is the honest baseline: most of the
   signal in survival problems is duration dependence. **If the full model does
   not beat this, the posting's content contributes nothing and the finding is
   that** — a real result, and a publishable one for the write-up.
3. **Per-board hazard**, i.e. predict each board's historical daily rate
   compounded over H. Beats #1 whenever boards differ; a content model that
   cannot beat it has learned only which company posted the job.

### Validation protocol

- **Time-based split.** Train on prediction points `t <= T_cut`; test on
  `t > T_cut + H`. The gap of H is not optional: without it, a test row's label
  window overlaps the training period and the same removal event appears on
  both sides.
- **Rolling-origin evaluation** once there are enough weeks, rather than one
  split — a single split on a short panel measures one week's weather.
- **Never `train_test_split(shuffle=True)`.** It puts rows from the same
  posting on both sides *and* puts the future in the training set. Two leaks,
  one line.
- **Report the confusion matrix at a stated threshold**, not just the curves.
  The threshold is a business decision (§8) and belongs in the write-up.

### Acceptance bar, set now so it cannot be moved later

> The model is worth keeping if, on a held-out later time window, it beats the
> `age_days`-only baseline on PR-AUC by a margin larger than the spread across
> rolling-origin folds, and its reliability curve is within ±5 percentage
> points of the diagonal in the 0–30% band where nearly all its mass will sit.

---

## 8. Business interpretation

### What the number is for

**Job search — the primary use, and the reason this project exists.** A daily
list of ML/Python postings ranked by "likely gone within a week" answers *what
do I apply to tonight*. It re-orders a board that is otherwise sorted by a date
that does not mean what it looks like.

**Ghost-job detection — the inverse reading.** A posting that is 200 days old
and carries a low removal probability is a posting the model believes is not
going anywhere. The README already states the claimed 18–22% ghost-job figure
as claimed rather than as fact; a fitted hazard is this dataset's own evidence
on it, and it comes from postings' behaviour rather than from a survey.

**Board and market structure.** Hazard by company, seniority and salary band is
a time-to-fill proxy that no single scrape can produce — the question
`questions.md` put in the "only answerable with history" table.

### The cost asymmetry, and what threshold follows

| Error | Cost |
|---|---|
| Predicted "closing soon", actually stays open | a rushed application. Hours. |
| Predicted "stays open", actually closes | a job never applied to. Unrecoverable. |

The costs are not symmetric and the second is worse, which argues for a
**low threshold and a recall-leaning operating point.** It also argues against
optimising a symmetric metric like accuracy or plain F1, and for reporting
precision at a fixed alert budget — 20 flagged postings a day is a list a human
can actually read.

### What a probability from this model does *not* mean

It means **"removed from this board"**. Not "filled". A posting comes down when
it is filled, when the req is cancelled, when the board is tidied, or when the
company reorganises its careers page. The model predicts an observable event,
and the write-up must say so in exactly those words rather than quietly
promoting it to "hired".

---

## 9. What the data cannot support yet

This section exists so that the honest answer is written down before it is
inconvenient.

**Today, 2026-09-04, the number of rows labelable at H=7 is zero.** Complete
runs at the current `rules_version` begin 2026-08-31. A prediction point on
that date needs runs through 09-07 to establish survival, and a corroborated
removal needs one further run. **The first complete labelled cohort settles on
or about 2026-09-08.**

**What accumulates from there**, at ~1,165 postings per daily complete run and
a 1.69% daily hazard:

| Complete-run days | Labelable job-days at H=7 | Expected removal events |
|---|---|---|
| 14 | ~7,000 | ~110 |
| 28 | ~23,000 | ~380 |
| 56 | ~56,000 | ~950 |

**The right column is the one that constrains the model.** Rows from one
posting are near-duplicates; the effective sample size is the event count. At
the usual rule of thumb of ~10 events per parameter, eight weeks of collection
supports on the order of 90 free parameters. That is a regularised logistic
regression or a small gradient-boosted model over a few dozen features — which
is, conveniently, exactly the classical-ML scope this project set out to build.
It is not a transformer over descriptions, and no amount of wanting one changes
the arithmetic.

**Therefore:**

- **Nothing to do but keep collecting.** The daily run at 03:45 UTC is the
  entire work item for the next several weeks.
- **Do not bump `rules_version`** without accepting a restart of the label
  series (§4).
- **Backfill the §5.5 archive fields before modelling, not after.** They are
  already captured; extracting them is offline work that can happen during the
  collection wait, and doing it later means re-running feature assembly.
- **Build the feature-assembly pipeline now against H=1**, where labels exist
  today. It proves the plumbing and — more to the point — proves the
  no-leakage discipline of §6 while the stakes are still zero.

---

## 10. Done when — the three answers

**What does one row represent?**
One observation of one posting by one complete scrape run: the pair
*(posting `4571182008`, run of 2026-09-01 03:45 UTC)*. A posting present for
five runs contributes five rows. Not one row per posting — because 1,135 of
1,240 postings were already on the board before this project first saw them,
and a per-posting lifetime is therefore unobservable at both ends.

**What are we predicting?**
A binary outcome: whether that posting is **absent from the board within 7 days
of that observation**, where absence means missing from two consecutive
complete runs and never returning. Not whether it was filled — whether it came
down. Rows whose 7-day window has not fully elapsed are dropped, never labelled
0.

**When is the prediction made?**
At the instant of that run — `runs.started_at`. Every feature is a fact
established at or before that instant, assembled from that run's snapshot file
and raw archive; the `jobs` table, which holds current state, is used only for
identity and for computing labels. Everything after that instant is the answer,
and touching it is the failure this document exists to prevent.
