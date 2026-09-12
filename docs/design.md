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
absence from two consecutive complete runs, final at that moment whatever the
posting does afterwards (§11, 2026-09-09), scanning from the posting's first
sighting (2026-09-11); rows whose horizon has not elapsed are dropped, never
labelled 0.

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

### MEASURED 2026-09-09 — the 7-day rate is **7.76%**, not 11.3%

First settled cohorts, on the 2026-09-08 snapshot: **174 positives in 2,242
labelable job-days**. The planning estimate overshot by about 45% relative, and
in the direction that makes the problem harder — rarer positives, less to learn
from per wave.

The extrapolation failed where it was warned it might. `1 - (1 - p)^7` treats
each day as an independent draw at the *transition* hazard, but the 7-day label
requires the posting to be gone **and never to return**, which is a strictly
smaller event than seven chances to disappear once. Reading back: the measured
7-day rate implies a daily equivalent of `1 - (1 - 0.0776)^(1/7)` = **1.15%**
against the 1.69% day-over-day transition rate.

**A number this project nearly got wrong in a way that would have looked right.**
Labelling on outcome alone reports **13.58%** on the same snapshot — much closer
to 11.3%, and wrong. A closure is knowable the moment it happens while survival
needs the whole window to elapse, so every cohort inside the horizon of the
panel's edge contains only closures: 151 rows, 100% positive, pooled into the
headline. The bias sat between the estimate and the truth and would have read as
a confirmation of the estimate. `compute_labels` now requires a settled cohort;
`docs/problem_definition.md` §10 always specified this and the implementation did
not.

### The hazard curve, measured on settled cohorts

| age at prediction | rows | closures | 7-day rate |
|---|---|---|---|
| 0–7 days | 225 | 19 | 8.4% |
| 7–14 | 276 | 14 | 5.1% |
| 14–30 | 368 | 25 | 6.8% |
| 30–60 | 442 | 41 | 9.3% |
| 60–120 | 387 | 42 | 10.9% |
| 120+ | 513 | 29 | 5.7% |

**Flat and non-monotone**, over a 2× range with 14–42 events per bucket. This is
the second of the two mind-changers below, and it fires: §7 of the problem
definition claims duration dependence carries most of the signal, and on this
data it carries very little. It is consistent with what the H=1 ladder already
found — `age_ceiling`, the best any age-only rule could do, scored 0.0229
against a 0.0190 base rate.

**Would change my mind:** a measured 7-day rate far from 11%, or a hazard curve
steep enough in the first week that a 7-day window averages away the signal.

**Verdict on H, 2026-09-09: unchanged at 7 days.** The rate moved and the hazard
curve is flat, so the first trigger fired and the second did not — the worry
there was a *steep* early curve being averaged away, and a flat one is not
averaged away by a wider window. What the flat curve costs is a feature, not a
horizon: it says `age_days` is weak, which is a finding to report rather than a
reason to re-cut the target. 7.76% is still a workable base rate, and it remains
the horizon that matches the decision the prediction feeds.

**What it costs is time**, and that is now the binding constraint. At H=7 the
embargo is 8d10h against daily waves, so each boundary discards **9** waves:
**20 labelled waves for a legal split, 31 for three folds.** On 2026-09-09 there
are 2. Projected: a legal split on **2026-09-19**, folds on **2026-09-30**.

*Revised 2026-09-11:* the 2026-09-09 crawl was missed, which widened the
embargo for the whole panel — each boundary now discards 10 waves, the gates
are **22** and **34** labelled waves, and the projections moved to
**2026-09-21** and **2026-10-03**. The figures above are what was true when
the decision was made and are left as written; the live ones are
`reports/readiness.md`, which is the only place they are maintained.

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

## 4. Is board identity a feature? — **DECIDED 2026-09-09: no**

**The deployment story, settled first because §4 cannot be answered without it:
score any posting, including one from a board never scraped.**

That is not an aspiration, it is a description of what already shipped.
`POST /predict` takes a posting, not a board id. `/contract` publishes what a
caller may send. Board-level fields are optional and imputed when absent (§12).
An interface shaped that way and a model that needs to know the board are not
the same product, and the interface is the one people can already use.

**The measurement, on the 2026-09-08 snapshot at H=7, settled cohorts.** Per-board
closure rates with 95% Wilson intervals, against a pooled rate of **7.76%**:

| board | rows | closures | rate | 95% interval |
|---|---|---|---|---|
| greenhouse:anthropic | 1,141 | 91 | 7.98% | 6.5–9.7% |
| greenhouse:gitlab | 444 | 44 | 9.91% | 7.5–13.0% |
| greenhouse:figma | 323 | 16 | 4.95% | 3.1–7.9% |
| greenhouse:duolingo | 169 | 7 | 4.14% | 2.0–8.3% |
| greenhouse:discord | 102 | 12 | 11.76% | 6.9–19.5% |
| greenhouse:airtable | 32 | 0 | 0.00% | 0.0–10.7% |
| python_org | 31 | 4 | 12.90% | 5.1–28.9% |

**All seven intervals contain the pooled rate.** No board is distinguishable
from the board average on this sample. Corroborated independently by the H=1
ladder, where the `board_hazard` rung — a predictor that *is* nothing but the
per-board historical rate — scored 0.018931 against a 0.018982 base rate, i.e.
below a constant.

**So board identity is excluded**, and the four columns that carry it move
together (`source`, `company`, `url`'s domain, and any missingness indicator over
the archive-derived columns). `include_board_identity=False` was already the
default in `src/features/preprocessing.py`; this makes it a decision rather than
an unexamined default.

**Stated honestly: this is not a demonstration that boards do not differ.** With
174 closures the intervals are wide — figma 3.1–7.9% against discord 6.9–19.5%
— and the sample cannot separate them. What it establishes is that including
board identity asks the model to learn a difference *this data does not show*,
at a real cost: a model that needs the board cannot score a posting from a board
it has never seen, which is the product. Absent a demonstrated benefit, the
cheaper error is to leave it out.

**Would change my mind:** a per-board rate whose interval clears the pooled rate
once depth accrues, or a leave-one-board-out transfer gap that is large — the
measurement `src/models/generalisation.py` runs, and which needs more positives
per board than this panel has. Either would say the boards genuinely differ, and
the deployment story would then have to buy that difference explicitly rather
than inherit it.

---

### The original argument, kept because the decision rests on it

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

**How it gets answered — added 2026-09-07.** This section and the model card's
"essentially a Greenhouse model" caveat are the same worry, and neither was
measured. The per-source breakdown `CLAUDE.md` §4.5 requires is weaker evidence
than it looks: it scores each board with a model **fitted on that board**, which
answers *does it work here* rather than *would it work somewhere new*.

`src/models/generalisation.py` answers the second question by holding a whole
board out of the fit, and it scores each fold **twice on the same rows** — once
with a model that never saw the board, once with a model that did. The gap
between the two is what board-specific learning was worth. The control is not
optional: boards differ in base rate from 0.0090 on figma to 0.0173 on discord,
so a low transfer score alone could be the board being harder rather than the
model failing to carry over.

**It cannot run on this panel, and the refusals are the finding for now.** Per
board, positives in the whole labelled frame: anthropic 52, gitlab 24, figma 10,
discord 6, duolingo 5, python_org 3, airtable 0. Testing on duolingo means a
five-positive test set; airtable's fold is undefined. And holding out anthropic
removes 52% of the training positives, so its transfer arm would be fitted on
half the data *and* one fewer board, with the two effects inseparable in the
result. Both guards are enforced — `MIN_HELD_OUT_POSITIVES = 10` and
`MIN_TRAIN_SHARE = 0.6` — and a refused fold is reported with its reason rather
than dropped, because a board missing from the table is a board nobody knows was
untested.

**Would change my mind about `source`:** a transfer gap consistently near zero
would say the model is using properties of postings rather than of boards, which
weakens the case for excluding `source` and strengthens the deployment story of
scoring a board the model has never seen. A large gap says the opposite, and is
the number that should sit beside the model card's caveat instead of the caveat
standing alone.

---

## 4a. Board *availability* patterns — **DECIDED 2026-09-11: allowed, with transfer as the criterion**

§4 removed the columns that name the board. It did not, and could not, remove
the information — and until 2026-09-11 nothing had measured how much stayed.
`python -m src.models.board_fingerprint` does: the production `Pipeline`, fitted
on the training block only, with its estimator's target swapped for `source`.
[`reports/board_fingerprint.md`](../reports/board_fingerprint.md).

**The board is recoverable from the production features at 100.0% accuracy**
(macro-F1 1.000, against 51.0% for guessing anthropic). Every one of the seven
boards, every validation row.

Where it lives is not where the missing-value policy suggested. The concern
that motivated the "deliberately not indicated" notes in `preprocessing.py` —
that `departments == __missing__` *means* python_org, and a sentinel is a
board id — is real for python_org and small overall: **missingness alone
scores 19.2%**, below the majority guess. The identity is in the *values*:

| feature alone | accuracy |
|---|---|
| `board_size_at_t` | **1.000** |
| `location` | 0.671 |
| `n_metadata` | 0.642 |
| `content_chars` | 0.544 |
| `departments` | 0.299 (python_org perfectly, nothing else) |

`board_size_at_t` is a board's name in integer form — anthropic ~600, gitlab
~230, figma ~160, duolingo ~85, discord ~45, python_org 30, airtable 16 — and
it is redundant: with it removed the rest still scores 99.3%, and with all
four board-context columns removed (§12's "absent" set, the nearest thing this
design has to a board-independent feature set) **97.6%**. `location` is each
employer's office cities; `n_metadata` and `content_chars` are each employer's
posting template. Remove those and what is left is not a posting.

### The decision

**Board-availability patterns and template-derived features are allowed,
explicitly.** They are as-of-`t`, they are what a posting *is*, and at serve
time a python_org posting genuinely has no `departments` and a posting from a
new board genuinely has its own location and template. None of it is leakage.

**Matrix neutrality was the wrong criterion and is dropped.** On seven boards
any representation rich enough to describe a posting identifies its employer;
a feature set that cannot is a feature set with nothing in it. The two options
were to say so or to test a board-independent set, and the second has now been
tested: it does not exist here.

**Transfer is the criterion, and it is measured on the model rather than
argued from the matrix.** What a fingerprint can cost is that the model learns
*this is gitlab* as a proxy for hazard and has nothing to say about a board it
has not seen. `generalisation.leave_one_board_out` removes a board from the fit
and scores it cold; `model_comparison.md` already reports it. From this
decision it is an **acceptance check at the freeze**: the chosen candidate's
leave-one-board-out table is read before `freeze --run`, and a held-out board
whose PR-AUC collapses to its base rate while the fitted-on score does not says
the model spent the fingerprint. That is a reason not to freeze that candidate,
not a number to note.

**`board_size_at_t` is named because it is the sharpest case.** §12 keeps board
context on a measured 0.0019 cost of removal, with serve-time imputation for a
board the batch cannot describe. This section adds what §12 did not know: the
imputed value is the *median board's identity*, so a new-board posting is scored
as if from a board of ~160 postings. That is acceptable only while the
leave-one-board-out check above passes with board context imputed the way the
service imputes it — `train.serve_time_regime` — and §12's decision is
conditional on that from now on.

### The gate — **DECIDED 2026-09-12: transfer collapse is a freeze refusal**

"Read before `freeze --run`" was a sentence, and two things made even the
reading weaker than it looked: the transfer driver in `evaluate` always
built XGBoost, whatever candidate the rule selected, and the fingerprint had
only ever been run at H=1. Both are fixed, and the check is now enforced in
code as the third evidence refusal, after depth and before the clean-tree
check, on the candidate being frozen, with board context withheld the way a
posting from an unknown board arrives (`generalisation.leave_one_board_out`,
`serve_time=True` — the regime §12 is conditional on).

The rule, fixed now while every leave-one-board-out table in the repository
is H=1 or refused for depth:

- **Collapsed** — over the held-out boards that clear the fold guards, the
  mean of (transfer PR-AUC − that board's base rate) is at or below zero, on
  at least two boards. On boards it has not seen the model is no better than
  the prior. **`freeze` refuses**, exit 3. `--accept-transfer-collapse`
  overrides it and is recorded on the artifact; the model is then described
  as fitted to these boards and nothing wider.
- **Board-specific** — the existing reading (mean gap larger than its spread
  across boards): measurably better on boards it has seen. Not a refusal —
  it may still be the best model for these seven — but the verdict travels on
  the artifact and the model is *not* described as applicable to boards it
  has not seen.
- **Unmeasured** — fewer than two boards could be held out. Depth, not a
  result; recorded; no claim about unseen boards.
- **Intact** — recorded, and the claim is allowed, scoped to a board *of the
  kind these seven are*. Never "arbitrary boards": seven employer boards on
  two platforms are not a sample of job boards.

`Metadata.transfer` carries the verdict, the per-board lifts, the skipped
boards with reasons and whether a collapse was overridden; `test_results.md`
has the table; `evaluate` states the gate's reading for the chosen candidate
beside the verdict prose, so the day's comparison says in advance what the
freeze will do. `rehearse.sh` now runs the fingerprint at H=7 too.

**Would change my mind:** a leave-one-board-out gap that is large once there
are enough positives per board to measure one, which would say the model is
spending the fingerprint and the §12 columns are the first to remove. Or an
eighth board whose posting template resembles none of the seven, scored live
through the API and returning probabilities that sit on the base rate — the
product's claim, tested on the product.

---

## 5. The metric and the cost asymmetry — **DECIDED 2026-09-04**

**PR-AUC (average precision) primary; Brier score and a reliability curve
co-primary.** Precision@20/day as the operational read. ROC-AUC reported for
comparability but not decisive. **Accuracy is not reported** — at an 11.3%
positive rate, always predicting "stays" scores 88.7%.

The output is consumed as a probability, so ranking well while calibrated badly
is a failure of the actual use, not a technicality — hence Brier alongside
PR-AUC rather than after it.

**Cost asymmetry:** a false "removal soon" costs a rushed application, measured
in hours. A false "stays open" costs a job never applied to, which is
unrecoverable. The second is worse, so the operating point leans to recall and
the threshold is chosen against a fixed alert budget rather than at 0.5.

**Every headline number carries an interval — added 2026-09-07.** The test block
will hold on the order of twenty positives, and a bare PR-AUC at that count is a
number whose second decimal is decoration. PR-AUC, precision, recall, Brier and
ECE each get a 95% percentile interval from **resampling postings, not rows**.

That unit is the decision worth defending. The panel is one row per (posting,
crawl) at about six rows per posting, so 8,037 labelled rows are not 8,037
independent observations — the board does not sample job-days, it accumulates
postings and observes them daily. A row-level bootstrap encodes the opposite
claim. Measured on a synthetic block of 241 rows and 99 postings the two
disagree, and the cluster interval came out *narrower* — [0.0969, 0.2508]
against [0.0939, 0.2782] — which is the opposite of the usual "clustering
inflates variance" intuition, and is not the argument. The cluster bootstrap is
right because it matches the sampling design; it would still be right if it came
out wider.

Threshold-dependent metrics are measured at the **frozen** threshold, held fixed
across resamples. Recomputing the budget threshold inside each resample would
mix how well the model separates with where the operating point happened to
land, and the artifact ships one threshold rather than a distribution.

**Recalibration — DECIDED 2026-09-12: by a rule, fixed before the number.**
§5 made calibration co-primary and said nothing about what to do when the
validation curve comes back bent — a decision that would then have been made
on the afternoon the numbers appeared, by the numbers. The rule is in
`src/models/calibration.py`, written while every validation ECE in the
repository is H=1 or synthetic:

> Recalibrate iff validation ECE > **0.25 × the validation base rate** and the
> block holds at least **30 positives**. Method: isotonic regression, fitted
> on the validation block only, wrapped around the pipeline's final estimator
> so the artifact stays one `Pipeline`.

Relative to the base rate rather than absolute, because at a 7.7% positive
rate an ECE of 0.02 is a quarter of everything there is to predict and at 50%
it would be noise; the positives floor is the bootstrap's own fragility line —
a monotone curve fitted to fewer events is fitted to those events. `evaluate`
states the verdict for the chosen model in the comparison report; `freeze`
applies it and records the decision on the artifact either way.

Why this is safe to pre-register: the operating point is a rank statistic (the
budget-th validation score) and isotonic regression is monotone, so
recalibration cannot reorder postings or drop one from the alert list. Its
one effect on *who* is flagged is at ties — isotonic pools neighbouring
scores into steps, and `probability >= threshold` includes ties — so it can
add a posting at the boundary, never remove one. What it changes is what the
percentage shown to a person means, which is the point. The test block's ECE,
opened after as before, is the honest measure of whether the curve
transferred. Tests hold the monotonicity, the alert-list property, and that
the artifact still serves.

**Would change my mind:** a validation block with enough positives that
Platt scaling's two parameters would generalise better than isotonic's
steps — at hundreds of events isotonic wins; at thirty it is a coin flip and
the floor is what keeps it honest.

**Intervals on validation too — added 2026-09-12.** The same posting-clustered
resampler, at the frozen threshold, on the validation block, so the
side-by-side table in `test_results.md` carries a spread on both sides; and the
comparison report carries it for the chosen model before the test block is
opened. And the **model's** performance on the incumbent stock against the
incident flow is reported beside the first-observation and seen/unseen slices —
the cohort audit says whether the *label* is indifferent to cohort, this says
whether the model is.

**The interval and the fold spread answer different questions**, and both are
reported. The interval asks how much this block's number would move on a
different sample of postings; the fold spread in `reports/model_comparison.md`
asks how much it would move on a different week. Neither substitutes for the
other, and a difference smaller than either is not a difference.

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

**It is measured twice, and only the second one can accept anything.**

| | What it measures | What it omits | Can it accept? |
|---|---|---|---|
| **Baseline** — the no-artifact image | process start, interpreter, the scikit-learn and XGBoost imports, on the real instance | joblib, unpickling the pipeline and the booster, the code path a prediction takes | **no** |
| **Definitive** — an image built from a real release | all of the above, plus the load path and a first prediction | nothing that a visitor would experience | **yes** |

The baseline is worth taking *now*, before the panel clears, because it is a real
measurement of a real instance and it can already fail: a baseline over the
criterion can only get worse once an artifact is added, and finding that out
today costs nothing. What it cannot do is pass. The image it comes from never
touches the model, so **whatever the unpickle costs on a tenth of a CPU is
exactly the part the baseline is missing** — and that part is the reason this
criterion exists at all.

**The baseline, measured 2026-09-06.** Taken against
`https://shelf-life-5hin.onrender.com` — the no-artifact image, `MODEL_TAG`
empty, `/health` reporting `model_loaded: false` — after 16 minutes of enforced
idle with nothing else touching the service:

| | |
|---|---|
| Cold `/health`, first request after 16 idle minutes | **32.65 s** (HTTP 200) |
| The criterion | 90 s |
| Left over for the entire load path | **57.35 s** |
| Verdict the script recorded | `BASELINE` — not `ACCEPTED` |

`/predict` was not timed. With no model it answers 503 without reaching the load
path, and that omission is precisely what makes this a lower bound.

**Thirty-six percent of the budget is gone before a single byte of model is
read.** That is the finding, and it is not a comfortable one. The definitive
measurement has to fit opening joblib, unpickling a `ColumnTransformer` and a
booster, and scoring one row into the remaining 57 seconds — on the same tenth of
a CPU that just spent 32 on a platform wake and the imports.

**A tempting cross-check, and why it is weaker than it looks.** The same
no-artifact image answered in **2.39 s** on this laptop, which invites a ~14×
"0.1 CPU factor" and an extrapolated prediction for tomorrow. It does not support
one. The local run restarted an already-created container against a warm page
cache on a full core of arm64; the Render figure includes platform scheduling and
starting a container from cold on amd64. The two numbers do not measure the same
events, so their ratio is an order-of-magnitude indication rather than a
calibration, and nothing should be forecast from it that tomorrow's measurement
would then be read as confirming. It is recorded because it was taken, not
because it is evidence.

**The architecture is not accepted until the definitive measurement is within the
criterion.** Until then it is provisional.

**And a passing baseline is not evidence that it will pass** — an earlier draft of
this entry said it was "evidence that the definitive one might", which is the
loose phrasing the whole distinction exists to prevent. The inference runs one
way only, because the baseline is a lower bound:

| Baseline result | What follows about the definitive measurement |
|---|---|
| **over** the criterion | it fails too, **conclusively** — the definitive can only be slower, so stop here and reassess without waiting for a model |
| **within** the criterion | **nothing.** The omitted work — opening joblib, unpickling a pipeline and a booster on 0.1 of a CPU — is not bounded by anything this run measured |

The asymmetry is the point. A baseline is worth taking because it can end the
question early, not because it can reassure anyone. The script says which kind of
run it just did rather than taking the caller's word for it — it asks `/health`
whether a model is loaded — because a caller who has to remember which sort of
measurement they are looking at will eventually file a lower bound as a result.

### The acceptance protocol — **ADDED 2026-09-11**

Everything above says *that* the definitive measurement is owed. This says
what it consists of, so that on the day it is a checklist rather than a
judgement, and so that the estimate below (§7e-ii: baseline 32.65 s plus a
locally measured load cost of 25.18 s, about 58 s) is never mistaken for it.
The estimate is two measurements of different things on different machines,
added; the acceptance is one measurement of the right thing on the right one.

| needed | produced by | recorded in |
|---|---|---|
| a true cold start with the **real** artifact | `scripts/cold_start.sh` against the public URL after `MODEL_TAG` names a real release; the script reads `/health` and files a run as `BASELINE` (no model), `REHEARSAL` (synthetic model) or `DEFINITIVE` (real) — only the last can accept | `reports/cold_start.md` |
| artifact download time | not part of a cold start: the artifact is fetched at *build* time and baked into the image, so the download is paid once, in Render's build log, never by a visitor | the build log; the `fetched … checksum verified` lines |
| deserialisation and startup time | `/health` reports `load_seconds` (the unpickle alone) and `ready_after_seconds` (process start to model ready); the platform wake is the cold `/health` minus the latter | the last three columns of the report |
| first successful `/predict` | timed, first and warm, every cycle | the report |
| first successful `/rank` | timed, first and warm, every cycle, on a three-posting batch — the shape the operating point was designed for (§15) | the report |
| memory under the actual model | `/health` reports peak RSS (`rss_mb`) from inside the process, against the instance's 512 MB | the report |
| repeat measurements | `REPEATS` cycles, default three, each after the full idle window; the criterion is applied to the worst request of the worst cycle, because the stranger who gets the slow one does not experience the median. A cycle answered by the *same process* as before its wait never went cold — `ready_after_seconds` is fixed for a process's life — and is marked and excluded (found 2026-09-11: a 0.34 s "cold start" right after a deploy) | one row per cycle |
| the ≤ 90 s criterion | enforced by the script's exit code, on the definitive kind only | the report's verdict line |

**What is settled now, and what is not.** The protocol and the machinery are:
every column above was exercised on 2026-09-11 against a local container
carrying the rehearsal release — filed as `REHEARSAL`, which accepts nothing —
and against the model-less image, filed as `BASELINE`. What is not settled is
the number, and nothing here can settle it before a real artifact is on the
public URL. **The Render architecture is provisional until `reports/cold_start.md`
exists with a `DEFINITIVE` verdict of `ACCEPTED`.** A README sentence quoting a
figure is not that file.

### The baseline, re-measured 2026-09-11 — and what it caught

The protocol's first run, three cycles against the live model-less service
after the lock (#72) and the runtime fields (#74) had deployed
(`reports/cold_start_baseline.md`):

| cycle | cold `/health` | ready after (inside) | peak RSS |
|---|---|---|---|
| 1 | 62.83 s | 42.21 s | 212 MB |
| 2 | 72.39 s | 46.39 s | 206 MB |
| 3 | 64.42 s | 38.54 s | 212 MB |

**Twice the 2026-09-06 baseline.** The decomposition said where: the process
itself took 38–46 s to become ready — interpreter and imports, no artifact —
against a whole cold start of 32.65 s five days earlier. Platform wake, the
difference, was 20–26 s and unchanged in kind. So the regression was in the
image, and the image had changed once: the switch from pip to uv. **pip
compiles bytecode at install time by default; uv does not.** Every cold start
was compiling scikit-learn, XGBoost, pandas and FastAPI to bytecode in memory
on a tenth of a CPU — and throwing it away, since the process cannot write to
site-packages. Confirmed locally under a hard `--cpus 0.1` quota: the same
imports take **341 s** from source and **177 s** from `.pyc`.

Fixed with one line, `UV_COMPILE_BYTECODE=1`, pinned by a test. **Re-measured
2026-09-12 on the fixed image**, three cycles, all of which went cold
(`reports/cold_start_baseline.md`, the baseline of record):

| cycle | cold `/health` | ready after (inside) | peak RSS |
|---|---|---|---|
| 1 | 52.51 s | 18.85 s | 204 MB |
| 2 | 43.64 s | 19.58 s | 208 MB |
| 3 | 52.43 s | 19.05 s | 200 MB |

The process now becomes ready in **19 s** against 38–46 s — the import cost
halved, as the local quota test said it would. The totals moved less, because
the platform wake in this run was 24–34 s against 20–26 s the day before: that
term is the platform's and varies by the hour, which is one more reason the
criterion is applied to the worst cycle. The 2026-09-06 single sample of
32.65 s is not reproduced by either run and should be read as one draw from a
wide distribution, not as the number this instance "really" does.

What the definitive run has to fit into: ~52 s worst-case baseline plus the
locally measured ~25 s load cost gives an estimate near **77 s** against 90 —
a margin of thirteen seconds on a platform whose wake alone varied by ten
between cycles. Whether it passes is the measurement's to say; the estimate's
only use is that it no longer says "comfortably".

Two things worth saying about the regression itself:

- **The estimate would have been wrong by the whole margin.** 58 s was the
  baseline plus the locally measured load cost. On the regressed image the
  same arithmetic gives 72 + 25 ≈ 97 s — over the criterion — and nothing
  short of the measurement would have said so. This is the case §7e says a
  baseline *can* decide, and it nearly did.
- **The decomposition is what made it a diagnosis instead of a number.** A
  cold `/health` of 72 s alone says "slower"; 46 s of it inside the process
  before any artifact says "imports", which says "bytecode", which says
  "uv". The three `/health` fields cost nothing and turned an hour of
  guessing into a `grep`.

### 7e-ii. The omitted work, measured on its own — **2026-09-10**

The `nothing` in the table above is what this subsection exists to attack. The
baseline cannot bound the load path, and the definitive measurement cannot be
taken until a model is frozen, so as written the largest unknown in the
deployment was scheduled to resolve itself on **the one day that cannot absorb a
bad answer**: freeze day opens the test block, spends it, and ships, and the
documented response to a blown criterion — reassess the architecture — is at its
most expensive precisely then.

`scripts/artifact_cost.sh` breaks that dependency by measuring the *difference*
the baseline is missing instead of the total it cannot reach. One image, started
twice under `--cpus 0.1 --memory 512m`, once with no artifact and once with
`models/` mounted read-only, timed from `docker run` to the first `/health` that
answers — the model loads in the app's lifespan, so that answer is already past
the unpickle. Three repeats per arm.

| | |
|---|---|
| median, no artifact (local, throttled) | 169.73 s |
| median, with artifact (local, throttled) | 194.91 s |
| **the artifact's cost — the difference** | **25.18 s** |
| measured remote baseline **+** that difference | **57.83 s** |
| the criterion | 90 s |
| headroom | 32.17 s |

**Why the difference is portable when neither total is.** The local no-artifact
arm takes 169.73 s where the real instance took 32.65 s. That gap is the evidence
that Docker's hard CFS quota and a free instance's burstable share are not the
same tenth of a CPU, and it is why no absolute figure here may ever be quoted
against the criterion. The difference survives because the two arms differ by
exactly one thing — same image, same limits, same launch site, same interpreter
start, same scikit-learn and XGBoost imports — and only one of them also reads
and unpickles a pipeline. Adding a local difference to the remote baseline is
then **pessimistic by construction**, because the slower of the two CPUs is the
one that produced the 25.18 s.

**What it still does not do, and the list is the point.** It does not accept the
architecture; nothing measured locally can, and §7e reserves acceptance for
`cold_start.sh` against a real release. The artifact it loads is the
synthetic-fixture one, so it exercises the real load path with a pipeline of the
real *shape* but not of the frozen model's *size*. And it is an estimate of a
term, not the term.

So the claim it supports is narrow, and stating it exactly is the whole
discipline: **the hosting decision is unlikely to fall over on freeze day, and if
it were going to, this run had a fair chance of saying so while the test block
was still unspent.** The architecture remains provisional. What changed is that
the risk is now quantified instead of merely acknowledged.

One arm returned 271.50 s against its own median of 194.91 s — a throttled
container losing a scheduling slice. The script reports medians for that reason;
a mean would have moved the headline by roughly seven seconds for reasons with no
connection to the model.

The criterion is applied to the **slowest request**, not to the wake alone. The
UI's timeout is per request and guards both, so a `/health` that wakes in 40
seconds followed by a `/predict` that takes 100 is a failure even though the wake
looked fine.

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

## 8a. Where the cut falls — **DECIDED 2026-09-07**

§8 settled *what the blocks are*. It did not settle *where the two cuts go*, and
the answer was left to whichever module was being written that day. There were
two, and both were wrong.

`train`, `evaluate` and `train_baseline` used `waves[0], waves[len // 2]`. That
pins `train_end` to the first wave at every depth, so the training block is one
wave deep on a panel of any size and `wave_forward_folds` cuts nothing from it —
**zero rolling-origin folds at 9 waves and at 17**, every wave the scraper added
going to the evaluation blocks. Every model comparison those modules produced
would have been a single number with no error bar, which is Obstacle 4 lost by
default.

`experiments` used 60/20/20 of the wave count, taking the fractions first and
letting the embargo come out of the result. The embargo is not a rounding error:
it discards three waves at each boundary, so a 20% validation slice is empty
until 20% of the panel exceeds three waves. Measured on the real panel, that
rule refuses every depth up to 15 and first returns a usable split at **16**
labelled waves — eight after one exists.

So two reports both saying "the split" described two different splits, and
neither described one worth having.

**Decided: `src.data.split.best_cuts`, a search over `feasible_cuts` rather than
a formula.** It keeps only cuts the acceptance check already marks valid,
prefers those yielding at least three rolling-origin folds, and among those
takes the cut closest to sharing the waves *that survive the embargo* 60/20/20.
All four modules call it, so "the split" now names one thing.

**Why proportional and not greedy**, since both greedy rules are simpler and
both are worse. Maximising folds deepens the training window without limit and
leaves validation one wave wide for ever — 201 training positives against 21 in
each evaluation block at 17 waves, and a test PR-AUC on 21 positives has an
error bar wider than any difference it could measure. Maximising the evaluation
blocks once three folds exist does the reverse and starves the fit: on the
synthetic panel it trains on 18 positives where proportional gets 35. Only
proportional grows all three together:

| labelled waves | train/val/test waves | folds | train pos | val pos | test pos |
| -------------- | -------------------- | ----- | --------- | ------- | -------- |
| 8              | 1/1/2                | 0     | 19        | 22      | 21       |
| 11             | 4/1/2                | 1     | 75        | 21      | 21       |
| 13             | 6/1/2                | 3     | 117       | 21      | 21       |
| 17             | 8/2/3                | 5     | 159       | 42      | 42       |
| 22             | 11/3/4               | 8     | 222       | 63      | 63       |

The fold target is a threshold rather than a quantity to maximise because on
this panel positives are the scarce resource — 96 in the whole frame — and past
three folds another fold buys less than the removals it takes out of the test
block. Three is `DEFAULT_TARGET_FOLDS` and it is a parameter, not a constant of
nature.

**Would change my mind:** a panel deep enough that positives stop being scarce,
where maximising folds costs nothing worth keeping; or a decision to report the
test metric with a bootstrap interval, which would raise the price of a thin
test block and argue for a larger test share than 20%.

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

## 11. The resurrection window — **DECIDED 2026-09-09: bounded at K = 2**

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

### DECIDED 2026-09-09 — bounded at K = 2: the label is final at corroboration

**The measurement that decides it.** Every absence-and-return in the 2026-09-08
snapshot, 1,530 postings across 122 complete runs:

| absent runs before returning | postings |
|---|---|
| 1 | 144 |
| 2 | **1** |
| 3 or more | **0** |

145 postings vanished and came back. **144 of them returned after a single
absent run** — which never reaches this clause at all, because the two-run
corroboration rule already refuses to call a single absence a removal. Exactly
one returned after two. None has ever returned after three.

So the distribution the earlier note asked for turns out to have almost no mass
past the corroboration threshold, and the choice of K is not delicate. **K = 2:
once a posting is absent at two consecutive complete runs it is gone, whatever
it does afterwards.** The `index > last_present` clause is removed from
`compute_labels`.

**What it costs, measured rather than bounded:** one posting in 1,530 —
**0.065%** — is now labelled removed when it did come back. On the panel the
change moves 174 positives to 175 and the H=7 base rate from 7.76% to 7.81%.
Nothing else moves.

> **Corrected 2026-09-11.** The clause removed above was doing a second job
> nobody had named: it was the only thing excluding the runs *before a posting
> was first listed* from the scan for two consecutive absences. Without it,
> every posting first seen at run 2 or later was dated gone at run 0 and was
> positive on every row it ever had. "Nothing else moves" was true on the
> 2026-09-09 panel, where almost every labelled posting was in the initial
> stock; the footprint grew with every wave of arrivals, to 671 of 785 H=1
> positives by 2026-09-11. The decision stands — K = 2, final at corroboration —
> and the scan now starts at first sight. `DEBUGGING.md`, 2026-09-11.

**What it buys is the embargo's correctness, not convenience.** The embargo has
always been computed as the horizon plus one run's reach. Under an unbounded
clause that arithmetic was false — the reach was the whole remaining panel, so
no embargo of any width sealed a training label from the evaluation period, and
the split was resting on a number that could not be right. K = 2 makes the reach
exactly what the embargo already assumes. It also makes a label final at a known
time, which is what lets one be recomputed for retraining.

**K = 3 was the alternative and is worse.** It would catch that single case, at
the cost of making the label read three runs beyond the horizon instead of one —
a wider embargo, more waves burnt at each block boundary, and a later date for
every gate. Paying panel depth, which is the binding constraint on this whole
build, to fix 0.065% of labels is the wrong trade.

**Would change my mind:** a return after three or more absent runs appearing at
any depth, or the two-run rate climbing above roughly 1%. Both are visible in
`src/data/split.py:resurrection_risk`, and the second would mean the corroboration
rule rather than K is what needs revisiting.
---

## 12. Board context at serve time — **DECIDED 2026-09-09: keep, accept-and-impute**

**The four board columns stay in the fitted model.** A caller who supplies them
gets them used; a caller who does not gets them imputed, and every response says
which happened via `board_context_supplied`. That is the handling shipped on
2026-09-05, and it is now the decision rather than a provisional arrangement.

**Why not drop them, given §4 just excluded board *identity*.** The two look
like the same question and are not. Board identity says *which* board this is —
a label the model can memorise a rate for, and the thing that stops a posting
from an unseen board being scorable. Board context says *what the board looked
like at `t`* — size, growth, how many like it were up — which is a property of
the situation, not of an identity, and which a caller from a board never scraped
can still supply if they have it. Excluding identity is what makes the product
work; excluding context would only throw away two of the more plausible
mechanisms in the feature set for nothing gained.

**The evidence §12 was waiting for, and the thing it got wrong.** This section
named an ablation as the deciding measurement. The ablation *refits* without the
four columns, which answers *what are they worth* — and that is not the serving
question. The deployed object is fitted **with** them; a caller who supplies none
sends nulls, the training fold's imputers fill them with constants, and the
fitted weights stay pointed at a column that no longer varies. A refit
redistributes that weight; the shipped model cannot.

`src/models/train.py:serve_time_regime` measures the actual regime — same model,
same split, same threshold, the only difference being whether the columns arrive.
On the H=1 panel:

| regime | PR-AUC | Brier | lost |
|---|---|---|---|
| board context supplied | 0.0226 | 0.0214 | — |
| absent, imputed | 0.0208 | 0.0227 | **0.0019** |

**The refit said 0.0005; the deployed model loses 0.0019 — roughly four times as
much.** So the evidence this section was waiting on would have understated the
thing it was meant to settle, by the mechanism above. Both numbers are now
reported side by side in `reports/model_results.md`, because reading either
alone is how a serving cost gets mistaken for a feature-importance one.

**What this does not settle.** 0.0019 is a single draw on a fold-less H=1 split,
which is a smoke test — the magnitude is not to be trusted, only the direction
and the fact that the two measurements disagree. The decision does not rest on
the number: it rests on the argument that a property of the situation is a
legitimate input and the interface already handles its absence honestly.

**Would change my mind:** a serve-time gap large enough that the imputed regime
falls to the baseline, at which point the honest move is not to drop the columns
but to serve two models and say so — one for callers who have board context and
one for callers who do not. That is a real option and it is deliberately not
being taken now, on a 0.0019 delta with no error bar.

### The rule at the freeze — **DECIDED 2026-09-12: imputed against a refit, not supplied against a refit**

"Would change my mind" above is a threshold with no number, decided by whoever
reads the serve-time table on the day. So the number is fixed now, in
`src/models/board_context.py`, while the only serve-time gaps on record are
H=1 and synthetic. Three validation PR-AUCs for the chosen candidate:

| regime | what it is |
|---|---|
| `supplied` | the full model, board context present — what `/rank` gets from a caller who can describe the board |
| `imputed` | the same model, board context withheld — what `/predict` gets from everyone else |
| `without` | a refit with the four columns removed — the alternative default |

> Ship the refit as the default public model iff `imputed + fold_sd < without`.

The full model, once its board columns are imputed to the training fold's
constants, does worse than a model that never had them by more than the
candidate's own fold-to-fold spread — the noise scale §14 already uses. Then
the caller this project actually has, someone holding one advert with no
board context, is better served by the refit, and it ships. Otherwise this
section stands and the full model ships with imputation. `evaluate` states
the verdict for the chosen candidate in the comparison report; `freeze`
applies it, runs the transfer gate (§4a) on *whichever pipeline ships*, and
records the three numbers and the decision on the artifact as `board_context`.

The comparison this rule deliberately does not make is `supplied` against
`without`. That asks whether the columns are worth having when present, which
this section answered on 2026-09-09 and which is the wrong question for a
default model that most callers reach with the columns absent. Nor does it
decide transfer: §4a's gate does, on the shipped pipeline.

What it does not do is serve two models. If the refit ships, callers who
*can* describe the board — the ranking mode — lose the four columns too; the
product change below does not change that, and serving two artifacts stays
the option named in "would change my mind" above.

### Two modes — **DECIDED 2026-09-12: `/predict` takes what a person can know; `/rank` derives the rest from a declared snapshot**

Two questions had lived under one contract. Someone holding one advert asks
*will this be gone soon?* and cannot know the four board fields — yet
`/predict` accepted them, almost always imputed them, and a caller who did
type them in was scoring a posting with numbers the model expects to describe
a whole board. Someone holding the board asks *which should I read first?*,
and for them the four are not unknowable: they are arithmetic over the very
batch being sent — which `/rank` refused to do, on the correct grounds that
an *undeclared* batch is not the board.

- **`/predict` — individual-posting mode.** The four board fields are not
  fields of its request. Sent, they are a 422 that names the other mode.
  `board_context_supplied` is always false and the response says so.
- **`/rank` — board-ranking mode.** `is_board_snapshot: true` declares the
  batch is one board, whole, as of `as_of`; the service then derives the
  four by the panel's own definitions (`src/inference/board_snapshot.py`,
  held equal to `assemble._board_context` by a test): rows in the batch,
  rows sharing the exact title, rows sharing `requisition_id` — accepted in
  this mode for the count and nothing else, an identifier never a feature —
  and growth against `previous_board_size`. A posting may not also carry the
  four (one source of truth); a batch naming two boards is refused; the
  response says `board_context_source` and the `board_size` the model was
  handed, so a partial board declared whole is visible as the caller's error.
  Undeclared, a batch is treated as before: supplied per posting, or imputed.

**Paging.** A snapshot larger than the service's batch cap cannot be
declared per page — a page's size is not the board's. So `app/client.py`
`rank_board` does the same arithmetic over the whole board client-side and
sends the result per posting; a test holds the client's copy equal to the
service's function, and the one-page case equal through both paths. The UI's
board mode has the switch; its one-posting form no longer has a board-context
expander, because that was the mixing made visible.

### The board as one logical collection — **DECIDED 2026-09-12: the server owns the pages**

The whole-board ranking cannot be one request on the free instance — a
day's board is ~106 s of scoring against a 90 s rule, warm — so it is
several, and the first version of the ranking mode put everything between
them in the client: page, merge by the service's rule, apply the budget once,
and in snapshot mode derive board context over the whole board before any
page. That was a second copy of the ranking rule and of the derivation, held
to the server's by tests, which is a way of saying the server did not own its
own product. Three shapes were considered:

- **Raise the cap to a whole board.** Out on measurement, not taste: 250
  postings is ~23 s on 0.1 vCPU and 1,150 is over the rule before any cold
  start. `MAX_BATCH` stays a measurement.
- **Score pages, finalise server-side over the scores.** Merges rankings, but
  cannot repair scores computed under per-page board context: size and
  same-title counts have to see the whole board *before* any page is scored.
- **A board as a stored collection, scored incrementally.** `POST /boards`
  opens it with its instant and snapshot declaration; pages are appended;
  `score` scores the next page, deriving board context over the whole stored
  board first, once; `rank` refuses until every page is scored and then orders
  the whole board by the rule `/rank` applies to a batch (`Predictor.order`,
  the same function). Chosen: the derivation and the rule live in
  `src/inference` once, and the client pages and loops.

What the free tier makes true is stated in the API rather than discovered:
boards live in memory on one instance for an hour and are gone when it
sleeps or redeploys — `404 … re-upload` — and the reference client retries
once from the top. Caps: 250 per page, 5,000 per board, 32 live boards.
`app/client.py` now holds neither the derivation nor the merge, and a test
says so.

**Would change my mind:** a second instance, at which point in-memory boards
become a bug rather than a shape and the store moves to something shared. Or
a caller with a legitimate partial view of a board
— a department's postings, say — for whom neither "impute" nor "this is the
whole board" is honest. That needs a third answer (supply the board's size,
derive the rest), and it is not built until someone has that view.

**Would change my mind:** a rule that fires on the day on a difference whose
interval spans zero — the fold spread is one noise scale, the posting-clustered
interval on validation (§5) is another, and if they disagree the wider one
should win.

---

### The original argument, kept because the decision rests on it

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

**Update 2026-09-07 — the deciding evidence is now built.** It was not, and the
gap was quiet: `ABLATIONS` in `src/models/train.py` priced the seven *engineered*
features and none of the four board ones, so the ablation this section was
waiting on would not have answered this section's question even after the panel
was deep enough to run it.

Two things were added, and both are needed. Each board column is now priced
singly, and all four are priced **together** — `BOARD_CONTEXT_ABLATION`, whose
withheld set is `contract.BOARD_CONTEXT` rather than a second list, so a field
that changes origin changes what is priced and neither can drift from the other.

The group ablation is the one that matches the question. Leave-one-out cannot
price these four: `board_size_at_t` and `board_growth` correlate at **0.42** on
the observed panel, so dropping either leaves its partner carrying the signal
and both deltas read near zero while the pair is worth something. And the
serving question is not what dropping one costs — it is *what a caller who
supplies none of them loses*, which is one refit with all four withheld.

Reading the two together is what makes the result actionable, which is why
`reports/model_results.md` now states it in words rather than leaving a reader to
subtract table rows:

- small singles, small group → the columns are worthless, and **this section can
  close by dropping them**;
- small singles, large group → they are redundant with each other, and a
  per-feature table alone would have licensed dropping columns that jointly
  carry signal;
- group ≈ the best single → the value sits in one column, not in the set.

**How it will be decided, fixed 2026-09-07 before the evidence exists.** Written
down now on purpose: a rule chosen after seeing the number is not a rule.

The comparison is the full model against the full model minus the board-context
group, refitted across the rolling-origin folds **inside the training window**
and paired fold by fold. Paired because both models see the same validation wave
in each fold and so share whatever made it easy or hard; differencing within the
fold removes that, where scoring each separately and subtracting the means
leaves it in. `board_context_folds` in `src/models/train.py` is that comparison,
and `test_the_board_comparison_never_reads_the_test_block` is what keeps it
honest — it poisons the test block and requires the output to be unchanged.

The reading, decided in advance:

- **inside one standard deviation → drop the four.** A tie is not a reason to
  keep them. If the columns cannot be shown to help, the simpler model is the
  one whose validated and served forms are the same object for every caller, and
  the imputation branch stops existing.
- **a lead clearing one standard deviation → keep them**, with the cost this
  section already names: a caller supplying nothing gets a model whose board
  columns are constant, and the response says so.
- **fewer than three folds → not evidence.** A standard deviation over two
  numbers is not a standard deviation, and the decision waits.

**The test block does not answer this.** Choosing a feature set is model
selection, and a test set consulted during selection is a validation set. It is
opened once, afterwards, for the performance claim.

**What this still does not decide.** Option B — requiring the caller to supply
board context — is not available under §5's decided user, a job seeker weighing
where to spend application effort, who cannot see a board at an instant. It
becomes available only if the deployment story in §4 changes to a board owner.
So the live choice is between dropping the four and keeping the imputation
fallback, and the number decides it.

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


---

## 14. The selection rule — **PRE-REGISTERED 2026-09-10, while every fold count was zero**

**The decision:** which model ships is decided by a rule written down *before the
numbers it will be applied to exist*, and applied mechanically once they do.

**Why the timing is the whole point.** A selection rule written after the fold
scores are visible is not a rule, it is a description of the winner. Every degree
of freedom in it — which metric, how to break a tie, how large a gap has to be,
whether to prefer the simpler model — is a place to arrive at the model one
already likes, and none of those choices looks arbitrary once there is a table on
the screen to justify it. On 2026-09-10 `reports/model_comparison.md` carried
`folds_scored = 0` for all ten candidates, at both horizons: **8 labelled waves at
H = 1 against the 13 that yield three folds, and 2 at H = 7 against 31.** There
was nothing to select on, which is exactly when the rule is cheapest to write
honestly.

### The rule, in four steps

Implemented in `src/models/evaluate.py:select`, whose docstring quotes this
section.

1. **Eligibility.** A candidate needs at least `MIN_SELECTION_FOLDS = 3` scored
   rolling-origin folds. Below that there is no spread to judge a lead against,
   and the verdict is that *nothing was selected* — not that the leader wins by
   default. Three is the same number `src/data/split.py` sizes the panel for.
2. **The tied set.** Rank by `cv_pr_auc_mean`. Take the leader, and collect every
   candidate whose *paired* per-fold difference from the leader is no larger than
   the standard deviation of that difference. Paired, because two models scored on
   the same fold share whatever made that fold hard; averaging each separately and
   subtracting leaves a week's weather in the answer.
3. **Parsimony decides among equals.** From the leader and everything tied with
   it, take the one earliest in `LADDER` — the simplest. Only a lead that survives
   fold variance buys complexity. `CLAUDE.md` §4.4: *the goal is not to reach the
   top rung, it is to learn whether increasing model complexity actually buys
   anything on this problem.*
4. **The heuristic floor is a gate.** If the pick is a fitted model whose lead over
   the best `HEURISTIC_RUNGS` entry does not itself survive fold variance, the
   heuristic is selected instead. `CLAUDE.md` §4.4 #12: *a model that does not beat
   the baseline is not a model, it is a slower baseline.*

### What this changes, and the bug it closes

The previous implementation documented step 3 and did not do it. `select` returned
the highest `cv_pr_auc_mean` whatever the spread, appending "treat them as tied"
to the *reason* while still crowning the winner — so a boosted model half a
standard deviation ahead of a logistic regression was selected, and the report
asserted both things at once. `learning_log/learning-log.md` had already recorded
the right conclusion on 2026-09-05 ("these two are tied"); the code never
implemented it. Step 4 did not exist at all.

**Step 4 is not redundant with step 3, and the synthetic panel is what showed
that.** Heuristic rungs sit at the simple end of the ladder, so a fitted model
that cannot separate from one *usually* finds it in its own tied set and loses on
parsimony. But "tied with the leader" and "better than a rule" are different
questions, and on a noisy label they come apart. With steps 1–3 only, the panel
whose label is drawn independently of every feature selected `age_only` — a fitted
model that does not beat a constant — because `prior` sat too far from the leader
to be tied with it, so the failure was never tested. With step 4 the same panel
returns `prior`.

### How it can be falsified

The rule is not a preference for simplicity; it is parsimony *among equals*, and
it must be able to choose a boosted model. Tests hold both directions:

- a lead larger than its own spread selects the complex model (`separation`)
- a lead smaller than its own spread selects the simpler one (`parsimony`)
- on the noise panel, a fitted model at 0.2078 against a base rate of 0.1000 loses
  to a heuristic — pinned end to end, because a rule that quietly began preferring
  the top of the ladder would produce a *better-looking* report and nothing else
  in the suite would notice

**Would change my mind:** a panel deep enough that one paired standard deviation
stops being the right width. At thirty folds rather than three it is a weak bar,
and a proper interval on the paired difference would be better; the current
arithmetic is chosen for a handful of folds and should be revisited if the panel
ever affords more. Separately: if a heuristic wins on the real panel, that is a
finding to write down, not a reason to weaken step 4.

### What it does not decide

Nothing here opens the test block. Selection happens on folds inside the training
window. `src/models/freeze.py` remains the only module that reads test, still
refuses on `SplitTooShallow` and `NoFoldEvidence`, and the freeze itself stays a
deliberate act with a `--run` argument naming the chosen model.

---

## 15. What the product scores — **DECIDED 2026-09-11: a day's whole board, with the first-observation slice reported**

Two prediction problems have been living in this repository under one name.
The README's opening promised a posting could be "scored at the moment it
first appears"; the dataset is job-day, `/rank` scores a whole board, and the
operating point (§5) is an alert budget *per day over the board*. Those are
related and different, and which one the model is evaluated on decides whether
`age_days` is a legitimate input or an artefact of when collection began.

**The product is: for today's entire board, rank which postings are most
likely to be gone within seven days.** Incumbent stock and newly appearing
postings both belong in the dataset, both are scored, and age is a feature —
the discrete-time hazard formulation §3 already chose.

Why this one and not the other:

- **It is what the operating point means.** The alert budget is "inspect the
  top twenty on the board today", and a rank statistic over a day's board is
  undefined for a stream of first sightings.
- **It is what the user does.** The cost asymmetry in §5 — a job never applied
  to is unrecoverable — belongs to someone scanning today's board, most of
  which is not new. A first-appearance model tells them nothing about the
  forty postings they saw last week and are still deciding on.
- **The narrower problem is not evaluable here.** Postings first seen after
  collection began are 254 labelled job-days at H=7 on the 2026-09-11 panel,
  and the deepest legal H=1 cut leaves a validation block with **no** first
  observations at all (`reports/cohort_audit_h1.md` §6). A primary evaluation
  on that slice would be a primary evaluation on nothing.
- **It is honest about left truncation rather than sidestepping it.** A
  first-appearance product avoids the incumbent stock by construction, and so
  never has to show that the label treats the two populations alike. This one
  does have to, every time — which is what `src/data/cohort_audit.py` is for.

**What the narrower promise becomes: a mandatory reported slice, not the
headline.** `reports/cohort_audit.md` scores the first-observation rows on
their own; when a validation block contains any, the age ranker and every
cohort table are shown for them separately, so a model that only works on the
stock cannot hide inside the board-wide number. The README's opening no longer
says "at the moment it first appears"; the API still accepts a posting on its
first day, and that is the case this slice measures.

**The condition the decision rests on, and the check that enforces it.** Job-
day scoring makes age a feature only if the label is indifferent to which
population a row came from. On the corrected label it is: incident rows close
at 9.8% against 7.6% for the stock at H=7, 0.7% against 1.1% at H=1, and
`age_days` as a bare ranker scores 0.013 against a base rate of 0.005 on the
H=1 validation block. On the label of 2026-09-09 to 2026-09-11 it was not —
every incident row was positive and no incumbent row was — and that is the
signature the audit's first verdict now names in one sentence.

**Implemented in the UI, 2026-09-12.** The Streamlit page opens on *Rank a
board* and keeps the single-posting form as the second mode, because a UI
that demonstrated `/predict` alone demonstrated the narrower promise this
section retired. The one design question it raised was the batch cap:
`/rank` accepts 250 postings (`MAX_BATCH`, a measurement) and a day's board
is ~1,150, so the board is paged — and under a budget, paging is only honest
if the pages are merged by the service's own rule and the budget applied once
to the union, which `app/client.py` `rank_board` does and a test holds equal
to the unpaged ranking. The cap itself was *not* raised: the 2026-09-12
baseline puts the worst-case cold start at 52 s, so 250 postings plus a wake
is ~75 s against the 90 s rule, and 500 would be over it. The cap is tighter
than the day it was set, not looser, and raising it stays a measurement
against the deployed instance with a real artifact.

**Would change my mind:** an incident cohort large enough to evaluate on its
own *and* a persistent gap between its rate and the stock's after the label is
known to be sound. That would mean arrival itself carries hazard, and the
product should then say so as a feature (`runs_seen`, already as-of-`t`)
rather than as a separate problem. Or a user whose decision genuinely happens
only on a posting's first day — none has been named.

---

## 16. Release discipline — **DECIDED 2026-09-11: a real freeze refuses a dirty tree**

Every generated report records the commit it was written at and whether the
tree was clean. For experiments that is the right trade: most runs happen
mid-edit, and a driver that will not run until you commit is a driver you stop
using. `dirty tree` on a report means *provisional*, and is read that way.

It is not enough for the artifact. A frozen model ships, and the commit
recorded on it is the claim that someone can reproduce the number it was
frozen with. **A SHA cannot reproduce a result if uncommitted source changes
affected the run**, and until 2026-09-11 nothing stopped that: several reports
on `main` carried `(dirty tree)`, correctly, and the same path would have
carried it onto an artifact.

**So `python -m src.models.freeze` on a real panel refuses unless
`git status --porcelain` is empty** — tracked and untracked alike, because a
new module the run imported is as unreproducible as an edited one. The check is
asked last, at the moment the held-out block would be opened, after both depth
refusals: it is about the code rather than the data, so it must not mask a
depth refusal, and it must not write a report, since writing one dirties the
tree. Its own exit code, `4`, so a caller can tell *commit first* from *wait
for depth*. Synthetic freezes are rehearsals, say `dataset=synthetic` on every
response, and are exempt.

The consequence for the day itself: `rehearse.sh` and `evaluate` regenerate
reports, which dirties the tree, so **the evidence is committed before the
block is opened** — the comparison a reader will check the result against is
in history before the result exists. That is the right order and the refusal
enforces it.

*Considered and rejected:* capturing the diff and checksumming it. It works,
and it is more machinery than a project with one author needs; a clean
committed tree is simpler and stricter.

### The environment is the lock — **DECIDED 2026-09-11**

The traceability table below names `lock_sha256` of `uv.lock`. On 2026-09-11,
when that row was written, the file was **gitignored** — the entry argued that a
lock "validated by nothing" would drift from what was actually installed. That
was true, and it made the hash worthless: it named a resolution that existed on
one laptop, that neither CI nor the image had ever installed, and that a reader
of the artifact could not fetch at the SHA beside it. Meanwhile
`pyproject.toml` carried lower bounds only, so training, CI and the image each
resolved afresh and could each land on a different scikit-learn or XGBoost.
For a pickled estimator that is the failure mode that matters: a booster
written by one version and read by another either fails to load — the
Dockerfile's build-time load would catch that — or loads and answers
differently from its recorded metrics, which nothing would.

**So the lock is committed, and everything installs from it.** Locally,
`uv sync --locked --extra dev --extra api --extra ui`; in CI, the same;
in the image, `uv sync --locked --extra api`. `--locked` is the half that
matters: it refuses a lock that has fallen behind `pyproject.toml`, so "add a
dependency, forget to re-lock" fails in CI and in the build rather than
resolving to something newer on the runner than on the laptop that froze the
model. The installer itself is pinned to one version in both places. The
lock's hash on the artifact now points at a file in the repository, which is
what the traceability row always claimed.

**Two consequences worth naming.** First, the image's load check became
strict: `load()` warns on a scikit-learn / XGBoost / joblib mismatch, because a
laptop opening an old artifact to look at it should not be refused, but the
image promotes that one warning to a build failure. With both sides installed
from the lock, the only way it fires is a lock bumped after the freeze — and
then the right answer is to re-freeze or not to bump, not to serve. Second, the
CUDA libraries XGBoost's Linux wheel drags in are still removed from the image,
but *after* the last sync, since a sync restores the environment to the lock
and the first draft had them reinstalled by the second sync. The lock keeps
naming them deliberately: it has to reproduce the freeze's environment
elsewhere, and a resolution edited by hand is no longer the lock.

**One Python, declared as a range of one.** `requires-python` said `>=3.11`
while every environment that ever ran the suite or froze a model was 3.12 — a
supported version nothing tested, on a project whose artifact is a pickle. It
is now `>=3.12,<3.13`, `.python-version` is committed, the lock resolves for
one interpreter instead of nine marker combinations, and a test asserts that
`pyproject.toml`, `.python-version`, `uv.lock`, the Dockerfile and the dev
container all name the same minor. Testing 3.11 as well was the alternative;
it would cost a CI matrix to support a version no environment here uses.

*The UI's `requirements.txt` is the one deliberate exception.* Streamlit
Community Cloud installs it unpinned, and it can afford to: the UI imports
nothing that could load a model (§7c), so drift there cannot change a
prediction.

**Would change my mind:** a second deployment target that cannot run `uv` —
then the lock would be exported to a pinned `requirements` file for it, still
generated from `uv.lock`, never written by hand.

### What the artifact is traceable to

Everything below travels on the artifact's metadata and its JSON sidecar, and
the test `test_the_artifact_names_everything_it_is_traceable_to` pins the list:

| what | where |
|---|---|
| source | `provenance.git_sha`, with `git_dirty` necessarily `false` on a real freeze |
| data | `provenance.panel_sha256`, `snapshot_date`, `panel_path` |
| the scraper's parsing epoch | `rules_version` — runs are only comparable within one, so the panel carries the value it was built at |
| horizon | `horizon_days`, `horizon_basis` |
| dependencies | `lock_sha256` of the committed `uv.lock` that every environment installs from, plus library versions in `versions` |
| training configuration | `run_name`, `params`, `features`, `fitted_on`, `threshold`, `budget_per_day`, `selection_folds` |
| random seed | `seed` — the one random state every fitted rung shares |
| the artifact itself | `artifact_sha256` in the sidecar (a file cannot contain its own hash), repeated by `SHA256SUMS` at release and verified on fetch |

**Would change my mind:** a second contributor, at which point "commit
everything" stops being one person's habit and a captured diff might be the
cheaper enforcement. Or a run so long that committing between the comparison
and the freeze is a real cost — it is seconds here.
