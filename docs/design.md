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

## 7. Where it deploys — **OPEN**

Not yet decided. Constraints known: free tiers sleep, and a cold container
holding a boosted model is slow to answer the first request. Whatever is chosen
must have its cold-start behaviour measured and stated in the README rather than
discovered by whoever is shown the link.

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
