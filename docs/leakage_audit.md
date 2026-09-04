# Leakage audit

A verdict for every column of the assembled job-day panel
(`data/processed/features/job_days_h*.parquet`, 44 columns, built by
`src/features/assemble.py`). The rejected rows are the point of the document.

This is not the same table as [`data_dictionary.md`](data_dictionary.md). That
one asks whether a *source* field is as-of-`t`. This one asks whether a column
of the *model matrix* may be shown to an estimator, which is a stricter question
and has more ways to fail.

**The ritual question** ([`problem_definition.md`](problem_definition.md) §6):
*at the moment I would need to make this prediction, would this value exist yet?*
Auditing the 2026-09-04 snapshot against it turned up columns that pass it and
still must not be features, so the verdicts below use four outcomes rather than
two.

## The four ways a column fails

| verdict | meaning | why it is a distinct failure |
|---|---|---|
| **leak** | the value does not exist at `t`, or the stored value is a later one | the classic. Trains on the answer |
| **axis** | exists at `t`, but the column's job is to place the row in time or to identify it | it is what the split is *made of*. As a feature it is a handle on the experiment rather than on the world |
| **skew** | exists at `t` in training, but is absent or degenerate when scoring a posting we have not been watching | not leakage — training/serving skew. The model learns a dependence that has no counterpart at serve time, which the offline metric cannot see |
| **dead** | no variance, or duplicated by a better column | not dangerous, but it inflates the feature count and hides real signal in a wide matrix |

**skew** is the verdict this dataset added. It is easy to see why `last_seen` is
a leak. It is much harder to see why `n_complete_runs_observed` — honestly
computed over a window ending at `t`, and knowable at `t` — still cannot be a
feature. The serving contract is that a stranger POSTs one posting; we have
watched it zero times, so the column is constant at serve time and every weight
the model learned on it is untrained in the only regime that matters.

---

## The table

`t` is the prediction point: the start instant of the complete run that observed
the posting.

### Identity and axis — 8 columns, all excluded

| column | verdict | reason |
|---|---|---|
| `t` | **axis** | it *is* the prediction point, and the thing the split cuts on. Calendar derivations of it are separate columns and judged separately |
| `run_id` | **axis** | names the crawl that produced the row |
| `run_index` | **axis** | the position of the run within its own source — and **not comparable across sources**. python_org missed the 2026-08-31 crawl, so its `run_index` 0 is 2026-09-01, the same instant as every Greenhouse board's `run_index` 1. Splitting on it would place 31 rows of the future in the earliest training block; using it as a feature encodes panel position, which is not a property of the posting |
| `source_id` | **axis** | the board's own id. Unique per posting, so it is a memorisation handle — and it is monotonic in creation order: Spearman 0.917 against `first_published` on the anthropic board. A surrogate key that correlates with age at ρ≈0.92 is an age feature with no missing values and no honesty |
| `url` | **axis** | unique per posting, and its domain names the board — it re-admits `source` through the back door |
| `requisition_id` | **axis** as a field | 1,120 distinct values over 1,240 postings: identity, not a category. Only the windowed count derived from it (`n_same_req_on_board`) is a feature |
| `horizon_days`, `horizon_basis` | **axis** | label-construction metadata, constant within a frame. They record how `y` was made |

### The label — 2 columns

| column | verdict | reason |
|---|---|---|
| `y` | — | the target |
| `label_observable` | — | the censoring flag. Rows where it is false are dropped before splitting, never coerced to 0 |

### As-of-`t` facts about the posting — usable

| column | verdict | reason |
|---|---|---|
| `title` | **yes** | present in the posting when first seen. Keyword and seniority flags derived from it are row-local and inherit the verdict |
| `location` | **yes** | free text as posted |
| `first_published` | **yes** | the employer's own publication instant, at full precision. The age origin |
| `updated_at` | **yes** | the employer's last edit *as recorded by that fetch*. Later edits are not visible because the archive is sealed per fetch |
| `departments` | **yes** | 120 distinct values. High cardinality — Component 7 must handle categories unseen at fit time without crashing |
| `offices` | **yes** | 88 distinct. Null on 8.4% of rows against `n_offices`' 2.2%, which is unreconciled and worth one look before use |
| `n_offices`, `n_metadata` | **yes** | counts from the sealed payload |
| `content_chars` | **yes** | description length. The cheap text feature; anything more is out of scope for this stage |
| `age_days` | **yes** | `t − first_published`. Measured from the employer's publication instant, never from `first_seen`, which records when this project started looking. Expected to carry most of the signal, which is why it is also a baseline in its own right |
| `days_since_update` | **yes** | `t − updated_at` |
| `posted_dow` | **yes** | day of week of publication. A property of the posting |
| `salary_stated` | **yes** | "the posting mentioned pay at all". Plausibly more predictive than the amount, and defined for every row |
| `salary_min_clean`, `salary_max_clean` | **yes** | re-derived from `salary_raw` by `src/data/clean.py` |
| `salary_currency_clean` | **yes** | not converted; an exchange rate is external and time-varying |

### Board context — usable *because* the window ends at `t`

Each of these reads other rows, which is exactly how a leak is usually
introduced. They are computed within a single run, so they describe the board as
it stood at `t`. A window that ran one run past `t` would make every one of them
a leak while leaving the frame looking identical.

| column | verdict | reason |
|---|---|---|
| `board_size_at_t` | **windowed → yes** | postings on that board in that run |
| `n_same_title_on_board` | **windowed → yes** | duplicate-title count within the run. 775 of 5,714 rows share a title with at least one other live posting |
| `n_same_req_on_board` | **windowed → yes** | postings sharing a requisition, within the run |
| `board_growth` | **windowed → yes** | change in board size against the source's previous run. Null on each source's first observed wave — a fact about the panel edge, not a defect |

### Excluded — leak, skew or dead

| column | verdict | reason |
|---|---|---|
| `n_complete_runs_observed` | **skew** | how many complete runs have seen this posting up to and including `t`. Passes the ritual question — it is knowable at `t` — and must still go. It measures how long *we* have been watching, not how long the posting has been alive, so it encodes the left-truncation of the panel; and at serve time a stranger's posting has been observed once by construction, making the column constant in the only regime that ships |
| `t_dow` | **axis proxy** | day of week of the prediction point. With five crawl waves it nearly identifies the wave, so it is a handle on the split rather than on the posting. Revisit when the panel spans several weeks and the two stop being the same column |
| `posted_month` | **skew** | publication month, spanning all twelve. On a four-day panel it is collinear with `age_days` and cannot survive a year boundary: a model that learned "March" meant "six months old" will read March next year as brand new |
| `remote` | **dead** | **0 non-null values in 5,714 rows.** Greenhouse never populates it and python_org does not either; arbeitnow was the only source that did, and arbeitnow is excluded from labelled rows (`problem_definition.md` §4). The column is a casualty of that exclusion, not of collection |
| `n_departments` | **dead** | constant `1.0` |
| `salary_period` | **dead** | constant `"unstated"`. Worth one look upstream: it means `annualise` never applied a multiplier on this population |
| `salary_parsed` | **dead** | identical to `salary_stated` on every row (1,535 false / 4,179 true, no disagreement). The two exist to separate "no pay mentioned" from "pay mentioned but unreadable", a distinction only arbeitnow produced |
| `salary_min`, `salary_max`, `currency` | **dead (superseded)** | the upstream parse. Agrees with the re-derived version on all 4,114 rows where both are present, has strictly worse coverage (28.0% null against 26.9%), and holds the £100/month value stored as 1,200,000,000 recorded in `DEBUGGING.md` |
| `posted_at` | **dead (superseded)** | a string date truncated to midnight. `first_published` is the same fact at full precision |
| `salary_raw` | **leak-adjacent, excluded as text** | the value is as-of-`t`, but as a feature it is high-cardinality free text whose *shape* fingerprints the board. Its information is already carried by `salary_stated` and the clean amounts |

### The two that Decision 4 owns

| column | verdict | reason |
|---|---|---|
| `source` | **open** | [`design.md`](design.md) §4. The strongest signal in the data and partly instrumental |
| `company` | **open, and tied to `source`** | see below |

---

## Findings that cut across columns

**1. `source` cannot be excluded on its own.** Each of the six Greenhouse boards
has exactly one company, and all 31 companies in the frame map to exactly one
source. `company` is therefore a lossless re-encoding of `source` for six of the
seven boards. Dropping `source` while keeping `company` drops nothing. The same
back door is open through `url` (its domain) and, less obviously, through
**archive-derived missingness**: `first_published`, `updated_at`, `departments`,
`requisition_id`, `n_metadata` and `content_chars` are null on exactly the 127
python_org rows and present on every Greenhouse row, because the archive covers
Greenhouse only. A missingness indicator over any of them reconstructs part of
`source` for free.

Decision 4 is therefore not "is `source` a feature?" but "is *board identity* a
feature?", and it has at least four columns and one missingness pattern in scope.
That is a wider question than [`design.md`](design.md) §4 currently states.

**2. Missingness here is structural, and it is a fingerprint.** This is the same
lesson the raw data taught, arriving again one layer up. Nothing in the frame is
missing at random: `remote` is missing because of who was excluded, the archive
columns are missing because of which boards have an API, and `board_growth` is
missing at the panel edge. Component 7 must encode missingness deliberately per
column rather than imputing it away, and must not add a generic
"missing-indicator for every column" step, which would hand the model the source
fingerprint in eight new places.

**3. Nine of 44 columns are dead.** `remote`, `n_departments`, `salary_period`,
`salary_parsed`, `salary_min`, `salary_max`, `currency`, `posted_at` and — for
modelling — `salary_raw`. None of them is dangerous. Together they are a fifth
of the matrix, and a wide matrix is where a leak hides.

**4. The label's forward reach is not bounded, so no embargo fully seals it.**
`t_gone` requires that a posting *never re-appeared*, and that clause reads the
whole remaining panel rather than a fixed window. `greenhouse:gitlab 8615319002`
was present at runs [0, 3, 4]: two consecutive absences — enough to satisfy the
corroboration guard — and then it returned, flipping a run-0 label from 1 to 0
when the panel deepened. Two of 1,240 postings do this (0.16%).
`src/data/split.py:resurrection_risk` measures it. Bounding the resurrection
window is a change to the label definition and belongs in
[`design.md`](design.md), not here.

**5. The split needs an embargo of `H + one run`, not `H`.** A row at `t` is
labelled by absence at the next run *corroborated at the one after*, so the
label reaches one run further than the horizon. The validation protocol in
[`problem_definition.md`](problem_definition.md) §7 specifies a gap of `H`,
which is one run too narrow. `src/data/split.py:embargo_width` computes it from
the horizon and the widest observed run gap.

---

## How this document is enforced

Prose drifts from code, so the table above is also data.
`src/features/preprocessing.py` carries it as `FEATURES` (17 columns),
`BOARD_IDENTITY` (2, gated together and off by default), `TEXT_FEATURES` (1, not
yet wired) and `EXCLUDED` (26, each with its verdict). Every panel column
appears in exactly one of them, `assert_known_columns` refuses a frame carrying
a column that appears in none, and a test asserts the four sets are disjoint. A
column added to the assembly step is therefore a build failure until a verdict
is written for it — it can neither become a silent feature nor be silently
dropped, which is the second way an audit becomes fiction.

Three rules from the table are implemented rather than described:

- **every transformer is fitted on the training fold only.** The one supported
  entry point is `fit_on_training_fold(pipeline, split)`, which takes a
  `SplitResult` and can only reach `.train`; `learned_statistics` exposes the
  fitted fills so that "which fold was this fitted on?" is an assertion rather
  than an assurance;
- **categorical handling survives categories unseen at fit time**
  (`handle_unknown="infrequent_if_exist"`), because `departments` has 120 values
  over four days and the tail will grow;
- **there is no blanket missing-indicator step.** Missingness is encoded only
  for categoricals, where "not stated" is a real level. An indicator over the
  archive-derived numerics would reconstruct board identity, which is the
  decision [`design.md`](design.md) §4 has not yet made.
