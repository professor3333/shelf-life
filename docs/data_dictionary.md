# Data dictionary

One row per column, with what it means and whether its value exists at the
moment a prediction would be made.

The measured facts — dtypes, null rates, per-source coverage — are regenerated
into `reports/` by `python -m src.data.profile`. This file holds the part that
requires a decision.

**"At prediction point?"** is the leakage question from
[`problem_definition.md`](problem_definition.md) §6: at time `t`, the start of a
complete run that observed the posting, does this value exist yet *and* is the
value we can read the one that existed then? A column answered **no** cannot be
a feature however predictive it looks.

The three answers:

| | meaning |
|---|---|
| **yes** | exists at `t`, and the stored value is the as-of-`t` value |
| **windowed** | safe only if computed over a window ending at `t` |
| **no** | does not exist at `t`, or the stored value is a later one |

**The structure of this table is the defence.** Features are assembled from the
per-run snapshot CSVs and the archived API payloads, both of which are sealed
observations of one instant. The `jobs` table is current state and supplies only
identity and labels. A feature builder that never opens `jobs` cannot commit the
"current state read as past state" error, which is the class of leak this
dataset is richest in — **420 of 1,207 postings were edited during
observation**.

---

## 1. `snapshots/<stamp>.csv` — the per-run observation

One file per run, one row per posting that run saw, never rewritten. As-of-`t`
by construction.

| column | meaning | at prediction point? | notes |
|---|---|---|---|
| `observed_at` | the run's start instant — **this is `t`** | n/a | not a feature; it *is* the prediction point. Calendar derivations of it (day-of-week) are features |
| `source` | which board the posting came from | **open** | see `design.md` §4. Strongest signal in the data and partly instrumental |
| `source_id` | the board's own id for the posting | no | identity key. A surrogate id, monotonic in creation order |
| `title` | job title as posted | **yes** | text feature; keyword and seniority flags derived from it are also as-of-`t` |
| `company` | employer name as posted | **yes** | normalised, not canonicalised — see `src/data/clean.py` |
| `location` | free-text location as posted | **yes** | unnormalised; `offices` from the archive is the structured version |
| `remote` | remote flag | **yes** | three-state. `NULL` means "not stated" and must stay distinct from `False`: Greenhouse never populates it |
| `salary_min`, `salary_max` | parsed pay bounds | **yes**, with a caveat | the upstream parser leaves 455 stated salaries NULL and holds one value six orders of magnitude wrong. Re-derive from `salary_raw` |
| `currency` | pay currency | **yes** | categorical. Not converted — an exchange rate is external and time-varying |
| `salary_raw` | the pay string as printed | **yes** | and `salary_raw IS NOT NULL` as its own indicator: "the posting stated pay at all" is plausibly more predictive than the amount |
| `posted_at` | publication date | **yes** | truncated to midnight UTC; `first_published` from the archive is the same instant at full precision |
| `url` | link to the posting | no | identity, and it encodes the board — using it re-admits `source` by the back door |

**Not present, and listed as a feature in §5.1:** `seniority`. It exists only in
the current-state `jobs` table, so reading it would leak later edits backwards.
It is derived from the title by a row-local rule and is re-derived from the
as-of-`t` title instead (`src/data/clean.py`). Recorded in `design.md` §9.

**Not present, deliberately:** `description` — employer copyright. The text is
available from the archive.

---

## 2. Archived API payloads — the as-of-`t` detail

`data/raw/boards-api.greenhouse.io/<board>/<stamp>.html.gz`, read by
`src/data/archive.py`. Immutable once written. Greenhouse boards only.

| column | meaning | at prediction point? | notes |
|---|---|---|---|
| `fetched_at` | the fetch instant — **this is `t`** | n/a | as `observed_at` above |
| `source_id` | the board's id for the posting | no | identity. Joins to `jobs.source_id` for all 1,207 postings |
| `first_published` | when the **employer** published it | **yes** | the age origin. Full precision, where `posted_at` is it truncated to midnight — matters because age is the most important feature in the problem |
| `updated_at` | employer's last-edit time, as of this fetch | **yes** | gives "days since last edit" directly, with none of the parser-attribution problem that makes `job_changes` unusable before 2026-09-04 |
| `departments` | Greenhouse department names | **yes** | 100% coverage, 119 distinct. A real category, not one inferred from the title |
| `n_departments` | how many | **yes** | |
| `offices`, `office_locations` | structured location | **yes** | 93% coverage |
| `n_offices` | how many | **yes** | multi-office postings may behave differently |
| `requisition_id` | the employer's req number | **yes** | the field itself is as-of-`t` and safe |
| — repost counts derived from it | postings sharing a req | **windowed** | 62 reqs cover more than one posting (150 postings). Counting them reads other rows, so the window must end at `t` |
| `metadata_json` | board-specific custom fields | **yes** | 85% coverage; stable JSON keyed by name |
| `n_metadata` | how many | **yes** | |
| `content` | the description HTML | **yes** | the text feature. Changes for 34 postings mid-panel, which is exactly why it must come from here and not from `jobs` |
| `content_chars` | its length | **yes** | the cheap derived feature; no NLP needed |
| `company_name` | employer name | **yes** | |
| `location_name` | free-text location | **yes** | changes for 9 postings mid-panel |
| `application_deadline` | stated deadline | n/a | **absent on every board.** Do not plan around it |
| `board_dir`, `board_token` | archive provenance | no | `board_token` is a proxy for `source` and inherits its open status; it is also absent for boards on custom domains |
| `internal_job_id` | Greenhouse's internal key | no | surrogate key, monotonic in creation order — the same objection as `jobs.id` |

---

## 3. `jobs` — identity and labels only

Current state, rewritten in place on every edit. **No feature is read from this
table.** Listed here so that each column's exclusion is on the record.

| column | meaning | at prediction point? | why not |
|---|---|---|---|
| `id` | surrogate primary key | no | monotonic in insertion time; a model given it learns calendar order |
| `source`, `source_id` | identity | n/a | used to join, never as a feature except as `design.md` §4 decides |
| `last_seen` | last run that saw the posting | **no — this is the label** | anything derived from it is the outcome. `last_seen - first_seen` is the outcome minus a constant |
| `first_seen` | first run that saw it | no | legal (`<= t`) but misleading: for 1,135 of 1,240 postings it records when *this project* started looking, not when the posting appeared |
| `title`, `company`, `location`, `remote` | current values | no | current state. A posting edited after `t` leaks its edit backwards |
| `salary_min`, `salary_max`, `currency`, `salary_raw` | current values | no | same, plus the parse faults above |
| `description` | current text | no | same; edits recorded 3,645 times, all after first sight |
| `seniority` | derived from title | no | current state, and absent from the snapshots — re-derive from the as-of-`t` title |
| `posted_at` | publication date | no *from here* | the value is fine but the source is not; take it from the snapshot or the archive |
| `content_hash` | hash of current content | no | current state by definition |
| `hash_version`, `parser_version` | which of **our** versions produced the row | no | facts about the instrument. They encode when our code was deployed, which correlates with calendar time, which determines which rows are censored |
| `url` | link | no | identity; encodes the board |

---

## 4. `runs` — label machinery, never features

Everything here describes **our crawl**, not the job market. §6.3: a model given
these can learn the deploy schedule and appear to predict removals.

| column | meaning | at prediction point? | notes |
|---|---|---|---|
| `id` | run key | n/a | identity |
| `source` | which board this run scraped | n/a | see `design.md` §4 |
| `started_at` | **this is `t`** | n/a | the prediction point |
| `finished_at` | when the run ended | no | after `t`, and instrumental |
| `status` | `ok` / `partial` / `failed` | **label machinery** | a *complete run* is `status='ok'` at the current `rules_version`. Absence is only evidence of removal when the observation was complete |
| `page_cap`, `pages_fetched` | pagination limit and use | **label machinery** | `pages_fetched >= page_cap` means the board was only partly seen. This is the arbeitnow defect — see `DEBUGGING.md` |
| `rules_version` | comparison epoch | **label machinery** | runs are only comparable within a version. Bumping it restarts the label series |
| `rows_parsed` | rows this run produced | no | a measurement of our crawl. It has already been misread once as hiring activity when it was a `--pages` setting changing |
| `parser_version` | our parser version | no | instrument |

---

## 5. `job_observations` — the panel, and the label

| column | meaning | at prediction point? | notes |
|---|---|---|---|
| `job_id`, `run_id` | posting seen by run | **split by time** | rows for runs at or before `t` are the observation history and are usable; **rows for runs after `t` are the label** and nothing else |

Derived counts need care. `n_complete_runs_observed` up to `t` is legal but is
strongly confounded with how long ago the panel started, so it is only
interpretable alongside age.

---

## 6. `job_changes` — edit history

| column | meaning | at prediction point? | notes |
|---|---|---|---|
| `job_id` | which posting | n/a | identity |
| `observed_at` | when the change was noticed | **windowed** | rows with `observed_at <= t` only; later rows are the future's edits |
| `field`, `old_value`, `new_value` | what changed | **windowed** | same window |
| `old_parser_version`, `new_parser_version` | attribution | no | instrument — but needed to *exclude* changes caused by our own parser rather than the employer |

**Caveat that limits this table.** Changes before 2026-09-04 cannot be
attributed to employer or parser, so they are unusable as employer-edit
features. The archive's `updated_at` supersedes this table for that purpose and
covers the whole panel.

---

## 7. Columns produced by cleaning

`src/data/clean.py`. All row-local and deterministic: each decides using only
the value in front of it, so none can leak.

| column | meaning | at prediction point? |
|---|---|---|
| `title_clean`, `company_clean`, `location_clean` | NFKC-normalised, whitespace-collapsed | **yes** — inherits its input |
| `seniority_clean` | validated against a closed vocabulary | **yes**, when derived from the as-of-`t` title |
| `salary_stated` | the posting mentioned pay at all | **yes** |
| `salary_parsed` | a figure could be read from it | **yes** — distinct from the above on purpose |
| `salary_min_clean`, `salary_max_clean` | figures as printed | **yes** |
| `salary_period` | hourly / monthly / yearly / unstated | **yes** — 1,503 of 1,666 are unstated |
| `salary_min_annual`, `salary_max_annual` | scaled to a year | **yes**, under a stated assumption: unstated periodicity is treated as annual |
| `salary_currency_clean`, `salary_is_range` | | **yes** |
