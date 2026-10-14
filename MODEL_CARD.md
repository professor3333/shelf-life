# Shelf-life model card

**Status: released as [`artifact-2026-10-14`](https://github.com/professor3333/shelf-life/releases/tag/artifact-2026-10-14)
— the H=7 freeze of `03-random_forest`.** Completed 2026-10-14 from the
artifact's sidecar (`shelf_life.json`), the release's `SHA256SUMS` and
[`reports/test_results.md`](reports/test_results.md); nothing here is from
memory or recomputed. The [README's model card](README.md#model-card) is the
short form of this one and the two say the same thing or one of them is wrong.
The service serves this model when `MODEL_TAG` names the tag.

> A positive prediction means likely disappearance from the monitored board within seven days; it does not establish that the vacancy was filled.

**The finding to read first.** On held-out postings the model also saw during
training (at earlier crawls), test PR-AUC is **0.304** against a base rate of
0.112; on postings it never saw it is **0.132** against 0.105, and on a
posting's first sighting **0.131** against 0.102. Most of the lift comes from
recognising postings already on the board, not from anything that carries to a
new one. Ranking a board's existing stock is the product, so this is useful
for that; it is close to the base rate for a posting nobody has seen yet. The
result is final — the test block has been opened and nothing is re-tuned
against it (design §20).

## Target, population and use

**Intended use:** help a job seeker prioritise today's available postings for
inspection and application. Rank the whole monitored board, including existing
stock and new arrivals, with an intended inspection budget of 20 postings per
day. A missed disappearance may cost an application opportunity; a false alert
costs attention. This is not a total-lifetime or first-appearance-only model.

| Item | Definition |
|---|---|
| Observation unit | One `(source, source_id, complete run)` pair, called a job-day. Repeated observations of a posting are correlated. |
| Prediction time | `t = runs.started_at` in UTC, when that complete run observed the posting. Features must exist at or before `t`. |
| Exact target | H=7, `horizon_basis=calendar`: disappearance by the UTC calendar date of `t + 7 days`, conditional on being observed at `t`. It is not an exact 168-hour guarantee. |
| Disappearance | `t_gone` is the earliest complete run after first sight at which the posting is absent, corroborated by absence in the next complete run (`K=2`). The label is final at corroboration; subsequent reappearance does not revise it. |
| Positive label | First require a complete run on or after the horizon date for that source. Then `y=1` if `t_gone` is defined and its date is on or before the horizon date. |
| Negative / censored | Otherwise `y=0` only if the posting was observed in a complete run on or after the horizon date. All other rows are excluded, including early known removals whose source's horizon has not elapsed. Unknown is never negative. |
| Eligible boards | Greenhouse: Airtable, Anthropic, Discord, Duolingo, Figma and GitLab; plus `python_org`. Complete runs use `status=ok`; the panel was built at scraper `rules_version=2`. |
| Excluded population | Arbeitnow: leaving its rolling feed cannot reliably distinguish removal from ageing out of the observation window. |

The label implementation is [compute_labels](src/features/assemble.py); its
contract is [the problem definition](docs/problem_definition.md#4-target-definition).
Age uses the employer's publication time, not collection `first_seen`.
Incumbent postings are retained despite unobserved arrival times.

## Model and inputs

**Selected: `03-random_forest`** — scikit-learn `RandomForestClassifier`
(`n_estimators=300`, `min_samples_leaf=5`, `class_weight="balanced_subsample"`,
`random_state=0`), inside the project `Pipeline`, wrapped in an isotonic
calibrator fitted on validation. Fitted on the **training block only**.

**How it was chosen.** By the rule pre-registered in design §14 and coded in
`select()` (`src/models/evaluate.py`), applied mechanically to four scored
rolling-origin folds inside the training window
([`reports/model_comparison.md`](reports/model_comparison.md), snapshot
2026-10-14, commit `b640656`):

| candidate | CV PR-AUC mean | SD |
|---|---:|---:|
| prior (constant) | 0.0791 | 0.0062 |
| best rule: `board_hazard` | 0.1004 | 0.0029 |
| logistic | 0.1021 | 0.0174 |
| xgboost (`05-xgboost_engineered`) | 0.1146 | 0.0138 |
| **random_forest** | **0.1255** | 0.0087 |

Random forest led; xgboost was inside fold variance of it and the rule's
parsimony step kept the simpler of the two; the lead over the best heuristic
rung is larger than fold variance, so the heuristic floor did not apply.

**Features: all 24 of the default pipeline** (the artifact's `features` list):

| Group | Features |
|---|---|
| Posting attributes | `age_days`, `days_since_update`, `content_chars`, `n_offices`, `n_metadata`, `salary_min_clean`, `salary_max_clean`, `departments`, `offices`, `location`, `posted_dow`, `salary_currency_clean`, `salary_stated` |
| Board context at `t` | `board_size_at_t`, `n_same_title_on_board`, `n_same_req_on_board`, `board_growth` |
| Stateless derivations inside the pipeline | `title_seniority`, `title_is_manager`, `title_words`, `title_chars`, `location_is_remote`, `n_locations`, `salary_band` |

As-of-run snapshots and archived payloads supply inputs. Raw title feeds
derivations rather than a text encoder; publication and update timestamps feed
age calculations. Imputation, scaling and category encoding are fitted inside
the training fold only. Missing numerics use per-column median/zero/one rules;
missing categoricals get explicit levels, and unseen categories are supported.

**Board context ships imputed.** By the rule in `src/models/board_context.py`
(design §12) — ship the refit without the four board columns iff the full
model, with them imputed, scores below it by more than fold spread. Validation
PR-AUC: supplied 0.2187, imputed 0.2189, refit without 0.2199, fold spread
0.0087 — inside the spread, so the full model ships and imputes when a caller
cannot supply them. `/rank` with `is_board_snapshot=true` derives them.

**Excluded inputs:** direct board identity (`source`, `company`,
`company_posting_volume`); identifiers and split axes (`source_id`, `url`,
`requisition_id`, `run_id`, `run_index`, raw `t`, `split`, `seen_in_train`);
target/censoring metadata (`y`, `label_observable`, `horizon_days`,
`horizon_basis`, `rules_version`); `first_seen`, `last_seen`, future edits,
lifetime observation/change counts, `n_observations_total`,
`days_on_board_total`; and the skewed or redundant fields
`n_complete_runs_observed`, `t_dow`, `posted_month`, `remote`, `n_departments`,
`salary_period`, `salary_parsed`, upstream `salary_min`, `salary_max`,
`currency`, `posted_at`, raw `salary_raw` text, `job_type_raw`, `tags_raw`.
See the [feature audit](docs/leakage_audit.md) and
[executable registry](src/features/preprocessing.py) for individual verdicts.

## Periods, split and operating point

| Item | Recorded value |
|---|---|
| Available observation period | 2026-08-31 03:45:25Z to 2026-10-14 03:48:29Z: 54,195 job-days in the H=7 panel, 47,199 labelled, 4,602 positive. |
| Training block | 2026-08-31 – 2026-09-13 (14 days): 14,924 job-days, 1,348 postings, 1,232 positives. |
| Validation block | 2026-09-23 – 2026-09-24 (2 days): 4,628 job-days, 1,177 postings, 463 positives, base rate 0.100. |
| Held-out test block | 2026-10-04 – 2026-10-07 (4 days): 4,580 job-days, 1,197 postings, 504 positives, base rate 0.110. Opened once, by `freeze` at `ab2d207`. |
| Claim scope | *Validated on the collected 2026-09-23 – 2026-10-07 period* (design §18). Seasons, holidays, hiring cycles, policy and schema changes are not demonstrated and not claimed. |
| Temporal policy | Earliest train, middle validation, latest test; never random rows. Rolling-origin folds inside train for selection. Simulates training on past boards and scoring later observations. |
| Posting overlap | The job-day split permits the same posting at different times across blocks (design §8): 881 of the 1,197 test postings were also seen in training. Their performance is reported separately below, and the difference is the headline finding. |
| Embargo | At each boundary: `H days + largest within-source complete-run gap` = **9 days 13:43:51**, discarding 10 crawl waves per boundary. |
| Threshold | **0.340909** — the validation score at rank 40 (20 a day × 2 validation days), frozen with the pipeline. |
| Serving modes | `/predict` is the individual-posting mode and does not accept the four board-context fields (imputed; `board_context_supplied=false`). `/rank` is the board-ranking mode: with `is_board_snapshot=true` the service derives the four from the batch and reports `board_context_source` and `board_size`. |
| Budget behaviour | `probability >= threshold` includes ties and need not yield exactly 20 alerts on a later day — on the test block it flagged 12 a day. Isotonic calibration maps scores onto a few plateaus, so many postings share one probability (the release smoke test ranked five postings all at 0.1222); `/rank` with an explicit budget breaks ties by input order and reports `threshold_source=batch_budget`. |

## Results and uncertainty

Every interval is a 95% posting-clustered percentile bootstrap (2,000 draws,
seed 0, `(source, source_id)` clusters with all their rows, model and threshold
held fixed). These intervals do not measure model-selection uncertainty or
unseen-board shift.

**At the frozen threshold — what the deployed model does:**

| metric | validation | test | test 95% interval |
|---|---:|---:|---|
| precision@20/day | 0.800 | **0.750** | [0.462, 1.000] |
| recall@20/day | 0.069 | **0.071** | [0.030, 0.116] |
| lift over the base rate | 8.00× | **6.82×** | [4.24×, 9.34×] |
| flagged per day | 20.0 | 12.0 | — |
| removals caught per day | 16.0 | 9.0 | — |
| TP / FP / FN / TN | 32 / 8 / 431 / 4,157 | 36 / 12 / 468 / 4,064 | — |
| F1 | 0.127 | 0.130 | — |
| PR-AUC (threshold-free) | 0.2082 | **0.2408** | [0.181, 0.308] |
| ROC-AUC (secondary) | 0.619 | 0.644 | — |
| Brier | 0.0835 | 0.0899 | [0.077, 0.103] |
| ECE | 0.000 (in-sample: the calibrator was fitted here) | 0.0067 | [0.0048, 0.0249] |

Test PR-AUC is above validation (gap −0.033), inside both intervals. Accuracy
is not reported: at a base rate of 0.110, predicting "stays" for everything
scores 89%.

**Read the frozen row, not the test report's headline table.** The first table
in `test_results.md` ("The shortlist") prints point estimates from a threshold
*recomputed on the test block's own budget* — 80 alerts, which ties at the
isotonic plateau expand to 138 flagged: precision 0.558, recall 0.153, lift
5.07×, 19.25 caught a day — beside intervals computed at the *frozen*
threshold. Recall 0.153 therefore sits outside its own printed interval. The
recomputed figures are a hindsight diagnostic of what a test-chosen threshold
would have done; the frozen figures above are the model's. The report is not
regenerated, because regenerating it would open the test block again; the
rendering is fixed for the next epoch (design §20).

**Calibration.** Recalibrated by the rule fixed 2026-09-12
(`src/models/calibration.py`, design §5): isotonic on validation, because
validation ECE 0.0756 exceeded 0.25 × base rate (0.0250) with 463 positives.
On test:

| predicted | n | mean predicted | observed |
|---|---:|---:|---:|
| 0.0–0.1 | 2,526 | 0.079 | 0.075 |
| 0.1–0.2 | 1,896 | 0.119 | 0.123 |
| 0.2–0.3 | 20 | 0.240 | 0.200 |
| 0.3–0.4 | 90 | 0.341 | 0.456 |
| 0.5–0.6 | 26 | 0.600 | 0.538 |
| 0.9–1.0 | 22 | 1.000 | 1.000 |

The curve transferred (ECE 0.0067), but 97% of rows sit below 0.2: the model
says "about the base rate" for almost everything and is confident about very
little.

**Carried over, cohort and first sighting** (test block):

| slice | n | postings | positives | base rate | PR-AUC [95%] | precision@20 [95%] | fragile |
|---|---:|---:|---:|---:|---|---|---|
| seen in training | 3,447 | 881 | 385 | 0.112 | 0.304 [0.233, 0.380] | 0.880 [0.740, 0.978] | no |
| **not seen in training** | 1,133 | 316 | 119 | 0.105 | **0.132** [0.090, 0.194] | 0.142 [0.093, 0.199] | no |
| incumbent stock | 2,824 | 721 | 303 | 0.107 | 0.312 [0.227, 0.404] | 0.688 [0.500, 0.856] | no |
| incident flow | 1,756 | 476 | 201 | 0.115 | 0.188 [0.124, 0.271] | 0.238 [0.058, 0.424] | no |
| first observation | 186 | 186 | 19 | 0.102 | 0.131 [0.079, 0.224] | 0.137 [0.080, 0.207] | **yes** |
| later observations | 4,394 | 1,178 | 485 | 0.110 | 0.247 [0.184, 0.319] | 0.589 [0.427, 0.756] | no |

**Per board** (test block; a model fitted on every board, so this answers
*does it work here*, not *would it work somewhere new*):

| board | n | postings | positives | base rate | PR-AUC [95%] | precision@20 [95%] | fragile |
|---|---:|---:|---:|---:|---|---|---|
| Greenhouse: Airtable | 16 | 4 | 2 | 0.125 | 0.417 [0.417, 0.417] | 0.125 [0.125, 0.375] | **yes** |
| Greenhouse: Anthropic | 2,556 | 669 | 257 | 0.101 | 0.273 [0.182, 0.369] | 0.613 [0.382, 0.801] | no |
| Greenhouse: Discord | 198 | 51 | 25 | 0.126 | 0.239 [0.059, 0.563] | 0.135 [0.045, 0.259] | **yes** |
| Greenhouse: Duolingo | 237 | 60 | 16 | 0.068 | 0.077 [0.018, 0.190] | 0.097 [0.000, 0.217] | **yes** |
| Greenhouse: Figma | 635 | 167 | 84 | 0.132 | 0.196 [0.103, 0.316] | 0.128 [0.055, 0.224] | no |
| Greenhouse: GitLab | 833 | 219 | 111 | 0.133 | 0.299 [0.171, 0.461] | 0.162 [0.104, 0.222] | no |
| python_org | 105 | 27 | 9 | 0.086 | 0.189 [0.052, 0.503] | 0.106 [0.023, 0.238] | **yes** |

Four of seven boards are fragile (fewer than thirty positives); their point
estimates are orders of magnitude, not bounds. Anthropic is 56% of the test
rows and carries the pooled shortlist precision.

**Unseen-board transfer** (leave-one-board-out on train and validation,
computed before the test block was opened; design §4a release gate):
**`intact`**. Transfer PR-AUC minus base rate: Discord +0.0648, Figma +0.0011,
GitLab −0.0246; mean +0.0138. Seeing a board at fit time was worth +0.0480
PR-AUC against a spread of 0.1121. Skipped: Airtable, Duolingo and python_org
(fewer than 10 validation positives); Anthropic (holding it out leaves 57% of
training positives, under 60%). `intact` licenses describing the model as
applicable to a Greenhouse-family board of the kind these are — and two of the
three measured boards sit at or below their base rate, so that licence is thin.

**Label counts** (H=7, all labelled rows of the 2026-10-14 panel, from
[`reports/cohort_audit.md`](reports/cohort_audit.md); descriptive, not model
performance):

| Board | Labelled job-days | Postings | Positive job-days | Rate |
|---|---:|---:|---:|---:|
| Greenhouse: Airtable | 414 | 19 | 113 | 27.3% |
| Greenhouse: Anthropic | 25,069 | 871 | 2,084 | 8.3% |
| Greenhouse: Discord | 1,973 | 76 | 220 | 11.2% |
| Greenhouse: Duolingo | 3,305 | 105 | 342 | 10.3% |
| Greenhouse: Figma | 6,486 | 215 | 495 | 7.6% |
| Greenhouse: GitLab | 8,771 | 360 | 1,240 | 14.1% |
| python_org | 1,181 | 40 | 108 | 9.1% |
| **Total** | **47,199** | **1,686** | **4,602** | **9.8%** |

Incident rows are removed at 9.8% against 9.7% for incumbent stock (10,809
against 36,390 rows): the label does not key on which population a row came
from. The model, per the cohort table above, does.

## Validity, limitations and prohibited interpretations

The [external board check](reports/label_check.md) on the 2026-10-14 snapshot
found **57 of 60** sampled labelled removals absent under their original IDs:
95.0%, with a **95% Wilson interval of 86.3%–98.3%**. Seven of the 57 had the
same title listed under a new ID. In the control arm, 0 of 60 postings live at
the final crawl had disappeared since. This supports board-disappearance
validity, not hiring-outcome validity. It is a later membership check, not
proof of the exact historical disappearance time; same-ID reinstatement remains
possible. Its Wilson interval is not a bootstrap interval for predictive
performance.

Limitations include daily interval censoring and horizon-boundary ambiguity,
missed observations, reappearance/relisting, a short panel (six weeks), few
independent removal events, a two-day validation block, unequal board coverage
and structural missingness. **The source family is narrow by construction**:
six Greenhouse boards and python_org, because the largest source (arbeitnow) is
unlabelable: its crawls stop at a page cap short of the feed's end and the board
reorders under them, and even a complete crawl could not tell a posting that
aged out of its roughly seven-day feed from one that was removed; nothing here
measures or claims performance on a board of a different kind. Fixing the
collector, an age-bounded label rule for rolling feeds, and genuinely different
families are v2 concerns, recorded in design §17. Excluding explicit identity
does not remove source confounding: the
[fingerprint diagnostic](reports/board_fingerprint.md) recovers the board from
the production features at 100%. **The model memorises postings**: training
PR-AUC is 0.93 against 0.22 on validation, and the held-out lift is concentrated
in postings seen during training (table above).

Do not interpret a positive as a filled vacancy, successful hire, applicant
success probability, hiring deadline, or proof that the role no longer exists
elsewhere. Do not interpret a negative as guaranteed continued availability. Do
not claim the model scores a never-seen posting much better than the base rate,
or that it generalises to boards outside the Greenhouse family or to periods
outside 2026-09-23 – 2026-10-07.

## Provenance

| Item | Identifier |
|---|---|
| Release tag | [`artifact-2026-10-14`](https://github.com/professor3333/shelf-life/releases/tag/artifact-2026-10-14), published 2026-10-14T10:26:49Z |
| Artifact SHA-256 (`shelf_life.joblib`) | `c5b9adebe1333b041848696d07f25137b187114eebb7cbf2ed2516d8d662124c` |
| Sidecar SHA-256 (`shelf_life.json`) | `73edbcabea91b0ab09211b395a497f08796d62867b3a8aef5c041b0b1da8c7e7` |
| Frozen at | 2026-10-14T10:26:40Z, commit `ab2d207dc889d5d92eaa64cfac73088cc98e4559` on `main`, clean tree |
| Evidence it was selected on | reports at `b640656`, all naming snapshot 2026-10-14 (`python -m src.models.evidence`: one bundle) |
| Dataset snapshot | `2026-10-14`, pinned 2026-10-14T09:40:05Z through SQLite's online backup; database SHA-256 `e8a6fedac29bcf3ffc0525ca8fc1d3f73b1869d45357a1e670e377de481ccd73` (473 runs, 36,852 jobs) |
| H=7 feature panel | `data/processed/features/job_days_h7_calendar.parquet`, 54,195 rows, SHA-256 `716c8cac7f63190fee809124729aa427d8475d84053201ff7bf9625f5bc7bc7b` |
| Scraper rules version | 2 |
| Dependency lock SHA-256 | `faed2403246939c85b469937506842fb1423348d2e1346b7f3a4db89ae253d14` ([uv.lock](uv.lock)); scikit-learn 1.9.0, xgboost 3.4.1, joblib 1.6.0 |
| Seed / budget / folds | seed 0 · 20 postings a day · `selection_folds=4` |
| Recalibration / board context / transfer | `isotonic` · `full` (imputed when absent) · `intact` — each with its rule and numbers on the artifact |
