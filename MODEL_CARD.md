# Shelf-life model card

**Status: awaiting final H=7 selection — no released model.** Reviewed
2026-09-11; the [README's model card](README.md#model-card) is the short form
of this one and the two say the same thing or one of them is wrong. This is the release summary to complete after selection; pending
fields are unavailable evidence, not zero scores. The current comparison is an
H=1 rehearsal, and both local model sidecars identify synthetic H=1 artifacts.
Neither supplies final seven-day results. `MODEL_TAG` has no release value.

> A positive prediction means likely disappearance from the monitored board within seven days; it does not establish that the vacancy was filled.

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
| Eligible boards | Greenhouse: Airtable, Anthropic, Discord, Duolingo, Figma and GitLab; plus `python_org`. Complete runs use `status=ok`, `rules_version=2` in the reviewed snapshot. |
| Excluded population | Arbeitnow: leaving its rolling feed cannot reliably distinguish removal from ageing out of the observation window. |

The label implementation is [compute_labels](src/features/assemble.py); its
contract is [the problem definition](docs/problem_definition.md#4-target-definition).
Age uses the employer's publication time, not collection `first_seen`.
Incumbent postings are retained despite unobserved arrival times.

## Model and inputs

**Selected family, run and parameters: pending.** The comparison includes a
constant prior, age rules, an age-only logistic fit, board hazard, logistic
regression, a decision tree, random forest and XGBoost. The registered rule
requires at least three scored rolling-origin folds, compares paired PR-AUC
differences against their spread, favours simplicity among tied candidates and
falls back to a heuristic when fitted complexity has no supported advantage.
Final selection remains a reviewed decision.

**Final feature subset: pending.** The current default pipeline allows the
following 24 features; this is a candidate inventory, not a claim that a final
model uses all of them:

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
There is no blanket numeric missingness-indicator step. Optional board context
is currently accepted and imputed when absent; its two serving regimes get
separate final results, and which pipeline ships as the default is decided by
the rule fixed 2026-09-12 in `src/models/board_context.py` (design §12): the
refit without the four columns ships iff the full model, with them imputed,
scores below the refit by more than the candidate's fold spread. The decision
and its three validation numbers travel on the artifact as `board_context`.

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
Raw `first_published` and `updated_at` are consumed by derivation, not passed
directly to the estimator. See the [feature audit](docs/leakage_audit.md) and
[executable registry](src/features/preprocessing.py) for individual verdicts.

## Periods, split and operating point

| Item | Recorded status / policy |
|---|---|
| Available observation period | 2026-08-31 03:45:25.789180Z to 2026-09-11 04:23:58.090801Z; 12,675 job-days in the reviewed H=7 panel. |
| Final training period | Pending: no legal H=7 three-way split. Record actual retained `t` bounds, rows, postings and positives after selection. Default final fit is training only. |
| Final evaluation period | Validation and held-out test periods both pending; the real test set has not been opened. Record both cut instants and actual retained bounds. |
| Temporal policy | Earliest train, middle validation, latest test; never random rows. `best_cuts` prefers three rolling-origin folds within train, then approximately 60/20/20 of retained waves after embargoes. This simulates training on past boards and scoring later observations. |
| Posting overlap | The implemented job-day split permits the same posting at different times across blocks, a deliberate departure recorded in [design §8](docs/design.md#8-the-split--decided-2026-09-04). It does not guarantee disjoint posting IDs; report carried-over versus unseen-in-training performance separately. |
| Embargo | At each boundary: `H days + largest observed within-source complete-run gap`, for one corroborating run. On this snapshot: **9 days 13:43:51.653374**, discarding 10 crawl waves per boundary. Recompute and record it for the final snapshot. |
| Current depth | On 2026-09-11: five labelled waves, zero legal cuts, zero folds, against 22 labelled waves for a legal split and 34 for three folds. The live figure is [readiness](reports/readiness.md); this row is the reviewed one and is not updated by hand. |
| Selected threshold | Pending. Default policy selects the validation score at rank `min(validation rows, 20 × prediction days)` and freezes it with the fitted pipeline. |
| Serving modes | `/predict` is the individual-posting mode and does not accept the four board-context fields (imputed; `board_context_supplied=false`). `/rank` is the board-ranking mode: with `is_board_snapshot=true` the service derives the four from the batch by the panel's definitions and reports `board_context_source` and `board_size`; undeclared, they are supplied per posting or imputed. Decided 2026-09-12, design §12. |
| Budget behaviour | `probability >= threshold` includes ties and need not yield exactly 20 alerts on a later day. `/rank` with an explicit budget selects exactly the requested top count, capped at batch size, using input order to break ties; it reports `threshold_source=batch_budget`. Without that override, the frozen threshold applies. A board larger than one request goes through `POST /boards` (upload in pages, score in pages, rank once server-side, same rule); the client never merges. A batch rank cutoff is not a calibrated probability guarantee. |

See [readiness](reports/readiness.md) and the
[selection record](reports/model_comparison.md). This card does not trigger
training, selection, freezing or another opening of the held-out test set.

## Results and uncertainty

**No final validation or test result is available.** Populate these fields from
the selected H=7 run and its frozen artifact, retaining the split and operating
point alongside every result. Do not substitute H=1 or synthetic scores.

| Final evidence | Status and required reporting |
|---|---|
| Validation | Pending. **Headline: precision@20/day, recall@20/day, lift over the base rate, removals caught per day** (the product's numbers, `metrics.at_budget`), then baseline and selected PR-AUC (average precision), fold mean/SD, F1, and TP/FP/FN/TN at the selected threshold; NDCG@20 secondary. Accuracy is not reported; ROC-AUC is secondary. |
| Held-out test | Pending: the same metrics, the shortlist table first with 95% posting-clustered intervals on precision@20, recall@20 and lift; threshold-dependent results at the frozen validation threshold distinguished from a recomputed test-budget diagnostic. |
| Bootstrap confidence intervals | Pending. Existing test protocol: 95% percentile intervals, 2,000 draws, seed 0; resample `(source, source_id)` posting clusters with all their rows, keeping the model and threshold fixed. Report PR-AUC, precision, recall, Brier and ECE intervals, retained resample counts and the fewer-than-30-positive fragility flag. Zero-positive draws are discarded. These intervals do not measure model-selection uncertainty or unseen-board shift. Validation intervals are emitted by both `evaluate` and `freeze` as of 2026-09-12, at the frozen threshold. |
| Calibration | Pending: Brier, ECE and the ten equal-width-bin reliability table, including bin counts. Whether to recalibrate is **decided by a rule fixed 2026-09-12** (`src/models/calibration.py`, design §5): isotonic on validation iff validation ECE > 0.25 × base rate and ≥ 30 positives; the decision and its numbers travel on the artifact as `recalibration`. No final calibration quality has been established. |
| Per-board / cohort performance | Pending: each eligible board, incident versus incumbent stock, first-observation-only rows, and carried-over versus unseen-in-training postings; include sample sizes, positive counts and unavailable/undefined metrics explicitly. All four slices are produced by `evaluate` (validation) and the first three by `freeze` (test) as of 2026-09-12. |
| Unseen-board transfer | Pending: leave-one-board-out score, board base rate, comparison with the board included in training, and skipped-board reasons — on the selected candidate, with board context withheld as at serve time (the driver built XGBoost regardless until 2026-09-12). **A release gate**: `freeze` refuses a candidate that collapses to the base rate on held-out boards (design §4a); the verdict travels on the artifact as `transfer`, and only `intact` licenses any claim about unseen boards, scoped to boards of this kind. |

The following are **descriptive H=7 label counts, not model performance**, from
the 2026-09-11 [cohort audit](reports/cohort_audit.md):

| Board | Labelled job-days | Positive job-days | Rate |
|---|---:|---:|---:|
| Greenhouse: Airtable | 80 | 0 | 0.0% |
| Greenhouse: Anthropic | 2,874 | 210 | 7.3% |
| Greenhouse: Discord | 245 | 29 | 11.8% |
| Greenhouse: Duolingo | 430 | 25 | 5.8% |
| Greenhouse: Figma | 795 | 40 | 5.0% |
| Greenhouse: GitLab | 1,107 | 115 | 10.4% |
| python_org | 126 | 17 | 13.5% |
| **Total** | **5,657** | **436** | **7.7%** |

Incident postings (first seen after that board's initial collection wave):
254 labelled rows, 105 postings, 25 positive rows (9.8%). Incumbent stock:
5,403 rows, 1,135 postings, 411 positive rows (7.6%). Incident does not mean
newly published, and its rows are not limited to first observations. These
small, correlated cohorts do not establish equal performance or equal hazards.

## Validity, limitations and prohibited interpretations

The [external board check](reports/label_check.md), using the 2026-09-08
snapshot, found **59/60** sampled labelled removals absent under their original
IDs: 98.3%, with a **95% Wilson interval of 91.1%–99.7%**. Seven had the same
title listed under a new ID. In the control arm, 2/60 postings had disappeared
since the last crawl. This supports board-disappearance validity, not hiring
outcome validity. It is a later membership check, not proof of the exact
historical disappearance time; same-ID reinstatement remains possible. Its
Wilson interval is not a bootstrap interval for predictive performance.

Limitations include daily interval censoring and horizon-boundary ambiguity,
missed observations, reappearance/relisting, short panel depth, few independent
removal events, a small incident cohort, unequal board coverage and structural
missingness. Excluding explicit identity does not remove source confounding:
the H=1 [fingerprint diagnostic](reports/board_fingerprint.md) recovered board
identity with 100% accuracy from production features. That diagnostic is not a
disappearance score or evidence of transfer. Imputed board context and repeated
posting observations create additional serving and memorisation risks.

Do not interpret a positive as a filled vacancy, successful hire, applicant
success probability, hiring deadline, or proof that the role no longer exists
elsewhere. Do not interpret a negative as guaranteed continued availability.
Do not claim validated seven-day probabilities, calibration, or generalisation
to unmonitored boards before the corresponding final evidence exists.

## Provenance and release completion

These identifiers pin the **reviewed evidence**, not a selected model:

| Item | Identifier |
|---|---|
| Reviewed source commit | `98cd96d3cc841b8e0559d6dca388352cb14f2b84` |
| H=7 report source commit | `bfa63a9469238c97b215469b3f2b0f812711c883`; reports record a dirty tree, so this alone cannot reproduce their full source state. |
| Dataset snapshot ID | `2026-09-11`, pinned at `2026-09-11T04:30:05.520890+00:00` |
| Snapshot database SHA-256 | `658cd4831d9cb7487dc7b3729fa5296d7d6f6feb20911425f88d7c7cb548d311` |
| H=7 feature panel SHA-256 | `bf14a21d28a678b67a791cef84f3966e82a10c9b1885a661c10e5cc6eafe8151` (`data/processed/features/job_days_h7_calendar.parquet`) |
| Reviewed dependency lock SHA-256 | `195c41368c2bac6e97a26038c745b990b22103b9b41632848af1595bfd82c0e1` ([uv.lock](uv.lock)); not yet a final model's lock attestation. |
| Final artifact SHA-256 / release tag | **Pending — no real H=7 artifact or release.** |
| Final training source / lock / snapshot | **Pending.** Copy the selected artifact's `provenance.git_sha`, `lock_sha256`, `snapshot_date`, `panel_sha256` and `rules_version`; do not assume the reviewed values above remain current. |

After final selection, complete this card from the matching
`models/shelf_life.json`, release `SHA256SUMS`, validation report and existing
[test report](reports/test_results.md). Verify the artifact checksum and clean
training commit, record actual periods, family, parameters, feature subset,
library versions, seed and budget, then replace pending results with the
recorded evidence. Preserve the distinction between validation, held-out test,
label audit and transfer; reading recorded results does not require rerunning
the test evaluation.
