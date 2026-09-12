# Depth ledger

Appended by `python -m src.models.evaluate` and `python -m src.models.freeze`.
Generated from `reports/depth_ledger.jsonl`; edit neither by hand.

## What is recorded

This is a depth history of **selected validation results and completed
held-out runs**, not an inventory of every metric-producing experiment.

- `evaluate` appends only when it selects a candidate, currently requiring
  at least three scored rolling-origin folds. Candidate metrics can appear
  in [model_comparison.md](model_comparison.md) without a ledger row.
- `freeze` appends after successfully saving an artifact, including
  synthetic rehearsals and explicit `--accept-no-folds` runs. An entry
  alone does not establish fold-qualified selection or a final release.
- H=1 and H=7 follow the same recording rules. Unselected H=1 rehearsals
  remain in [model_results.md](model_results.md), the comparison report
  and experiment tracking; `train` and `experiments` do not append here.

One row per `(stage, dataset, panel_sha256, git_sha)`: rerunning that
combination replaces its row. Different candidates or budgets alone do
not create separate rows. Read each metric against the available fold
spread or bootstrap interval and its sample size.

## Real panel

No selected real-data validation result or completed real-data
held-out run has been recorded here. H=1 candidate metrics in the
reports above do not require a selected model or a ledger row.

## Synthetic panel

Machinery checks on `tests/panels.py`, whose label is drawn independently of every feature. **No number here is a finding about job postings.**

| snapshot_date | waves | positives | base_rate | folds | chosen | period | pr_auc | pr_auc_ci | precision_at_budget | lift_at_budget | cv_pr_auc_mean | cv_pr_auc_sd | stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | 18 | 143 | 0.1230 | 5 | 05-xgboost_engineered | — | 0.1830 | [0.1012, 0.3102] | — | — | — | — | held_out |

A dash in `cv_pr_auc_sd` means CV spread was not recorded for that row.
It can be unavailable when fewer than two folds scored; `held_out` rows
do not store CV summaries even when folds exist. Their posting-clustered
bootstrap interval is reported separately in `pr_auc_ci`. A missing CV
spread does not imply that the run has no uncertainty estimate.

1 of 1 row(s) were recorded from a **dirty tree** and are provisional: the code that produced them is not any commit.
