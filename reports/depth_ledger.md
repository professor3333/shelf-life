# Depth ledger

Appended by `python -m src.models.evaluate` and `python -m src.models.freeze`.
Generated from `reports/depth_ledger.jsonl`; edit neither by hand.

One row per run, keyed by the code and the data it ran on. Read the metric
**against its interval and the positives column**, never on its own: this
panel accrues about 19 closures a day, so an early row is a number with an
interval wide enough to swallow most differences between models, and saying
so is the finding.

## Synthetic panel

Machinery checks on `tests/panels.py`, whose label is drawn independently of every feature. **No number here is a finding about job postings.**

| snapshot_date | waves | positives | base_rate | folds | chosen | pr_auc | pr_auc_ci | cv_pr_auc_mean | cv_pr_auc_sd | stage |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-11 | 18 | 143 | 0.1230 | 5 | 05-xgboost_engineered | 0.1830 | [0.1012, 0.3102] | — | — | held_out |

A dash in `cv_pr_auc_sd` means fewer than two folds scored, so there is no
spread to report — the run has a metric and no error bar, which is the
state this ledger exists to make visible rather than to hide.

1 of 1 row(s) were recorded from a **dirty tree** and are provisional: the code that produced them is not any commit.
