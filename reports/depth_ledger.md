# Depth ledger

Appended by `python -m src.models.evaluate` and `python -m src.models.freeze`.
Generated from `reports/depth_ledger.jsonl`; edit neither by hand.

One row per run, keyed by the code and the data it ran on. Read the metric
**against its interval and the positives column**, never on its own: this
panel accrues about 19 closures a day, so an early row is a number with an
interval wide enough to swallow most differences between models, and saying
so is the finding.

## Nothing recorded yet

No run has produced a metric. `reports/test_results.md` records the
shortfall and `scripts/rehearse.sh` is the readiness check.
