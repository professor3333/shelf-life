# shelf-life

A job posting has a shelf life: it sits on a job board until it is pulled, and
this project predicts how long that takes. Using a panel of job postings
collected daily by my own scraper, it builds an end-to-end classical machine
learning system — a reproducible dataset, a leakage-audited feature set, a
temporal train/validation/test split, a baseline, and a gradient-boosted model
that has to beat it — and serves the result over HTTP so a single posting can be
scored at the moment it first appears. The label is *removed from the board*,
which is not the same thing as *filled*; the name of the project is deliberately
chosen not to claim otherwise.

**Status:** in development. This README will be replaced with full setup, usage
and evaluation documentation once the system is complete.
