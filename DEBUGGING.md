# Debugging record

What broke, why, and the rule that stops it recurring. Newest entry first.

---

## 2026-09-04 — "Disappeared from the board" was measuring the scraper's page cap, not job closures

- **Problem:** the planned label — a posting is *closed* on the first day it
  stops appearing in a scrape — is invalid for `arbeitnow`, which is 74% of the
  dataset (3,531 of 4,771 jobs). Nothing crashed and no test failed; the label
  simply meant something other than what it claimed. Taken at face value it
  implies the board turns over 40–55% per day (2026-09-03: 404 postings appear,
  540 vanish), which no real job board does. Had this gone unnoticed, roughly
  2,600 rows of confidently wrong labels would have dominated training, and the
  resulting model would have scored well while predicting nothing but recency.

- **Root cause:** `arbeitnow` runs stop at a page cap. Every run since
  2026-08-31 records `page_cap = 8, pages_fetched = 8, rows_parsed = 950,
  status = 'partial'` — the cap was reached, so there were further pages that
  were never fetched. The daily visible set is therefore pinned at the cap
  (949, 934, 935, 950, 929) rather than reflecting the true size of the board.
  Because the listing is ordered newest-first, a posting drifts past page 8 as
  newer ones arrive and is never seen again. The label logic assumed *absent
  from a scrape* ⇒ *removed from the board*, but for a truncated listing the
  correct reading is *absent from the fetched window*, which is a statement
  about the scraper's position in the list, not about the posting.
  Gap analysis does not catch this: a posting pushed past the window never
  reappears either, so the reappearance rate is ~0% under both the true and the
  false hypothesis. What separates them is that the visible set is pinned to the
  cap and inflow tracks outflow.
  The seven `greenhouse:*` sources fetch the full board (`status ok`, cap never
  reached) and show credible churn instead — `anthropic` moves 571 → 590 visible
  with 6–15 postings leaving per day, ~1–3%.

- **Solution:** not yet fixed; the label is quarantined rather than repaired.
  `arbeitnow` and the pre-2026-09-01 `python_org` runs are excluded from
  labelling, leaving the `greenhouse:*` sources as the only ones where absence
  is evidence of removal. That is ~1,150 labellable rows and 39 positives at a
  3-day horizon, which is not enough to train and evaluate against a temporal
  split — so the choice of horizon and of which sources carry labels is
  deferred to `docs/design.md` (Decisions 1–3), and run completeness is carried
  through ingestion (`src/data/load.py` keeps `status`, `page_cap`,
  `pages_fetched` on every run) so that no downstream step can silently treat a
  truncated scrape as a complete one.

- **Lesson:** absence is only evidence of removal when the observation was
  complete. Before deriving any label from *a record stopped appearing*, prove
  the collector actually looked at the whole population on that day — for a
  paginated source that means the cap was not reached. More generally: when a
  label is derived from a collection process rather than observed directly, the
  metadata about that process (`status`, `page_cap`, `pages_fetched`) is part of
  the label definition, not logging. And a churn rate that would be implausible
  in the real world is a bug report about the measurement, not a finding about
  the world.
