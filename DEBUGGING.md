# Debugging record

What broke, why, and the rule that stops it recurring. Newest entry first.

---

## 2026-09-06 — A platform chosen from a pricing table instead of the documentation

- **Problem:** `docs/design.md` §7c committed the Streamlit UI to a Hugging Face
  Space on 2026-09-05, and the following day the plan was extended to put the
  FastAPI service on a Docker Space too. Both were wrong. The Hub's own
  documentation says Spaces that run on compute — Gradio and Docker — **require
  a paid plan to create**; only Static Spaces are free, plus up to two
  Gradio-on-ZeroGPU Spaces. Nothing failed and nothing errored: the decision was
  written, defended in prose, and a deploy path was built on top of it.

- **Root cause:** the pricing table says **"CPU Basic — 2 vCPU — 16 GB — FREE"**,
  and that sentence is true about the *hardware's hourly cost* while saying
  nothing about the *right to create a Space that uses it*. The reading that fit
  was chosen over the reading that was checked. The evidence I did consult —
  search results and summaries — repeated the same table, so agreement between
  sources was mistaken for verification when all of them had one origin.

- **Solution:** re-verified against `huggingface.co/docs/hub` directly before
  writing any code, along with Render's compute plans and Streamlit Community
  Cloud. §7 rewritten with a comparison table dated to the day it was checked,
  each row naming the requirement it fails. The architecture moved to a Render
  free web service plus Streamlit Community Cloud, and `render.yaml` now carries
  `plan: free` with `tests/test_deploy.py::test_the_service_plan_is_free`
  asserting it, because that field is the one whose drift costs money.

- **Lesson:** **a price is not a permission.** For any "free tier", the question
  to answer is not "what does this cost per hour" but "what must be true about my
  account before I am allowed to create one", and only the vendor's documentation
  answers the second. Corroboration from three summaries of one table is one
  source, not three. This is the same failure as the 139.6 MiB reading directly
  below: a single observation, taken once, believed because it was convenient.

## 2026-09-05 — A memory reading taken before the process had finished starting

- **Problem:** `docker stats --no-stream` on the freshly started API container
  reported **139.6 MiB** resident. Repeated with a wait, the same container on
  the same image reported **370.8 MiB**. Nothing failed; both commands exited 0
  and printed a plausible number. The low reading was about to be written into
  `docs/design.md` §7 as evidence that a 512 MB free tier is comfortable.

- **Root cause:** `--no-stream` takes a single sample, and the sample raced the
  container reaching steady state. The model is loaded in the FastAPI lifespan
  hook, so at the instant the first `/health` answers, the artifact is loaded
  but the allocator has not settled and the sample lands mid-startup. The
  process was not lying and neither was Docker — the number was simply of a
  moment that does not describe the running service.

- **Solution:** measured at three points instead of one — after start with a
  wait, after one prediction, after thirty-one — which agree at 371–377 MiB and
  show no growth under repeated requests. The figures and the method are in
  `docs/design.md` §7d, and the Render rejection in §7b was rewritten: it had
  been justified by a *guess* of 300–500 MB, and the measurement (377 MiB, which
  fits in 512 MB) does not support that reason. The platform choice stands on
  sleep behaviour and cold-start controls instead.

- **Lesson:** a single sample of a process that is still starting is not a
  measurement of that process. Sample repeatedly, with the load you care about,
  and only trust a figure that two readings agree on. And the direction matters:
  this error was *flattering* — it made a rejected option look viable — which is
  the kind that survives review, because a number that supports the easier
  decision does not get a second look.

---

## 2026-09-05 — The pinned prediction only held on the machine that wrote it

- **Problem:** three tests asserting a fixed posting scores a fixed probability
  passed locally and failed on the first CI run. macOS/arm64 returned
  `0.04348672926425934`; Linux/x86 returned `0.06030833721160889`. Not float
  noise — a 39% difference, from the same code, the same seed and the same
  synthetic panel.

- **Root cause:** the pin was taken against `05-xgboost_engineered`, and
  **gradient boosting is not bit-reproducible across platforms.** Histogram
  construction sums gradients in a thread-dependent order and the arithmetic is
  not associative, so two machines can disagree in the last decimal place of a
  split gain. On a 220-posting panel whose label is drawn independently of every
  feature, competing splits are near-ties by construction, and a last-decimal
  difference is enough to choose a different one — after which the two runners
  have fitted genuinely different trees, not the same tree with rounding error.
  `random_state` does not help: it seeds sampling, not summation order.

- **Solution:** the pinned artifact moved to `02-logistic`
  (`tests/conftest.py:FROZEN_RUN`). A convex fit reaches the same optimum from
  the same data on any machine, so the number means "the features changed"
  rather than "the runner changed". The stronger assertion was added alongside
  it: `test_the_feature_vector_for_a_fixed_posting_is_unchanged` pins the
  preprocessed matrix — width, sum and leading values — which is the feature
  logic itself and involves no fitted tree at all. Boosting is still frozen,
  saved, loaded and scored by `test_a_boosted_artifact_also_freezes_and_serves`;
  it just no longer carries a constant only one laptop can verify.

- **Lesson:** a regression pin is only as portable as the computation behind it.
  Before pinning a number, ask *what in this chain is allowed to differ between
  two machines* — thread counts, BLAS kernels, summation order, library builds —
  and pin at the last point upstream of all of it. Here that point is the
  feature matrix, and the end-to-end number is worth pinning only through an
  estimator whose fit is deterministic. This is also the argument for running CI
  on a different platform than you develop on: the failure was invisible on one
  machine and is not a rare edge case.

---

## 2026-09-05 — A serve-time row whose numbers arrived as text

- **Problem:** caught by printing `dtypes` on the first row `build_row`
  produced, not by an exception. Every numeric field the caller omitted —
  `board_size_at_t`, `n_offices`, `content_chars` — came back as `object` dtype
  holding a single `None`. The pipeline would have run: `select_columns` routes
  on dtype and would have handed those columns to the categorical branch, while
  the `ColumnTransformer` routes on *name* and would have sent them to a median
  imputer. The request would have returned a confident probability computed
  from a column the model had never seen in that form.

- **Root cause:** `pd.DataFrame([row])` infers each column's dtype from one
  value, and the only value was `None`. Training never sees this because the
  panel is built from thousands of rows at once, so a column of mostly-numbers
  is numeric whatever any single row holds. At serve time the frame is one row
  wide by definition, and **every payload omits something** — so the pathological
  case is the ordinary case, and the shapes of the two paths diverge exactly
  where nothing compares them.

- **Solution:** `src/inference/contract.py:build_row` now constructs each column
  as a typed `pd.Series` from the field's declared kind, using pandas' nullable
  dtypes so "absent" survives as NA rather than collapsing the dtype.
  `tests/test_inference.py::test_serve_time_row_local_features_match_the_training_frame`
  compares the served row against one built by the panel's own
  `row_local_features`, so a future divergence fails rather than serves.

- **Lesson:** dtype inference is a statistic, and a one-row frame is a sample of
  size one. Anywhere training sees a batch and serving sees a single record,
  the schema must be declared rather than inferred — and the test that catches
  it is the one that builds the same record down both paths and compares, not
  the one that checks the response was a float between 0 and 1.

---

## 2026-09-05 — A synthetic panel that could not exhibit the bug it existed to show

- **Problem:** caught while reading the fixture rather than from a failure, and
  recorded because a green suite would have hidden it. Had the history been
  replayed on `make_panel`, run 06 (leaky features admitted) and run 07
  (removed) would have scored identically, the deliberate leak would have
  measured 0.0000 PR-AUC, and every test around it would have passed while
  demonstrating nothing.

- **Root cause:** `make_panel` keeps every posting alive in every wave. The
  leaky feature is `n_observations_total`, a count of a posting's rows across
  the panel — and with no attrition that count is the same number for every
  posting, a constant column carrying no information. The leak is a property of
  *postings leaving the board*, and the fixture had no leaving in it. The
  fixture was not wrong for its original purpose; it was wrong for this one, and
  nothing in either the fixture or the test said which purposes it served.

- **Solution:** `make_closing_panel` in `tests/panels.py` gives each posting a
  drawn lifetime, staggered entry, and a right-censored tail.
  `tests/test_experiments.py::test_the_closing_panel_has_attrition_and_censoring`
  asserts the attrition exists, and `test_the_leaky_columns_separate_the_classes`
  asserts the planted leak actually carries the outcome — so a fixture that
  stopped being able to show the bug would fail rather than quietly pass.

- **Lesson:** a test that demonstrates a phenomenon must first assert that its
  fixture can *exhibit* that phenomenon. Otherwise the demonstration passes on a
  fixture where the effect is structurally impossible, and a green suite records
  the shape of the lesson without its substance. This is the second fixture bug
  in two days — see the 2026-09-04 entry on a fixture that encoded its own
  label. Both times the fixture was data, and both times it was wrong in a way
  no assertion was looking at.

---

## 2026-09-05 — A default split that silently produced no error bars

- **Problem:** replaying the experiment history on a twenty-wave panel crashed
  with `KeyError: 'pr_auc'` inside `summarise_folds`, because `cross_validate`
  had returned an empty DataFrame.

- **Root cause:** the default cuts, copied from the earlier components, are
  `Cuts(waves[0], waves[len(waves) // 2])`. That pair was written for a
  five-wave panel, where it is the only choice leaving anything on both sides.
  It puts `train_end` at the *first* wave, so the training block is one wave
  wide however deep the panel is — and `wave_forward_folds` needs at least two
  waves in the training window to cut a single fold. The crash was luck: had
  `summarise_folds` tolerated an empty frame, every run would have reported no
  cross-validation and the tables would have carried a mean with no spread,
  which reads as "cross-validation ran and found nothing".

- **Solution:** `default_cuts` in `src/models/experiments.py` places the cuts at
  60% and 80% of the available waves, and `test_default_cuts_leave_a_training_
  window_deep_enough_for_folds` asserts at least three folds survive. `execute`
  now records `folds_scored=0` rather than crashing when no fold exists.

- **Lesson:** a default chosen under one constraint becomes a bug when the
  constraint lifts. The five-wave panel forced `waves[0]`; nothing re-examined
  it when the panel got deeper, because it was never written down as a
  concession. A constant that encodes a temporary limitation needs the
  limitation in a comment beside it, or it outlives its reason silently.

---

## 2026-09-05 — A cross-validation fold with only one class to learn from

- **Problem:** `IndexError: index 1 is out of bounds for axis 1 with size 1`
  from `model.predict_proba(features)[:, 1]` in
  `src/models/evaluate.py:cross_validate`, on the very first run of the history.

- **Root cause:** on an expanding window the earliest folds are the shallowest,
  and on a rare-event panel a training slice covering the first wave or two can
  contain no positives at all. A scikit-learn classifier fitted on a single
  class has one entry in `classes_`, so `predict_proba` returns one column
  instead of two and the `[:, 1]` index is out of range. The traceback describes
  an array shape and says nothing about the cause. This had never fired before
  because the real panel is too shallow to split, so cross-validation had never
  actually executed — the code path was written in Component 10 and first *run*
  in Component 11.

- **Solution:** `cross_validate` checks `train_block["y"].nunique() < 2` before
  fitting and emits a `NaN` row, which `summarise_folds` already excludes from
  the mean. Checked before the fit rather than caught after it, because "this
  fold had nothing to learn from" is a fact about the fold, not an exception.

- **Lesson:** code that has never run is not tested code, however carefully it
  was reviewed. Two components' worth of machinery sat behind a `SplitTooShallow`
  guard and looked finished. When a blocker stops a path from executing, the path
  needs a fixture that makes it execute anyway — otherwise the first real run
  becomes the first test, at the worst possible moment.

---

## 2026-09-05 — `pip install` into the wrong interpreter

- **Problem:** `mlflow` was installed successfully and then `import mlflow`
  raised `ModuleNotFoundError` inside the activated virtualenv. The install had
  also uninstalled `protobuf 5.29.3` and replaced it with `6.33.6` somewhere.

- **Root cause:** `.venv` is a `uv` virtualenv and ships no `pip` binary. With
  the venv activated, `which pip` still resolved to `/opt/anaconda3/bin/pip`,
  because activation prepends `.venv/bin` to `PATH` and the shell simply found
  the next `pip` along it. So the packages went into the Anaconda base
  environment — a different interpreter entirely — and its `protobuf` was the
  one upgraded. `python -m pip` then failed honestly with "No module named pip",
  which is what a `uv` venv should say.

- **Solution:** installed with `uv pip install`, which targets the project venv.
  `mlflow` is declared in `pyproject.toml` under a `tracking` extra.

- **Lesson:** an activated virtualenv guarantees which `python` you get, not
  which `pip`. Install with `python -m pip` or the environment's own tool
  (`uv pip`), never the bare `pip` on `PATH` — and read the *destination* in the
  install output rather than trusting the exit code, because installing into the
  wrong environment succeeds.

---

## 2026-09-04 — A synthetic fixture that encoded its own label

- **Problem:** the test asserting that no rung of the baseline ladder can beat
  the base rate on an unlearnable label failed, with the random forest scoring
  PR-AUC 1.000 and logistic regression 0.667. The obvious reading was a leak in
  the preprocessing pipeline — the target reaching the model through some
  column — which would have invalidated the previous component.

- **Root cause:** the fixture, not the pipeline. The synthetic panel's label was
  `index < 2`, and three of its categoricals were `index % 3` (location),
  `index % 2` (departments, offices) and `index % 7` (posted_dow). By the
  Chinese remainder theorem those three residues identify `index` uniquely for
  any panel narrower than 42 postings, so the features jointly determined the
  label exactly. The forest was learning real structure that a human had put
  there by accident while trying to write varied-looking test data.

- **Solution:** `make_panel(random_labels=True)` in `tests/panels.py` draws
  positives per wave from a seeded generator independent of every feature, and
  `test_no_rung_beats_the_base_rate_when_the_label_is_noise` uses it. The
  default fixture keeps the learnable label — it is useful for the tests that
  need a model to fit — with the coincidence documented on the builder.

- **Lesson:** a synthetic fixture is data, and data can leak. Modular arithmetic
  over a row index looks like harmless variety and is a hash of the index, so
  any label that is also a function of the index becomes recoverable. When a
  test asserts that a model *cannot* learn something, the label must be drawn
  from a generator that never touches the feature values — and if such a test
  fails, suspect the fixture before the system, because a fixture that encodes
  its own label is far more common than a pipeline that leaks.

---

## 2026-09-04 — A missing category that was not missing enough

- **Problem:** the per-column missing-value policy for categoricals was silently
  skipped. `salary_currency_clean` and `offices` have nulls, the pipeline fills
  them with the explicit level `__missing__`, and the fitted encoder produced no
  `__missing__` level at all. Nothing raised, the matrix had no NaNs, and the
  model fitted and scored normally — the only visible symptom was a test
  asserting that the level exists.

- **Root cause:** `select_columns` normalised absent values in object columns to
  `None`. `SimpleImputer` looks for `np.nan`, and `None` in an object array is
  not `np.nan`, so the imputer passed those rows through untouched and
  `OneHotEncoder` learned `None` as an ordinary category. Two consequences, both
  quiet: the documented fill policy did not run, and `None` became a level that
  exists only where training data happened to have nulls — so the same column
  arriving null at serve time in a board that never had nulls would be an
  *unknown* category, handled by the unknown branch rather than the missing one.
  Both spellings mean "absent" to a reader and only one means it to sklearn.

- **Solution:** `select_columns` in `src/features/preprocessing.py` normalises
  missing object values to `np.nan`, and `tests/test_preprocessing.py`
  asserts a `__missing__` level appears in the encoded feature names. Confirmed
  on the real panel: `offices___missing__` and
  `salary_currency_clean___missing__` are now present.

- **Lesson:** "missing" is not one value. pandas has `NA`, numpy has `nan`,
  Python has `None`, and a library that documents "missing values" means exactly
  one of them — sklearn means `np.nan` unless told otherwise. When crossing from
  pandas into sklearn, normalise the sentinel explicitly at the boundary and
  assert the downstream step actually fired, because a skipped imputation looks
  identical to a successful one in every shape, dtype and NaN count.

---

## 2026-09-04 — A one-day horizon that no removal could reach

- **Problem:** at H=1 the label rule produced **0 positives in 1,116 rows** at
  run index 0, and a positive rate that moved 0.00% / 1.77% / 1.06% / 0.00%
  across the four labelable runs. Nothing raised. The frame had the right
  columns, the right row count, a plausible 32 positives overall, and the
  censoring rule was working correctly — it simply discarded nineteen removals
  the scraper had observed.

- **Root cause:** `t_gone <= t + H` was compared as instants, on a panel that
  looks once a day at a time that drifts. Complete runs are 34.3601h, 13.6407h,
  23.9993h and 24.0075h apart, so from run 0 the very next observation lands
  34.4h out and cannot fall inside a 24h horizon; a removal detected as fast as
  the panel physically permits is therefore not "removed within a day". The
  deeper fault is that removals here are interval-censored — a posting vanished
  somewhere in `(t_last_seen, t_first_absent]` and we never learn when — so a
  horizon expressed in continuous time is not identifiable from this data at
  all. The 2.6 seconds by which the run 2 → run 3 gap undershot 24h is what
  kept that run's twelve positives, and 27.0 seconds of drift at run 3 is what
  discarded thirteen rows.

- **Solution:** the comparison is made on calendar dates
  (`compute_labels(basis="calendar")`, now the default in
  `src/features/assemble.py`), which on this schedule is exactly *"absent at the
  next complete run"*. `basis="instant"` is kept so the two remain comparable
  rather than a claim in prose, and `tests/test_assemble.py` asserts that
  seconds of clock drift cannot change a label. Recorded as design decision §10.

- **Lesson:** a horizon may not be measured in units finer than the grid you
  observe on. If the data arrives once a day, "within one day" is a statement
  about *the next observation*, not about 86,400 seconds, and writing it as
  arithmetic quietly converts scheduler punctuality into label noise. Worse, it
  is noise that lines up with `run_index`, `t_dow` and `age_days` — features —
  so the model can learn it as signal. Whenever a threshold is compared against
  a timestamp the collection process produced, ask what the spacing of that
  process is and whether the threshold can even be resolved.

---

## 2026-09-04 — A monthly salary stored as 1,200,000,000

- **Problem:** `jobs.salary_min` holds `1200000000` for a posting whose
  `salary_raw` reads `£100 month`. The correct annual figure is 1,200 — the
  stored value is six orders of magnitude too large. Nothing raised: it is a
  plausible integer in an integer column, and it survives every non-null check.
  Found by re-parsing all 1,666 `salary_raw` strings and comparing: 1,210 of
  1,211 rows agreed, and this was the one that did not.

- **Root cause:** the upstream parser annualises by multiplying, and for this
  string it applied a multiplier that is not 12. The same pass also fails to
  read 455 other stated salaries — single values with no range (`€1,000`,
  `15,50 €`) and the uppercase `BIS` variant of the German range word — leaving
  `salary_min` NULL where a number was plainly present. Both faults share a
  cause: the parser recognises a narrow set of shapes and has no signal for
  "this looked like money and I could not read it", so a miss and a
  misinterpretation are equally silent.

- **Solution:** cleaning re-derives salary from `salary_raw` in
  `src/data/clean.py` rather than trusting `salary_min`. `parse_salary` returns
  the figures, the currency and the periodicity separately, and `annualise`
  applies the multiplier as an explicit, tested step. `salary_stated` and
  `salary_parsed` are kept as distinct columns so "no pay mentioned" and "pay
  mentioned but unreadable" cannot collapse into each other. The upstream bug
  is not yet fixed in the scraper.

- **Lesson:** a derived column is a claim, and a claim needs somewhere to record
  that it failed. When a parser can only return a value or NULL, a
  misinterpretation is indistinguishable from an absence and both look like
  clean data. Prefer parsers that return the reading *and* its status, and when
  a second implementation exists, diff them across the whole corpus rather than
  spot-checking — one row in 1,211 is not something eyes find.

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
