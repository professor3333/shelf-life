# Feature hypotheses and their outcomes

Every feature was a claim about the world before it was a column. This is the
ledger of those claims, dated 2026-09-04, and it distinguishes three outcomes
that are easy to blur together: **rejected before building** (the data cannot
support the claim, so no code was written), **built, untested** (the feature
exists and the experiment that would judge it has not run), and **built and
judged**.

Nothing here is yet in the last category, and the reason is stated at the
bottom. Saying so is the point of the document.

---

## Rejected before building

Four of the roadmap's suggested features did not survive contact with the data.
None of them cost a line of code, which is the value of stating the hypothesis
first.

| Suggested feature | Hypothesis | Why it was not built |
|---|---|---|
| `description_length` | thin postings are speculative and get pulled sooner | already exists as `content_chars`, straight from the archived payload. Building it again would be a duplicate under a new name |
| `has_salary` | salary disclosure signals a serious, well-defined role | already exists as `salary_stated` |
| `posted_day_of_week` | Monday postings differ from Friday postings | already exists as `posted_dow` |
| `title_language` | German and English postings behave differently | **the data cannot test it.** 12 of 1,240 titles contain a non-ASCII character. German-language postings came almost entirely from arbeitnow, which is excluded from labelled rows because its page cap makes absence unreadable. The hypothesis may well be true; this dataset has no evidence either way |
| `is_remote` from the `remote` column | remote roles draw a larger applicant pool | the column is dead — 0 non-null values in 5,714 rows, for the same reason. **Rebuilt from text instead**, see `location_is_remote` below |

## Built, and awaiting evidence

Seven features, each computed inside the pipeline by `src/features/derive.py`.

| Feature | Hypothesis | Distribution on the 1,240 postings | Status |
|---|---|---|---|
| `title_seniority` | how senior a role is relates to how long it takes to fill | unspecified 744, lead 216, senior 202, executive 76, **junior 2** | built; see the caveat below |
| `title_is_manager` | management roles have longer, more deliberate hiring processes | 28% of postings | built |
| `title_words` | terse titles are boilerplate on high-volume requisitions and churn faster | median 5 words, range 1–11 | built |
| `title_chars` | the same claim, by another measure | median 37 characters | built |
| `location_is_remote` | remote roles draw a larger pool and close faster | 27% of postings | built |
| `n_locations` | a posting open in several places is a wider net and fills faster | 1,080 single-location, 157 multi (max 8) | built |
| `salary_band` | pay level relates to fill speed, non-linearly | unstated 336, 250k–350k 294, 150k–200k 187, 200k–250k 136, over 350k 136, 100k–150k 120, under 100k 31 | built |

**The seniority caveat, which is a finding rather than a footnote.** The
roadmap's phrasing of this hypothesis is "junior roles fill faster than staff
roles", and that specific contrast **cannot be tested here at all**: two
postings out of 1,240 are junior. The greenhouse boards in the labelled
population are senior-heavy technology employers, so the feature can distinguish
lead from senior from unspecified, and the junior arm is empty. If the finished
model reports anything about junior roles, it will be extrapolation dressed as
evidence.

**`salary_band` carries a known distortion.** Bands are nominal, with no
currency conversion, because an exchange rate is external and time-varying. 69
of 1,240 postings are GBP or EUR and are banded by their face figure.
`salary_currency_clean` remains a separate feature so a model can separate them,
but a reader should know the bands are not strictly comparable across the three
currencies.

## Built, gated, and probably a mistake to use

| Feature | Hypothesis | Verdict |
|---|---|---|
| `company_posting_volume` | high-volume posters churn listings faster | **implemented as a fitted transformer, gated behind board identity, and expected to be dropped** |

The roadmap flags this one as a leakage trap, and it is — a per-company count
computed over the full frame is target leakage through an aggregate.
`CompanyVolumeEncoder` avoids that by learning the lookup on the training fold
alone.

But the more interesting problem is the one the trap warning does not mention.
**On this data the feature is board identity with a numeric face.** Six of the
seven sources have exactly one company each, so the count reproduces `source`
almost exactly; the seventh has 25 companies of one to three postings apiece and
so carries almost no variance either. The hypothesis "high-volume posters churn
faster" is, on this population, indistinguishable from "Anthropic behaves
differently from Airtable" — which is `design.md` §4's open question, not a new
feature. It is gated with `source` and `company` accordingly.

---

## Why nothing has been judged yet

The experiment that decides each of these is the leave-one-out ablation in
`src/models/train.py`: refit without the feature, and see what the validation
PR-AUC does. It has not run, because **no honest three-way split exists on this
snapshot** — `feasible_cuts` reports 0 of 6 candidate cuts usable, and the
reason is panel depth rather than a cut that could be moved. Details in
`reports/baseline_results.md`.

So the ledger stands at five rejected, seven built and unjudged, one built and
gated. `python -m src.models.train` is the command that fills in the last
column, and it will start doing so on its own as the scraper adds waves.

What can be said today, without the ablation:

- the constant-predictor reference is PR-AUC **0.0140** on 6,874 labelled rows
  (2026-09-06 snapshot);
- every engineered feature is computed inside the pipeline, so each one exists
  identically for a single posting at serve time — verified by a test that
  scores one raw row;
- the deliberate overfit works as intended on synthetic data whose label is
  independent of every feature: the train/validation gap opens from 0.09 to 0.82
  as depth goes from 1 to 12, and `min_child_weight` is the knob that closes it
  (0.82 → 0.41) while `reg_lambda` at 50 barely moves it. That ordering is worth
  re-checking on the real panel, because on real signal it may well invert.
