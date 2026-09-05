# Learning log

One entry per problem: what happened → what I didn't understand → what I
learned → where I applied it → how I'd catch it next time.

This is my interview preparation. In twelve months it is the most valuable
thing I own.

Newest entry at the top. `DEBUGGING.md` records what broke in *the system*;
this file records what I didn't understand.

Stage 0 was about making software work. This build is about a harder question:
the code runs, but **is the answer right, and how would I know?** Most of the
entries below are versions of that question.

---

## 2026-09-05 — A score that only holds on one computer is not a fact

- **What happened:** I saved a test that said "this exact job posting must score
  this exact probability." It passed on my Mac and failed on the machine that
  runs the tests automatically. My Mac said `0.0435`. The other machine said
  `0.0603`. Same code, same settings, same data.
- **What I didn't understand:** that adding up numbers on a computer is not
  quite like adding up numbers on paper. Change the order and the last few
  digits can change. A boosted model adds up thousands of small numbers, and it
  splits that work across several cores, so the order depends on the machine. On
  a small dataset, two possible splits are often almost exactly tied — so a
  difference in the last digit is enough to pick a different one. After that,
  the two machines have grown genuinely different trees.
- **What I learned:** a "fixed number" test is only fixed if the thing producing
  it is fully repeatable. Setting a random seed does not help here, because the
  seed controls which rows get picked, not the order the machine adds things up.
- **Where I applied it:** the pinned number now comes from the logistic
  regression, which is repeatable everywhere, and there is a second test that
  pins the **table of features** the posting turns into. That is the part I
  actually want to protect — if a refactor changes the features, I want to know.
- **How I'd catch it next time:** before writing down any exact number, ask
  "would this come out identical on a different computer?" If the answer is no,
  pin the input instead of the output.

## 2026-09-05 — Code that has never run is not tested code

- **What happened:** the very first real run of the experiment history crashed
  with `IndexError: index 1 is out of bounds`. The code it crashed in had been
  written two components earlier, reviewed, and looked finished.
- **What I didn't understand:** that the code had never actually executed. The
  real data was too shallow to split, so everything behind that check had been
  sitting there unrun. I had been reading it and calling it done.
- **What I learned:** reading code is not running code. A section that is
  blocked by a "not enough data yet" guard is not half-finished — it is
  completely untested, and the first time it runs will be the worst possible
  moment to find that out.
- **Where I applied it:** the actual bug was that on the earliest slice of time
  there were no closed postings at all, so the model only ever saw one answer
  and returned one column instead of two. `cross_validate` now checks for that
  before fitting and records "this slice had nothing to learn from" instead of
  crashing.
- **How I'd catch it next time:** when something blocks a code path from
  running, build fake data that makes it run anyway. Otherwise the guard is
  hiding the code from me, not protecting it.

## 2026-09-05 — A test can only prove something if its fake data could contain it

- **What happened:** I found this by reading, not from a failure, which is why
  it is worth writing down. I had built a demonstration of a cheating feature —
  admit it, watch the score jump, remove it, watch the score fall back. On the
  fake data I was about to use, the jump would have been exactly zero, and every
  test around it would have passed.
- **What I didn't understand:** the cheating feature counts how many times a
  posting was seen. It only cheats because postings that stay on the board keep
  being seen and postings that close stop. My fake data kept every posting alive
  for every day — so the count was the same number for everyone, and a column
  where everyone is the same tells you nothing. The fake data could not contain
  the effect I was trying to show.
- **What I learned:** a test that demonstrates something must first prove its
  own fake data is *capable* of showing it. Otherwise the test passes and proves
  nothing, which is worse than failing.
- **Where I applied it:** `make_closing_panel` gives every posting a lifetime,
  so some of them leave. Two tests now check the fake data itself — that
  postings really do disappear, and that the cheating column really does
  separate the two answers.
- **How I'd catch it next time:** for any test of the form "X makes a
  difference", add a test that the fixture can produce that difference at all.

## 2026-09-05 — Training sees a crowd; serving sees one person

- **What happened:** I printed the column types of the single row built from an
  incoming request and found that every number the caller left out had turned
  into text. Nothing crashed. The request would have come back with a confident
  answer computed from columns the model had never seen in that shape.
- **What I didn't understand:** pandas guesses what type a column is by looking
  at the values in it. When training, a column is guessed from thousands of
  rows, so a few blanks don't matter. When serving, the table is one row wide —
  so a blank is the *only* value, and the guess is made from nothing.
- **What I learned:** guessing types is a kind of averaging, and you cannot
  average one thing. Everywhere training sees a batch and serving sees a single
  record, the types must be **declared**, not guessed. And every real request
  leaves something out, so the awkward case is the normal case.
- **Where I applied it:** `build_row` now states the type of every column up
  front. A test builds the same posting down both paths — the training path and
  the serving path — and compares them, so if they ever drift apart the test
  fails instead of the user getting a wrong number.
- **How I'd catch it next time:** whenever the same data travels two different
  routes, write the test that sends one example down both and compares. Not the
  test that checks the answer is a number between 0 and 1.

## 2026-09-05 — A default that was right once goes wrong quietly

- **What happened:** replaying the experiments on a deeper set of data crashed.
  The cause was a default setting I had copied from an earlier stage.
- **What I didn't understand:** that setting was written when I only had five
  days of data, where it was the only choice that left anything on both sides of
  the split. It puts the training cut at the very first day — so no matter how
  much data I collect, the training window stays one day wide. It made sense
  once, and nothing re-examined it when the reason went away.
- **What I learned:** the crash was luck. If the code had quietly tolerated an
  empty result, every experiment would have reported an average with no spread,
  and a table like that reads as "we checked and found nothing" rather than "we
  never checked."
- **Where I applied it:** the default now sits at 60% and 80% of however much
  data exists, and a test asserts at least three slices survive.
- **How I'd catch it next time:** when a constant exists because of a temporary
  limit, write the limit down next to it. Otherwise it outlives its reason and
  nobody notices.

## 2026-09-05 — Perfectly honest and completely useless can be the same model

- **What happened:** in the comparison table, the dumbest possible model — the
  one that ignores every feature and always says "the average" — came out with a
  *perfect* calibration score.
- **What I didn't understand:** calibration only asks "when you say 10%, does it
  happen 10% of the time?" A model that says 10% for absolutely everything is
  right about that, always. It has told you nothing about which posting to look
  at, and it scores perfectly on that measure.
- **What I learned:** there are two separate questions and you need both. *Can
  it rank?* — put the ones about to close near the top. *Are its numbers
  honest?* — when it says 10%, is it really 10%? A model can be perfect at one
  and useless at the other, in both directions.
- **Where I applied it:** the reports never show calibration on its own. It sits
  next to PR-AUC, which measures how well the model finds the rare cases, and
  the dumb model is left in the table on purpose as the illustration.
- **How I'd catch it next time:** whenever a metric looks great, ask what the
  laziest possible model scores on it. If the lazy model also scores great, the
  metric is not measuring what I want.

## 2026-09-05 — Two scores are not a comparison

- **What happened:** on the fake data, the random forest scored 0.2532 and the
  logistic regression 0.1937. The forest looks clearly better. The verdict
  refuses to say so.
- **What I didn't understand:** those numbers are averages over seven slices of
  time, and the slices are not equally hard. If one slice happens to be easy,
  *both* models score high on it. Averaging each model separately and then
  subtracting leaves all of that shared luck in the answer.
- **What I learned:** compare the two models **on the same slice**, then average
  the differences. That cancels out whatever made a slice easy or hard and
  leaves only the part that is about the models. The forest led by 0.0595 and
  won 4 slices out of 7 — inside the normal wobble, so the honest report is
  "these two are tied."
- **Where I applied it:** `paired_fold_difference` in `src/models/evaluate.py`,
  and `select` refuses to declare a winner when the gap does not survive.
- **How I'd catch it next time:** a single number with no spread beside it is
  not evidence. Ask "how much does this bounce around?" before asking "which is
  bigger?"

## 2026-09-05 — Waiting can be the correct output

- **What happened:** the split, the comparison and the test-set report all
  produce the same thing right now: a clear statement that they cannot run yet,
  and a count of how much longer. Five days of usable data against a minimum of
  seven.
- **What I didn't understand:** I expected "not enough data" to be a failure to
  work around. It is a finding. Splitting the data three ways — past, middle,
  recent — with gaps between them costs days, and I do not have the days yet.
  Forcing it would have produced a test result computed on a slice with no
  closed postings, which is not a hard test. It is a number that means nothing.
- **What I learned:** the honest output of a step that cannot run is a
  measurement of *why* and *how much longer*, not a number produced anyway.
  A meaningless number is worse than no number, because a number gets quoted.
- **Where I applied it:** `minimum_waves` and `feasible_cuts` in
  `src/data/split.py` answer "can I split today?" and "how much longer?"
  separately. The reports print the blocker instead of a result. The scraper
  keeps running, so the wait costs nothing but patience.
- **How I'd catch it next time:** when a result is disappointing or impossible,
  ask whether the honest version is a measurement. Usually it is.

---

## 2026-09-04 — "It's gone" only means something if you looked everywhere

- **What happened:** my whole prediction target is "this posting disappeared
  from the board." For 74% of my data — 3,531 of 4,771 postings — that was not
  what I was measuring. Nothing crashed. No test failed. The label was simply
  about something else.
- **What I didn't understand:** the scraper stops after 8 pages of that site.
  The site lists newest first. So as new postings arrive, older ones slide past
  page 8 and are never seen again — while still sitting on the board, perfectly
  open. "Not in what I fetched" is a fact about where my scraper stopped
  reading. "Removed from the board" is a fact about the world. I had been
  reading the first as the second.
- **What I learned:** two things. First, absence is only evidence if the
  observation was **complete**. Second — and this is the part I nearly missed —
  the obvious check does not work here. A posting pushed off page 8 never comes
  back either, so "does it ever reappear?" gives the same answer under both
  stories. What actually gave it away was a number that was silly: it implied
  the board turned over 40–55% every day, and no real job board does that.
- **Where I applied it:** those sources are excluded from labelling entirely.
  Only the Greenhouse boards, which fetch the whole board every time, carry a
  label. That took me from 4,771 postings to about 1,150 usable rows — and it is
  why this project has been waiting for more data ever since. The cost of
  catching it was the ability to train at all, and it was still worth it.
- **How I'd catch it next time:** before building a label out of "the record
  stopped appearing," prove the collector looked at the whole population that
  day. And when a rate would be absurd in the real world, treat it as a bug
  report about my measurement, not a discovery about the world.

## 2026-09-04 — A feature doesn't cheat because of what it measures, but when it stops looking

- **What happened:** I let two columns into the model on purpose to see what
  cheating looks like. The score jumped from 0.1915 to 0.8385. Taking them out
  again brought it back to 0.1915 exactly.
- **What I didn't understand:** my first rule was "counting a posting's history
  is suspicious." That rule is wrong, and it is wrong in the expensive
  direction — it throws away good features and still lets bad ones through. The
  actual problem is *when the counting stops*. `n_observations_total` counts a
  posting's appearances across the whole dataset, including the future. A
  posting that stays open keeps being counted; one that closes stops. So the
  count is just the answer, written a different way.
- **What I learned:** the same shape of feature is fine if the window closes at
  the moment of prediction. `board_growth` and `n_same_title_on_board` count
  history too, and they are legitimate, because they stop counting at *now*. The
  rule is not "no counting." It is **"nothing whose window is still open."** That
  version I can apply to a column I have never seen before.
- **Where I applied it:** the leak lives in `src/features/leaky.py`, committed
  on purpose so the size of the lie is measured rather than imagined, and every
  legitimate feature has its window closed at `t`.
- **How I'd catch it next time:** the honest version of the real damage is not
  the score. A posting arriving at the live service has been seen exactly
  **once**, so that count is 1 for every request anyone ever makes. The model
  learned low count means closing. It would have flagged everything.

## 2026-09-04 — Fake data is still data, and it can cheat too

- **What happened:** a test that said "no model should be able to beat random
  guessing on this unlearnable answer" failed. The random forest scored a
  perfect 1.000. The obvious explanation was a leak in my pipeline, which would
  have invalidated the previous week of work.
- **What I didn't understand:** the pipeline was fine. My *fake data* was
  cheating. I had made the answer "row number under 2" and then, trying to make
  the columns look varied, filled three of them with row number divided by 3, by
  2 and by 7 — leaving the remainder. Those three remainders together identify
  the row number exactly. So the features secretly contained the answer, and the
  forest found it. It was learning real structure that I had put there by
  accident.
- **What I learned:** test data is data, and it can leak like any other data.
  Arithmetic on a row number looks like harmless variety and is actually a
  fingerprint of the row.
- **Where I applied it:** the "unlearnable" fixture now draws its answers from a
  random generator that never touches any feature value.
- **How I'd catch it next time:** when a test asserting a model *cannot* learn
  something fails, suspect the fixture before the system. A fixture that
  contains its own answer is far more common than a pipeline that leaks.

## 2026-09-04 — "Missing" is not one thing

- **What happened:** I had written a rule that blank categories should be filled
  with an explicit label, `__missing__`, so that "we don't know" becomes a real
  category rather than a hole. The rule silently did not run. No error, no
  blanks left over, a perfectly normal score.
- **What I didn't understand:** Python's `None` and numpy's `nan` both mean
  "nothing here" to a human and are different values to a library. The tool I
  was using only looks for `nan`. My earlier step had turned blanks into `None`.
  So the filling step passed them straight through and the encoder learned
  `None` as if it were an ordinary category like "Berlin".
- **What I learned:** when data crosses from one library into another,
  "missing" has to be spelled the way the receiving library spells it — and I
  should assert that the step actually fired, because a skipped fill looks
  identical to a successful one in every count I could check.
- **Where I applied it:** `select_columns` normalises blanks properly, and a
  test asserts the `__missing__` category really appears in the finished feature
  list. It does now, on the real data.
- **How I'd catch it next time:** the serving consequence is the one that
  matters, and it is what makes this more than tidiness. A column arriving blank
  from a board that never had blanks in training would take the "never seen
  this" path instead of the "we don't know" path — two different behaviours for
  the same situation.

## 2026-09-04 — You can't measure time more finely than you look

- **What happened:** with a one-day horizon, the label produced **zero** closed
  postings out of 1,116 rows on the first day, and a rate that jumped 0.00% /
  1.77% / 1.06% / 0.00% across four days. Nothing raised an error. The rule was
  working exactly as written.
- **What I didn't understand:** I look once a day, but not at the same second
  each day. The gaps were 34.4h, 13.6h, 24.0h and 24.0h. So "did it disappear
  within 24 hours?" compared against the next look, which was 34.4 hours away,
  can only ever answer no. I was measuring the punctuality of my own schedule
  and calling it a property of job postings.
- **What I learned:** if the data arrives once a day, "within one day" means
  *by the next observation*, not "within 86,400 seconds." Writing it as seconds
  turns clock drift into noise in the answers. And it is the worst kind of
  noise, because it lines up with day-of-week and age — which are features — so
  the model can learn my scheduler and think it has learned the world.
- **Where I applied it:** the comparison is done on calendar dates. The old
  version is kept so the two can be compared rather than argued about, and a
  test asserts that seconds of drift cannot change a label.
- **How I'd catch it next time:** whenever a threshold is compared against a
  timestamp my own collection process produced, ask how far apart those
  timestamps are, and whether the threshold is even reachable.

## 2026-09-04 — A number can be badly wrong and look completely normal

- **What happened:** a posting whose pay reads `£100 month` was stored as
  `1200000000`. The right answer is 1,200. It is out by a factor of a million.
- **What I didn't understand:** nothing checks this. It is a plausible whole
  number in a whole-number column. It passes every "is this blank?" test. I only
  found it by re-reading all 1,666 original pay strings myself and comparing:
  1,210 of 1,211 agreed, and this one did not.
- **What I learned:** the same parser also failed to read 455 other salaries
  that were plainly there, and left them blank. Those two faults have one cause:
  the parser can return a value or nothing, and has no way to say *"that looked
  like money and I couldn't read it."* So a misreading and an absence look
  identical, and both look like clean data.
- **Where I applied it:** cleaning re-derives pay from the original text rather
  than trusting the stored number, and returns the figure, the currency and the
  period separately. "No pay mentioned" and "pay mentioned but unreadable" are
  now two different columns, because they mean different things.
- **How I'd catch it next time:** prefer a parser that returns the reading *and*
  how it went. And when a second version exists, compare them across the whole
  dataset rather than spot-checking — one row in 1,211 is not something eyes
  find.

## 2026-09-04 — Being right 98.7% of the time can mean I learned nothing

- **What happened:** closed postings are rare. On the labelled data the rate is
  1.32% — 75 closures against 5,618 that stayed. A model that says "it'll stay
  open" about everything, always, with no thinking whatsoever, is right 98.7%
  of the time.
- **What I didn't understand:** accuracy counts every row equally, and almost
  every row is the boring answer. The rare answer is the entire reason the model
  exists, and accuracy barely notices whether I get it right.
- **What I learned:** the trap has a second, subtler version — ROC-AUC. It looks
  like a proper measure and it flatters here, because one of the things it
  divides by is the count of postings that stayed open, and there are 5,618 of
  those. A model can raise a huge number of false alarms and barely move that
  number. Precision divides the same false alarms by how many postings I
  flagged, where they cannot hide.
- **Where I applied it:** accuracy is not reported at all. PR-AUC is the
  headline, with calibration beside it. ROC-AUC is reported for comparison with
  other people's work, and is not allowed to decide anything.
- **How I'd catch it next time:** ask what the do-nothing model scores. If it
  scores well, the metric is wrong for the problem — not the model.

## 2026-09-04 — Which mistake is worse is a decision, not a default

- **What happened:** I had been about to use 0.5 as the cut-off between "likely
  to close" and "likely to stay", because that is what everything defaults to.
- **What I didn't understand:** 0.5 is not a decision anybody made. It is just
  the middle. The right cut-off depends on which mistake hurts more, and that is
  a question about the person using it, not about the data.
- **What I learned:** for someone deciding whether to apply for a job today: a
  false "closing soon" costs them a rushed application — a few hours. A false
  "plenty of time" costs them a job they never applied to, which they cannot get
  back. The second mistake is much worse. So the model should lean towards
  warning too often rather than too rarely.
- **Where I applied it:** the cut-off is set from an **alert budget** — how many
  postings a person will realistically read in a day — rather than a probability
  threshold picked out of the air. It ships inside the saved model file, not
  beside it, so the number and the cut-off it is compared against cannot drift
  apart.
- **How I'd catch it next time:** before choosing a cut-off, name the person,
  name both mistakes, and say which one they would rather make. If I cannot, I
  do not understand the problem well enough to set one.
