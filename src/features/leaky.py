"""Features that leak, added on purpose.

Nothing in this module should ever reach a model you believe. It exists so that
one specific leak — *an observation count accumulates **because** the posting
stayed open* — can be committed, measured, and then removed, with the size of
the lie written down rather than imagined. `docs/leakage_audit.md` rules this
class of column out in prose; this module is what makes the ruling checkable,
by building the column the audit forbids and pricing it.

**Why these two columns and not something more obvious.** A leak that announces
itself teaches nothing. `last_seen` as a feature is not a trap, it is a typo:
anyone reading the column list stops at it. The trap is a column that reads like
ordinary metadata and whose name contains no hint of the outcome. Both of these
pass that test — *how many times have we seen this posting* and *how long have
we had it on file* sound like data-quality measures, the sort of thing you add
to let a model discount thinly-observed rows.

**Why they leak.** The panel is one row per (posting, crawl). A posting that
stays on the board accrues a row per wave; a posting that is pulled stops
accruing. So a count over the **whole panel** is a measurement of how long the
posting survived — which is the label, integrated. The giveaway is the word
*whole*: both quantities are perfectly legitimate when the window ends at `t`,
and `board_growth` and `n_same_title_on_board` in the real feature set are
exactly that, windowed. These are the same idea with the window left open at the
right-hand end, and that single difference is the whole bug.

**Why the value is not knowable at the prediction point.** Apply the question
every entry in `docs/leakage_audit.md` is answered with: *at the moment I would
need to make this prediction, would this value exist yet?* On 2026-09-02, looking
at a posting first seen on 2026-08-31, `n_observations_total` counts crawls that
have not happened. At serve time it
cannot be computed at all — a posting arriving at `POST /predict` has been seen
exactly once, so the honest value is 1 for every request, and a model that
learned "1 means closing" would flag every posting a stranger ever sent it.
That last consequence is worth more than the metric: the leak does not merely
inflate a score, it produces a model that is *inverted* in production.

**Why the column is computed before the split, and must be.** The bug this
reproduces is not "someone used a forbidden column". It is "someone added a
column to the assembly step, computed over the frame they had". Gating it inside
the pipeline would make it a different, milder bug — the pipeline would compute
the count within whatever fold it was fitted on, and the leak would partly
cancel. `add_leaky_features` therefore runs over the full panel, exactly as the
real mistake would, and the fix demonstrated in the experiment history is to
delete the column rather than to move it.
"""

from __future__ import annotations

import pandas as pd

#: The leaky columns, by name. Registered in `src/features/preprocessing.py` as
#: a gated set so that they cannot become features by accident, and so that
#: `assert_known_columns` still has a verdict for every column of a panel that
#: has been through `add_leaky_features`.
LEAKY_COLUMNS: tuple[str, ...] = ("n_observations_total", "days_on_board_total")


def _posting_id(frame: pd.DataFrame) -> pd.Series:
    """Identity across sources — `source_id` is unique only within a board.

    Deliberately the same rule as `src/data/split.py:_posting_id`. A leak
    grouped on a *different* notion of identity would be a second bug on top of
    the intended one, and the experiment history is only readable if run 06
    differs from run 07 by exactly one thing.
    """
    return frame["source"].astype(str) + "|" + frame["source_id"].astype(str)


def add_leaky_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the panel with the leaky columns attached.

    Computed over every row of `frame`, with no window and no reference to `t`.
    That is the defect, stated in code: the aggregation reads each posting's
    future, so a row's feature value depends on crawls that occur after the row's
    own prediction point — and, once the frame is split, on rows in the
    validation and test blocks.
    """
    out = frame.copy()
    pid = _posting_id(out)

    # Rows per posting across the entire panel. A posting pulled on day three
    # has three; one that never closes has one per wave.
    out["n_observations_total"] = pid.map(pid.value_counts()).astype("float64")

    # The same information as a duration. Kept alongside the count because the
    # two survive different fixes — dropping duplicate crawls would blunt the
    # count and leave this one untouched — and a leak with two spellings is
    # closer to how the mistake actually arrives.
    span = out.groupby(pid)["t"].transform(lambda t: t.max() - t.min())
    out["days_on_board_total"] = (span / pd.Timedelta(days=1)).astype("float64")

    return out
