"""A synthetic job-day panel carrying every column the real one does.

Shared by the preprocessing and baseline tests so that "what a panel looks like"
has one definition. Nothing here reads `data/`; the real panel is not committed
and a test that depends on today's scrape is a test that changes its mind
overnight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = pd.Timedelta(days=1)
WAVE0 = pd.Timestamp("2026-08-31T03:45:00Z")


def make_panel(
    n_waves: int = 9,
    per_wave: int = 40,
    seed: int = 0,
    random_labels: bool = False,
    positives_per_wave: int = 2,
) -> pd.DataFrame:
    """Build the panel.

    Numeric values drift upward wave by wave, so that a statistic learned on an
    early block is measurably different from one learned on the whole frame —
    which is the only way to tell the two apart in a test.

    **The default label is learnable, and more so than it looks.** It is
    `index < 2`, and the categoricals are `index % 3` (location), `index % 2`
    (departments, offices) and `index % 7` (posted_dow) — which by the Chinese
    remainder theorem identify `index` uniquely for any `per_wave` below 42. A
    forest reaches PR-AUC 1.0 on it, legitimately. Pass `random_labels=True` for
    a panel whose label is independent of every feature, which is what a test
    needs when it wants to assert that a model *cannot* do better than the base
    rate.
    """
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(seed + 1_000)
    positive_ids = {
        wave: set(label_rng.choice(per_wave, size=positives_per_wave, replace=False))
        for wave in range(n_waves)
    }
    rows = []
    for wave in range(n_waves):
        for index in range(per_wave):
            rows.append(
                {
                    "source": "greenhouse:acme",
                    "source_id": f"p{index}",
                    "title": f"Engineer {index}",
                    "company": "Acme",
                    "location": ["Remote", "London", "Berlin"][index % 3],
                    "remote": None,
                    "salary_min": None,
                    "salary_max": None,
                    "currency": None,
                    "salary_raw": None,
                    "posted_at": "2026-01-01",
                    "url": f"https://acme.example/{index}",
                    "run_id": wave,
                    "t": WAVE0 + wave * DAY,
                    "run_index": wave,
                    "y": int(index in positive_ids[wave])
                    if random_labels
                    else int(index < 2),  # positives present in every wave either way
                    "label_observable": True,
                    "first_published": WAVE0 - 30 * DAY,
                    "updated_at": WAVE0 - DAY,
                    "departments": ["Eng", "Sales"][index % 2],
                    "n_departments": 1.0,
                    "offices": ["HQ", "Remote"][index % 2],
                    "n_offices": 1.0,
                    "requisition_id": f"R{index}",
                    "n_metadata": 3.0,
                    "content_chars": 1000.0 + 100 * wave + rng.integers(0, 10),
                    "board_size_at_t": float(per_wave),
                    "n_same_title_on_board": 1.0,
                    "n_same_req_on_board": 1.0,
                    "board_growth": 0.0,
                    "n_complete_runs_observed": wave + 1,
                    "age_days": 30.0 + wave + rng.integers(0, 3),
                    "days_since_update": 1.0 + wave,
                    "t_dow": (wave + 6) % 7,
                    "posted_dow": float(index % 7),
                    "posted_month": 1.0,
                    "salary_stated": True,
                    "salary_parsed": True,
                    "salary_min_clean": 50_000.0 + 5_000 * wave,
                    "salary_max_clean": 70_000.0 + 5_000 * wave,
                    "salary_period": "unstated",
                    "salary_currency_clean": "USD",
                    "horizon_days": 1,
                    "horizon_basis": "calendar",
                }
            )
    frame = pd.DataFrame(rows)
    frame["y"] = frame["y"].astype("Int8")
    return frame


def make_closing_panel(
    n_waves: int = 20,
    n_postings: int = 220,
    seed: int = 7,
    mean_lifetime: float = 6.0,
    censor_tail_waves: int = 2,
) -> pd.DataFrame:
    """A panel in which postings actually leave the board.

    `make_panel` keeps every posting alive in every wave, which is fine for the
    preprocessing tests and useless for anything about the *shape* of the panel:
    with no attrition, a count of a posting's rows is the same number for every
    posting, so the leak in `src/features/leaky.py` cannot be demonstrated on
    it — run 06 and run 07 would come out identical and the demonstration would
    prove nothing.

    So this builder models the three things the real panel does:

    **Attrition.** Each posting draws a lifetime and stops appearing after it.
    That is what makes an observation count informative about the outcome, which
    is the entire mechanism of the leak.

    **Staggered entry.** Postings arrive across the window rather than all at
    wave 0, so `age_days` varies for reasons unrelated to the label and the
    panel has the left-truncated look of the real one.

    **Right-censoring.** A posting still on the board near the end of the window
    has an outcome nobody has observed yet. Those rows carry
    `label_observable=False` and are dropped before splitting, exactly as
    `problem_definition.md` §4 requires — labelling them negative is the bug
    that teaches a model that recent means open.

    **The label is drawn independently of every feature**, as in
    `make_panel(random_labels=True)`: lifetimes come from their own generator
    and nothing about a posting's title, location or salary influences them. So
    the honest runs *should* score at the base rate, and any run that beats it
    is reading something it should not — which is what makes this fixture able
    to tell a leak from a finding.
    """
    rng = np.random.default_rng(seed)
    lifetime_rng = np.random.default_rng(seed + 5_000)

    rows = []
    for index in range(n_postings):
        entry = int(lifetime_rng.integers(0, max(1, n_waves - 2)))
        lifetime = int(1 + lifetime_rng.geometric(1.0 / mean_lifetime))
        last = min(entry + lifetime - 1, n_waves - 1)
        # A posting that runs to the end of the window did not close; one that
        # stops earlier did, and its final row is the positive.
        closed = last < n_waves - 1

        for wave in range(entry, last + 1):
            observable = wave <= n_waves - 1 - censor_tail_waves
            rows.append(
                {
                    "source": "greenhouse:acme",
                    "source_id": f"p{index}",
                    "title": f"Engineer {index}",
                    "company": "Acme",
                    "location": ["Remote", "London", "Berlin"][index % 3],
                    "remote": None,
                    "salary_min": None,
                    "salary_max": None,
                    "currency": None,
                    "salary_raw": None,
                    "posted_at": "2026-01-01",
                    "url": f"https://acme.example/{index}",
                    "run_id": wave,
                    "t": WAVE0 + wave * DAY,
                    "run_index": wave,
                    "y": int(closed and wave == last and observable),
                    "label_observable": observable,
                    "first_published": WAVE0 + entry * DAY - 30 * DAY,
                    "updated_at": WAVE0 + entry * DAY - DAY,
                    "departments": ["Eng", "Sales"][index % 2],
                    "n_departments": 1.0,
                    "offices": ["HQ", "Remote"][index % 2],
                    "n_offices": 1.0,
                    "requisition_id": f"R{index}",
                    "n_metadata": 3.0,
                    "content_chars": 800.0 + rng.integers(0, 400),
                    "board_size_at_t": 100.0,
                    "n_same_title_on_board": 1.0,
                    "n_same_req_on_board": 1.0,
                    "board_growth": 0.0,
                    "n_complete_runs_observed": wave + 1,
                    "age_days": float(wave - entry + 30),
                    "days_since_update": float(wave - entry + 1),
                    "t_dow": (wave + 6) % 7,
                    "posted_dow": float(index % 7),
                    "posted_month": 1.0,
                    "salary_stated": True,
                    "salary_parsed": True,
                    "salary_min_clean": 50_000.0 + 1_000 * (index % 20),
                    "salary_max_clean": 70_000.0 + 1_000 * (index % 20),
                    "salary_period": "unstated",
                    "salary_currency_clean": "USD",
                    "horizon_days": 1,
                    "horizon_basis": "calendar",
                }
            )

    frame = pd.DataFrame(rows).sort_values(["t", "source_id"], kind="stable")
    frame["y"] = frame["y"].astype("Int8")
    return frame.reset_index(drop=True)
