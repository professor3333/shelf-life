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
