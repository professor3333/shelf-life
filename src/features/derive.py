"""Row-local features derived from the as-of-`t` text.

Each function here is a **hypothesis about the world** before it is a line of
code; `reports/feature_hypotheses.md` states each one and records what happened
to it. Nothing is added because it might help.

**Everything in this module is stateless.** Every value is a function of that
row's own text and nothing else — no medians, no quantiles, no per-company
counts, no vocabulary learned from a corpus. That is a deliberate constraint
rather than a coincidence, and it has two consequences. A stateless derivation
cannot leak across a split however it is called, so it can safely run before the
train/validation boundary. And it produces the same value for a posting seen
alone at serve time as it did in training, so it cannot cause training/serving
skew.

The one proposed feature that is *not* row-local — posting volume per company —
therefore does not live here. It is a fitted transformer in
`src/features/preprocessing.py`, because computing it over the whole frame is
target leakage through an aggregate, which is the subtlest kind and does not
announce itself.

**Salary bands use fixed thresholds, not quantiles.** A quantile is a statistic
of the data it was computed on. Binning by the frame's own quartiles would make
the encoding depend on which rows are present, so the bands are constants,
chosen once, and written down. No currency conversion is applied — an exchange
rate is external and time-varying — so a GBP or EUR posting is banded by its
nominal figure, which is 69 of 1,240 postings in the 2026-09-04 snapshot.
`salary_currency_clean` remains a separate feature so a model can tell them
apart.
"""

from __future__ import annotations

import pandas as pd

#: Seniority, checked in this order because titles combine words: "Senior
#: Engineering Manager" is a manager, "Staff Backend Engineer" is a lead, and
#: whichever pattern is tested first wins. Derived from the as-of-`t` title
#: rather than read from `jobs.seniority`, which is current state and would
#: leak later edits backwards (`docs/design.md` §9).
SENIORITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("executive", r"\b(?:chief|c[teofi]o|vp|vice president|head of|director)\b"),
    ("lead", r"\b(?:staff|principal|lead|distinguished|fellow)\b"),
    ("senior", r"\b(?:senior|sr\.?)\b"),
    ("junior", r"\b(?:junior|jr\.?|intern|internship|graduate|entry[- ]level|apprentice)\b"),
)

MANAGER_PATTERN = r"\b(?:manager|management|managing)\b"
REMOTE_PATTERN = r"\b(?:remote|anywhere|distributed|work from home|wfh)\b"

#: Nominal currency units. Chosen once against the observed distribution
#: (median 245,000 USD on this population) and fixed, so that the encoding does
#: not move when the frame does.
SALARY_BAND_EDGES: tuple[float, ...] = (100_000.0, 150_000.0, 200_000.0, 250_000.0, 350_000.0)
SALARY_BAND_LABELS: tuple[str, ...] = (
    "under_100k",
    "100k_150k",
    "150k_200k",
    "200k_250k",
    "250k_350k",
    "over_350k",
)
SALARY_BAND_UNSTATED = "unstated"

#: Columns this module adds. Named here so the pipeline and the audit agree.
DERIVED_COLUMNS: tuple[str, ...] = (
    "title_seniority",
    "title_is_manager",
    "title_words",
    "title_chars",
    "location_is_remote",
    "n_locations",
    "salary_band",
)


def title_seniority(titles: pd.Series) -> pd.Series:
    """Seniority level from the title, or `unspecified`.

    *Hypothesis: how senior a role is relates to how long it takes to fill, so a
    posting's level predicts how long it stays on the board.*

    Note what this cannot test on the present data: junior roles are 0.5% of
    postings once arbeitnow is excluded, so the "junior fills faster than staff"
    contrast the roadmap suggests has almost no support here. The level is
    recorded anyway; the junior arm is a hypothesis this dataset cannot answer.
    """
    text = titles.fillna("").astype(str)
    out = pd.Series("unspecified", index=titles.index, dtype="object")
    assigned = pd.Series(False, index=titles.index)
    for label, pattern in SENIORITY_PATTERNS:
        matches = text.str.contains(pattern, case=False, regex=True) & ~assigned
        out[matches] = label
        assigned |= matches
    return out


def title_is_manager(titles: pd.Series) -> pd.Series:
    """*Hypothesis: management roles have longer, more deliberate hiring
    processes, so they sit on the board longer.* Kept separate from seniority
    because the two are orthogonal — an Engineering Manager and a Staff Engineer
    are the same level and different jobs."""
    return titles.fillna("").astype(str).str.contains(MANAGER_PATTERN, case=False, regex=True)


def location_is_remote(locations: pd.Series) -> pd.Series:
    """*Hypothesis: remote roles draw a larger applicant pool and so close
    faster.*

    Read from the location text because the structured `remote` column is dead —
    zero non-null values in 5,714 rows, since arbeitnow was the only source that
    populated it and arbeitnow is excluded from labelled rows. The text carries
    it for 27% of postings, so this recovers a feature the exclusion destroyed.
    """
    return locations.fillna("").astype(str).str.contains(REMOTE_PATTERN, case=False, regex=True)


def n_locations(locations: pd.Series) -> pd.Series:
    """*Hypothesis: a posting open in several places is a wider net and fills
    faster.* Greenhouse joins multiple offices with `;`."""
    text = locations.fillna("").astype(str)
    return text.str.count(";").add(1).where(text.str.len() > 0, 0).astype("float64")


def title_words(titles: pd.Series) -> pd.Series:
    """*Hypothesis: terse titles are boilerplate on high-volume requisitions and
    churn faster than specific ones.* The cheap counterpart of
    `content_chars`, which says the same thing about the description."""
    return titles.fillna("").astype(str).str.split().str.len().astype("float64")


def salary_band(amounts: pd.Series) -> pd.Series:
    """*Hypothesis: pay level relates to fill speed non-linearly — the effect of
    going from 100k to 150k is not the effect of going from 300k to 350k.*

    A band rather than the raw number because a linear model cannot express that
    on its own, and because it gives "no pay stated" a level of its own rather
    than an imputed number.
    """
    values = pd.to_numeric(amounts, errors="coerce")
    banded = pd.cut(
        values,
        bins=[-float("inf"), *SALARY_BAND_EDGES, float("inf")],
        labels=list(SALARY_BAND_LABELS),
        right=False,
    )
    return banded.astype("object").where(values.notna(), SALARY_BAND_UNSTATED)


def derive_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every derived column. Stateless; safe to run before any split.

    Returns a copy. The inputs it reads — `title`, `location`, `salary_min_clean`
    — are as-of-`t` by construction, coming from the sealed per-run snapshot and
    the archived payload, so every output is a fact that existed at the moment
    the prediction would be made.
    """
    out = frame.copy()
    titles = out["title"] if "title" in out else pd.Series("", index=out.index)
    locations = out["location"] if "location" in out else pd.Series("", index=out.index)

    out["title_seniority"] = title_seniority(titles)
    out["title_is_manager"] = title_is_manager(titles).astype("float64")
    out["title_words"] = title_words(titles)
    out["title_chars"] = titles.fillna("").astype(str).str.len().astype("float64")
    out["location_is_remote"] = location_is_remote(locations).astype("float64")
    out["n_locations"] = n_locations(locations)
    out["salary_band"] = salary_band(
        out["salary_min_clean"] if "salary_min_clean" in out else pd.Series(dtype="float64")
    )
    return out
