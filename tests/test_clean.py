"""Table-driven tests for the cleaning rules.

Every parser gets at least one case that should *fail* to produce a number, and
the separator cases are drawn from real strings in the collected data — the
German thousands dot and the comma decimal are the pair that silently turns
sixty thousand euros into sixty.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import (
    HOURS_PER_YEAR,
    MONTHS_PER_YEAR,
    annualise,
    clean_jobs,
    normalize_seniority,
    normalize_whitespace,
    parse_salary,
)

# raw, min, max, currency, period, is_range
SALARY_CASES = [
    # US ranges, em dash / en dash / word
    ("$320,000 — $405,000", 320_000, 405_000, "USD", "unstated", True),
    ("€60,000 – €90,000", 60_000, 90_000, "EUR", "unstated", True),
    ("$150,000 to $200,000", 150_000, 200_000, "USD", "unstated", True),
    # German: dot is a thousands separator, and "bis" appears in both cases
    ("90.000 € bis 130.000 € / Jahr", 90_000, 130_000, "EUR", "yearly", True),
    ("60.000 € BIS 80.000 €", 60_000, 80_000, "EUR", "unstated", True),
    # German: comma is the decimal separator
    ("15,50 €", 15.5, 15.5, "EUR", "unstated", False),
    ("2.000 €", 2_000, 2_000, "EUR", "unstated", False),
    # Single values, symbol on either side, currency as a token
    ("€1,000", 1_000, 1_000, "EUR", "unstated", False),
    ("120,000 USD/Year", 120_000, 120_000, "USD", "yearly", False),
    ("50000 EUR", 50_000, 50_000, "EUR", "unstated", False),
    # Periodicity, including the bare word with no "per" or slash
    ("$200/month", 200, 200, "USD", "monthly", False),
    ("150€ Month", 150, 150, "EUR", "monthly", False),
    ("£13.00 - £18.20 Per Hour", 13.0, 18.2, "GBP", "hourly", True),
    ("530 EUR bis 3.850 EUR monatlich", 530, 3_850, "EUR", "monthly", True),
]


@pytest.mark.parametrize("raw,low,high,currency,period,is_range", SALARY_CASES)
def test_parse_salary(raw, low, high, currency, period, is_range):
    result = parse_salary(raw)
    assert result.ok and result.present
    assert result.minimum == pytest.approx(low)
    assert result.maximum == pytest.approx(high)
    assert result.currency == currency
    assert result.period == period
    assert result.is_range is is_range


@pytest.mark.parametrize("raw", ["Competitive", "DOE", "Negotiable", "$", "—", ""])
def test_stated_but_unreadable_yields_no_number(raw):
    """'The posting mentioned pay' and 'we could read a figure' are different
    facts, and the model may want both."""
    result = parse_salary(raw)
    assert result.minimum is None and result.maximum is None
    assert result.ok is False
    assert result.present is (raw != "")


@pytest.mark.parametrize("raw", [None, pd.NA, float("nan")])
def test_absent_salary_is_not_stated(raw):
    """A `string`-dtype column yields pd.NA, not None. Treating it as text turns
    the literal '<NA>' into a stated salary for every empty row."""
    result = parse_salary(raw)
    assert result.present is False and result.ok is False


def test_thousands_dot_is_not_a_decimal_point():
    """The case that matters: German '60.000' is sixty thousand, not sixty."""
    assert parse_salary("60.000 €").minimum == 60_000
    assert parse_salary("$60.50").minimum == pytest.approx(60.50)


def test_annualise_uses_the_stated_constants():
    assert annualise(200, "monthly") == 200 * MONTHS_PER_YEAR
    assert annualise(13, "hourly") == 13 * HOURS_PER_YEAR
    assert annualise(90_000, "yearly") == 90_000
    assert annualise(90_000, "unstated") == 90_000, "unstated is assumed annual"
    assert annualise(None, "monthly") is None


@pytest.mark.parametrize(
    "raw,expected",
    [("  Senior  ", "senior"), ("SENIOR", "senior"), ("Intern", "intern"), ("head", "head")],
)
def test_normalize_seniority_known_values(raw, expected):
    assert normalize_seniority(raw) == expected


@pytest.mark.parametrize("raw", ["Sr.", "mid-level", "IC5", "", None])
def test_unknown_seniority_becomes_none(raw):
    """An unrecognised token means the upstream parser changed. Admitting it
    would create a category that first appears part-way through the panel."""
    assert normalize_seniority(raw) is None


def test_normalize_whitespace_is_encoding_only():
    assert normalize_whitespace("  Acme   GmbH \n") == "Acme GmbH"
    assert normalize_whitespace("Ｗolt") == "Wolt"  # NFKC folds fullwidth forms
    assert normalize_whitespace(None) is None
    assert normalize_whitespace("   ") is None


def test_company_suffixes_are_preserved_not_stripped():
    """Canonicalising 'Wolt - English' to 'Wolt' needs to know what else is in
    the corpus, so it is not row-local and does not belong in cleaning. It also
    merges nothing here: no ' - ' company in the data has a bare twin."""
    assert normalize_whitespace("Wolt - English") == "Wolt - English"
    assert (
        normalize_whitespace("Deutsche Gesellschaft für Nachhaltiges Bauen - DGNB e.V.")
        == "Deutsche Gesellschaft für Nachhaltiges Bauen - DGNB e.V."
    )


def test_clean_jobs_adds_columns_without_dropping_originals():
    frame = pd.DataFrame(
        {
            "title": ["  Data   Engineer "],
            "company": ["Wolt - English"],
            "location": ["Berlin"],
            "seniority": ["Senior"],
            "salary_raw": ["90.000 € bis 130.000 € / Jahr"],
        }
    ).astype("string")

    out = clean_jobs(frame)
    assert list(frame.columns) == [c for c in frame.columns if c in out.columns]
    assert out.loc[0, "title_clean"] == "Data Engineer"
    assert out.loc[0, "company_clean"] == "Wolt - English"
    assert out.loc[0, "seniority_clean"] == "senior"
    assert out.loc[0, "salary_min_clean"] == 90_000
    assert out.loc[0, "salary_period"] == "yearly"
    assert out.loc[0, "salary_raw"] == "90.000 € bis 130.000 € / Jahr"


def test_clean_jobs_leaves_missing_salary_unstated():
    frame = pd.DataFrame({"salary_raw": pd.array([None], dtype="string")})
    out = clean_jobs(frame)
    assert out.loc[0, "salary_stated"] is not True
    assert pd.isna(out.loc[0, "salary_min_clean"])
