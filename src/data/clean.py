"""Row-local cleaning of raw posting fields.

    "90.000 € bis 130.000 € / Jahr"  ->  90000, 130000, EUR, yearly
    "$320,000 — $405,000"            ->  320000, 405000, USD, unstated
    "Competitive"                    ->  no numbers, has_salary = False

Every rule here is **deterministic and row-local**: it decides using only the
value in front of it, never by looking at other rows. That is the line between
this module and the preprocessing pipeline. A step that needs to know what other
rows contain — a median to impute with, a category vocabulary, a canonical
company name learned from the corpus — learns a statistic, and a statistic
fitted outside the training fold is leakage. Such steps do not belong here.

Two consequences of that rule, both of which cost us a tempting "fix":

* **Company names are normalised, not canonicalised.** Whitespace and Unicode
  form are row-local; deciding that ``"Wolt - English"`` is the same employer as
  ``"Wolt"`` is not — it requires knowing what else is in the corpus. It also
  does not pay: of the 19 companies containing ``" - "``, **none** has a bare
  twin anywhere in the data, so stripping the suffix merges nothing while
  risking real names like ``"Deutsche Gesellschaft für Nachhaltiges Bauen -
  DGNB e.V."``.
* **Currencies are not converted.** Turning GBP into USD needs an exchange rate,
  which is external, time-varying, and not a property of the row. Currency stays
  categorical.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

#: Periodicity multipliers to annualise a figure. Stated as constants because
#: they are an assumption, not a fact: 2,080 hours is 40h x 52w with no holiday.
HOURS_PER_YEAR = 2080
MONTHS_PER_YEAR = 12

#: The seniority vocabulary the scraper already emits. Anything outside it is
#: returned as None rather than guessed at.
SENIORITY_VOCABULARY = frozenset(
    {"intern", "junior", "senior", "staff", "lead", "principal", "head", "director"}
)

_CURRENCY_BY_SYMBOL = {"$": "USD", "€": "EUR", "£": "GBP"}
_CURRENCY_BY_TOKEN = {"USD": "USD", "EUR": "EUR", "GBP": "GBP"}

#: Range separators seen in the data: em dash, en dash, hyphen, and the English
#: and German words. ``bis`` appears in both cases upstream ("BIS"), which is
#: why matching is case-insensitive.
_RANGE_SPLIT = re.compile(r"\s*(?:—|–|-|\bto\b|\bbis\b|\buntil\b)\s*", re.IGNORECASE)

#: Order matters: the first match wins, so the narrower unit is tested first.
#: Each alternative accepts the bare word too ("150 EUR Month"), not only the
#: "per"/"/" forms, because the data carries both.
_PERIOD_PATTERNS = (
    ("hourly", re.compile(r"\bhour(?:s|ly)?\b|\bhr\b|\bstunde\w*", re.I)),
    ("monthly", re.compile(r"\bmonth(?:s|ly)?\b|\bmonat\w*", re.I)),
    (
        "yearly",
        re.compile(r"\byear(?:s|ly)?\b|\byr\b|\bannum\b|\bannual\w*|\bjahr\w*", re.I),
    ),
)

_NUMBER = re.compile(r"\d[\d.,]*")


@dataclass(frozen=True)
class SalaryParse:
    """Outcome of reading one ``salary_raw`` string.

    ``ok`` is False both when the string was absent and when it contained no
    number ("Competitive"). The two are distinguished by ``present``, because
    "the posting stated pay at all" is a different signal from "the posting
    stated a number we could read".
    """

    minimum: float | None = None
    maximum: float | None = None
    currency: str | None = None
    period: str = "unstated"
    present: bool = False
    ok: bool = False
    is_range: bool = False


def normalize_whitespace(value: str | None) -> str | None:
    """NFKC-normalise, collapse internal whitespace, strip. Row-local and safe:
    it changes the encoding of a name, never which employer it refers to."""
    # pd.NA, None and NaN all mean "absent". A `string`-dtype column yields
    # pd.NA, which is neither None nor a float — missing it here turns the
    # literal text "<NA>" into a value and marks every empty row as stated.
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_seniority(value: str | None) -> str | None:
    """Lower-case and validate against the known vocabulary.

    Returns None for anything unrecognised rather than passing it through: an
    unexpected token here means the upstream parser changed, and silently
    admitting it would create a category that appears mid-panel.
    """
    text = normalize_whitespace(value)
    if text is None:
        return None
    text = text.casefold()
    return text if text in SENIORITY_VOCABULARY else None


def _parse_number(token: str) -> float | None:
    """Read one numeric token, resolving separator ambiguity.

    ``60.000`` is sixty thousand in German and sixty in English, and both
    conventions appear in this data. The rule: when both separators are present
    the *last* one is the decimal point; when only one is present it is a
    thousands separator if it occurs once and is followed by exactly three
    digits, and a decimal point otherwise.
    """
    token = token.strip()
    if not token:
        return None
    has_dot, has_comma = "." in token, "," in token

    if has_dot and has_comma:
        decimal_sep = "." if token.rfind(".") > token.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        token = token.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_dot or has_comma:
        sep = "." if has_dot else ","
        head, _, tail = token.rpartition(sep)
        if token.count(sep) >= 1 and len(tail) == 3 and head:
            token = token.replace(sep, "")
        else:
            token = token.replace(sep, ".")

    try:
        return float(token)
    except ValueError:
        return None


def _detect_currency(text: str) -> str | None:
    for symbol, code in _CURRENCY_BY_SYMBOL.items():
        if symbol in text:
            return code
    upper = text.upper()
    for token, code in _CURRENCY_BY_TOKEN.items():
        if re.search(rf"\b{token}\b", upper):
            return code
    return None


def _detect_period(text: str) -> str:
    for name, pattern in _PERIOD_PATTERNS:
        if pattern.search(text):
            return name
    return "unstated"


def parse_salary(raw: str | None) -> SalaryParse:
    """Read ``salary_raw`` into numbers, a currency and a periodicity."""
    text = normalize_whitespace(raw)
    if text is None:
        return SalaryParse()

    currency = _detect_currency(text)
    period = _detect_period(text)

    # Strip the periodicity clause before splitting, so "/ Jahr" is not mistaken
    # for a range separator and "per hour" does not contribute a number.
    body = re.sub(r"(?:per|/)\s*\w+", " ", text, flags=re.IGNORECASE)

    numbers: list[float] = []
    for part in _RANGE_SPLIT.split(body):
        for match in _NUMBER.finditer(part):
            value = _parse_number(match.group())
            if value is not None:
                numbers.append(value)

    if not numbers:
        # Present but unreadable — "Competitive", "DOE", or a currency alone.
        return SalaryParse(currency=currency, period=period, present=True, ok=False)

    low, high = min(numbers), max(numbers)
    return SalaryParse(
        minimum=low,
        maximum=high,
        currency=currency,
        period=period,
        present=True,
        ok=True,
        is_range=len(numbers) > 1 and low != high,
    )


def annualise(value: float | None, period: str) -> float | None:
    """Scale a figure to a yearly rate using the stated constants.

    Kept separate from :func:`parse_salary` because it is an *assumption*, not a
    reading: an hourly rate only becomes a salary if you decide how many hours a
    year is. A row whose periodicity is unstated is returned unchanged, which
    silently assumes it was annual — true for most postings here and false for
    some, so ``salary_period`` is retained as a column to model that doubt.
    """
    if value is None:
        return None
    if period == "hourly":
        return value * HOURS_PER_YEAR
    if period == "monthly":
        return value * MONTHS_PER_YEAR
    return value


def clean_jobs(frame: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned columns to a postings frame, leaving the originals intact.

    Nothing is dropped and nothing is imputed. The raw columns stay so that any
    disagreement between them and the parsed values remains auditable.
    """
    out = frame.copy()

    for column in ("title", "company", "location"):
        if column in out.columns:
            out[f"{column}_clean"] = out[column].map(normalize_whitespace).astype("string")

    if "seniority" in out.columns:
        out["seniority_clean"] = out["seniority"].map(normalize_seniority).astype("string")

    if "salary_raw" in out.columns:
        parsed = out["salary_raw"].map(parse_salary)
        out["salary_stated"] = parsed.map(lambda p: p.present).astype("boolean")
        out["salary_parsed"] = parsed.map(lambda p: p.ok).astype("boolean")
        out["salary_is_range"] = parsed.map(lambda p: p.is_range).astype("boolean")
        out["salary_period"] = parsed.map(lambda p: p.period).astype("string")
        out["salary_currency_clean"] = parsed.map(lambda p: p.currency).astype("string")
        out["salary_min_clean"] = parsed.map(lambda p: p.minimum).astype("Float64")
        out["salary_max_clean"] = parsed.map(lambda p: p.maximum).astype("Float64")
        out["salary_min_annual"] = parsed.map(lambda p: annualise(p.minimum, p.period)).astype(
            "Float64"
        )
        out["salary_max_annual"] = parsed.map(lambda p: annualise(p.maximum, p.period)).astype(
            "Float64"
        )

    return out
