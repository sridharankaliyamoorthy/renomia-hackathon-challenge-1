"""
Regex/rule-based extraction helpers for insurance offer fields.
Zero Gemini calls — used for the baseline /solve implementation.
"""

import re
from typing import Optional

from preprocess import build_preferred_offer_text, get_offer_text_debug

# ---------------------------------------------------------------------------
# Money regex
# Captures: (number_part) (optional multiplier) (currency)
# Handles: "5 000 000 Kč", "5.000.000 Kč", "5,000,000 Kč",
#          "5 mil. Kč", "250 tis. Kč", "5,5 mil. Kč", "CZK"
# ---------------------------------------------------------------------------
_MONEY_RE = re.compile(
    r"(\d[\d\s.,]*\d|\d)"           # number (single digit OR multi-digit)
    r"\s*"
    r"(mil\.?|milion[ůu]?|tis\.?|tisíc[ůu]?)?"   # optional multiplier
    r"\s*"
    r"(?:Kč|Kc|kc|CZK)",
    re.IGNORECASE | re.UNICODE,
)

# ---------------------------------------------------------------------------
# Keyword lists per field  (ordered: most-specific first)
# ---------------------------------------------------------------------------
_LIMIT_KEYWORDS = [
    "limit pojistného plnění",
    "limit plnění",
    "pojistný limit",
    "základní limit",
    "pojistné plnění do výše",
    "plnění do výše",
]

_DEDUCTIBLE_KEYWORDS = [
    "základní spoluúčast",
    "spoluúčast pojistníka",
    "spoluúčast:",
    "spoluúčast ",
    "spoluúčastnění",
]

_PREMIUM_KEYWORDS = [
    # Most-specific phrases first (ČPP-style and equivalents)
    "pojistné za první pojistné období",
    "pojistné za první období",
    "celkové roční pojistné",
    "roční pojistné celkem",
    "roční pojistné včetně",
    "pojistné za pojistné období",
    "pojistné za rok",
    "pojistné celkem",
    "celkové pojistné",
    "výsledné pojistné",
    "roční pojistné",
    "běžné pojistné",
    "výše pojistného",
    "celkem pojistné",
    "pojistné:",
]

# Terms that, when appearing between a premium keyword and a money value,
# indicate the amount belongs to a different concept — skip the match.
_PREMIUM_FP_RE = re.compile(
    r"splátk|záloha|dlužné\s+pojistné|penále|limit\b|spoluúčast",
    re.IGNORECASE | re.UNICODE,
)

# Characters to search ahead after a keyword
_WINDOW = 300


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Collapse runs of spaces/tabs and repeated newlines."""
    if not text:
        return ""
    text = re.sub(r"[ \t\u00a0]+", " ", text)   # non-breaking space → regular
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_money_czk(raw_num: str, multiplier: Optional[str] = None) -> Optional[int]:
    """
    Convert a raw captured number string + optional multiplier token into CZK int.

    Examples
    --------
    normalize_money_czk("5 000 000")        → 5_000_000
    normalize_money_czk("5.000.000")        → 5_000_000
    normalize_money_czk("5,000,000")        → 5_000_000
    normalize_money_czk("5", "mil.")        → 5_000_000
    normalize_money_czk("5,5", "mil.")      → 5_500_000
    normalize_money_czk("250", "tis.")      → 250_000
    """
    s = raw_num.strip().replace("\u00a0", "").replace(" ", "")

    if re.fullmatch(r"\d+([.,]\d{3})+", s):
        # "5.000.000" or "5,000,000" — all separators are thousands separators
        s = re.sub(r"[.,]", "", s)
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        # "5,5" — decimal comma
        s = s.replace(",", ".")
    elif re.fullmatch(r"\d+\.\d{1,2}", s):
        # "5.5" — decimal dot (already valid float string)
        pass
    else:
        # Remove any remaining separators (shouldn't be needed but guards edge cases)
        s = re.sub(r"[.,]", "", s)

    try:
        value = float(s)
    except ValueError:
        return None

    mult = 1
    if multiplier:
        m = multiplier.lower().replace(".", "").strip()
        if m.startswith("mil"):
            mult = 1_000_000
        elif m.startswith("tis") or m.startswith("tisí"):
            mult = 1_000

    return int(round(value * mult))


# ---------------------------------------------------------------------------
# Keyword-window extraction internals
# ---------------------------------------------------------------------------

def _first_money_after_keywords(text: str, keywords: list) -> Optional[int]:
    """
    For each keyword (in order), scan all occurrences in *text*.
    Return the first CZK amount found within _WINDOW chars after any occurrence.
    """
    text_lower = text.lower()
    for kw in keywords:
        kw_lower = kw.lower()
        search_from = 0
        while True:
            pos = text_lower.find(kw_lower, search_from)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]
            m = _MONEY_RE.search(window)
            if m:
                return normalize_money_czk(m.group(1), m.group(2))
            search_from = pos + 1
    return None


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def extract_basic_limit_czk(text: str) -> Optional[int]:
    return _first_money_after_keywords(text, _LIMIT_KEYWORDS)


def extract_basic_deductible_czk(text: str) -> Optional[int]:
    return _first_money_after_keywords(text, _DEDUCTIBLE_KEYWORDS)


def extract_premium_czk(text: str) -> Optional[int]:
    """
    Premium extraction with false-positive guard.

    Iterates _PREMIUM_KEYWORDS (most-specific first).  For each occurrence,
    checks the text between the keyword end and the money match for terms
    that would indicate a different concept (instalment, deposit, penalty, etc.).
    Returns the first clean CZK amount found.
    """
    text_lower = text.lower()
    for kw in _PREMIUM_KEYWORDS:
        kw_lower = kw.lower()
        search_from = 0
        while True:
            pos = text_lower.find(kw_lower, search_from)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]
            m = _MONEY_RE.search(window)
            if m:
                between = window[len(kw): m.start()]
                if not _PREMIUM_FP_RE.search(between):
                    return normalize_money_czk(m.group(1), m.group(2))
            search_from = pos + 1
    return None


# ---------------------------------------------------------------------------
# Per-offer baseline parser
# ---------------------------------------------------------------------------

def parse_offer_baseline(offer: dict) -> dict:
    """
    Extract the 3 core numeric fields from all documents of one offer.
    All other schema fields are returned as None (to be filled later).
    Zero Gemini calls.

    Uses preprocess.build_preferred_offer_text to build a clean, prioritised
    input (rfp/ujednani first, VPP last) instead of a blind raw concatenation.
    """
    docs = offer.get("documents") or []

    combined = build_preferred_offer_text(docs)

    # Internal diagnostics — not exposed in the API response
    _debug = get_offer_text_debug(docs)  # noqa: F841

    return {
        # Identity
        "id": offer.get("id"),
        "insurer": offer.get("insurer"),
        "label": offer.get("label"),
        # Extracted numeric fields
        "basic_limit_czk": extract_basic_limit_czk(combined),
        "basic_deductible_czk": extract_basic_deductible_czk(combined),
        "premium_czk": extract_premium_czk(combined),
        # All other required schema fields — null for this baseline
        "covered_activities": None,
        "territorial_scope": None,
        "limit_multiplier_per_year": None,
        "aggregate_limit_czk": None,
        "limit_persons_in_custody_czk": None,
        "limit_pure_financial_loss_czk": None,
        "limit_taken_items_czk": None,
        "limit_cross_liability_czk": None,
        "limit_recourse_czk": None,
        "limit_non_pecuniary_damage_czk": None,
        "deductible_recourse_czk": None,
        "deductible_non_pecuniary_czk": None,
        "deductible_brought_items_czk": None,
        "deductible_financial_loss_czk": None,
    }
