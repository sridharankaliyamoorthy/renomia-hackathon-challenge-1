"""
Rule-based extraction and ranking for the "auta" (motor vehicle) segment.

Handles Czech auto insurance quote layouts including:
- Variant-style quotes (Komfort / Plus / Extra / Max tiers)
- Povinné ručení (MTPL liability) + Havarijní pojištění (comprehensive/collision)
- Spoluúčast (deductible) in "3 % / 3 000 Kč" format
- Limit povinného ručení in "200 / 200 mil. Kč" format

Document priority (highest signal first):
  0 — quote / calculation / nabídka / RfP
  1 — smlouva (policy contract)
  2 — general terms / VPP / podmínky
  3 — other (GDPR, broker info, etc.)

Zero Gemini calls.
"""

import logging
import re
from typing import Optional

from extractors import normalize_money_czk, _MONEY_RE as _CZK_MONEY_RE
from preprocess import clean_ocr_text

logger = logging.getLogger(__name__)

# Characters to scan ahead of a keyword when searching for a value
_WINDOW = 300


# ---------------------------------------------------------------------------
# Document prioritisation for auto offers
# ---------------------------------------------------------------------------

def _auto_doc_priority(doc: dict) -> int:
    """
    Assign a priority integer to an auto document (lower = higher signal).

      0 — quote / calculation / nabídka / RfP
      1 — smlouva (policy contract)
      2 — general terms / VPP / podmínky
      3 — other (GDPR, broker info, etc.)
    """
    fn = (doc.get("filename") or "").lower()
    snippet = (doc.get("ocr_text") or "")[:1_000].lower()

    if re.search(r"rfp|nabidka|nab[ií]dka|quote|kalkulac|výpočet|vypocet", fn):
        return 0
    if re.search(r"smlouva|pojistn[aá][\s_-]*smlouva", fn):
        return 1
    if re.search(r"vpp|podm[ií]nky|podminky|general|terms|condition", fn):
        return 2

    # Content-based fallback
    if re.search(r"(nab[ií]dka|kalkulac|výpočet|cena celkem|roční pojistné)", snippet):
        return 0
    if re.search(r"pojistn[aá]\s+smlouva", snippet):
        return 1
    if re.search(r"(vpp|v[sš]eobecn[eé]\s+pojistn[eé])", snippet):
        return 2

    return 3


def _build_auto_offer_text(documents: list) -> str:
    """
    Build combined, cleaned offer text for regex extraction.

    Priority order:
      quote/calculation  → up to 12 000 chars
      contract           → up to  5 000 chars
      general terms      → up to  4 000 chars
      other              → up to  2 000 chars

    Hard cap: 20 000 chars total.
    """
    if not documents:
        return ""

    sorted_docs = sorted(documents, key=_auto_doc_priority)
    per_priority_limits = {0: 12_000, 1: 5_000, 2: 4_000, 3: 2_000}
    max_total = 20_000

    parts = []
    total = 0
    for doc in sorted_docs:
        raw = doc.get("ocr_text") or ""
        if not raw.strip():
            continue
        cleaned = clean_ocr_text(raw)
        priority = _auto_doc_priority(doc)
        limit = per_priority_limits.get(priority, 2_000)
        remaining = max_total - total
        if remaining <= 0:
            break
        chunk = cleaned[: min(limit, remaining)]
        if chunk.strip():
            parts.append(chunk)
            total += len(chunk)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CZK amount helpers
# ---------------------------------------------------------------------------

def _first_czk_after_label(text: str, labels: list) -> Optional[int]:
    """Return the first CZK amount found within _WINDOW chars after any label."""
    text_lower = text.lower()
    for label in labels:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]
            m = _CZK_MONEY_RE.search(window)
            if m:
                return normalize_money_czk(m.group(1), m.group(2))
            start = pos + 1
    return None


# ---------------------------------------------------------------------------
# Field label groups
# ---------------------------------------------------------------------------

_TOTAL_PREMIUM_LABELS = [
    "Cena celkem",
    "Celkem / rok",
    "Celkem/rok",
    "Celkem za rok",
    "Roční platba",
    "Roční pojistné celkem",
    "Celkové roční pojistné",
    "Roční pojistné",
    "Pojistné celkem",
    "Celkem pojistné",
    "Pojistné za rok",
    "Celkem",
]

_LIABILITY_LIMIT_LABELS = [
    "Limit povinného ručení",
    "Pojistný limit odpovědnosti",
    "Limit odpovědnosti",
    "Povinné ručení - limit",
    "PO - limit",
    "Limit plnění",
    "Pojistné plnění do výše",
]

_DEDUCTIBLE_LABELS = [
    "Spoluúčast havarijního pojištění",
    "Spoluúčast HP",
    "Spoluúčast - havárie",
    "Havarijní spoluúčast",
    "Spoluúčast:",
    "Spoluúčast ",
    "Spoluúčast",
]

_VEHICLE_VALUE_LABELS = [
    "Pojistná částka",
    "Pojistná hodnota vozidla",
    "Cena vozidla",
    "Hodnota vozidla",
    "Tržní hodnota",
]

_VEHICLE_MODEL_LABELS = [
    "Vozidlo",
    "Model",
    "Typ vozidla",
    "Značka a typ",
    "Registrační značka",
    "Spz",
    "SPZ",
]

_MILEAGE_LABELS = [
    "Roční nájezd",
    "Předpokládaný roční nájezd",
    "Roční kilometrický limit",
    "Kilometrický limit",
    "Km/rok",
    "km ročně",
]

_COVERAGE_DETECT_LABELS = [
    "Havarijní pojištění",
    "Povinné ručení",
    "Asistenční služba",
    "Úrazové pojištění",
    "Střet se zvěří",
    "Živel",
    "Odcizení",
    "Sklo",
    "Zavazadla",
    "Náhradní vozidlo",
]


# ---------------------------------------------------------------------------
# Liability limit extraction (supports "200 / 200 mil. Kč" format)
# ---------------------------------------------------------------------------

# Matches: "200 / 200 mil. Kč", "100/200 mil. Kč"
_LIABILITY_SPLIT_RE = re.compile(
    r"(\d[\d\s.,]*)\s*/\s*(\d[\d\s.,]*)\s*(mil\.?|milion[ůu]?|tis\.?|tisíc[ůu]?)?\s*(?:Kč|Kc|kc|CZK)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_liability_limit(text: str) -> tuple:
    """
    Extract liability limit text and CZK value.

    Handles:
      "200 / 200 mil. Kč"  → text="200 / 200 mil. Kč", czk=200_000_000
      "100/200 mil. Kč"    → text="100/200 mil. Kč", czk=200_000_000 (higher)
      "50 mil. Kč"         → text="50 mil. Kč", czk=50_000_000

    Returns (limit_text, limit_czk_if_possible).  Either may be None.
    """
    text_lower = text.lower()
    for label in _LIABILITY_LIMIT_LABELS:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]

            # Try split "A / B mul Kč" first
            m = _LIABILITY_SPLIT_RE.search(window)
            if m:
                raw_a = m.group(1).strip().replace(" ", "")
                raw_b = m.group(2).strip().replace(" ", "")
                mult_str = m.group(3)
                limit_text = m.group(0).strip()
                val_a = normalize_money_czk(raw_a, mult_str)
                val_b = normalize_money_czk(raw_b, mult_str)
                if val_a is not None and val_b is not None:
                    limit_czk = max(val_a, val_b)
                elif val_a is not None:
                    limit_czk = val_a
                else:
                    limit_czk = val_b
                return limit_text, limit_czk

            # Try simple CZK amount
            m = _CZK_MONEY_RE.search(window)
            if m:
                val = normalize_money_czk(m.group(1), m.group(2))
                return m.group(0).strip(), val

            start = pos + 1

    return None, None


# ---------------------------------------------------------------------------
# Deductible extraction (supports "3 % / 3 000 Kč" and "10 % min. 5 000 Kč")
# ---------------------------------------------------------------------------

# "3 % / 3 000 Kč" or "5%/5000 Kč"
_DEDUCTIBLE_PCT_SLASH_RE = re.compile(
    r"(\d+(?:[,\.]\d+)?)\s*%\s*[/,]\s*"
    r"(\d[\d\s.,]*\d|\d)\s*(?:Kč|Kc|kc|CZK)",
    re.IGNORECASE | re.UNICODE,
)

# "10 % min. 5 000 Kč"
_DEDUCTIBLE_PCT_MIN_RE = re.compile(
    r"(\d+(?:[,\.]\d+)?)\s*%\s*(?:min\.?|minimáln[eě]?)?\s*"
    r"(\d[\d\s.,]*\d|\d)\s*(?:Kč|Kc|kc|CZK)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_deductible(text: str) -> tuple:
    """
    Extract deductible text, percent, and CZK value.

    Handles:
      "3 % / 3 000 Kč"       → text, pct=3.0, czk=3000
      "5%/5000 Kč"           → text, pct=5.0, czk=5000
      "10 % min. 5 000 Kč"   → text, pct=10.0, czk=5000
      "5 000 Kč"             → text, pct=None, czk=5000

    Returns (deductible_text, deductible_percent, deductible_czk).
    """
    text_lower = text.lower()
    for label in _DEDUCTIBLE_LABELS:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]

            m = _DEDUCTIBLE_PCT_SLASH_RE.search(window)
            if m:
                pct = float(m.group(1).replace(",", "."))
                czk = normalize_money_czk(m.group(2))
                return m.group(0).strip(), pct, czk

            m = _DEDUCTIBLE_PCT_MIN_RE.search(window)
            if m:
                pct = float(m.group(1).replace(",", "."))
                czk = normalize_money_czk(m.group(2))
                return m.group(0).strip(), pct, czk

            m = _CZK_MONEY_RE.search(window)
            if m:
                val = normalize_money_czk(m.group(1), m.group(2))
                return m.group(0).strip(), None, val

            start = pos + 1

    return None, None, None


# ---------------------------------------------------------------------------
# Vehicle model extraction
# ---------------------------------------------------------------------------

def _extract_vehicle_model(text: str) -> Optional[str]:
    """
    Extract vehicle model/type text from the offer.

    Only matches the label when it appears at the start of a line (preceded by
    a newline or at position 0) to avoid matching labels embedded in phrases
    like "Náhradní vozidlo".
    """
    text_lower = text.lower()
    for label in _VEHICLE_MODEL_LABELS:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            # Require label at start of a line (pos 0 or preceded by newline)
            if pos > 0 and text_lower[pos - 1] not in ("\n", "\r"):
                start = pos + 1
                continue
            after = text[pos + len(label): pos + len(label) + 120]
            after = re.sub(r"^[\s:;/|-]+", "", after)
            line = after.split("\n")[0].strip()
            if line and len(line) >= 3:
                return line[:80]
            start = pos + 1
    return None


# ---------------------------------------------------------------------------
# Annual mileage extraction
# ---------------------------------------------------------------------------

_MILEAGE_RE = re.compile(
    r"(\d[\d\s]*\d|\d)\s*(?:km/rok|km\s*ročně|\bkm\b)",
    re.IGNORECASE | re.UNICODE,
)


def _extract_annual_mileage_km(text: str) -> Optional[int]:
    """Extract annual mileage in km."""
    text_lower = text.lower()
    for label in _MILEAGE_LABELS:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]
            m = _MILEAGE_RE.search(window)
            if m:
                raw = m.group(1).strip().replace(" ", "").replace("\u00a0", "")
                try:
                    return int(raw)
                except ValueError:
                    pass
            start = pos + 1
    return None


# ---------------------------------------------------------------------------
# Coverage summary extraction
# ---------------------------------------------------------------------------

def _extract_coverage_summary(text: str) -> Optional[str]:
    """
    Scan for coverage-type labels and return a comma-joined summary string.
    Returns None if nothing is found.
    """
    text_lower = text.lower()
    found = [label for label in _COVERAGE_DETECT_LABELS if label.lower() in text_lower]
    return ", ".join(found[:8]) if found else None


# ---------------------------------------------------------------------------
# Coverage richness heuristic for ranking
# ---------------------------------------------------------------------------

_RICHNESS_KW = [
    "havarijní pojištění", "povinné ručení",
    "asistenční", "úrazové", "střet se zvěří",
    "živel", "odcizení", "sklo",
    "zavazadla", "náhradní vozidlo",
]


def _coverage_richness_score(coverage_summary: Optional[str]) -> int:
    """Return an integer richness score (0–10). Each recognised keyword adds 1 point."""
    if not coverage_summary:
        return 0
    text_lower = coverage_summary.lower()
    return sum(1 for kw in _RICHNESS_KW if kw in text_lower)


# ---------------------------------------------------------------------------
# Min-max normalisation
# ---------------------------------------------------------------------------

def _minmax_norm(values: list, higher_is_better: bool) -> list:
    """
    Min-max normalise a list of Optional[float] to [0, 1].
    Missing values receive 0.0 (penalty score).
    When all values are equal, non-missing entries receive 0.5.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return [0.5] * len(values)
    vmin, vmax = min(valid), max(valid)
    result = []
    for v in values:
        if v is None:
            result.append(0.0)
        elif vmax == vmin:
            result.append(0.5)
        else:
            norm = (v - vmin) / (vmax - vmin)
            result.append(norm if higher_is_better else 1.0 - norm)
    return result


# ---------------------------------------------------------------------------
# Public: parse one offer
# ---------------------------------------------------------------------------

def parse_auto_offer(offer: dict) -> dict:
    """
    Extract auto insurance fields from a single offer's documents.

    Returns a dict with all schema fields; missing values are None (never guessed).
    Zero Gemini calls.
    """
    docs = offer.get("documents") or []
    text = _build_auto_offer_text(docs)

    total_premium = _first_czk_after_label(text, _TOTAL_PREMIUM_LABELS)
    liability_limit_text, liability_limit_czk = _extract_liability_limit(text)
    deductible_text, deductible_percent, deductible_czk = _extract_deductible(text)
    vehicle_value = _first_czk_after_label(text, _VEHICLE_VALUE_LABELS)
    vehicle_model = _extract_vehicle_model(text)
    annual_mileage = _extract_annual_mileage_km(text)
    coverage_summary = _extract_coverage_summary(text)

    logger.debug(
        "[auto] offer=%s premium=%s liability_czk=%s deductible_czk=%s",
        offer.get("id"), total_premium, liability_limit_czk, deductible_czk,
    )

    return {
        "id":                              offer.get("id"),
        "insurer":                         offer.get("insurer"),
        "label":                           offer.get("label"),
        "total_premium_czk":               total_premium,
        "liability_limit_text":            liability_limit_text,
        "liability_limit_czk_if_possible": liability_limit_czk,
        "deductible_text":                 deductible_text,
        "deductible_czk":                  deductible_czk,
        "deductible_percent":              deductible_percent,
        "vehicle_value_czk":               vehicle_value,
        "vehicle_model":                   vehicle_model,
        "annual_mileage_km":               annual_mileage,
        "coverage_summary_text":           coverage_summary,
    }


# ---------------------------------------------------------------------------
# Public: rank offers
# ---------------------------------------------------------------------------

def rank_auto_offers(offers_parsed: list) -> tuple:
    """
    Rank auto insurance offers deterministically.

    Weights
    -------
    0.50  total_premium_czk               lower is better
    0.20  deductible_czk                  lower is better
    0.20  liability_limit_czk_if_possible  higher is better
    0.10  coverage richness score          higher is better (keyword count 0–10)

    Tie-breakers (in order):
      1. lower total_premium_czk
      2. lower deductible_czk
      3. higher liability_limit_czk_if_possible
      4. original offer order (stable)

    Returns
    -------
    (ranking: list[str], best_offer_id: Optional[str])
    """
    if not offers_parsed:
        return [], None

    premiums    = [o.get("total_premium_czk") for o in offers_parsed]
    deductibles = [o.get("deductible_czk") for o in offers_parsed]
    liab_limits = [o.get("liability_limit_czk_if_possible") for o in offers_parsed]
    cov_scores  = [
        float(_coverage_richness_score(o.get("coverage_summary_text")))
        for o in offers_parsed
    ]

    prem_s  = _minmax_norm(premiums,    higher_is_better=False)
    dedt_s  = _minmax_norm(deductibles, higher_is_better=False)
    liab_s  = _minmax_norm(liab_limits, higher_is_better=True)
    cov_s   = _minmax_norm(cov_scores,  higher_is_better=True)

    scores = [
        0.50 * prem_s[i] + 0.20 * dedt_s[i] + 0.20 * liab_s[i] + 0.10 * cov_s[i]
        for i in range(len(offers_parsed))
    ]

    _INF = float("inf")
    indexed = list(enumerate(offers_parsed))
    indexed.sort(
        key=lambda x: (
            -scores[x[0]],
            premiums[x[0]]    if premiums[x[0]]    is not None else _INF,
            deductibles[x[0]] if deductibles[x[0]] is not None else _INF,
            -(liab_limits[x[0]] or 0),
            x[0],
        )
    )

    ranking = [offers_parsed[i]["id"] for i, _ in indexed]
    best_offer_id = ranking[0] if ranking else None
    return ranking, best_offer_id
