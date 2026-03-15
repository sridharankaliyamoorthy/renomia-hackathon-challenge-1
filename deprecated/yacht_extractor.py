"""
Rule-based extraction and ranking for the "lode" (yacht) segment.

Primary source document: YACHT POOL.pdf (high-signal quote).
Secondary: Conditions.pdf, Mandatory Information.pdf (reference/fallback only).
Scanned Application / Quotation PDFs are lowest priority and only used if no
other document provides a value.

Zero Gemini calls.
"""

import logging
import re
from typing import Optional

from extractors import normalize_money_czk, _MONEY_RE as _CZK_MONEY_RE
from preprocess import clean_ocr_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EUR money regex
# Handles: "1 500 EUR", "1 500 €", "7,5mil €", "7.5 mil. EUR", "1.500,00 €"
# ---------------------------------------------------------------------------
_EUR_MONEY_RE = re.compile(
    r"(\d[\d\s.,]*\d|\d)"                             # number
    r"\s*"
    r"(mil\.?|milion[ůu]?|tis\.?|tisíc[ůu]?)?"       # optional multiplier
    r"\s*"
    r"(?:€|EUR\b)",
    re.IGNORECASE | re.UNICODE,
)

# Characters to scan ahead of a keyword when searching for a value
_WINDOW = 300


# ---------------------------------------------------------------------------
# EUR money normalisation
# ---------------------------------------------------------------------------

def _normalize_money_eur(raw_num: str, multiplier: Optional[str] = None) -> Optional[float]:
    """
    Convert a raw number string + optional multiplier token to EUR float.

    Examples
    --------
    _normalize_money_eur("1 500")          → 1500.0
    _normalize_money_eur("7,5", "mil.")    → 7_500_000.0
    _normalize_money_eur("1.500")          → 1500.0  (European thousands sep)
    _normalize_money_eur("1,500")          → 1500.0
    """
    s = raw_num.strip().replace("\u00a0", "").replace(" ", "")

    if re.fullmatch(r"\d+([.,]\d{3})+", s):
        # "1.500" or "1,500" — thousands separator only
        s = re.sub(r"[.,]", "", s)
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        # "7,5" — decimal comma
        s = s.replace(",", ".")
    elif re.fullmatch(r"\d+\.\d{1,2}", s):
        # "7.5" — decimal dot (already valid)
        pass
    else:
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

    return value * mult


# ---------------------------------------------------------------------------
# Keyword-window helpers
# ---------------------------------------------------------------------------

def _first_eur_after_label(text: str, labels: list) -> Optional[float]:
    """Return the first EUR amount found within _WINDOW chars after any label."""
    text_lower = text.lower()
    for label in labels:
        label_lower = label.lower()
        start = 0
        while True:
            pos = text_lower.find(label_lower, start)
            if pos == -1:
                break
            window = text[pos: pos + _WINDOW]
            m = _EUR_MONEY_RE.search(window)
            if m:
                return _normalize_money_eur(m.group(1), m.group(2))
            start = pos + 1
    return None


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
# TPL limit extraction (supports embedded "Up to X mil €" phrases)
# ---------------------------------------------------------------------------

def _extract_tpl_limit_eur(text: str) -> Optional[float]:
    """
    Extract TPL / third-party liability limit in EUR.

    Tries label-based extraction first, then falls back to scanning for
    "up to X mil €" style phrases.
    """
    tpl_labels = [
        "TPL limit",
        "TPL - limit",
        "Third party liability limit",
        "Liability limit",
        "Pojistná částka - odpovědnost",
        "Pojistný limit odpovědnosti",
        "Limit odpovědnosti",
        "Pojištění odpovědnosti za škodu - limit",
        "Odpovědnost - limit",
        "Limit of liability",
        "Cover limit",
    ]
    val = _first_eur_after_label(text, tpl_labels)
    if val is not None:
        return val

    # Phrase-based: "up to 7,5mil €" or "up to 7.5 mil. EUR"
    m = re.search(
        r"up\s+to\s+(\d[\d\s.,]*\d|\d)\s*(mil\.?|tis\.?)?\s*(?:€|EUR\b)",
        text,
        re.IGNORECASE,
    )
    if m:
        return _normalize_money_eur(m.group(1), m.group(2))

    return None


# ---------------------------------------------------------------------------
# Cruising area extraction and scoring
# ---------------------------------------------------------------------------

def _extract_cruising_area_text(text: str) -> Optional[str]:
    """Return the raw cruising area value string from the document, or None."""
    labels = [
        "Cruising area",
        "Cruising Area",
        "Oblast plavby",
        "Area of navigation",
        "Navigační oblast",
        "Geographic area",
        "Sailing area",
    ]
    text_lower = text.lower()
    for label in labels:
        label_lower = label.lower()
        pos = text_lower.find(label_lower)
        if pos == -1:
            continue
        after = text[pos + len(label): pos + len(label) + 150]
        after = re.sub(r"^[\s:;/|-]+", "", after)
        line = after.split("\n")[0].strip()
        if line:
            return line
    return None


# Breadth score patterns — checked in descending score order.
# Each entry: (score, [regex_patterns]).
_AREA_BREADTH_PATTERNS = [
    (4, [
        r"worldwide", r"world\s*wide", r"global",
        r"\bocean\b", r"atlantic", r"offshore",
        r"international\s+waters", r"celý\s+svět", r"svet",
        r"world-wide",
    ]),
    (3, [
        r"mediterranean", r"středo.*moř", r"stredomorsk",
        r"\beurope\b", r"european\s+seas", r"\badriat", r"tyrrhenian",
        r"black\s+sea", r"aegean", r"north\s+sea", r"baltic",
        r"nordic\s+sea", r"inland.*sea", r"european.*coast",
        r"celá\s+evrop", r"whole\s+europe",
    ]),
    (2, [
        r"european\s+river", r"european\s+lake", r"alpine\s+lake",
        r"\břek[ay]\b", r"\bjezer", r"inland\s+water",
        r"vnitrozemsk", r"fluss", r"\blago\b", r"rivers?\s+and\s+lakes?",
    ]),
    (1, [
        r"česk[oá]", r"\bczech\b", r"\bbohemia\b", r"\bmorava\b",
        r"inland\s+only", r"local\s+only", r"pouze\s+česk",
        r"\bČR\b",
    ]),
]


def score_cruising_area(area_text: Optional[str]) -> int:
    """
    Return a cruising area breadth score (1–4):
      1 = Czech / local inland only
      2 = European rivers and lakes
      3 = Mediterranean / broader European waters
      4 = Worldwide / international

    Defaults to 2 if text is present but unrecognised (conservative but not
    pessimistic). Defaults to 1 if area_text is None.
    """
    if not area_text:
        return 1
    for score, patterns in _AREA_BREADTH_PATTERNS:
        for pat in patterns:
            if re.search(pat, area_text, re.IGNORECASE):
                return score
    return 2


# ---------------------------------------------------------------------------
# Document prioritisation for yacht offers
# ---------------------------------------------------------------------------

def _yacht_doc_priority(doc: dict) -> int:
    """
    Assign a priority integer to a yacht document (lower = higher signal).

      0 — YACHT POOL, quotation, nabídka (primary quote document)
      1 — application, application form
      2 — conditions, general terms, VPP
      3 — mandatory information, other
    """
    fn = (doc.get("filename") or "").lower()

    if re.search(r"yacht\s*pool|pool|quotation|nabidka|nab[ií]dka|quote|rfp", fn):
        return 0
    if re.search(r"application", fn):
        return 1
    if re.search(r"condition|podm[ií]nky|vpp|general\s+term", fn):
        return 2
    return 3


def _build_yacht_offer_text(documents: list) -> str:
    """
    Build combined, cleaned offer text for regex extraction.

    Priority order:
      YACHT POOL / quotation  → up to 12 000 chars
      application             → up to  4 000 chars
      conditions              → up to  6 000 chars
      other                   → up to  3 000 chars

    Hard cap: 20 000 chars total.
    """
    if not documents:
        return ""

    sorted_docs = sorted(documents, key=_yacht_doc_priority)

    per_priority_limits = {0: 12_000, 1: 4_000, 2: 6_000, 3: 3_000}
    max_total = 20_000

    parts = []
    total = 0
    for doc in sorted_docs:
        raw = doc.get("ocr_text") or ""
        if not raw.strip():
            continue
        cleaned = clean_ocr_text(raw)
        priority = _yacht_doc_priority(doc)
        limit = per_priority_limits.get(priority, 3_000)
        remaining = max_total - total
        if remaining <= 0:
            break
        chunk = cleaned[: min(limit, remaining)]
        if chunk.strip():
            parts.append(chunk)
            total += len(chunk)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Field label groups
# ---------------------------------------------------------------------------

_TOTAL_COST_LABELS = [
    "Total insurance cost",
    "Total insurance premium",
    "Total premium",
    "celková cena pojištění",
    "Celkové pojistné",
    "Total cost",
    "Grand total",
]

_HULL_PREMIUM_LABELS = [
    "Hull insurance premium",
    "Hull premium",
    "Havarijní pojištění - pojistné",
    "Havarijní pojistné",
    "Casco premium",
    "Hull:",
]

_TPL_PREMIUM_LABELS = [
    "TPL premium",
    "TPL insurance premium",
    "Pojištění odpovědnosti za škodu - pojistné",
    "Odpovědnostní pojistné",
    "Third party liability premium",
    "Liability premium",
    "TPL:",
]

_PAX_PREMIUM_LABELS = [
    "Passenger accident insurance premium",
    "Passenger accident premium",
    "Pojištění posádky - pojistné",
    "Posádka pojistné",
    "Crew accident premium",
    "Personal accident premium",
    "PA premium",
]

_HULL_DEDUCTIBLE_LABELS = [
    "Hull insurance deductible",
    "Hull deductible",
    "Havarijní pojištění - spoluúčast",
    "Havarijní spoluúčast",
    "Casco deductible",
    "Deductible hull",
    "Deductible:",
    "Spoluúčast:",
    "Excess:",
]

_BOAT_VALUE_LABELS = [
    "Boat value",
    "Hodnota jachty se zabudovanou výbavou",
    "Hodnota lodi",
    "Agreed value",
    "Insured value of the vessel",
    "Sum insured",
    "Vessel value",
]

_INVENTORY_VALUE_LABELS = [
    "Boat Inventory value",
    "Inventory value",
    "Dodatečná výbava a osobní ohodnocení",
    "Výbava a příslušenství",
    "Additional equipment",
    "Personal effects",
]

_TRAILER_VALUE_LABELS = [
    "Trailer value",
    "Hodnota přívěsu",
    "Přívěs",
    "Trailer:",
]


# ---------------------------------------------------------------------------
# Public: parse one offer
# ---------------------------------------------------------------------------

def parse_yacht_offer(offer: dict) -> dict:
    """
    Extract yacht insurance fields from a single offer's documents.

    Returns a dict with all schema fields; missing values are None (never guessed).
    """
    docs = offer.get("documents") or []
    text = _build_yacht_offer_text(docs)

    total_cost    = _first_eur_after_label(text, _TOTAL_COST_LABELS)
    hull_prem     = _first_eur_after_label(text, _HULL_PREMIUM_LABELS)
    tpl_prem      = _first_eur_after_label(text, _TPL_PREMIUM_LABELS)
    pax_prem      = _first_eur_after_label(text, _PAX_PREMIUM_LABELS)
    hull_deduct   = _first_eur_after_label(text, _HULL_DEDUCTIBLE_LABELS)
    tpl_limit     = _extract_tpl_limit_eur(text)

    boat_val      = _first_czk_after_label(text, _BOAT_VALUE_LABELS)
    inventory_val = _first_czk_after_label(text, _INVENTORY_VALUE_LABELS)
    trailer_val   = _first_czk_after_label(text, _TRAILER_VALUE_LABELS)

    cruising_area = _extract_cruising_area_text(text)

    return {
        "id":                             offer.get("id"),
        "insurer":                        offer.get("insurer"),
        "label":                          offer.get("label"),
        "total_insurance_cost_eur":       total_cost,
        "hull_premium_eur":               hull_prem,
        "tpl_premium_eur":                tpl_prem,
        "passenger_accident_premium_eur": pax_prem,
        "hull_deductible_eur":            hull_deduct,
        "tpl_limit_eur":                  tpl_limit,
        "boat_value_czk":                 boat_val,
        "inventory_value_czk":            inventory_val,
        "trailer_value_czk":              trailer_val,
        "cruising_area":                  cruising_area,
    }


# ---------------------------------------------------------------------------
# Public: rank offers
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


def rank_yacht_offers(offers_parsed: list) -> tuple:
    """
    Rank yacht offers deterministically.

    Weights
    -------
    0.45  total_insurance_cost_eur   lower is better
    0.25  tpl_limit_eur              higher is better
    0.15  hull_deductible_eur        lower is better
    0.15  cruising_area breadth      higher is better (score 1–4)

    Tie-breakers (in order):
      1. lower total_insurance_cost_eur
      2. higher tpl_limit_eur
      3. lower hull_deductible_eur
      4. original offer order (stable)

    Returns
    -------
    (ranking: list[str], best_offer_id: Optional[str])
    """
    if not offers_parsed:
        return [], None

    costs    = [o.get("total_insurance_cost_eur") for o in offers_parsed]
    tpl_lims = [o.get("tpl_limit_eur") for o in offers_parsed]
    deducts  = [o.get("hull_deductible_eur") for o in offers_parsed]
    areas    = [float(score_cruising_area(o.get("cruising_area"))) for o in offers_parsed]

    cost_s   = _minmax_norm(costs,    higher_is_better=False)
    tpl_s    = _minmax_norm(tpl_lims, higher_is_better=True)
    deduct_s = _minmax_norm(deducts,  higher_is_better=False)
    area_s   = _minmax_norm(areas,    higher_is_better=True)

    scores = [
        0.45 * cost_s[i] + 0.25 * tpl_s[i] + 0.15 * deduct_s[i] + 0.15 * area_s[i]
        for i in range(len(offers_parsed))
    ]

    _INF = float("inf")
    indexed = list(enumerate(offers_parsed))
    indexed.sort(
        key=lambda x: (
            -scores[x[0]],
            costs[x[0]] if costs[x[0]] is not None else _INF,
            -(tpl_lims[x[0]] or 0),
            deducts[x[0]] if deducts[x[0]] is not None else _INF,
            x[0],
        )
    )

    ranking = [offers_parsed[i]["id"] for i, _ in indexed]
    best_offer_id = ranking[0] if ranking else None
    return ranking, best_offer_id
