"""
rank.py — Deterministic win-count ranking for insurance offers.

No Gemini calls anywhere in this file. Pure Python math only.
"""

from typing import Optional

from normalize import parse_number, normalize_text_for_compare


# ---------------------------------------------------------------------------
# Direction keyword sets (pre-normalized — no diacritics)
# ---------------------------------------------------------------------------

_LOWER_KEYWORDS = [
    "spoluucast",
    "pojistne",
    "premiu",       # matches "premium", "prémium", etc.
    "premium",
    "celkem",
    "rocni pojistne",
    "deductible",
    "excess",
    "self-retention",
    "cena celkem",
    "total",
]

_HIGHER_KEYWORDS = [
    "limit",
    "plneni",
    "sublimit",
    "kryti",
    "pojistna castka",
    "odskodne",
    "pojistna suma",
    "insured sum",
    "combined single",
]


def infer_field_direction(field_name: str, field_type: str) -> str:
    """
    Return "lower", "higher", or "qualitative" for a given field.

    "lower"      — cost/deductible fields where smaller is better
    "higher"     — coverage/limit fields where larger is better
    "qualitative" — string fields scored via score_qualitative_string,
                   and any number field with no keyword match defaults to "higher"
    """
    normalized = normalize_text_for_compare(field_name)

    for kw in _LOWER_KEYWORDS:
        if kw in normalized:
            return "lower"

    for kw in _HIGHER_KEYWORDS:
        if kw in normalized:
            return "higher"

    if field_type == "string":
        return "qualitative"

    # number field with no keyword match — default to higher
    return "higher"


def score_qualitative_string(field_name: str, value: str) -> Optional[float]:
    """
    Score a qualitative string field to a float in [0.0, 1.0], or None.

    Returns None for N/A, empty, unrecognized values (no points, no penalty).
    Rules are checked in order; first match wins.
    """
    if value is None:
        return None
    v = normalize_text_for_compare(value)
    if not v or v == "n/a":
        return None

    # Boolean / inclusion
    if v in ("ano", "yes", "included"):
        return 1.0
    if v in ("ne", "no", "excluded"):
        return 0.0

    # Unlimited / limited
    if v == "neomezeno":
        return 1.0
    if v == "omezeno":
        return 0.0

    # Geographic coverage
    if "cely svet" in v or "worldwide" in v:
        return 1.0
    if "evropa" in v or "europe" in v:
        return 0.7
    if "ceska republika" in v or " cr " in v or v.strip() == "cr":
        return 0.3

    # Coverage breadth
    if "allrisk" in v or "all risk" in v or "all risks" in v:
        return 1.0
    if "mini" in v or "basic" in v:
        return 0.3

    # Settlement basis
    if "nova cena" in v or "new value" in v or "replacement" in v:
        return 1.0
    if "casova cena" in v or "time value" in v or "current value" in v:
        return 0.5

    return None


def rank_offers_dynamic(
    offers_parsed: list,
    fields_to_extract: list,
    field_types: dict,
    rfp_text: str = "",
) -> list:
    """
    Return list of offer IDs sorted best to worst using deterministic win-count.

    rfp_text is accepted for API compatibility but is NEVER used for ranking —
    it is only passed to extraction prompts.

    Algorithm:
      - For each field, find the best value across all offers.
      - Award `weight` points to every offer that matches the best value.
      - NUMBER fields: weight=3.0; STRING fields: weight=1.0.
      - Ties broken by: (1) non-N/A field count, (2) first premium value,
        (3) stable original input order.
    """
    wins = {offer["id"]: 0.0 for offer in offers_parsed}

    for field in fields_to_extract:
        ftype = field_types.get(field, "string")
        direction = infer_field_direction(field, ftype)
        weight = 3.0 if ftype == "number" else 1.0

        values: dict = {}

        if ftype == "number":
            for offer in offers_parsed:
                raw = offer["fields"].get(field, "N/A")
                parsed = parse_number(raw)
                if parsed is not None:
                    values[offer["id"]] = parsed
        else:
            for offer in offers_parsed:
                raw = offer["fields"].get(field, "N/A")
                score = score_qualitative_string(field, raw)
                if score is not None:
                    values[offer["id"]] = score

        if not values:
            continue

        best = min(values.values()) if direction == "lower" else max(values.values())

        for offer_id, v in values.items():
            if v == best:
                wins[offer_id] += weight

    # -----------------------------------------------------------------------
    # N/A penalty: offers with >50% missing fields should not win by default
    # -----------------------------------------------------------------------
    total_fields = len(fields_to_extract)
    if total_fields > 0:
        for offer in offers_parsed:
            na_count = sum(
                1 for f in fields_to_extract
                if offer["fields"].get(f, "N/A") == "N/A"
            )
            na_ratio = na_count / total_fields
            if na_ratio > 0.50:
                penalty = wins[offer["id"]] * na_ratio * 0.5
                wins[offer["id"]] -= penalty

    # -----------------------------------------------------------------------
    # Tiebreakers
    # -----------------------------------------------------------------------

    # TB1: count non-N/A fields per offer
    def non_na_count(offer: dict) -> int:
        return sum(
            1 for v in offer["fields"].values()
            if v and str(v).strip().upper() != "N/A"
        )

    # TB2: value of first number field that has a valid parse_number result
    def first_premium_value(offer: dict) -> float:
        for field in fields_to_extract:
            if field_types.get(field, "string") == "number":
                raw = offer["fields"].get(field, "N/A")
                parsed = parse_number(raw)
                if parsed is not None:
                    return parsed
        return float("inf")

    # Build index map for TB3 (stable original order)
    input_index = {offer["id"]: idx for idx, offer in enumerate(offers_parsed)}

    sorted_offers = sorted(
        offers_parsed,
        key=lambda o: (
            -wins[o["id"]],
            -non_na_count(o),
            first_premium_value(o),
            input_index[o["id"]],
        ),
    )

    return [o["id"] for o in sorted_offers]
