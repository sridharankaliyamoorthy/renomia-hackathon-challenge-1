"""
Segment-aware routing for the /solve pipeline.

Normalises incoming segment strings to canonical values and dispatches
to the appropriate solver.  Only "odpovednost" has a full implementation;
all other segments return a valid response shape with null fields and
preserve original offer order for ranking.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segment normalisation map
# ---------------------------------------------------------------------------

_SEGMENT_ALIASES: dict = {
    # odpovednost
    "odpovědnost": "odpovednost",
    "odpovednost": "odpovednost",
    # auta
    "auta":    "auta",
    "auto":    "auta",
    "vozidla": "auta",
    # lode
    "lodě":     "lode",
    "lode":     "lode",
    "plavidla": "lode",
}

_CANONICAL_SEGMENTS = ["odpovednost", "auta", "lode"]


def normalize_segment(segment: str) -> str:
    """Normalise a raw segment string to its canonical form.

    Returns the canonical segment name, or "unknown" if not recognised.
    """
    if not segment:
        return "unknown"
    key = segment.strip().lower()
    return _SEGMENT_ALIASES.get(key, "unknown")


def supported_segments() -> list:
    """Return the list of canonical supported segment names."""
    return list(_CANONICAL_SEGMENTS)


# ---------------------------------------------------------------------------
# Stub helpers for unimplemented segments
# ---------------------------------------------------------------------------

def _null_offer_stub(offer: dict) -> dict:
    """Return a valid parsed-offer shape with only identity fields populated."""
    return {
        "id":                              offer.get("id"),
        "insurer":                         offer.get("insurer"),
        "label":                           offer.get("label"),
        "basic_limit_czk":                 None,
        "basic_deductible_czk":            None,
        "premium_czk":                     None,
        "covered_activities":              None,
        "territorial_scope":               None,
        "limit_multiplier_per_year":       None,
        "aggregate_limit_czk":             None,
        "limit_persons_in_custody_czk":    None,
        "limit_pure_financial_loss_czk":   None,
        "limit_taken_items_czk":           None,
        "limit_cross_liability_czk":       None,
        "limit_recourse_czk":              None,
        "limit_non_pecuniary_damage_czk":  None,
        "deductible_recourse_czk":         None,
        "deductible_non_pecuniary_czk":    None,
        "deductible_brought_items_czk":    None,
        "deductible_financial_loss_czk":   None,
    }


def _stub_response(payload: dict) -> dict:
    """Return a valid /solve response shape with null fields, original-order ranking."""
    offers = payload.get("offers") or []
    offers_parsed = [_null_offer_stub(o) for o in offers]
    ranking = [o["id"] for o in offers_parsed if o.get("id") is not None]
    best_offer_id = ranking[0] if ranking else None
    return {
        "offers_parsed": offers_parsed,
        "ranking":       ranking,
        "best_offer_id": best_offer_id,
    }


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def solve_segment(payload: dict, gemini_client, db_conn=None) -> dict:
    """
    Route a solve request to the appropriate segment handler.

    Segment dispatch:
      - "odpovednost" → full existing solver (_solve_core in main.py), unchanged
      - "auta"        → stub: null fields, original-order ranking
      - "lode"        → stub: null fields, original-order ranking
      - unknown       → stub: null fields, original-order ranking

    The public response shape is always:
        { offers_parsed, ranking, best_offer_id }

    The _debug key is stripped before returning so the public API stays clean.
    db_conn is accepted for API consistency but the odpovednost solver manages
    its own connection internally.
    """
    raw_segment = payload.get("segment") or ""
    canonical = normalize_segment(raw_segment)

    logger.info("[router] segment=%r → canonical=%r", raw_segment, canonical)

    if canonical == "odpovednost":
        # Import here to avoid circular imports at module load time.
        from main import _solve_core  # noqa: PLC0415
        result = _solve_core(payload)
        result.pop("_debug", None)
        return result

    if canonical == "lode":
        from yacht_extractor import parse_yacht_offer, rank_yacht_offers  # noqa: PLC0415
        offers = payload.get("offers") or []
        offers_parsed = [parse_yacht_offer(o) for o in offers]
        ranking, best_offer_id = rank_yacht_offers(offers_parsed)
        return {
            "offers_parsed": offers_parsed,
            "ranking":       ranking,
            "best_offer_id": best_offer_id,
        }

    if canonical == "auta":
        from auto_extractor import parse_auto_offer, rank_auto_offers  # noqa: PLC0415
        offers = payload.get("offers") or []
        offers_parsed = [parse_auto_offer(o) for o in offers]
        ranking, best_offer_id = rank_auto_offers(offers_parsed)
        return {
            "offers_parsed": offers_parsed,
            "ranking":       ranking,
            "best_offer_id": best_offer_id,
        }

    logger.warning(
        "[router] unknown segment=%r — returning null stub with original-order ranking",
        raw_segment,
    )
    return _stub_response(payload)
