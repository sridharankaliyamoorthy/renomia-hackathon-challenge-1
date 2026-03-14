"""
Text field extraction helpers for covered_activities and territorial_scope.
Uses local regex/heuristic pass first; falls back to Gemini only
when local heuristics leave one or both fields unresolved.

Pass 1 Gemini (max 2500 chars): high-signal docs only
  (rfp_nabidka → ujednani → smlouva_ps; VPP excluded unless nothing else).
Pass 2 Gemini (max 1800 chars): VPP/PP docs only, for still-missing fields.
Maximum 2 Gemini calls per offer total.

Zero ranking logic. Zero numeric extraction.
"""

import json
import logging
import re
from typing import Optional

from preprocess import clean_ocr_text, extract_keyword_windows, prioritize_documents

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

_TERRITORIAL_KEYWORDS = [
    "územní rozsah",
    "územní platnost",
    "místo pojištění",
    "česká republika",
    "Česká republika",
    "ČR",
    "Evropa",
    "Europe",
    "svět",
    "worldwide",
    "celý svět",
]

_ACTIVITY_KEYWORDS = [
    # Most-specific multi-word phrases first — avoids noise from bare "činnost"
    "rozsah pojištěné činnosti",
    "pojištěná činnost",
    "pojištěné činnosti",
    "pojistné činnosti",
    "podnikatelská činnost",
    "předmět pojištění",
    "předmět činnosti",
    "předmětem pojištění",
    "činnosti pojistníka",
    "druh činnosti",
    "pojistník provozuje",
]

# Label patterns ordered most-specific → least-specific; first match wins.
# Each captures the value in group 1 up to the first newline, CR, or semicolon.
_ACTIVITY_LABEL_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"rozsah\s+pojišt[eě]n[eé]\s+č[íi]nnosti?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"pojišt[eě]n[aáé]\s+č[íi]nnost[íi]?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"pojistn[eé]\s+č[íi]nnosti?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"podnikatelsk[aáé]\s+č[íi]nnost[íi]?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"p[řr]edm[eě]t(?:em)?\s+poji[sš]t[eě]n[íi]?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"p[řr]edm[eě]t\s+č[íi]nnosti?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"č[íi]nnosti\s+pojistn[íi]?ka"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"druh\s+č[íi]nnosti?"
        r"\s*:?\s*([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
    re.compile(
        r"pojistn[íi]?k\s+provozuje\s+([^\n\r;]{5,150})",
        re.IGNORECASE,
    ),
]

# Additional word-level stop markers not already handled by [^\n\r;] in the
# label patterns above.
_ACTIVITY_STOP_RE = re.compile(
    r"územní\s+rozsah"
    r"|limit\s+pln[eě]ní"
    r"|\bspoluúčast\b"
    r"|\bpojistné\b",
    re.IGNORECASE,
)

# Extracted values that are too vague to be useful — return null instead.
_ACTIVITY_TOO_GENERIC: frozenset[str] = frozenset(
    {
        "odpovědnost",
        "činnost",
        "činnosti",
        "dle živnostenského oprávnění",
    }
)

# Canonical territorial scope values (all Czech):
#   "Česká republika" | "Evropa" | "celý svět"
#
# Ordered most-specific → least-specific so the first match wins.
# "worldwide", "Europe", "celý svět / worldwide" all map to "celý svět".
_TERRITORY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"celý\s+svět", re.IGNORECASE), "celý svět"),
    (re.compile(r"\bworldwide\b", re.IGNORECASE), "celý svět"),
    (re.compile(r"\bEvropa\b|\bEuropa\b|\bEurope\b", re.IGNORECASE), "Evropa"),
    (re.compile(r"\bČeská\s+republika\b|\bčeská\s+republika\b", re.IGNORECASE), "Česká republika"),
    (re.compile(r"\bČR\b", re.UNICODE), "Česká republika"),
    (re.compile(r"\bčesko\b", re.IGNORECASE), "Česká republika"),
]

# Normalise any raw string (including Gemini output) to a canonical value.
# Returns the original string unchanged when no mapping applies.
def normalize_territorial_scope(raw: str) -> str:
    """Map any raw territorial scope string to the canonical Czech form."""
    if not raw:
        return raw
    for pattern, canonical in _TERRITORY_PATTERNS:
        if pattern.search(raw):
            return canonical
    return raw

# Doc-type sets used to partition documents for each pass
_HIGH_SIGNAL_TYPES = {"rfp_nabidka", "ujednani", "smlouva_ps"}
_VPP_TYPES = {"vpp_pp"}


# ---------------------------------------------------------------------------
# Local heuristics
# ---------------------------------------------------------------------------

def extract_territorial_scope_local(text: str) -> Optional[str]:
    """
    Extract territorial scope from text via keyword windows + pattern matching.
    Returns None when no confident match is found.
    """
    if not text:
        return None

    snippets = extract_keyword_windows(text, _TERRITORIAL_KEYWORDS, window_chars=200)
    combined = " ".join(snippets)

    # Search snippets first (higher signal), then full text
    for search_target in (combined, text):
        for pattern, label in _TERRITORY_PATTERNS:
            if pattern.search(search_target):
                return label

    return None


def extract_covered_activities_local(text: str) -> Optional[str]:
    """
    Extract covered activities from text via keyword windows + label-pattern matching.

    Strategy:
    - Extract windows around the activity keywords (300-char radius).
    - Try each label pattern in priority order (most-specific first); first hit wins.
    - Truncate the captured value at hard stop markers (newline/CR/semicolon handled
      by the pattern itself; word-level stops handled by _ACTIVITY_STOP_RE).
    - Strip trailing punctuation and excess whitespace.
    - Return None when no confident, non-generic match is found.
    """
    if not text:
        return None

    snippets = extract_keyword_windows(text, _ACTIVITY_KEYWORDS, window_chars=300)
    if not snippets:
        return None

    for pattern in _ACTIVITY_LABEL_PATTERNS:
        for snippet in snippets:
            m = pattern.search(snippet)
            if not m:
                continue

            raw = m.group(1)

            # Truncate at word-level stops not already handled by [^\n\r;]
            stop = _ACTIVITY_STOP_RE.search(raw)
            if stop:
                raw = raw[: stop.start()]

            value = raw.strip()
            value = re.sub(r"[,.\s]+$", "", value)       # trailing punctuation/space
            value = re.sub(r"\s{2,}", " ", value)        # internal whitespace runs
            value = re.sub(r"[,;]{2,}", ",", value)      # repeated separators

            if len(value) < 5:
                continue

            # Reject values that are too generic to be informative
            normalised = value.lower().strip(".,;:- ")
            if normalised in _ACTIVITY_TOO_GENERIC:
                continue

            return value

    return None


# ---------------------------------------------------------------------------
# Context builders for Gemini — never send full OCR
# ---------------------------------------------------------------------------

def _build_context_from_docs(
    docs: list[dict],
    keywords: list[str],
    max_chars: int,
    label_override: Optional[str] = None,
) -> str:
    """
    Shared context-building logic: extract keyword windows from *docs*, label
    each section, deduplicate snippets, and hard-cap at max_chars.

    label_override: if set, every section is labelled with this string instead
    of the doc's own _doc_type (used for VPP pass to always show [vpp_pp]).
    """
    parts: list[str] = []
    total_chars = 0
    seen_snippets: set[str] = set()

    for doc in docs:
        doc_type = label_override or doc.get("_doc_type", "unknown")
        raw_text = doc.get("ocr_text") or ""
        if not raw_text.strip():
            continue

        cleaned = clean_ocr_text(raw_text)
        snippets = extract_keyword_windows(cleaned, keywords, window_chars=200)
        if not snippets:
            continue

        doc_snippets: list[str] = []
        for snippet in snippets:
            if snippet in seen_snippets:
                continue
            seen_snippets.add(snippet)
            label = f"[{doc_type}]\n"
            available = max_chars - total_chars - len(label) - 5
            if available <= 0:
                break
            chunk = snippet[:available]
            doc_snippets.append(chunk)
            total_chars += len(chunk)

        if doc_snippets:
            section = f"[{doc_type}]\n" + "\n---\n".join(doc_snippets)
            parts.append(section)

        if total_chars >= max_chars:
            break

    return "\n\n".join(parts)


def build_first_pass_text_context(documents: list[dict], max_chars: int = 2500) -> str:
    """
    Build compact, labelled context from high-signal docs only
    (rfp_nabidka, ujednani, smlouva_ps). VPP/PP docs are excluded unless
    no non-VPP docs exist at all.

    Uses keyword windows only — never sends full OCR.
    Hard-capped at max_chars total.
    """
    all_keywords = _TERRITORIAL_KEYWORDS + _ACTIVITY_KEYWORDS
    prioritized = prioritize_documents(documents)

    # Partition: high-signal first, then unknown, VPP excluded from pass 1
    high_signal = [d for d in prioritized if d.get("_doc_type") in _HIGH_SIGNAL_TYPES]
    other = [
        d for d in prioritized
        if d.get("_doc_type") not in _HIGH_SIGNAL_TYPES
        and d.get("_doc_type") not in _VPP_TYPES
    ]
    candidates = high_signal + other

    # Only fall back to VPP if there are truly no non-VPP documents
    if not candidates:
        candidates = prioritized

    return _build_context_from_docs(candidates, all_keywords, max_chars)


def build_second_pass_vpp_context(
    documents: list[dict],
    missing_fields: list[str],
    max_chars: int = 1800,
) -> str:
    """
    Build compact context from VPP/PP docs only, using keyword windows
    scoped to the still-missing fields.

    Hard-capped at max_chars. Never sends full VPP OCR.
    Returns empty string if no VPP docs are found or no keywords match.
    """
    # Build keyword list from only the still-missing fields
    keywords: list[str] = []
    if "territorial_scope" in missing_fields:
        keywords.extend(_TERRITORIAL_KEYWORDS)
    if "covered_activities" in missing_fields:
        keywords.extend(_ACTIVITY_KEYWORDS)

    if not keywords:
        return ""

    prioritized = prioritize_documents(documents)
    vpp_docs = [d for d in prioritized if d.get("_doc_type") in _VPP_TYPES]
    if not vpp_docs:
        return ""

    return _build_context_from_docs(vpp_docs, keywords, max_chars, label_override="vpp_pp")


# ---------------------------------------------------------------------------
# Legacy wrapper — identical signature to the old function
# ---------------------------------------------------------------------------

def build_text_field_context(documents: list[dict], max_chars: int = 2500) -> str:
    """Backward-compatible wrapper; delegates to build_first_pass_text_context."""
    return build_first_pass_text_context(documents, max_chars=max_chars)


# ---------------------------------------------------------------------------
# Decision gate
# ---------------------------------------------------------------------------

def needs_text_field_llm(
    covered_activities: Optional[str],
    territorial_scope: Optional[str],
) -> bool:
    """Return True if either field is still unresolved after local heuristics."""
    return covered_activities is None or territorial_scope is None


# ---------------------------------------------------------------------------
# Shared Gemini prompt builder
# ---------------------------------------------------------------------------

def _build_extraction_prompt(missing_fields: list[str], context: str) -> str:
    return (
        "You are an insurance document analyst. "
        "Extract only the following fields from the provided Czech insurance document snippets.\n"
        f"Fields to extract: {', '.join(missing_fields)}\n\n"
        "Rules:\n"
        "- covered_activities: short description of insured business activities (1-2 sentences max)\n"
        "- territorial_scope: geographic scope of coverage. "
        "Use ONLY one of these exact canonical values: "
        "'Česká republika', 'Evropa', 'celý svět'. "
        "Map 'worldwide', 'Europe', 'celý svět / worldwide' to the canonical form.\n"
        "- Return null for any field you cannot confidently determine from the text\n"
        "- No explanations. No markdown. No extra keys.\n"
        "- Output ONLY valid JSON with exactly these two keys: "
        "covered_activities, territorial_scope\n\n"
        f"Document snippets:\n{context}\n\n"
        "JSON output:"
    )


def _apply_gemini_result(
    data: dict,
    parsed_offer: dict,
    missing_fields: list[str],
) -> dict:
    """Merge Gemini JSON result into parsed_offer. Never overwrites a non-null value."""
    for field in ("covered_activities", "territorial_scope"):
        if field in missing_fields and parsed_offer.get(field) is None:
            val = data.get(field)
            if val:
                text = str(val).strip()
                if field == "territorial_scope":
                    text = normalize_territorial_scope(text)
                parsed_offer[field] = text
            else:
                parsed_offer[field] = None
    return parsed_offer


# ---------------------------------------------------------------------------
# Two-pass Gemini enrichment — max 2 Gemini calls per offer
# ---------------------------------------------------------------------------

def enrich_text_fields_two_pass(
    offer: dict,
    parsed_offer: dict,
    gemini_client,
) -> dict:
    """
    Fill covered_activities and territorial_scope using:
      1. Local heuristics (zero Gemini cost)
      2. Pass 1 Gemini (≤2500 chars): high-signal docs (rfp_nabidka, ujednani, smlouva_ps)
      3. Pass 2 Gemini (≤1800 chars): VPP/PP docs only, only for still-missing fields

    Second pass is triggered only when all of:
      - at least one of covered_activities / territorial_scope is still null after pass 1
      - gemini_client.enabled is True
      - at least one VPP/PP document exists
      - build_second_pass_vpp_context returns non-empty text

    Never overwrites a non-null field. Mutates and returns parsed_offer.
    """
    docs = offer.get("documents") or []
    offer_id = parsed_offer.get("id", "?")

    if not docs:
        logger.info("[text_fields] offer=%s — no documents, skipping", offer_id)
        return parsed_offer

    # --- Local heuristic pass (zero cost) ---
    combined_parts: list[str] = []
    for doc in prioritize_documents(docs):
        raw = doc.get("ocr_text") or ""
        if raw.strip():
            combined_parts.append(clean_ocr_text(raw))
    combined_text = "\n\n".join(combined_parts)

    territorial_scope = extract_territorial_scope_local(combined_text)
    covered_activities = extract_covered_activities_local(combined_text)

    if territorial_scope is not None:
        parsed_offer["territorial_scope"] = territorial_scope
    if covered_activities is not None:
        parsed_offer["covered_activities"] = covered_activities

    if not needs_text_field_llm(
        parsed_offer.get("covered_activities"),
        parsed_offer.get("territorial_scope"),
    ):
        logger.info(
            "[text_fields] offer=%s — both fields resolved locally, Gemini skipped", offer_id
        )
        return parsed_offer

    if not getattr(gemini_client, "enabled", False):
        logger.info(
            "[text_fields] offer=%s — Gemini disabled, keeping local results", offer_id
        )
        return parsed_offer

    # --- Pass 1: high-signal docs (rfp_nabidka, ujednani, smlouva_ps) ---
    missing_pass1 = [
        f for f in ("covered_activities", "territorial_scope")
        if parsed_offer.get(f) is None
    ]

    pass1_context = build_first_pass_text_context(docs, max_chars=2500)
    if pass1_context.strip():
        logger.info(
            "[text_fields] offer=%s — pass1 Gemini called, fields=%s, context_chars=%d",
            offer_id, missing_pass1, len(pass1_context),
        )
        prompt1 = _build_extraction_prompt(missing_pass1, pass1_context)
        for attempt in range(2):
            try:
                response = gemini_client.generate(
                    prompt1,
                    generation_config={"response_mime_type": "application/json"},
                )
                data = json.loads(response.text.strip())
                _apply_gemini_result(data, parsed_offer, missing_pass1)
                logger.info(
                    "[text_fields] offer=%s — pass1 result: covered_activities=%r territorial_scope=%r",
                    offer_id,
                    parsed_offer.get("covered_activities"),
                    parsed_offer.get("territorial_scope"),
                )
                break  # success
            except json.JSONDecodeError as exc:
                if attempt == 0:
                    logger.warning(
                        "[text_fields] offer=%s — pass1 Gemini malformed JSON (%s), retrying",
                        offer_id, exc,
                    )
                else:
                    logger.warning(
                        "[text_fields] offer=%s — pass1 Gemini failed after retry (%s), keeping nulls",
                        offer_id, exc,
                    )
            except Exception as exc:
                logger.warning(
                    "[text_fields] offer=%s — pass1 Gemini failed (%s), keeping nulls",
                    offer_id, exc,
                )
                break  # don't retry on non-JSON errors
    else:
        logger.info(
            "[text_fields] offer=%s — pass1 Gemini skipped (no keyword snippets in high-signal docs)",
            offer_id,
        )

    # --- Pass 2: VPP/PP only, only for still-missing fields ---
    still_missing = [
        f for f in ("covered_activities", "territorial_scope")
        if parsed_offer.get(f) is None
    ]

    if not still_missing:
        return parsed_offer

    prioritized_docs = prioritize_documents(docs)
    has_vpp = any(d.get("_doc_type") in _VPP_TYPES for d in prioritized_docs)

    if not has_vpp:
        logger.info(
            "[text_fields] offer=%s — pass2 skipped (no VPP/PP docs), fields=%s still null",
            offer_id, still_missing,
        )
        return parsed_offer

    pass2_context = build_second_pass_vpp_context(docs, still_missing, max_chars=1800)
    if not pass2_context.strip():
        logger.info(
            "[text_fields] offer=%s — pass2 skipped (no keyword snippets in VPP/PP docs)",
            offer_id,
        )
        return parsed_offer

    logger.info(
        "[text_fields] offer=%s — pass2 Gemini called (VPP only), fields=%s, context_chars=%d",
        offer_id, still_missing, len(pass2_context),
    )

    prompt2 = _build_extraction_prompt(still_missing, pass2_context)
    before = {f: parsed_offer.get(f) for f in still_missing}
    for attempt in range(2):
        try:
            response = gemini_client.generate(
                prompt2,
                generation_config={"response_mime_type": "application/json"},
            )
            data = json.loads(response.text.strip())
            _apply_gemini_result(data, parsed_offer, still_missing)
            filled = [f for f in still_missing if before[f] is None and parsed_offer.get(f) is not None]
            logger.info(
                "[text_fields] offer=%s — pass2 result: filled=%s "
                "covered_activities=%r territorial_scope=%r",
                offer_id, filled,
                parsed_offer.get("covered_activities"),
                parsed_offer.get("territorial_scope"),
            )
            break  # success
        except json.JSONDecodeError as exc:
            if attempt == 0:
                logger.warning(
                    "[text_fields] offer=%s — pass2 Gemini malformed JSON (%s), retrying",
                    offer_id, exc,
                )
            else:
                logger.warning(
                    "[text_fields] offer=%s — pass2 Gemini failed after retry (%s), keeping nulls",
                    offer_id, exc,
                )
        except Exception as exc:
            logger.warning(
                "[text_fields] offer=%s — pass2 Gemini failed (%s), keeping nulls",
                offer_id, exc,
            )
            break  # don't retry on non-JSON errors

    return parsed_offer


# ---------------------------------------------------------------------------
# Legacy alias — kept so existing callers continue to work unchanged
# ---------------------------------------------------------------------------

def enrich_text_fields_with_gemini(
    offer: dict,
    parsed_offer: dict,
    gemini_client,
) -> dict:
    """Backward-compatible alias for enrich_text_fields_two_pass."""
    return enrich_text_fields_two_pass(offer, parsed_offer, gemini_client)
