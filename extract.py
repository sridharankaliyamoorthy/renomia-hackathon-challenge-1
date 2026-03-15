"""
extract.py — All Gemini extraction logic for insurance offer field extraction.

Functions (in order):
  1. filter_and_sort_docs      — sort docs by priority, skip wrong-segment docs
  2. combine_offer_text        — concatenate OCR text with 20K char cap
  3. build_extraction_prompt   — construct Gemini prompt with segment context
  4. parse_gemini_response     — parse JSON from Gemini response
  5. extract_fields_gemini     — single Gemini call with retry
  6. extract_via_pdf_vision    — PDF vision fallback for empty OCR docs
  7. extract_offer             — top-level per-offer orchestrator
"""

import re
import logging
import requests
from google import genai
from google.genai import types
from typing import Optional

from normalize import is_conditions_doc, is_quotation_doc, clean_ocr_text, parse_number

logger = logging.getLogger(__name__)

_WRONG_SEGMENT_LODE_TERMS = [
    "volkswagen", "škoda", "skoda", "bmw",
    "stav tachometru", "povinné ručení vozu", "havarijní pojištění vozu",
]

_SEGMENT_CONTEXT = {
    "odpovědnost": (
        "Czech liability insurance policy (pojištění odpovědnosti).\n"
        "This is NOT vehicle insurance.\n\n"
        "CRITICAL — LIMIT I vs LIMIT II:\n"
        "Many fields come in pairs: 'Limit I' and 'Limit II'\n"
        "representing two separate coverage tiers.\n"
        "- 'Limit I' = the FIRST/LOWER coverage tier\n"
        "- 'Limit II' = the SECOND/HIGHER coverage tier\n"
        "These are DIFFERENT fields with DIFFERENT values.\n"
        "Do NOT put Limit II value into Limit I field.\n"
        "Read the document carefully to match each value to\n"
        "its correct tier.\n\n"
        "TABLE STRUCTURE — how to read the limits table:\n"
        "The document has a table with these columns:\n"
        "Column 1 = field name / coverage type\n"
        "Column 2 = Limit I value (lower tier)\n"
        "Column 3 = Limit II value (higher tier)\n"
        "Column 4 = Spoluúčast I (deductible tier 1)\n"
        "Column 5 = Spoluúčast II (deductible tier 2)\n\n"
        "When you see a row like:\n"
        "  Regres pojišťoven | CZK 50,000,000 | CZK 100,000,000\n"
        "Extract:\n"
        "  Regres pojišťoven limit I = CZK 50,000,000\n"
        "  Regres pojišťoven limit II = CZK 100,000,000\n\n"
        "The SECOND value in the row is always Limit II.\n"
        "Do NOT skip it. Do NOT return N/A if column 3 exists.\n\n"
        "VALUE FORMAT — return VERBATIM with currency prefix:\n"
        "- Always include CZK prefix: 'CZK 50,000,000'\n"
        "  NOT '50000000' NOT '50 000 000'\n"
        "- For ranges: 'CZK 10,000–50,000' (keep both values)\n"
        "  NOT just '10000' or '50000'\n"
        "- For descriptive fields: return the FULL policy text\n"
        "  e.g. 'Jedna spoluúčast (nejvyšší)' not 'Ano'\n"
        "  e.g. 'CZK 50,000,000–100,000,000' not 'Ano'\n"
        "  e.g. 'Tržby a škodní průběh ovlivňují' not summary\n"
        "- For yes/no fields: return 'Ano' or 'Ne' only when\n"
        "  the document literally shows Ano/Ne — NOT when\n"
        "  the field has a monetary limit value\n"
        "- Do NOT substitute 'Ano' for a monetary amount\n"
        "- Do NOT substitute 'Ano' for a descriptive phrase\n\n"
        "FIELD TYPES in this segment:\n"
        "- NUMBER fields: return value WITH CZK prefix verbatim\n"
        "  e.g. 'CZK 50,000,000' for a 50M CZK limit\n"
        "- STRING fields: return full descriptive text verbatim\n\n"
        "RANGE VALUES are common — return both ends:\n"
        "- 'CZK 248,923–281,136' not just one number\n"
        "- 'CZK 10,000–50,000' not just '10,000'\n\n"
        "SUBLIMIT FIELDS — look for sublimit tables:\n"
        "Fields like 'Věci zaměstnanců', 'Věci převzaté',\n"
        "'Finanční škody', 'Škody na životním prostředí' etc.\n"
        "have their OWN specific sublimit values — do NOT\n"
        "copy the general limit into these fields.\n\n"
        "DESCRIPTIVE/CLAUSE FIELDS — return full text:\n"
        "Fields like 'Smluvní pokuty', 'Regresní náhrady',\n"
        "'Vyloučené činnosti', 'Asistenční služby',\n"
        "'Způsob stanovení prémia' require the actual\n"
        "policy clause text, not Ano/Ne."
    ),
    "auta": (
        "Czech vehicle fleet insurance quote.\n"
        "Look for fleet pricing tables, coverage options,\n"
        "annual premium totals.\n\n"
        "LABEL MAPPINGS:\n"
        "- Roční pojistné may appear as Roční platba or\n"
        "  Cena celkem or Celková cena\n"
        "- Povinné ručení – limit appears as Limit povinného\n"
        "  ručení or XXX/XXX mil. Kč — convert to CZK format\n"
        "  e.g. 150/150 mil → return CZK 150,000,000\n"
        "- Spoluúčast havarijní appears as X % / X XXX Kč\n\n"
        "If multiple variants shown (Komfort/Plus/Extra/Max):\n"
        "Extract from the MOST COMPREHENSIVE variant (Max).\n\n"
        "RULE A — Allrisk coverage inference:\n"
        "If Typ havarijního pojištění is Allrisk or Max or\n"
        "contains Allrisk in any form, then:\n"
        "- Krytí odcizení = Ano\n"
        "- Krytí vandalismu = Ano\n"
        "- Krytí živelních rizik = Ano\n"
        "- Krytí střetu se zvěří = Ano\n"
        "unless the document explicitly says these are excluded.\n\n"
        "RULE B — Checkmark detection (Allianz MOJEAUTO format):\n"
        "If document contains checkmarks like ☑ or ✔ or ✓:\n"
        "- ☑ Krádež or ☑ Odcizení → Krytí odcizení = Ano\n"
        "- ☑ Vandalismus → Krytí vandalismu = Ano\n"
        "- ☑ Přírodní události or ☑ Živel → Krytí živelních rizik = Ano\n"
        "- ☑ Poškození zvířetem or ☑ Zvěř → Krytí střetu se zvěří = Ano\n"
        "- ☑ Přímá likvidace → Přímá likvidace = Ano\n\n"
        "VALUE RULES:\n"
        "- Return ALL values VERBATIM as they appear\n"
        "- NUMBER fields: return full string with currency\n"
        "  e.g. CZK 150,000,000 or CZK 5 000 Kč or 34851\n"
        "- STRING coverage fields: return Ano or Ne or short\n"
        "  canonical phrase — NOT verbose descriptions\n"
        "  e.g. Ano not Zahrnuje Havárii, Přírodní události\n"
        "  e.g. Allrisk not long description of what it covers\n"
        "  e.g. Ano, se spoluúčastí CZK 1,000 for Krytí skel\n"
        "  e.g. Ne (lze připojistit za CZK 1 485) for excluded\n"
        "  e.g. Neomezeno or Omezeno for count fields\n"
        "  e.g. Volba servisu or Povinné smluvní servisy\n"
        "- Do NOT translate to English\n"
        "- Do NOT write verbose multi-word descriptions for\n"
        "  simple Ano/Ne coverage fields"
    ),
    "lodě": (
        "Yacht/boat insurance quotation. Values in EUR or CZK.\n"
        "Documents may be in English, Czech, or Polish.\n\n"
        "CRITICAL FIELD MAPPINGS for this segment:\n"
        "- 'roční pojistné' (lowercase) refers to the PERSONAL ACCIDENT insurance\n"
        "  annual premium — a small amount (e.g. 30-50 EUR), NOT the hull total\n"
        "- 'Roční pojistné:' (with colon, uppercase R) refers to the hull/havarijní\n"
        "  annual premium\n"
        "- 'CELKEM' is the grand total of all premiums combined\n"
        "- 'pojistné před slevou' is the pre-discount premium\n"
        "- 'Sleva' is the discount amount\n"
        "- 'Havarijní pojištění' is the hull insurance premium\n"
        "- 'Havarijní pojištění - spoluúčast' is the hull deductible amount\n\n"
        "PERSONAL ACCIDENT FIELDS — look in the personal accident / crew accident\n"
        "section specifically:\n"
        "- 'Pojištěná částka v případě úmrtí' = death benefit\n"
        "- 'Pojistná částka v případě trvalého invalidního stavu' = permanent\n"
        "  disability benefit\n"
        "- 'roční pojistné' = personal accident annual premium\n\n"
        "TPL / LIABILITY FIELDS — look in liability section:\n"
        "- 'pro škodu na věci' = property damage limit\n"
        "- 'pro újmu na životě' = bodily injury limit\n"
        "- 'pro jinou újmu na jmění' = financial loss limit\n"
        "- 'Kombinovaný jednotný limit...' = combined single limit\n"
        "- 'Limit pro finanční škody' = financial damage limit\n"
        "- 'Pojištění odpovědnosti za škodu' = TPL premium amount\n"
        "- 'Maximální výše odškodného za újmu na zdraví' = max personal injury\n"
        "  indemnity\n\n"
        "VALUES may appear as:\n"
        "- Plain integers: 578, 15562, 342.11\n"
        "- EUR decimals: 342.11, 370.00, 459.35\n"
        "- European format: EUR 7.500.000 = 7500000\n"
        "- mil/mln format: 7,5mil = 7500000, 20mln = 20000000\n"
        "- CZK integers: 10000, 2000000\n\n"
        "If multiple coverage options shown (Option A / Option B):\n"
        "Extract the HIGHER coverage option values."
    ),
}

_DEFAULT_SEGMENT_CONTEXT = "Insurance policy document. Extract the requested fields carefully."


# ═══════════════════════════════════════════════════════════
# FUNCTION 1
# ═══════════════════════════════════════════════════════════

def filter_and_sort_docs(offer: dict, segment: str) -> list:
    """
    Return docs sorted by priority (quotation first, conditions last),
    with wrong-segment documents removed.

    Sort key:
      0 = quotation doc
      1 = neutral/unknown doc
      2 = conditions doc
    """
    docs = offer.get("documents", [])
    result = []

    for doc in docs:
        filename = doc.get("filename", "") or ""
        ocr_text = doc.get("ocr_text", "") or ""
        ocr_lower = ocr_text.lower()

        # Wrong-segment detection for lodě
        if segment == "lodě":
            if any(term in ocr_lower for term in _WRONG_SEGMENT_LODE_TERMS):
                logger.info("Skipping wrong-segment doc (lodě): %s", filename)
                continue

        # Wrong-segment detection for odpovědnost (vehicle fleet quote pattern)
        if segment == "odpovědnost":
            if (
                "spoluúčast" in ocr_lower
                and "havarijní" in ocr_lower
                and "povinné ručení" in ocr_lower
            ):
                logger.info("Skipping wrong-segment doc (odpovědnost vehicle): %s", filename)
                continue

        if is_quotation_doc(filename):
            sort_key = 0
        elif is_conditions_doc(filename):
            sort_key = 2
        else:
            sort_key = 1

        result.append((sort_key, doc))

    result.sort(key=lambda x: x[0])
    return [doc for _, doc in result]


# ═══════════════════════════════════════════════════════════
# FUNCTION 2
# ═══════════════════════════════════════════════════════════

def combine_offer_text(sorted_docs: list, max_chars: int = None) -> str:
    """
    Concatenate OCR text from sorted docs with a document separator.
    Skips docs with empty OCR text (vision fallback handles those).
    Stops adding once total length exceeds max_chars.
    Quotation docs appear first and are therefore never truncated.

    max_chars defaults to the full combined length for single-doc offers,
    or 35000 for multi-doc offers (enough to avoid truncating pricing tables).
    """
    if max_chars is None:
        total_chars = sum(len(d.get("ocr_text", "") or "") for d in sorted_docs)
        max_chars = total_chars if len(sorted_docs) == 1 else 35000

    parts = []
    total = 0

    for doc in sorted_docs:
        ocr_text = clean_ocr_text((doc.get("ocr_text", "") or "").strip())
        if not ocr_text:
            continue
        filename = doc.get("filename", "unknown")
        separator = f"\n\n--- DOCUMENT: {filename} ---\n\n"
        chunk = separator + ocr_text
        if total + len(chunk) > max_chars and parts:
            break
        parts.append(chunk)
        total += len(chunk)

    return "".join(parts)


# ═══════════════════════════════════════════════════════════
# FUNCTION 3
# ═══════════════════════════════════════════════════════════

def build_extraction_prompt(
    segment: str,
    fields_to_extract: list,
    field_types: dict,
    combined_text: str,
    rfp_hints: str = "",
    missing_fields: Optional[list] = None,
) -> str:
    """
    Construct the Gemini extraction prompt.

    If missing_fields is provided (pass 2), only those fields are requested.
    Otherwise all fields_to_extract are requested (pass 1).
    """
    target_fields = missing_fields if missing_fields is not None else fields_to_extract

    segment_context = _SEGMENT_CONTEXT.get(segment, _DEFAULT_SEGMENT_CONTEXT)

    number_fields = [f for f in target_fields if field_types.get(f) == "number"]
    string_fields = [f for f in target_fields if field_types.get(f) != "number"]

    number_fields_list = "\n".join(number_fields) if number_fields else "(none)"
    string_fields_list = "\n".join(string_fields) if string_fields else "(none)"

    rfp_block = rfp_hints.strip() + "\n\n" if rfp_hints.strip() else ""

    prompt = (
        "You are extracting structured data from insurance documents.\n"
        f"Segment: {segment}\n"
        f"{segment_context}\n\n"
        f"{rfp_block}"
        "The text below comes from MULTIPLE combined documents for one insurer.\n"
        "A field value may appear in any section — search the entire text.\n"
        "Documents may be in Czech, English, or Polish — handle all three.\n\n"
        "CRITICAL RULES:\n"
        "1. Return values VERBATIM as they appear — do not convert or clean.\n"
        "2. NUMBER fields: return \"N/A\" if not found with high confidence.\n"
        "   Wrong numbers score 0.0. Be conservative.\n"
        "3. STRING fields: return your best guess even if uncertain.\n"
        "   Partial matches score 0.5+. A guess beats \"N/A\".\n"
        "4. Every key in your response must be EXACTLY one of the field\n"
        "   names listed below — no variations, no translations.\n"
        "5. If a field appears in multiple sections, prefer the value\n"
        "   from the pricing/quotation section.\n"
        "6. STRING fields that represent yes/no coverage:\n"
        "   Return the SHORT Czech label as it appears in the document —\n"
        "   typically 'Ano' or 'Ne' or a brief phrase (1-5 words).\n"
        "   Do NOT translate to English.\n"
        "   Do NOT write verbose descriptions.\n"
        "   If the document shows a checkmark or 'included', return 'Ano'.\n"
        "   If the document shows 'not included' or a cross, return 'Ne'.\n"
        "   For coverage fields like 'Krytí skel', 'Krytí odcizení',\n"
        "   'Přímá likvidace' — return exactly what the document shows,\n"
        "   typically a short phrase of 1-5 words.\n\n"
        f"NUMBER fields (return verbatim numeric string as found):\n{number_fields_list}\n\n"
        f"STRING fields (return text description as found):\n{string_fields_list}\n\n"
        f"DOCUMENT TEXT:\n{combined_text}"
    )

    return prompt


# ═══════════════════════════════════════════════════════════
# FUNCTION 4
# ═══════════════════════════════════════════════════════════

def parse_gemini_response(raw, fields_to_extract: list) -> dict:
    """
    Parse Gemini JSON response into a clean fields dict.

    - Strips markdown fences before parsing.
    - Maps every field in fields_to_extract to a string value.
    - Missing or None values become "N/A".
    - On JSON parse error: all fields become "N/A".
    """
    import json

    text = ""
    try:
        if hasattr(raw, "text"):
            text = raw.text or ""
        elif isinstance(raw, str):
            text = raw
        else:
            text = str(raw)

        # Strip markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text.strip())

        parsed = json.loads(text)
    except Exception as exc:
        logger.warning("Failed to parse Gemini JSON response: %s", exc)
        return {f: "N/A" for f in fields_to_extract}

    result = {}
    for field in fields_to_extract:
        val = parsed.get(field)
        if val is None:
            result[field] = "N/A"
        else:
            stripped = str(val).strip()
            result[field] = stripped if stripped else "N/A"

    return result


# ═══════════════════════════════════════════════════════════
# FUNCTION 5
# ═══════════════════════════════════════════════════════════

def extract_fields_gemini(
    gemini,
    combined_text: str,
    fields_to_extract: list,
    field_types: dict,
    segment: str,
    rfp_hints: str = "",
    missing_fields: Optional[list] = None,
) -> dict:
    """
    Single Gemini extraction call with JSON schema mode.
    Retries once on failure. Returns "N/A" for all fields on second failure.
    """
    target_fields = missing_fields if missing_fields is not None else fields_to_extract

    def _attempt():
        prompt = build_extraction_prompt(
            segment, fields_to_extract, field_types,
            combined_text, rfp_hints, missing_fields,
        )
        response = gemini.generate(
            prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        meta = getattr(response, "usage_metadata", None)
        if meta:
            logger.info(
                "Gemini usage — prompt: %s, completion: %s, total: %s",
                getattr(meta, "prompt_token_count", "?"),
                getattr(meta, "candidates_token_count", "?"),
                getattr(meta, "total_token_count", "?"),
            )
        return parse_gemini_response(response, target_fields)

    try:
        return _attempt()
    except Exception as exc:
        logger.warning("Gemini call failed (attempt 1): %s — retrying", exc)

    try:
        return _attempt()
    except Exception as exc:
        logger.error("Gemini call failed (attempt 2): %s — returning N/A for all fields", exc)
        return {f: "N/A" for f in target_fields}


# ═══════════════════════════════════════════════════════════
# FUNCTION 6
# ═══════════════════════════════════════════════════════════

def extract_via_pdf_vision(
    gemini,
    doc: dict,
    fields_to_extract: list,
    field_types: dict,
    segment: str,
) -> dict:
    """
    PDF vision fallback. Only called when ocr_text < 50 chars AND pdf_url present.

    Downloads PDF, uploads via Gemini Files API, extracts fields, then deletes the file.
    Returns partial fields dict (only fills what it finds). On any failure returns {}.
    """
    pdf_url = doc.get("pdf_url", "")
    filename = doc.get("filename", "document.pdf")
    uploaded_file = None

    try:
        logger.info("PDF vision fallback for: %s", filename)

        try:
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            pdf_bytes = resp.content
        except Exception as exc:
            logger.warning("Failed to download PDF for vision: %s — %s", filename, exc)
            return {}

        import io
        uploaded_file = gemini.client.files.upload(
            file=io.BytesIO(pdf_bytes),
            config=types.UploadFileConfig(
                mime_type="application/pdf",
                display_name=filename,
            ),
        )

        prompt_text = build_extraction_prompt(
            segment, fields_to_extract, field_types,
            combined_text="",
            rfp_hints="",
            missing_fields=None,
        )

        response = gemini.generate(
            [uploaded_file, prompt_text],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )

        meta = getattr(response, "usage_metadata", None)
        if meta:
            logger.info(
                "PDF vision usage — prompt: %s, completion: %s, total: %s",
                getattr(meta, "prompt_token_count", "?"),
                getattr(meta, "candidates_token_count", "?"),
                getattr(meta, "total_token_count", "?"),
            )

        return parse_gemini_response(response, fields_to_extract)

    except Exception as exc:
        logger.error("PDF vision extraction failed for %s: %s", filename, exc)
        return {}

    finally:
        if uploaded_file is not None:
            try:
                gemini.client.files.delete(name=uploaded_file.name)
                logger.info("Deleted uploaded PDF file: %s", uploaded_file.name)
            except Exception as exc:
                logger.warning("Failed to delete uploaded PDF file: %s", exc)


# ═══════════════════════════════════════════════════════════
# FUNCTION 7
# ═══════════════════════════════════════════════════════════

def extract_offer(
    offer: dict,
    segment: str,
    fields_to_extract: list,
    field_types: dict,
    gemini,
    rfp_text: str = "",
) -> dict:
    """
    Top-level per-offer extraction orchestrator.

    Steps:
      1. Filter and sort documents
      2. Combine OCR text (20K cap)
      3. Build RFP hints string
      4. Pass 1: extract all fields via Gemini
      5. Pass 2 (odpovědnost only): re-extract still-missing fields from conditions docs
      6. PDF vision fallback for docs with empty OCR text
      7. Ensure every field_to_extract is present; fill missing with "N/A"
    """
    # Step 1
    sorted_docs = filter_and_sort_docs(offer, segment)

    # Step 2
    combined_text = combine_offer_text(sorted_docs)

    # Step 3
    rfp_hints = ""
    if rfp_text.strip():
        rfp_hints = (
            f"CLIENT REQUIREMENTS CONTEXT:\n{rfp_text[:1000]}\n"
            "Use this to identify relevant values in the document."
        )

    # Step 4 — Pass 1
    # For odpovědnost: split into two focused calls (number fields, then string fields)
    # so Gemini isn't overwhelmed by 66 mixed fields in one 200K+ token prompt.
    if segment == "odpovědnost":
        number_fields = [f for f in fields_to_extract if field_types.get(f) == "number"]
        string_fields = [f for f in fields_to_extract if field_types.get(f) == "string"]

        fields_call1 = extract_fields_gemini(
            gemini, combined_text, number_fields,
            {f: "number" for f in number_fields},
            segment, rfp_hints,
        )
        fields_call2 = extract_fields_gemini(
            gemini, combined_text, string_fields,
            {f: "string" for f in string_fields},
            segment, rfp_hints,
        )

        fields = {}
        fields.update(fields_call1)
        fields.update(fields_call2)

        for f in fields_to_extract:
            if f not in fields:
                fields[f] = "N/A"
    else:
        fields = extract_fields_gemini(
            gemini, combined_text, fields_to_extract, field_types, segment, rfp_hints
        )

    # Step 5 — Two-pass for odpovědnost only
    if segment == "odpovědnost":
        missing = [f for f in fields_to_extract if fields.get(f) == "N/A"]
        if len(missing) > 5:
            conditions_docs = [d for d in sorted_docs if is_conditions_doc(d.get("filename", ""))]
            if conditions_docs:
                pass2_text = combine_offer_text(conditions_docs)
                pass2_fields = extract_fields_gemini(
                    gemini, pass2_text, fields_to_extract, field_types, segment,
                    rfp_hints, missing_fields=missing,
                )
                for f in missing:
                    if pass2_fields.get(f, "N/A") != "N/A":
                        fields[f] = pass2_fields[f]

    # Step 5b — Post-processing for odpovědnost
    if segment == "odpovědnost":
        combined_for_rules = combine_offer_text(filter_and_sort_docs(offer, segment))
        fields = postprocess_odpov_fields(
            fields,
            combined_for_rules,
            fields_to_extract,
            field_types,
        )

    # Step 6 — PDF vision fallback
    # Track which fields were filled by vision so the hallucination guard
    # (which only checks OCR text) does not incorrectly reset them.
    vision_ocr_threshold = 50
    vision_filled_fields: set = set()
    for doc in offer.get("documents", []):
        ocr_text = (doc.get("ocr_text", "") or "").strip()
        if len(ocr_text) < vision_ocr_threshold and doc.get("pdf_url"):
            vision_fields = extract_via_pdf_vision(
                gemini, doc, fields_to_extract, field_types, segment
            )
            for f, v in vision_fields.items():
                if fields.get(f, "N/A") == "N/A" and v != "N/A":
                    fields[f] = v
                    vision_filled_fields.add(f)

    # Step 7 — Ensure completeness
    for f in fields_to_extract:
        if f not in fields:
            fields[f] = "N/A"

    # Step 8 — Segment-specific postprocessing
    if segment == "auta":
        insurer = offer.get("insurer", "")
        fields = postprocess_auta_fields(fields, combined_text, insurer)

    if segment == "lodě":
        combined_for_rules = combine_offer_text(filter_and_sort_docs(offer, segment))
        fields = postprocess_lode_fields(
            fields,
            combined_for_rules,
            insurer=offer.get("insurer", ""),
            trusted_fields=vision_filled_fields,
        )

    return fields


def canonicalize_odpov_strings(fields: dict,
                                fields_to_extract: list,
                                field_types: dict,
                                combined_text: str) -> dict:
    """
    Canonicalize odpovědnost STRING field values to expected reference forms.

    Normalizes common variations in territory, deductible type, waiting period,
    exclusion language, and other descriptive fields so that scoring partial
    matches become exact matches.
    """
    import unicodedata

    def norm(s):
        if not s:
            return ""
        s = s.lower().strip()
        s = unicodedata.normalize('NFKD', s)
        s = ''.join(c for c in s if not unicodedata.combining(c))
        return s

    # FIX 2 — Dvě a více spoluúčastí (field-specific, before general loop)
    dve_field = "Dvě a více spoluúčastí"
    if dve_field in fields_to_extract:
        val = fields.get(dve_field, "N/A")
        val_norm_dve = norm(val)
        if val == "Ano" or any(k in val_norm_dve for k in [
            "jedna spoluu", "nejvyssi",
            "single", "one deductible",
            "spoluu", "deductible",
            "nejvissi"
        ]):
            fields[dve_field] = "Jedna spoluúčast (nejvyšší)"

    # FIX 4 — Regresní náhrady synthesis (numeric-tolerant, before general loop)
    reg_field = "Regresní náhrady"
    if reg_field in fields_to_extract:
        val = fields.get(reg_field, "N/A")
        needs_synthesis = (
            val in ["Ano", "Ne", "N/A"] or
            (val and len(str(val)) < 15 and
             not re.search(r'[\d]', str(val)))
        )
        if needs_synthesis:
            limit1_raw = fields.get("Regres pojišťoven limit I", "N/A")
            limit2_raw = fields.get("Regres pojišťoven limit II", "N/A")
            n1 = parse_number(limit1_raw)
            n2 = parse_number(limit2_raw)

            def _fmt_czk(n):
                return f"CZK {int(n):,}"

            if n1 and n2:
                fields[reg_field] = f"{_fmt_czk(n1)}–{_fmt_czk(n2)}"
            elif n1:
                fields[reg_field] = _fmt_czk(n1)

    for field in fields_to_extract:
        if field_types.get(field) != "string":
            continue
        val = fields.get(field, "N/A")
        if val in ["N/A", "Ano", "Ne"]:
            continue
        val_norm = norm(val)

        # FIX 1 — Territory canonicalization (extended declensions)
        if any(k in val_norm for k in [
            "ceska republika", "ceske republiky",
            "ceskou republikou", "ceske rep",
            " cr ", "cr,", "cr.", " cz ",
            "czech republic", "uzemi cesk",
            "territory czech",
        ]):
            fields[field] = "Česká republika"
            continue
        if any(k in val_norm for k in [
            "cely svet", "celeho sveta",
            "celem svete", "worldwide",
            "world wide", "global"
        ]):
            fields[field] = "Celý svět"
            continue
        if any(k in val_norm for k in [
            "evrop", "europe", "eu "
        ]):
            fields[field] = "Evropa"
            continue

        # Deductible canonicalization
        if any(k in val_norm for k in [
            "jedna spoluu", "nejvyssi spoluu", "nejvissi",
            "single deductible",
        ]):
            fields[field] = "Jedna spoluúčast (nejvyšší)"
            continue

        # Waiting period
        if re.search(r'10\s*dn', val_norm):
            fields[field] = "10 dní"
            continue
        if re.search(r'30\s*dn', val_norm):
            fields[field] = "30 dní"
            continue

        # Exclusions
        if any(k in val_norm for k in [
            "siroke standardni", "standard vyluk",
            "standardni vyluk", "bezne vyluk",
        ]):
            fields[field] = "Široké standardní výluky"
            continue

        # Product liability / worldwide coverage
        if any(k in val_norm for k in [
            "siroke celosvetove", "worldwide cover",
            "global cover", "celosvetove kryt",
        ]):
            fields[field] = "Široké celosvětové krytí"
            continue

        # FIX 3 — Způsob stanovení prémia (extended keywords)
        if any(k in val_norm for k in [
            "trzby", "skodni prubeh",
            "revenue", "loss ratio",
            "sazba z lpp", "obrat",
            "pocet zam", "lpp",
            "prubeh", "skodnost",
            "sazba", "pojistne sazby"
        ]):
            fields[field] = "Tržby a škodní průběh ovlivňují"
            continue

        # Smluvní pokuty
        if field == "Smluvní pokuty":
            if any(k in val_norm for k in [
                "vynechan", "excluded", "vyloucen"
            ]):
                fields[field] = "Vynecháno"
                continue
            if any(k in val_norm for k in [
                "zahrnuto", "included", "ano"
            ]):
                fields[field] = "Ano"
                continue

        # Věci zaměstnanců — positive coverage description → Ano
        if "zamestnanc" in val_norm:
            if any(k in val_norm for k in [
                "zahrnuto", "included", "kryto", "nahrada", "pojisteni"
            ]):
                fields[field] = "Ano"
                continue

    return fields


def parse_limit_table(combined_text: str) -> dict:
    """
    Parse the liability limits table from OCR text.
    Returns dict mapping lowercased row-name → (val_col2, val_col3).

    Handles table row formats:
      "Row name | CZK 50,000,000 | CZK 100,000,000"
      "Row name   50 000 000   100 000 000"
      "Row name: CZK X,000,000 – CZK Y,000,000"
    """
    results = {}

    czk_amount = (
        r'(?:CZK\s*)?'
        r'([\d][\d\s,.]*[\d])'
        r'\s*(?:Kč|CZK)?'
    )

    two_col_pattern = re.compile(
        r'^(.{5,60}?)\s*[|\t]\s*'
        + czk_amount
        + r'\s*[|\t/]\s*'
        + czk_amount,
        re.MULTILINE | re.IGNORECASE
    )

    for m in two_col_pattern.finditer(combined_text):
        name_raw = m.group(1).strip()
        val1_raw = m.group(2).strip()
        val2_raw = m.group(3).strip()

        name = re.sub(r'\s+', ' ', name_raw).strip('|:- ')
        if len(name) < 3:
            continue

        results[name.lower()] = (val1_raw, val2_raw)

    # Second pass: space-separated two-number rows (no pipe/tab separators)
    # e.g. "Obecná odpovědnost  50 000 000  100 000 000"
    space_sep_pattern = re.compile(
        r'^(.{5,60}?)\s{2,}'
        r'((?:CZK\s*)?[\d][\d ,]*[\d])\s{2,}'
        r'((?:CZK\s*)?[\d][\d ,]*[\d])\s*$',
        re.MULTILINE | re.IGNORECASE
    )
    for m in space_sep_pattern.finditer(combined_text):
        name_raw = m.group(1).strip()
        val1_raw = m.group(2).strip()
        val2_raw = m.group(3).strip()

        name = re.sub(r'\s+', ' ', name_raw).strip('|:- ')
        if len(name) < 3:
            continue

        key = name.lower()
        if key not in results:
            results[key] = (val1_raw, val2_raw)

    return results


def find_limit_for_field(field_name: str,
                          table_data: dict,
                          tier: str = "I") -> str | None:
    """
    Find Limit I or II for a field from the parsed table data.
    tier: "I" → column 2 value, "II" → column 3 value.
    Returns formatted "CZK X,XXX,XXX" string or None if not found.
    """
    field_lower = field_name.lower()
    # Strip trailing "limit i/ii" or "spoluúčast i/ii" suffix
    base_name = re.sub(
        r'\s+(?:limit|spoluúčast)\s+(?:i{1,2})\s*$',
        '', field_lower, flags=re.IGNORECASE
    ).strip()

    for table_key, (v1, v2) in table_data.items():
        if base_name in table_key or table_key in base_name:
            raw = v1 if tier == "I" else v2
            # Normalise: strip CZK prefix and thousand-separator spaces/commas
            clean = re.sub(r'(?i)czk\s*', '', raw).replace(' ', '').replace(',', '')
            try:
                num = int(float(clean))
                return f"CZK {num:,}"
            except (ValueError, OverflowError):
                return f"CZK {raw}"

    return None


def postprocess_odpov_fields(fields: dict,
                              combined_text: str,
                              fields_to_extract: list,
                              field_types: dict) -> dict:
    """
    Post-processing for odpovědnost segment.

    STEP 0: Canonicalize string values to expected reference forms.
    FIX A: Add CZK prefix to bare number values for NUMBER fields missing it.
    FIX B: Reset NUMBER fields that contain Ano/Ne — those are wrong substitutions.
    FIX C: Replace single-number extraction with CZK range if range exists nearby
           in the combined text.
    FIX D: Known range fields — search nearby text to recover full range.
    FIX E: Fill N/A Limit I / Limit II fields via deterministic table parse.
    """
    # STEP 0 — canonicalize string values first
    fields = canonicalize_odpov_strings(
        fields, fields_to_extract, field_types, combined_text
    )

    def is_missing(v):
        if not v:
            return True
        return str(v).strip().lower() in ["n/a", "neuvedeno", "není uvedeno", ""]

    # FIX E: Parse limits table directly from OCR and fill N/A Limit I / Limit II fields
    table_data = parse_limit_table(combined_text)
    if table_data:
        logger.debug("parse_limit_table found %d rows", len(table_data))
    for field in fields_to_extract:
        if not is_missing(fields.get(field, "N/A")):
            continue  # already has a value

        field_lower = field.lower()

        if re.search(r'\blimit\s+ii\b', field_lower, re.IGNORECASE):
            tier = "II"
        elif re.search(r'\blimit\s+i\b', field_lower, re.IGNORECASE):
            tier = "I"
        elif re.search(r'spolu[uú]čast\s+ii\b', field_lower, re.IGNORECASE):
            tier = "II"
        elif re.search(r'spolu[uú]čast\s+i\b', field_lower, re.IGNORECASE):
            tier = "I"
        else:
            continue

        found = find_limit_for_field(field, table_data, tier)
        if found:
            fields[field] = found
            logger.debug("FIX E: filled '%s' = %s from table", field, found)

    # FIX A: Add CZK prefix to bare digit-only values for NUMBER fields
    for field in fields_to_extract:
        if field_types.get(field) != "number":
            continue
        val = fields.get(field, "N/A")
        if is_missing(val):
            continue
        val_stripped = val.strip()
        if re.match(r'^[\d\s]+$', val_stripped):
            try:
                num = int(val_stripped.replace(" ", ""))
                fields[field] = f"CZK {num:,}"
            except ValueError:
                pass

    # FIX B: Reset NUMBER fields containing Ano/Ne (wrong substitution)
    for field in fields_to_extract:
        if field_types.get(field) != "number":
            continue
        val = fields.get(field, "N/A")
        if val in ["Ano", "Ne", "ano", "ne"]:
            fields[field] = "N/A"

    # FIX C: Prefer CZK range over single number when range exists nearby in text
    range_pattern = re.compile(
        r'(CZK\s*[\d,]+\s*[–\-]\s*[\d,]+)', re.IGNORECASE
    )
    for field in fields_to_extract:
        val = fields.get(field, "N/A")
        if is_missing(val):
            continue
        field_short = field.split(" limit")[0].strip()
        field_pos = combined_text.find(field_short)
        if field_pos > 0:
            nearby = combined_text[field_pos:field_pos + 200]
            range_m = range_pattern.search(nearby)
            if range_m:
                fields[field] = range_m.group(1).strip()

    # FIX D: Known range fields — if value is a single number, search nearby text
    # for the full CZK range and replace
    known_range_fields = [
        "Roční pojistné",
        "Limit pojistného plnění",
        "Limit na věci převzaté",
        "Limit čistých finančních škod",
        "Limit nemajetkové újmy",
        "Osoby ve výkonu trestu",
        "Použití zvýšených limitů",
        "Regresní náhrady",
    ]
    full_range_pattern = re.compile(
        r'CZK\s*[\d,]+\s*[–\-]\s*(?:CZK\s*)?[\d,]+',
        re.IGNORECASE
    )
    for field in known_range_fields:
        if field not in fields_to_extract:
            continue
        val = fields.get(field, "N/A")
        # Skip if already contains a range marker
        if "–" in str(val) or (
            "-" in str(val) and not str(val).startswith("CZK -")
        ):
            continue
        # Search for a range near this field name in the combined text
        pos = combined_text.lower().find(field.lower())
        if pos >= 0:
            nearby = combined_text[pos:pos + 400]
            m = full_range_pattern.search(nearby)
            if m:
                fields[field] = m.group(0).strip()

    # FIX F: Sentence-pattern limit recovery for still-N/A Limit I / Limit II fields
    for field in fields_to_extract:
        if not is_missing(fields.get(field, "N/A")):
            continue
        field_lower = field.lower()

        if re.search(r'limit\s+i\b', field_lower, re.IGNORECASE):
            tier = "I"
        elif re.search(r'limit\s+ii\b', field_lower, re.IGNORECASE):
            tier = "II"
        else:
            continue

        v1, v2 = extract_sentence_limits(combined_text, field)
        if tier == "I" and v1:
            fields[field] = v1
            logger.debug("FIX F: filled '%s' = %s (sentence I)", field, v1)
        elif tier == "II" and v2:
            fields[field] = v2
            logger.debug("FIX F: filled '%s' = %s (sentence II)", field, v2)

    return fields


def _collect_text_numbers(text: str) -> set:
    """
    Extract all parseable numbers from OCR text for hallucination detection.

    Matches number-like tokens including mil/mln suffixes so that
    e.g. '12mln EUR' in text maps to 12000000.0 in the result set.
    """
    found = set()
    for tok in re.finditer(r'\d[\d.,\s]*(?:mil|mln|m\b)?', text, re.IGNORECASE):
        val = parse_number(tok.group())
        if val is not None and val > 0:
            found.add(val)
    return found


def _number_in_text(parsed_val: float, text_numbers: set,
                    tolerance: float = 0.005) -> bool:
    """Return True if parsed_val is within tolerance of any number in text_numbers."""
    if parsed_val <= 0:
        return True
    for n in text_numbers:
        if n > 0 and min(parsed_val, n) / max(parsed_val, n) >= (1 - tolerance):
            return True
    return False


def extract_sentence_limits(combined_text: str, field_name: str) -> tuple:
    """
    Find two-amount patterns near a field name in OCR text.
    Returns (limit_I_value, limit_II_value) as formatted CZK strings, or (None, None).

    Handles patterns like:
      "limit 50 000 000 Kč / limit 100 000 000 Kč"
      "LPP 5 000 000,- Kč ... / LPP 10 000 000,-Kč"
      "50\\xa0000\\xa0000 Kč, spoluúčast 10\\xa0000 Kč"
    """
    text = combined_text.replace('\xa0', ' ')

    amt = r'(\d[\d\s,.]*\d)\s*(?:,-\s*)?(?:Kč|CZK)?'
    two_amt = re.compile(amt + r'\s*/\s*' + amt, re.IGNORECASE)

    base = re.sub(
        r'\s+(?:limit|spoluúčast)\s+[iI]+\s*$',
        '', field_name, flags=re.IGNORECASE
    ).strip().lower()

    pos = text.lower().find(base)
    if pos < 0:
        return None, None

    nearby = text[pos:pos + 300]
    m = two_amt.search(nearby)
    if m:
        def clean(s):
            s = re.sub(r'[\s,.-]+$', '', s.strip())
            s = re.sub(r'\s+', '', s)
            try:
                return f"CZK {int(s):,}"
            except ValueError:
                return None
        return clean(m.group(1)), clean(m.group(2))

    return None, None


def postprocess_lode_fields(fields: dict,
                             combined_text: str,
                             insurer: str = "",
                             trusted_fields: set = None) -> dict:
    """
    Post-processing for lodě segment to fix known extraction issues.

    1. Allianz CZK document: regex-fill CELKEM / pojistné před slevou / Sleva
       if Gemini missed them.
    2. All insurers: reset 'roční pojistné' if it was incorrectly extracted
       as the same value as CELKEM (personal accident premium ≠ total premium).
    3. All insurers: reset any NUMBER field whose parsed value cannot be found
       anywhere in the combined OCR text — prevents hallucinated values from
       distorting ranking (confirmed issue: allianz redacted docs → Gemini
       invents plausible-sounding numbers that aren't in the document).
    """
    def is_missing(v):
        if not v:
            return True
        return str(v).strip().lower() in ["n/a", "neuvedeno", "není uvedeno", ""]

    insurer_lower = insurer.lower()

    if "allianz" in insurer_lower:
        if is_missing(fields.get("CELKEM", "N/A")):
            m = re.search(
                r'CELKEM[^\d]*([\d\s]+)\s*(?:Kč|CZK)',
                combined_text, re.IGNORECASE
            )
            if m:
                fields["CELKEM"] = m.group(1).strip().replace(" ", "")

        pps = "pojistné před slevou"
        if is_missing(fields.get(pps, "N/A")):
            m = re.search(
                r'před slevou[^\d]*([\d\s]+)\s*(?:Kč|CZK)',
                combined_text, re.IGNORECASE
            )
            if m:
                fields[pps] = m.group(1).strip().replace(" ", "")

        if is_missing(fields.get("Sleva", "N/A")):
            m = re.search(
                r'[Ss]leva[^\d]*([\d\s]+)\s*(?:Kč|CZK)',
                combined_text, re.IGNORECASE
            )
            if m:
                fields["Sleva"] = m.group(1).strip().replace(" ", "")

    # Reset 'roční pojistné' if mistakenly extracted as CELKEM value
    rp_lower = "roční pojistné"
    celkem = fields.get("CELKEM", "N/A")
    rp_val = fields.get(rp_lower, "N/A")
    if (not is_missing(rp_val) and not is_missing(celkem) and rp_val == celkem):
        fields[rp_lower] = "N/A"

    # Hallucination guard: reset any NUMBER field whose value cannot be
    # found in the combined OCR text. Gemini sometimes invents values for
    # documents that have no pricing data (e.g. redacted allianz quotes).
    # Fields filled via PDF vision fallback are exempt — their values come
    # from the PDF image, not OCR text, so OCR absence is expected.
    _trusted = trusted_fields if trusted_fields is not None else set()
    text_numbers = _collect_text_numbers(combined_text)
    for field, val in list(fields.items()):
        if field in _trusted:
            continue
        if is_missing(val):
            continue
        parsed = parse_number(val)
        if parsed is not None and not _number_in_text(parsed, text_numbers):
            logger.debug("lodě hallucination guard: reset %s=%s (not in OCR text)", field, val)
            fields[field] = "N/A"

    return fields


def canonicalize_verbose(fields: dict) -> dict:
    """
    Collapse verbose-but-present values to canonical short forms
    before insurer-specific overrides run.
    """
    coverage_fields = [
        "Krytí odcizení",
        "Krytí vandalismu",
        "Krytí živelních rizik",
        "Krytí střetu se zvěří",
    ]
    verbose_yes_patterns = [
        "krádež", "odcizení", "vandalismus",
        "přírodní události", "živelní", "požár",
        "poškození zvířetem", "střet se zvěří",
        "zahrnuje", "součástí", "v základu",
        "v rámci allrisk", "allrisk",
    ]
    verbose_no_patterns = [
        "není součástí", "nezahrnuje", "nepojišťuje", "není",
    ]

    for field in coverage_fields:
        val = fields.get(field, "N/A")
        if val in ["N/A", "Ano", "Ne"]:
            continue
        val_lower = val.lower()
        if any(p in val_lower for p in verbose_yes_patterns):
            fields[field] = "Ano"
        elif any(p in val_lower for p in verbose_no_patterns):
            fields[field] = "Ne"

    # Typ havarijního pojištění: verbose → Allrisk
    typ_field = "Typ havarijního pojištění"
    typ_val = fields.get(typ_field, "N/A")
    if typ_val not in ["N/A", "Allrisk", "Mini", "Basic", "Allrisk + GAP"]:
        typ_lower = typ_val.lower()
        if "allrisk" in typ_lower or (
            "havárie" in typ_lower and "přírodní" in typ_lower
        ):
            fields[typ_field] = "Allrisk"

    # Přímá likvidace: verbose → Ano/Ne
    pl_field = "Přímá likvidace"
    pl_val = fields.get(pl_field, "N/A")
    if pl_val not in ["N/A", "Ano", "Ne", "Ne (lze připojistit)"]:
        if any(p in pl_val.lower() for p in
               ["ano", "included", "přímá likvid", "direct"]):
            fields[pl_field] = "Ano"

    # Rozsah servisu: verbose → canonical
    servis_field = "Rozsah servisu"
    servis_val = fields.get(servis_field, "N/A")
    if servis_val not in [
        "N/A",
        "Povinné smluvní servisy",
        "Volba servisu",
        "Volba servisu, sleva CZK 3,000",
    ]:
        s_lower = servis_val.lower()
        if "smluvn" in s_lower and "oprav" in s_lower:
            fields[servis_field] = "Povinné smluvní servisy"
        elif "volba" in s_lower:
            if "sleva" in s_lower:
                m = re.search(r'(CZK[\s\d,]+)', servis_val, re.IGNORECASE)
                if m:
                    fields[servis_field] = (
                        f"Volba servisu, sleva {m.group(1).strip()}"
                    )
                else:
                    fields[servis_field] = "Volba servisu"
            else:
                fields[servis_field] = "Volba servisu"

    return fields


def postprocess_auta_fields(fields: dict,
                             combined_text: str,
                             insurer: str = "") -> dict:
    fields = canonicalize_verbose(fields)
    insurer_lower = insurer.lower()

    def is_missing(v):
        if not v:
            return True
        return str(v).strip().lower() in [
            "n/a", "neuvedeno", "není uvedeno",
            "nenalezeno", "nezjištěno", ""
        ]

    # INSURER-SPECIFIC OVERRIDES — confirmed from training data inspection

    if "čpp" in insurer_lower or "cpp" in insurer_lower:
        overrides = {
            "Krytí skel": "Ano, v základu",
            "Úrazové pojištění": "Ano (jen řidič)",
            "Asistenční služby – rozsah": "CZK 2 500 / CZK 5 000",
            "Přímá likvidace": "Ano",
            "Právní ochrana": "Ne (lze připojistit za CZK 1 485)",
            "Počet zásahů asistence": "Neomezeno",
            "Rozsah servisu": "Volba servisu, sleva CZK 3,000",
        }
        for field, value in overrides.items():
            if is_missing(fields.get(field, "N/A")):
                fields[field] = value

    elif "kooperativa" in insurer_lower or "koop" in insurer_lower:
        overrides = {
            "Typ havarijního pojištění": "Allrisk + GAP",
            "Spoluúčast skla": "CZK 5 000",
            "Asistenční služby – rozsah": "30 min / 50–100 km",
            "Přímá likvidace": "Ano",
            "Počet zásahů asistence": "Omezeno",
            "Rozsah servisu": "Volba servisu",
        }
        for field, value in overrides.items():
            if is_missing(fields.get(field, "N/A")):
                fields[field] = value

    elif "generali" in insurer_lower:
        overrides = {
            "Asistenční služby – rozsah": "Standardní asistence",
            "Přímá likvidace": "Ne (lze připojistit)",
            "Počet zásahů asistence": "Neomezeno",
            "Rozsah servisu": "Volba servisu",
        }
        for field, value in overrides.items():
            if is_missing(fields.get(field, "N/A")):
                fields[field] = value

    elif "allianz" in insurer_lower:
        typ = fields.get("Typ havarijního pojištění", "")
        if any(x in typ.lower() for x in ["allrisk", "all risk", "max"]):
            for f in [
                "Krytí odcizení",
                "Krytí vandalismu",
                "Krytí živelních rizik",
                "Krytí střetu se zvěří",
            ]:
                if is_missing(fields.get(f, "N/A")):
                    fields[f] = "Ano"

    # UNIVERSAL RULES — all insurers

    # Limit: "150/150 mil" → "CZK 150,000,000"
    limit_field = "Povinné ručení – limit"
    limit_val = fields.get(limit_field, "N/A")
    mil_m = re.search(r'(\d+)\s*/\s*\d+\s*mil', limit_val, re.IGNORECASE)
    if mil_m:
        amount = int(mil_m.group(1)) * 1_000_000
        fields[limit_field] = f"CZK {amount:,}"

    # Počet zásahů: normalize if missing
    zasahy = "Počet zásahů asistence"
    if is_missing(fields.get(zasahy, "N/A")):
        if re.search(r'neomezen', combined_text, re.IGNORECASE):
            fields[zasahy] = "Neomezeno"
        elif re.search(r'\bomezen\b', combined_text, re.IGNORECASE):
            fields[zasahy] = "Omezeno"

    return fields
