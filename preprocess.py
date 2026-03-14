"""
Preprocessing utilities for insurance offer OCR text.
Clean, classify, prioritize, and build compact high-signal input for extraction.
Zero Gemini calls.
"""

import re
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Document type priority (lower = higher signal)
# ---------------------------------------------------------------------------
_TYPE_PRIORITY: dict[str, int] = {
    "rfp_nabidka": 0,
    "ujednani":    1,
    "smlouva_ps":  2,
    "vpp_pp":      3,
    "unknown":     4,
}

# Per-type char cap when combining offer text
_TYPE_CHAR_LIMITS: dict[str, int] = {
    "rfp_nabidka": 8_000,
    "ujednani":    6_000,
    "smlouva_ps":  6_000,
    "vpp_pp":      4_000,
    "unknown":     3_000,
}

# Hard cap on total combined text (keeps things token-efficient for later Gemini use too)
_MAX_COMBINED_CHARS = 20_000

# Pattern for page-marker lines
_PAGE_MARKER_RE = re.compile(
    r"^[ \t]*(Strana\s+\d+(\s+z\s+\d+)?|Page\s+\d+(\s+of\s+\d+)?|[-\u2013\u2014]\s*\d+\s*[-\u2013\u2014])[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)
# Isolated line that is only a number (page number artefact)
_LONE_NUMBER_RE = re.compile(r"(?m)^[ \t]*\d{1,3}[ \t]*$")


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """
    Return a cleaner version of raw OCR text:
    - normalize whitespace (spaces, tabs, special spaces)
    - normalize OCR currency variants: Kc / kc → Kč
    - normalize dash / bullet variants → hyphen
    - strip page markers (Strana N, Page N, isolated lone numbers)
    - deduplicate lines that repeat 3+ times (headers/footers)
    - collapse 3+ blank lines → one blank line
    """
    if not text:
        return ""

    # Special / non-breaking spaces → regular space
    text = re.sub(r"[\u00a0\u200b\u2009\u202f\u00ad]", " ", text)

    # Currency OCR normalization
    text = re.sub(r"\bKc\b", "Kč", text)
    text = re.sub(r"\bkc\b", "Kč", text)

    # Dash / bullet variants → simple hyphen
    text = re.sub(r"[–—•·▪▸►◆❑□■●]", "-", text)

    # Collapse horizontal whitespace within a line
    text = re.sub(r"[ \t]+", " ", text)

    # Strip page marker lines
    text = _PAGE_MARKER_RE.sub("", text)
    text = _LONE_NUMBER_RE.sub("", text)

    # Deduplicate lines that appear 3+ times (repeated headers/footers)
    lines = text.split("\n")
    counts: Counter = Counter(ln.strip() for ln in lines if ln.strip())
    seen: set = set()
    deduped: list[str] = []
    for ln in lines:
        stripped = ln.strip()
        if stripped and counts[stripped] >= 3:
            if stripped not in seen:
                seen.add(stripped)
                deduped.append(ln)
            # repeated header/footer — skip
        else:
            deduped.append(ln)
    text = "\n".join(deduped)

    # Collapse runs of 3+ newlines → double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def detect_document_type(filename: str, text: str) -> str:
    """
    Classify a document into one of:
      "rfp_nabidka"  — offer / RFP
      "ujednani"     — specific agreement / ujednání
      "smlouva_ps"   — policy contract / pojistná smlouva
      "vpp_pp"       — general terms / VPP / PP
      "unknown"
    Uses filename first, then content hints.
    """
    fn = (filename or "").lower()
    # Strip file extension so dots don't interfere with suffix checks
    fn_base = re.sub(r"\.[^.]+$", "", fn)
    snippet = (text or "")[:2_000].lower()

    # --- filename-based hints (fast path) ---
    # Use simple substring / token checks; avoid \b since _ counts as \w
    if re.search(r"rfp|nabidka|nab[ií]dka", fn_base):
        return "rfp_nabidka"
    if re.search(r"ujednani|ujednán", fn_base):
        return "ujednani"
    # "smlouva" anywhere, or "ps" as a standalone token (segment bounded by [-_] or ^/$)
    if re.search(r"smlouva|pojistn[aá].smlouva", fn_base):
        return "smlouva_ps"
    if re.search(r"(^|[-_])ps($|[-_])", fn_base):
        return "smlouva_ps"
    # "vpp" or "pp" as standalone tokens, or podminky
    if re.search(r"vpp|podm[ií]nky", fn_base):
        return "vpp_pp"
    if re.search(r"(^|[-_])pp($|[-_])", fn_base):
        return "vpp_pp"

    # --- content-based hints (first 2 k, case-insensitive) ---
    if re.search(
        r"(nab[ií]?dka\s+poji[sš]?t[eě]?n[íi]?|rfp|request\s+for\s+proposal|nab[ií]?dka\s+poji[sš]?t)",
        snippet,
        re.IGNORECASE,
    ):
        return "rfp_nabidka"
    if re.search(r"ujednán[íi]?", snippet, re.IGNORECASE):
        return "ujednani"
    if re.search(
        r"(pojistn[aá]\s+smlouva|smlouva\s+[cč]\.?\s*\d)",
        snippet,
        re.IGNORECASE,
    ):
        return "smlouva_ps"
    if re.search(
        r"(v[sš]?eobecn[eé]\s+pojistn[eé]\s+podm[ií]?nky|pojistn[eé]\s+podm[ií]?nky\s+\w+\s*[-–]|vpp\s+\d|^pp\s+\d)",
        snippet,
        re.IGNORECASE | re.MULTILINE,
    ):
        return "vpp_pp"

    # --- broader fallback scan (first 5 k) ---
    broader = (text or "")[:5_000].lower()
    if re.search(r"(nab[ií]?dka|nab\.)", broader):
        return "rfp_nabidka"
    if re.search(r"ujednán[íi]?", broader):
        return "ujednani"
    if re.search(r"pojistn[aá]\s+smlouva", broader):
        return "smlouva_ps"
    if re.search(r"(vpp|v[sš]?eobecn[eé]\s+pojistn[eé]|pojistn[eé]\s+podm[ií]?nky)", broader):
        return "vpp_pp"

    return "unknown"


def prioritize_documents(documents: list[dict]) -> list[dict]:
    """
    Return a new list of documents sorted highest-signal first:
      1. rfp_nabidka
      2. ujednani
      3. smlouva_ps
      4. vpp_pp
      5. unknown

    Each returned dict is a shallow copy of the original with an added
    "_doc_type" key.  The caller can remove it if not needed.
    """
    def _with_type(doc: dict) -> dict:
        filename = doc.get("filename") or ""
        text = doc.get("ocr_text") or ""
        doc_type = detect_document_type(filename, text)
        return {**doc, "_doc_type": doc_type}

    typed = [_with_type(d) for d in documents]
    typed.sort(key=lambda d: _TYPE_PRIORITY.get(d.get("_doc_type", "unknown"), 99))
    return typed


def extract_keyword_windows(
    text: str,
    keywords: list[str],
    window_chars: int = 160,
) -> list[str]:
    """
    Return a deduplicated list of short snippets, each centred on an
    occurrence of any keyword in *text*.

    Overlapping windows are merged before extraction.
    Useful for building compact context for later Gemini calls.
    """
    if not text or not keywords:
        return []

    text_lower = text.lower()
    ranges: list[tuple[int, int]] = []

    for kw in keywords:
        kw_lower = kw.lower()
        start = 0
        while True:
            pos = text_lower.find(kw_lower, start)
            if pos == -1:
                break
            lo = max(0, pos - window_chars // 4)
            hi = min(len(text), pos + len(kw) + window_chars)
            ranges.append((lo, hi))
            start = pos + 1

    if not ranges:
        return []

    # Merge overlapping / adjacent ranges
    ranges.sort()
    merged: list[list[int]] = [list(ranges[0])]
    for lo, hi in ranges[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    snippets = [text[lo:hi].strip() for lo, hi in merged]

    # Deduplicate identical snippets
    seen: set = set()
    unique: list[str] = []
    for s in snippets:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def build_preferred_offer_text(documents: list[dict]) -> str:
    """
    Build a compact, high-signal combined text ready for regex extraction:
      1. Clean each document's OCR text.
      2. Prioritize by document type (rfp_nabidka first, VPP last).
      3. Concatenate in priority order, applying per-type char limits.
      4. Hard-cap total output at _MAX_COMBINED_CHARS.

    Returns a single string.  If high-signal docs already cover the key
    fields, VPP content is appended only if space remains.
    """
    if not documents:
        return ""

    prioritized = prioritize_documents(documents)

    parts: list[str] = []
    total = 0
    for doc in prioritized:
        raw = doc.get("ocr_text") or ""
        if not raw.strip():
            continue

        cleaned = clean_ocr_text(raw)
        doc_type = doc.get("_doc_type", "unknown")
        char_limit = _TYPE_CHAR_LIMITS.get(doc_type, 3_000)

        remaining_budget = _MAX_COMBINED_CHARS - total
        if remaining_budget <= 0:
            break

        chunk = cleaned[: min(char_limit, remaining_budget)]
        if not chunk.strip():
            continue

        parts.append(chunk)
        total += len(chunk)

    return "\n\n".join(parts)


def get_offer_text_debug(documents: list[dict]) -> dict:
    """
    Return diagnostic info about how documents for one offer were processed.
    Not exposed in the API response — for logging / testing only.
    """
    if not documents:
        return {"doc_types": [], "combined_len": 0, "vpp_included": False}

    prioritized = prioritize_documents(documents)
    doc_types = [d.get("_doc_type", "unknown") for d in prioritized]
    combined = build_preferred_offer_text(documents)
    return {
        "doc_types": doc_types,
        "combined_len": len(combined),
        "vpp_included": "vpp_pp" in doc_types,
    }
