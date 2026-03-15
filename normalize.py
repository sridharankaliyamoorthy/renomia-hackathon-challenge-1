"""
normalize.py — Number parsing and text normalization utilities.

Used by rank.py for ranking comparisons. Never used to modify output values —
extracted field values are always returned verbatim to the scorer.
"""

import re
import unicodedata
from typing import Optional


def clean_ocr_text(text: str) -> str:
    """
    Clean LaTeX-style OCR artifacts from scanned Czech insurance documents.

    Replaces encoded diacritics (e.g. \\tilde{c} -> č, \\v{s} -> š) and
    removes stray LaTeX commands and tilde separators that appear in raw OCR output.
    """
    if not text:
        return text
    replacements = {
        r'\\tilde\{c\}': 'č',
        r'\\tilde\{C\}': 'Č',
        r'\\tilde\{s\}': 'š',
        r'\\tilde\{S\}': 'Š',
        r'\\tilde\{z\}': 'ž',
        r'\\tilde\{Z\}': 'Ž',
        r'\\tilde\{r\}': 'ř',
        r'\\tilde\{R\}': 'Ř',
        r'\\acute\{e\}': 'é',
        r'\\acute\{a\}': 'á',
        r'\\acute\{i\}': 'í',
        r'\\acute\{o\}': 'ó',
        r'\\acute\{u\}': 'ú',
        r'\\acute\{y\}': 'ý',
        r'\\acute\{E\}': 'É',
        r'\\acute\{A\}': 'Á',
        r'\\acute\{I\}': 'Í',
        r'\\acute\{U\}': 'Ú',
        r'\\acute\{Y\}': 'Ý',
        r'\\v\{c\}': 'č',
        r'\\v\{C\}': 'Č',
        r'\\v\{s\}': 'š',
        r'\\v\{S\}': 'Š',
        r'\\v\{z\}': 'ž',
        r'\\v\{Z\}': 'Ž',
        r'\\v\{r\}': 'ř',
        r'\\v\{R\}': 'Ř',
        r'\\v\{e\}': 'ě',
        r'\\v\{E\}': 'Ě',
        r'\\v\{n\}': 'ň',
        r'\\v\{d\}': 'ď',
        r'\\v\{t\}': 'ť',
        r'\\u\{u\}': 'ů',
        r'\\u\{U\}': 'Ů',
        r'\\%': '%',
        r'\\,': ' ',
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Tilde between digits = thousands separator space (e.g. "56~326")
    text = re.sub(r'(?<=\d)~(?=\d)', ' ', text)
    # Remaining tildes = space
    text = re.sub(r'~', ' ', text)
    # Remove remaining unknown LaTeX commands with braces
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    # Remove remaining bare LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\b', '', text)
    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def parse_number(s) -> Optional[float]:
    """
    Parse a Czech/EUR insurance value string into a float for ranking comparisons.

    Handles all confirmed formats from training data:
      "50 000 000 Kč"        -> 50000000.0
      "50.000.000,- Kč"      -> 50000000.0
      "CZK 150,000,000"      -> 150000000.0
      "CZK 248,923–281,136"  -> 248923.0   (range: first number)
      "459.35"               -> 459.35
      "342.11"               -> 342.11
      "EUR 342,11"           -> 342.11     (European decimal comma)
      "3 % / CZK 3 000"      -> 3000.0    (skip % token, take money value)
      "34851"                -> 34851.0
      "15562"                -> 15562.0
      "N/A"                  -> None
      ""                     -> None
      None                   -> None

    Algorithm (in order):
      1. None / empty / "N/A" -> None
      2. Strip % token if % present and other digits remain
      3. Strip currency symbols (CZK, EUR, Kč, €)
      4. Handle range: take content before en-dash/em-dash only
      5. Strip trailing noise (spaces, commas, periods, hyphens)
      6. European decimal comma: comma + exactly 2 digits at end -> decimal point
      7. Multiple separator groups (e.g. "50.000.000") -> remove all [,.]
      8. Single separator before 3 trailing digits -> thousand separator, remove
      9. Extract first valid float from remaining string
    """
    if s is None:
        return None
    s = clean_ocr_text(str(s)) if s else str(s)
    s = s.strip()
    if not s:
        return None
    if s.upper() == "N/A":
        return None

    # Step 2: Skip % token, keep the money value
    # e.g. "3 % / CZK 3 000" -> " / CZK 3 000"
    if "%" in s:
        without_pct = re.sub(r'\d+\s*%', '', s)
        if re.search(r'\d', without_pct):
            s = without_pct

    # Step 3: Strip currency symbols
    s = re.sub(r'(?i)\b(czk|eur)\b', ' ', s)
    s = re.sub(r'[Kč€]', ' ', s)

    # Step 4: Handle ranges — take content before en-dash or em-dash only
    # e.g. "248,923–281,136" -> "248,923"
    range_match = re.search(r'([\d][\d\s.,]*)[–—]', s)
    if range_match:
        s = s[:range_match.end(1)]

    # Step 5: Strip trailing noise: spaces, commas, periods, hyphens
    # e.g. "50.000.000,- " -> "50.000.000"
    s = re.sub(r'[\s,.\-]+$', '', s).strip()

    # Step 6: European decimal comma — comma followed by EXACTLY 2 digits at end
    # e.g. "342,11" -> "342.11"; "248,923" has 3 digits -> not a decimal comma
    if re.search(r',\d{2}$', s):
        s = re.sub(r',(\d{2})$', r'.\1', s)
        s = s.replace(' ', '')

    # Step 7: Multiple separator groups -> all are thousand separators
    # Detected by: digit-separator-3digits-separator pattern
    # e.g. "50.000.000", "150,000,000"
    elif re.search(r'\d{1,3}[.,]\d{3}[.,]', s):
        s = re.sub(r'[,.]', '', s)
        s = s.replace(' ', '')

    # Step 8: Single separator before exactly 3 trailing digits -> thousand separator
    # e.g. "248,923" -> "248923"
    elif re.search(r'[.,]\d{3}$', s):
        s = re.sub(r'[.,](\d{3})$', r'\1', s)
        s = s.replace(' ', '')

    else:
        # No recognized separator pattern — just remove spaces
        s = s.replace(' ', '')

    # Step 9: Extract first valid float from whatever remains
    match = re.search(r'\d+(?:\.\d+)?', s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def normalize_text_for_compare(s: str) -> str:
    """
    Normalize a string for fuzzy comparison in ranking.

    Used only for ranking logic — never to modify output values.
    Steps: lowercase -> strip diacritics (NFKD) -> collapse whitespace -> strip.

    Examples:
      "Česká republika" -> "ceska republika"
      "Allrisk"         -> "allrisk"
      "Celý svět"       -> "cely svet"
    """
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def is_conditions_doc(filename: str) -> bool:
    """
    Return True if the filename suggests a general conditions / legal text document.

    These docs contain policy terms but rarely contain pricing data.
    They are sorted LAST in document concatenation order.
    """
    if not filename:
        return False
    name = filename.lower()
    patterns = ["conditions", "vpp", "pp_", "podmínky", "podminky", "pyc", "glossary", "general"]
    return any(p in name for p in patterns)


def is_quotation_doc(filename: str) -> bool:
    """
    Return True if the filename suggests a pricing / quotation document.

    These docs contain the actual offer numbers and are sorted FIRST
    in document concatenation order to avoid truncation at the 20K cap.
    """
    if not filename:
        return False
    name = filename.lower()
    patterns = ["quotation", "nabídka", "nabidka", "kalkulace", "quote",
                "proposal", "application", "q15", "redigov"]
    return any(p in name for p in patterns)
