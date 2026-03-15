"""
Per-offer PostgreSQL cache helpers.

Cache key is a SHA-256 of normalized offer content (id, insurer, segment,
document filenames, OCR text). Ranking output is never part of the key.

The existing `cache(key TEXT PRIMARY KEY, value JSONB, created_at TIMESTAMPTZ)`
table is reused without schema changes.
"""

import hashlib
import json
import logging
import re

logger = logging.getLogger(__name__)


def _norm_ws(s: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return re.sub(r"\s+", " ", s or "").strip()


def canonicalize_offer_documents(documents: list) -> str:
    """
    Produce a deterministic string representation of the documents list.

    Documents are sorted by filename so JSON array order doesn't affect the key.
    Whitespace in filenames and OCR text is normalized before hashing.
    """
    parts = []
    for doc in sorted(documents, key=lambda d: _norm_ws(d.get("filename") or "")):
        filename = _norm_ws(doc.get("filename") or "")
        ocr_text = _norm_ws(doc.get("ocr_text") or "")
        parts.append(f"file:{filename}|text:{ocr_text}")
    return "\n".join(parts)


def compute_offer_cache_key(offer: dict, segment: str = "") -> str:
    """
    Compute a deterministic SHA-256 cache key for one offer.

    Inputs included in the key:
      - offer id (if present)
      - insurer (if present)
      - segment
      - normalized document filenames
      - normalized OCR text content

    Ranking output and parsed fields are intentionally excluded.
    """
    offer_id = _norm_ws(offer.get("id") or "")
    insurer = _norm_ws(offer.get("insurer") or "")
    seg = _norm_ws(segment)
    doc_canon = canonicalize_offer_documents(offer.get("documents") or [])

    raw = f"id:{offer_id}|insurer:{insurer}|segment:{seg}\n{doc_canon}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cached_offer(db_conn, cache_key: str):
    """
    Try to load a previously parsed offer dict from the cache table.

    Returns the parsed offer dict on a hit, or None on miss / error.
    Never raises — all exceptions are caught and logged as warnings.
    """
    try:
        with db_conn.cursor() as cur:
            cur.execute("SELECT value FROM cache WHERE key = %s", (cache_key,))
            row = cur.fetchone()
        if row is None:
            return None
        value = row[0]
        # psycopg2 may return JSONB as dict or as str depending on adapter registration
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, dict):
            logger.warning(
                "[cache] malformed cached value for key=%s… (type=%s), ignoring",
                cache_key[:12],
                type(value).__name__,
            )
            return None
        return value
    except Exception as exc:
        logger.warning("[cache] load failed for key=%s…: %s", cache_key[:12], exc)
        return None


def save_cached_offer(db_conn, cache_key: str, parsed_offer: dict) -> None:
    """
    Upsert the parsed offer dict into the cache table.

    Never raises — all exceptions are caught, logged, and the transaction is
    rolled back so the connection stays usable.
    """
    try:
        with db_conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cache (key, value)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
                """,
                (cache_key, json.dumps(parsed_offer)),
            )
        db_conn.commit()
    except Exception as exc:
        logger.warning("[cache] save failed for key=%s…: %s", cache_key[:12], exc)
        try:
            db_conn.rollback()
        except Exception:
            pass
