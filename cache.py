import hashlib
import json
import logging

logger = logging.getLogger(__name__)


def compute_cache_key(offer, segment, fields_to_extract, model_version):
    """Return a SHA-256 hex string cache key for the given offer extraction request."""
    components = [
        offer["id"],
        offer.get("insurer", ""),
        segment,
        "|".join(sorted(fields_to_extract)),
        "|".join(sorted(
            hashlib.sha256(doc.get("ocr_text", "").encode()).hexdigest()
            for doc in offer.get("documents", [])
        )),
        model_version,
    ]
    raw = "|".join(components)
    return hashlib.sha256(raw.encode()).hexdigest()


def load_cached_result(conn, key):
    """Return cached dict for key, or None if not found or on any error."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM cache WHERE key = %s", (key,))
            row = cur.fetchone()
        if row is None:
            return None
        value = row[0]
        if isinstance(value, dict):
            return value
        return json.loads(value)
    except Exception:
        return None


def save_cached_result(conn, key, value):
    """Upsert value into cache table. Logs a warning on failure; never raises."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO cache (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, json.dumps(value)),
            )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to save cache key %s: %s", key, exc)
