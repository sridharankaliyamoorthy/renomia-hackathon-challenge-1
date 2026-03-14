"""
Challenge 1: Porovnání pojistných nabídek (Insurance Offer Comparison)
Domain: Odpovědnost (Liability Insurance)

Input:  Multiple insurance offers with OCR text from documents
Output: Parsed parameters per offer, ranking, best offer identification
"""

import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
from fastapi import FastAPI
import uvicorn

from cache_utils import compute_offer_cache_key, load_cached_offer, save_cached_offer
from extractors import parse_offer_baseline
from segment_router import normalize_segment, solve_segment
from text_fields import enrich_text_fields_two_pass

logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Challenge 1: Insurance Offer Comparison")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.enabled = bool(api_key)
        if self.enabled:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        response = self.model.generate_content(prompt, **kwargs)
        with self._lock:
            self.request_count += 1
            meta = getattr(response, "usage_metadata", None)
            if meta:
                self.prompt_tokens += getattr(meta, "prompt_token_count", 0)
                self.completion_tokens += getattr(meta, "candidates_token_count", 0)
                self.total_tokens += getattr(meta, "total_token_count", 0)
        return response

    def get_metrics(self):
        with self._lock:
            return {
                "gemini_request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0


gemini = GeminiTracker(GEMINI_API_KEY)


def get_db():
    return psycopg2.connect(DATABASE_URL)


@app.on_event("startup")
def init_db():
    for _ in range(15):
        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                """CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )"""
            )
            conn.commit()
            cur.close()
            conn.close()
            return
        except Exception:
            time.sleep(1)


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return gemini.get_metrics()


@app.post("/metrics/reset")
def reset_metrics():
    gemini.reset()
    return {"status": "reset"}


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


def rank_offers(parsed_offers: list) -> list:
    """
    Deterministic ranking using only the 3 baseline numeric fields.

    Weights: 0.50 limit (higher better)
             0.25 deductible (lower better)
             0.25 premium (lower better)

    Tie-breakers (in order):
      1. higher basic_limit_czk
      2. lower premium_czk
      3. lower basic_deductible_czk
      4. original offer order
    """
    if not parsed_offers:
        return []

    limits = [o.get("basic_limit_czk") for o in parsed_offers]
    deductibles = [o.get("basic_deductible_czk") for o in parsed_offers]
    premiums = [o.get("premium_czk") for o in parsed_offers]

    limit_s = _minmax_norm(limits, higher_is_better=True)
    deduct_s = _minmax_norm(deductibles, higher_is_better=False)
    prem_s = _minmax_norm(premiums, higher_is_better=False)

    scores = [
        0.50 * limit_s[i] + 0.25 * deduct_s[i] + 0.25 * prem_s[i]
        for i in range(len(parsed_offers))
    ]

    _INF = float("inf")
    indexed = list(enumerate(parsed_offers))
    indexed.sort(
        key=lambda x: (
            -scores[x[0]],
            -(limits[x[0]] or 0),
            premiums[x[0]] if premiums[x[0]] is not None else _INF,
            deductibles[x[0]] if deductibles[x[0]] is not None else _INF,
            x[0],
        )
    )

    return [parsed_offers[i]["id"] for i, _ in indexed]


def _solve_core(payload: dict) -> dict:
    """
    Internal pipeline: parse, enrich, rank.

    Returns the public fields PLUS a '_debug' key for the local eval harness.
    The public /solve endpoint always strips '_debug' before responding.
    """
    offers = payload.get("offers") or []
    # Normalise so cache keys are consistent regardless of input spelling
    # (e.g. "odpovědnost" and "odpovednost" produce the same key).
    segment = normalize_segment(payload.get("segment") or "")

    db_conn = None
    try:
        db_conn = get_db()
    except Exception as exc:
        logger.warning("[solve] DB connection failed, caching disabled: %s", exc)

    db_available = db_conn is not None
    offers_parsed = []
    _debug_offers: list = []

    for o in offers:
        offer_id = o.get("id", "?")
        cache_key = compute_offer_cache_key(o, segment)

        # --- Cache lookup ---
        cached = None
        cache_status = "db_unavailable"
        if db_conn is not None:
            cached = load_cached_offer(db_conn, cache_key)
            cache_status = "hit" if cached is not None else "miss"

        if cached is not None:
            logger.info(
                "[solve] offer=%s cache=HIT key=%s… (Gemini skipped)", offer_id, cache_key[:12]
            )
            offers_parsed.append(cached)
            _debug_offers.append(
                {"id": offer_id, "cache_status": "hit", "gemini_called": False}
            )
            continue

        logger.info("[solve] offer=%s cache=%s key=%s…", offer_id, cache_status.upper(), cache_key[:12])

        # --- Parse and enrich ---
        _req_before = gemini.get_metrics()["gemini_request_count"]
        parsed = parse_offer_baseline(o)
        parsed = enrich_text_fields_two_pass(o, parsed, gemini)
        _req_after = gemini.get_metrics()["gemini_request_count"]

        # --- Persist to cache ---
        if db_conn is not None:
            save_cached_offer(db_conn, cache_key, parsed)

        offers_parsed.append(parsed)
        _debug_offers.append(
            {
                "id": offer_id,
                "cache_status": cache_status,
                "gemini_called": _req_after > _req_before,
            }
        )

    if db_conn is not None:
        try:
            db_conn.close()
        except Exception:
            pass

    ranking = rank_offers(offers_parsed)
    best_offer_id = ranking[0] if ranking else None

    return {
        "offers_parsed": offers_parsed,
        "ranking": ranking,
        "best_offer_id": best_offer_id,
        "_debug": {
            "db_available": db_available,
            "offers": _debug_offers,
        },
    }


@app.post("/solve")
def solve(payload: dict):
    """
    Compare insurance offers and identify the best option.

    Supported segments: odpovednost, auta, lode.
    Unknown segments return a valid null-field response with original-order ranking.

    Input example:
    {
        "offers": [
            {
                "id": "generali_current",
                "insurer": "Generali ČP",
                "label": "Stávající smlouva",
                "documents": [
                    {
                        "filename": "nabidka_generali.pdf",
                        "ocr_text": "... OCR extracted text ..."
                    }
                ]
            },
            {
                "id": "csob_1",
                "insurer": "ČSOB",
                "label": "ČSOB I.",
                "documents": [{"filename": "...", "ocr_text": "..."}]
            }
        ],
        "segment": "odpovědnost"
    }

    Expected output:
    {
        "offers_parsed": [...],
        "ranking": ["csob_1", "generali_current"],
        "best_offer_id": "csob_1"
    }
    """
    # Dispatch to the segment-specific solver; strips _debug before returning.
    result = solve_segment(payload, gemini)
    result.pop("_debug", None)  # safety guard — solve_segment already strips it
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
