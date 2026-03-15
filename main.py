"""
Challenge 1: Porovnání pojistných nabídek (Insurance Offer Comparison)

Input:  Multiple insurance offers with OCR text, plus a list of fields to extract
Output: Parsed fields per offer, ranking, best offer identification
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()

from google import genai
import psycopg2
from fastapi import FastAPI
import uvicorn

import cache
import extract
import rank

app = FastAPI(title="Challenge 1: Insurance Offer Comparison")

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class GeminiTracker:
    """Wrapper around Gemini that tracks token usage."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        self.enabled = bool(api_key)
        self.model_name = model_name
        self.client = None
        if self.enabled:
            self.client = genai.Client(api_key=api_key)
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._lock = threading.Lock()

    def generate(self, prompt, **kwargs):
        if not self.enabled:
            raise RuntimeError("Gemini API key not configured")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            **kwargs,
        )
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

GEMINI_MODEL_VERSION = "gemini-2.5-flash-split-v1"


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


def _get_db_safe():
    try:
        return get_db()
    except Exception:
        return None


def _solve_core(payload: dict, gemini_client, conn) -> dict:
    segment = payload.get("segment", "")
    fields_to_extract = payload.get("fields_to_extract", [])
    field_types = payload.get("field_types", {})
    offers = payload.get("offers", [])
    rfp_text = (payload.get("rfp") or {}).get("ocr_text", "")

    def process_one(offer):
        key = cache.compute_cache_key(
            offer, segment, fields_to_extract, GEMINI_MODEL_VERSION
        )
        cached = cache.load_cached_result(conn, key) if conn else None
        if cached:
            logging.info(f"Cache hit for offer {offer.get('id')}")
            return cached

        fields = extract.extract_offer(
            offer, segment, fields_to_extract,
            field_types, gemini_client, rfp_text
        )
        parsed = {
            "id": offer["id"],
            "insurer": offer.get("insurer", ""),
            "fields": fields
        }
        if conn:
            cache.save_cached_result(conn, key, parsed)
        return parsed

    with ThreadPoolExecutor(max_workers=4) as executor:
        offers_parsed = list(executor.map(process_one, offers))

    ranking = rank.rank_offers_dynamic(
        offers_parsed, fields_to_extract, field_types, rfp_text
    )

    return {
        "offers_parsed": offers_parsed,
        "ranking": ranking,
        "best_offer_id": ranking[0] if ranking else None
    }


@app.post("/solve")
def solve(payload: dict):
    """
    Compare insurance offers and identify the best option.

    The input contains:
    - segment: insurance segment (odpovědnost, auta, lodě, majetek, ...)
    - fields_to_extract: list of field names to extract from each offer
    - field_types: dict mapping field name → "number" or "string"
    - offers: list of offers, each with id, insurer, label, and documents
    - rfp: (optional) request for proposal document

    Expected output:
    {
        "offers_parsed": [
            {
                "id": "allianz",
                "insurer": "Allianz",
                "fields": {
                    "Roční pojistné": "125000",
                    "Povinné ručení – limit": "100000000",
                    ...
                }
            }
        ],
        "ranking": ["allianz", "generali", ...],
        "best_offer_id": "allianz"
    }
    """
    return _solve_core(payload, gemini, _get_db_safe())


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
