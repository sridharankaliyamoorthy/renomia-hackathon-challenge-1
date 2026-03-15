"""
eval_local.py — Local evaluation script against the training database.

Usage:
    python eval_local.py
    python eval_local.py --segment odpovědnost
"""

import argparse
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

import psycopg2
from difflib import SequenceMatcher

from normalize import parse_number, normalize_text_for_compare
from main import _solve_core, gemini, GEMINI_MODEL_VERSION
import cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DB = {
    "host": "35.234.124.49",
    "port": 5432,
    "dbname": "hackathon_training",
    "user": "hackathon_reader",
    "password": "ReadOnly2025hack",
}

LOCAL_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hackathon:hackathon@localhost:5432/hackathon"
)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 1: fetch_training_data
# ─────────────────────────────────────────────────────────────────────────────

def fetch_training_data(segment=None):
    """
    Fetch training rows from the remote training DB.

    Returns list of (input_data, expected_output) tuples.
    Optionally filtered to rows whose input matches the given segment.
    """
    conn = psycopg2.connect(**TRAINING_DB)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT input, expected_output FROM training_data WHERE challenge_id = 1"
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    results = []
    for input_raw, expected_raw in rows:
        input_data = input_raw if isinstance(input_raw, dict) else json.loads(input_raw)
        expected_output = expected_raw if isinstance(expected_raw, dict) else json.loads(expected_raw)

        if segment is not None:
            row_segment = input_data.get("segment", "")
            if row_segment != segment:
                continue

        results.append((input_data, expected_output))

    logger.info(f"Fetched {len(results)} training row(s)" +
                (f" for segment '{segment}'" if segment else ""))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 2: score_number_field
# ─────────────────────────────────────────────────────────────────────────────

def score_number_field(predicted, expected) -> float:
    """Score a number field: 0.0 to 1.0."""
    if not predicted or not expected:
        return 0.0

    # Range fields (e.g. 'CZK 248,923–281,136') — treat as string comparison
    exp_str = str(expected)
    if "–" in exp_str or ("-" in exp_str and not exp_str.strip().startswith("-")):
        return score_string_field(predicted, expected)

    p = parse_number(str(predicted))
    e = parse_number(str(expected))

    if p is None or e is None:
        return 0.0
    if p == e:
        return 1.0

    if max(p, e) == 0:
        return 1.0

    ratio = min(p, e) / max(p, e)
    if ratio >= 0.90:
        return 1.0
    if ratio >= 0.80:
        return 0.75
    if ratio >= 0.75:
        return 0.5
    if ratio >= 0.70:
        return 0.25
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 3: score_string_field
# ─────────────────────────────────────────────────────────────────────────────

def score_string_field(predicted, expected) -> float:
    """Score a string field: 0.0 to 1.0."""
    predicted_norm = normalize_text_for_compare(str(predicted))
    expected_norm = normalize_text_for_compare(str(expected))

    if predicted_norm == expected_norm:
        return 1.0

    ratio = SequenceMatcher(None, predicted_norm, expected_norm).ratio()
    if ratio > 0.5:
        return ratio
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 4: score_ranking
# ─────────────────────────────────────────────────────────────────────────────

def score_ranking(predicted_ranking: list, expected_ranking: list) -> float:
    """Score ranking order: 0.0 to 1.0."""
    if not expected_ranking:
        return 1.0

    scores = []
    for expected_pos, offer_id in enumerate(expected_ranking):
        try:
            predicted_pos = predicted_ranking.index(offer_id)
        except ValueError:
            predicted_pos = len(predicted_ranking)

        displacement = abs(expected_pos - predicted_pos)
        position_score = max(0.0, 1.0 - displacement * 0.25)
        scores.append(position_score)

    return sum(scores) / len(scores)


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 5: evaluate_segment
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_segment(input_data: dict, expected_output: dict, conn) -> dict:
    """
    Run _solve_core on input_data and compare result to expected_output.
    Returns a dict with all scores and diagnostic info.
    """
    segment = input_data.get("segment", "unknown")
    field_types = input_data.get("field_types", {})

    logger.info(f"Evaluating segment: {segment}")
    result = _solve_core(input_data, gemini, conn)

    # ── Debug print for odpovědnost first offer ───────────────────────────────
    if segment == 'odpovědnost':
        first = result['offers_parsed'][0] if result.get('offers_parsed') else None
        if first:
            na_count = sum(1 for v in first['fields'].values() if v == 'N/A')
            print(f'\nDEBUG odpov first offer: {first["id"]}')
            print(f'N/A count: {na_count}/66')
            # Build expected lookup for this offer
            exp_lookup = {}
            for o in expected_output.get('offers_parsed', []):
                if o['id'] == first['id']:
                    exp_lookup = o.get('fields', {})
            for f, v in first['fields'].items():
                exp_val = exp_lookup.get(f, 'NOT_IN_EXPECTED')
                if v == 'N/A':
                    score_marker = '← N/A'
                elif exp_val not in ('NOT_IN_EXPECTED', None):
                    score_marker = '← has value'
                else:
                    score_marker = ''
                print(f'  {repr(f)}: {repr(v)} | expected: {repr(exp_val)} {score_marker}')

    # Build lookup: offer_id -> fields dict (from predicted result)
    predicted_by_id = {
        o["id"]: o.get("fields", {})
        for o in result.get("offers_parsed", [])
    }

    # ── Field extraction score ────────────────────────────────────────────────
    field_scores = []   # list of (field_name, score)
    for offer in expected_output.get("offers_parsed", []):
        offer_id = offer["id"]
        predicted_fields = predicted_by_id.get(offer_id, {})

        for field, expected_val in offer.get("fields", {}).items():
            predicted_val = predicted_fields.get(field, "N/A")
            ftype = field_types.get(field, "string")

            if ftype == "number":
                score = score_number_field(predicted_val, expected_val)
            else:
                score = score_string_field(predicted_val, expected_val)

            field_scores.append((field, score))

    extraction_score = (
        sum(s for _, s in field_scores) / len(field_scores)
        if field_scores else 0.0
    )

    # ── Ranking score ─────────────────────────────────────────────────────────
    ranking_score = score_ranking(
        result.get("ranking", []),
        expected_output.get("ranking", [])
    )

    # ── Best offer score ──────────────────────────────────────────────────────
    best_score = (
        1.0
        if result.get("best_offer_id") == expected_output.get("best_offer_id")
        else 0.0
    )

    # ── Total score ───────────────────────────────────────────────────────────
    total = extraction_score * 0.60 + ranking_score * 0.25 + best_score * 0.15

    # ── Hardest fields (10 lowest scores) ────────────────────────────────────
    sorted_fields = sorted(field_scores, key=lambda x: x[1])
    hardest_fields = sorted_fields[:10]

    return {
        "segment": segment,
        "extraction_score": extraction_score,
        "ranking_score": ranking_score,
        "best_offer_score": best_score,
        "total_score": total,
        "hardest_fields": hardest_fields,
        "field_count": len(field_scores),
        "offer_count": len(expected_output.get("offers_parsed", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCTION 6: main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against training data")
    parser.add_argument(
        "--segment",
        type=str,
        default=None,
        help="Filter to a single segment (e.g. odpovědnost, auta, lodě)"
    )
    args = parser.parse_args()

    rows = fetch_training_data(segment=args.segment)
    if not rows:
        print("No training rows found.")
        return

    # Connect to local cache DB
    try:
        local_conn = psycopg2.connect(LOCAL_DATABASE_URL)
        # Ensure cache table exists
        cur = local_conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )"""
        )
        local_conn.commit()
        cur.close()
        logger.info("Connected to local cache DB")
    except Exception as e:
        logger.warning(f"Could not connect to local cache DB: {e}. Running without cache.")
        local_conn = None

    segment_results = []
    for input_data, expected_output in rows:
        result = evaluate_segment(input_data, expected_output, local_conn)
        segment_results.append(result)

    if local_conn:
        local_conn.close()

    # ── Print report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 46)
    print("EVALUATION RESULTS")
    print("=" * 46)

    total_weighted = 0.0
    for r in segment_results:
        print(f"\nSegment: {r['segment']}")
        print(f"  Fields scored:     {r['field_count']}")
        print(f"  Extraction score:  {r['extraction_score']:.3f}  (weight 60%)")
        print(f"  Ranking score:     {r['ranking_score']:.3f}  (weight 25%)")
        print(f"  Best offer score:  {r['best_offer_score']:.3f}  (weight 15%)")
        print(f"  TOTAL SCORE:       {r['total_score']:.3f}")

        if r["hardest_fields"]:
            print(f"\n  Hardest fields (lowest scores):")
            for field_name, score in r["hardest_fields"]:
                print(f"    {field_name:<45s}  {score:.2f}")

        print("-" * 46)

        total_weighted += r["total_score"]

    overall = total_weighted / len(segment_results) if segment_results else 0.0

    print(f"\n{'=' * 46}")
    print(f"OVERALL WEIGHTED AVERAGE: {overall:.3f}")
    print("=" * 46 + "\n")


if __name__ == "__main__":
    main()
