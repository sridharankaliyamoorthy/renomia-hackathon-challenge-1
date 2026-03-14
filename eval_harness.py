"""
Developer evaluation harness for the /solve pipeline.

Runs the solve pipeline directly in Python (no HTTP) on a local JSON payload
and prints a compact report covering: ranking, per-offer fields, runtime,
Gemini request count, and cache effectiveness.

Usage:
    python eval_harness.py examples/sample_payload.json
    python eval_harness.py examples/sample_payload.json --repeat 2

The --repeat 2 flag runs the pipeline twice.  On the second pass, if the DB
cache is reachable, Gemini calls should be 0 and runtime should drop
noticeably, demonstrating that the PostgreSQL cache is working.
"""

import argparse
import json
import sys
import time

# Import the pipeline and Gemini tracker directly — no HTTP round-trip.
# FastAPI app object is created at import time but never served here.
from main import _solve_core, gemini


# ---------------------------------------------------------------------------
# Cache status helpers
# ---------------------------------------------------------------------------

_CACHE_LABELS = {
    # (cache_status, gemini_called) → display label
    ("hit", False):           "HIT  (PostgreSQL cache)",
    ("miss", True):           "MISS (Gemini called)",
    ("miss", False):          "MISS (Gemini skipped — local heuristics resolved fields)",
    ("db_unavailable", True):  "DB UNAVAILABLE (Gemini called)",
    ("db_unavailable", False): "DB UNAVAILABLE (Gemini skipped — local heuristics resolved fields)",
}


def _cache_label(cache_status: str, gemini_called: bool) -> str:
    return _CACHE_LABELS.get((cache_status, gemini_called), f"{cache_status.upper()}")


# ---------------------------------------------------------------------------
# Core harness functions
# ---------------------------------------------------------------------------

def load_sample_payload(path: str) -> dict:
    """Load a JSON payload file from disk."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def call_local_solve(payload: dict) -> dict:
    """Invoke the internal pipeline directly (includes _debug; never hits HTTP)."""
    return _solve_core(payload)


def summarize_result(result: dict) -> dict:
    """
    Extract the fields of interest from a solve result.

    Returns a compact dict suitable for reporting — does not alter the
    original result.
    """
    offers_parsed = result.get("offers_parsed") or []
    offer_summaries = []
    for o in offers_parsed:
        offer_summaries.append(
            {
                "id": o.get("id"),
                "insurer": o.get("insurer"),
                "basic_limit_czk": o.get("basic_limit_czk"),
                "basic_deductible_czk": o.get("basic_deductible_czk"),
                "premium_czk": o.get("premium_czk"),
                "covered_activities": o.get("covered_activities"),
                "territorial_scope": o.get("territorial_scope"),
            }
        )
    return {
        "num_offers": len(offers_parsed),
        "ranking": result.get("ranking"),
        "best_offer_id": result.get("best_offer_id"),
        "offers": offer_summaries,
    }


def print_eval_report(
    payload_path: str,
    result: dict,
    elapsed_s: float,
    gemini_before: dict,
    gemini_after: dict,
    run_index: int = 1,
) -> None:
    """Print a compact developer evaluation report to stdout."""
    summary = summarize_result(result)

    # Read accurate cache status from the _debug signal; never infer from
    # Gemini call counts alone.
    debug = result.get("_debug") or {}
    db_available: bool = debug.get("db_available", None)
    debug_offers: list = debug.get("offers") or []
    # Build a lookup by offer id for quick access
    debug_by_id = {d["id"]: d for d in debug_offers}

    calls_this_run = (
        gemini_after["gemini_request_count"] - gemini_before["gemini_request_count"]
    )
    tokens_this_run = (
        gemini_after["total_tokens"] - gemini_before["total_tokens"]
    )

    sep = "─" * 64
    print(f"\n{sep}")
    print(f"  EVAL REPORT  run={run_index}  payload={payload_path}")
    print(sep)
    print(f"  Offers       : {summary['num_offers']}")
    print(f"  Ranking      : {summary['ranking']}")
    print(f"  Best offer   : {summary['best_offer_id']}")
    print()

    for o in summary["offers"]:
        oid = o["id"]
        d = debug_by_id.get(oid) or {}
        cache_status = d.get("cache_status", "unknown")
        gemini_called = d.get("gemini_called", False)
        label = _cache_label(cache_status, gemini_called)

        print(f"  [{oid}]  insurer={o['insurer']}")
        print(f"    basic_limit_czk       : {o['basic_limit_czk']}")
        print(f"    basic_deductible_czk  : {o['basic_deductible_czk']}")
        print(f"    premium_czk           : {o['premium_czk']}")
        print(f"    covered_activities    : {o['covered_activities']}")
        print(f"    territorial_scope     : {o['territorial_scope']}")
        print(f"    cache                 : {label}")

    print()
    print(f"  Runtime      : {elapsed_s:.3f}s")
    print(
        f"  Gemini calls : {calls_this_run} this run  "
        f"(cumulative: {gemini_after['gemini_request_count']})"
    )
    print(
        f"  Tokens used  : {tokens_this_run} this run  "
        f"(cumulative: {gemini_after['total_tokens']})"
    )

    # Summary DB line — sourced from _debug, not inferred from Gemini counts
    if db_available is None:
        print("  DB           : status unknown (no _debug signal)")
    elif not db_available:
        print("  DB           : UNAVAILABLE — caching disabled this run")
    else:
        hits = sum(1 for d in debug_offers if d.get("cache_status") == "hit")
        misses = sum(1 for d in debug_offers if d.get("cache_status") == "miss")
        print(f"  DB           : reachable — {hits} hit(s), {misses} miss(es)")

    print(sep)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline evaluation harness for the /solve pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python eval_harness.py examples/sample_payload.json\n"
            "  python eval_harness.py examples/sample_payload.json --repeat 2\n"
        ),
    )
    parser.add_argument("payload", help="Path to a JSON payload file")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run the pipeline N times (run 2+ demonstrates cache effectiveness)",
    )
    args = parser.parse_args()

    try:
        payload = load_sample_payload(args.payload)
    except FileNotFoundError:
        print(f"ERROR: payload file not found: {args.payload}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON in {args.payload}: {exc}", file=sys.stderr)
        sys.exit(1)

    num_offers = len(payload.get("offers") or [])
    print(
        f"\nLoaded payload: {args.payload}  "
        f"({num_offers} offer(s), segment={payload.get('segment', '?')})"
    )
    print(f"Repeats: {args.repeat}")

    for run_idx in range(1, args.repeat + 1):
        before = gemini.get_metrics()
        t0 = time.perf_counter()
        result = call_local_solve(payload)
        elapsed = time.perf_counter() - t0
        after = gemini.get_metrics()

        print_eval_report(
            payload_path=args.payload,
            result=result,
            elapsed_s=elapsed,
            gemini_before=before,
            gemini_after=after,
            run_index=run_idx,
        )


if __name__ == "__main__":
    main()
