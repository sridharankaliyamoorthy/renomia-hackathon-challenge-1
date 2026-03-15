"""
Minimal gold-check: compare solver output against ground truth derived
directly from the OCR source text in examples/real_payload.json.

Usage:
    python3 gold_check.py

Checks only the 5 scored fields:
    basic_limit_czk, basic_deductible_czk, premium_czk,
    covered_activities, territorial_scope
"""

import json
import sys

from main import _solve_core

# ---------------------------------------------------------------------------
# Ground truth — values read directly from OCR text in real_payload.json
# ---------------------------------------------------------------------------
GOLD = {
    "allianz_1": {
        "basic_limit_czk":        30_000_000,
        "basic_deductible_czk":   15_000,
        "premium_czk":            72_000,
        "covered_activities":     "provádění stavebních a montážních prací, výstavba bytových a nebytových objektů",
        "territorial_scope":      "Česká republika",
    },
    "kooperativa_1": {
        "basic_limit_czk":        50_000_000,
        "basic_deductible_czk":   10_000,
        "premium_czk":            89_000,
        "covered_activities":     "stavební a montážní práce na pozemních stavbách, rekonstrukce a modernizace stávajících objektů",
        "territorial_scope":      "Evropa",
    },
    "cpp_1": {
        "basic_limit_czk":        25_000_000,
        "basic_deductible_czk":   20_000,
        "premium_czk":            65_000,
        "covered_activities":     "provádění stavebních prací, montáž technologických zařízení",
        "territorial_scope":      "celý svět",
    },
}

SCORED_FIELDS = [
    "basic_limit_czk",
    "basic_deductible_czk",
    "premium_czk",
    "covered_activities",
    "territorial_scope",
]

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_field(field: str, expected, actual) -> str:
    """Return a short status tag for one field."""
    if expected is None and actual is None:
        return "both_null"
    if actual is None:
        return "NULL_vs_expected"
    if expected is None:
        return "unexpected_value"
    if isinstance(expected, int):
        if actual == expected:
            return "MATCH"
        pct = abs(actual - expected) / expected * 100
        return f"NUMERIC_MISMATCH  Δ={actual - expected:+,}  ({pct:.1f}%)"
    # string comparison: case-insensitive, strip
    if str(actual).strip().lower() == str(expected).strip().lower():
        return "MATCH"
    return "STRING_MISMATCH"


def run_gold_check(payload_path: str) -> None:
    with open(payload_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    result = _solve_core(payload)
    offers_by_id = {o["id"]: o for o in (result.get("offers_parsed") or [])}

    sep = "─" * 68
    any_miss = False

    print(f"\n{sep}")
    print("  GOLD CHECK  —  examples/real_payload.json")
    print(sep)

    for offer_id, gold in GOLD.items():
        actual_offer = offers_by_id.get(offer_id) or {}
        print(f"\n  [{offer_id}]")
        for field in SCORED_FIELDS:
            expected = gold.get(field)
            actual   = actual_offer.get(field)
            status   = compare_field(field, expected, actual)
            icon     = "✓" if status == "MATCH" else "✗"
            print(f"    {icon}  {field:<30}  {status}")
            if "MATCH" not in status and "both_null" not in status:
                print(f"         expected : {expected!r}")
                print(f"         actual   : {actual!r}")
                any_miss = True

    print(f"\n{sep}")
    print(f"  Ranking  : {result.get('ranking')}")
    print(f"  Best     : {result.get('best_offer_id')}")
    print(sep)

    if any_miss:
        sys.exit(1)   # non-zero exit so CI can catch regressions


if __name__ == "__main__":
    run_gold_check("examples/real_payload.json")
